"""Hugging Face checkpoint validation and streaming tensor materialization for Inkling-2-Mini."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from safetensors import safe_open

from . import inventory


_EXPECTED_TEXT_CONFIG = {
    "model_type": "inkling2_loop_text",
    "hidden_size": inventory.HIDDEN_SIZE,
    "num_hidden_layers": inventory.PHYS_LAYERS,
    "vocab_size": inventory.VOCAB_SIZE,
    "unpadded_vocab_size": inventory.UNPADDED_VOCAB_SIZE,
    "num_attention_heads": inventory.NUM_Q_HEADS,
    "num_key_value_heads": inventory.NUM_KV_HEADS,
    "head_dim": inventory.HEAD_DIM,
    "d_rel": inventory.D_REL,
    "rel_extent": inventory.REL_EXTENT,
    "sliding_window_size": inventory.SLIDING_WINDOW,
    "sconv_kernel_size": inventory.SCONV_KERNEL,
    "n_layers_per_attn_res_block": inventory.ATTN_RES_BLOCK_SIZE,
    "n_routed_experts": inventory.NUM_EXPERTS,
    "num_experts_per_tok": inventory.TOP_K,
    "n_shared_experts": 0,
    "intermediate_size": inventory.MOE_INTERMEDIATE,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 8.0,
    "final_logit_softcapping": 30.0,
    "rms_norm_eps": 1e-6,
    "log_scaling_n_floor": 8192,
    "log_scaling_alpha": 0.1,
    "qk_l2_clip_threshold": 16.0 * (2.0**0.5),
    "rel_l2_clip_threshold": 8.0,
    "swa_head_dim": inventory.HEAD_DIM,
    "swa_num_attention_heads": inventory.NUM_Q_HEADS,
    "swa_num_key_value_heads": inventory.NUM_KV_HEADS,
    "q_bias": False,
    "o_bias": False,
    "use_value_norm_no_gamma": True,
    "attn_res_normalize_values": True,
    "use_gate_bias": True,
}

_EXPECTED_LOOP_CONFIG = {
    "core_start": inventory.LOOP_START,
    "core_end": inventory.LOOP_END,
    "num_passes": inventory.LOOP_PASSES,
    "gate_rank": inventory.LOOP_GATE_RANK,
    "is_varloop": False,
}


def _build_e4m3_lut() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint8)
    exponent = ((codes & 0x7F) >> 3).astype(np.int32)
    mantissa = (codes & 0x7).astype(np.float32)
    values = np.where(
        exponent == 0,
        np.ldexp(mantissa, -9),
        np.ldexp(1.0 + mantissa / 8.0, exponent - 7),
    )
    values = np.where((codes & 0x80) != 0, -values, values)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return values.astype(np.float32)


_E4M3_LUT = _build_e4m3_lut()
_E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def decode_nvfp4_expert_np(
    packed_u8: np.ndarray,
    scale_u8: np.ndarray,
    amax_val: float,
    denom_val: float,
    *,
    name: str,
    expert: int,
) -> np.ndarray:
    """Decode one NVFP4 expert weight slice ``[rows, K//2]`` to FP32 ``[rows, K]``."""

    if packed_u8.dtype != np.uint8 or scale_u8.dtype != np.uint8:
        raise TypeError(f"{name} expert {expert}: expected uint8 bytes for NVFP4 decode")
    if not np.isfinite(amax_val) or amax_val < 0.0:
        raise ValueError(f"{name} expert {expert}: invalid NVFP4 amax={amax_val}")
    if not np.isfinite(denom_val) or denom_val <= 0.0:
        raise ValueError(f"{name} expert {expert}: invalid NVFP4 outer_scale_denominator={denom_val}")
    block_scale = _E4M3_LUT[scale_u8]
    if np.isnan(block_scale).any():
        raise ValueError(f"{name} expert {expert}: E4M3FN NaN in NVFP4 block scales")

    rows, half_k = packed_u8.shape
    values = np.empty((rows, half_k * 2), dtype=np.float32)
    values[:, 0::2] = _E2M1_LUT[packed_u8 & 0x0F]
    values[:, 1::2] = _E2M1_LUT[packed_u8 >> 4]
    values.reshape(rows, -1, 16)[:] *= block_scale[:, :, None]
    with np.errstate(over="ignore", invalid="ignore"):
        values *= np.float32(amax_val) / np.float32(denom_val)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} expert {expert}: non-finite decoded NVFP4 weight")
    return values


def decode_mxfp8_expert_np(
    packed_u8: np.ndarray,
    scale_u8: np.ndarray,
    *,
    name: str,
    expert: int,
) -> np.ndarray:
    """Decode one MXFP8 expert weight slice ``[rows, K]`` to FP32 ``[rows, K]``."""

    if packed_u8.dtype != np.uint8 or scale_u8.dtype != np.uint8:
        raise TypeError(f"{name} expert {expert}: expected uint8 bytes for MXFP8 decode")
    if np.any(scale_u8 == 255):
        raise ValueError(f"{name} expert {expert}: UE8M0 NaN (255) in MXFP8 block scales")
    values = _E4M3_LUT[packed_u8]
    if np.isnan(values).any():
        raise ValueError(f"{name} expert {expert}: E4M3FN NaN in MXFP8 weights")
    block_scale = np.ldexp(np.ones(scale_u8.shape, dtype=np.float32), scale_u8.astype(np.int32) - 127)
    rows = values.shape[0]
    with np.errstate(over="ignore", invalid="ignore"):
        values.reshape(rows, -1, 32)[:] *= block_scale[:, :, None]
    if not np.isfinite(values).all():
        raise ValueError(f"{name} expert {expert}: non-finite decoded MXFP8 weight")
    return values


def deinterleave_w13_expert(expert_fp32: np.ndarray) -> np.ndarray:
    """Reorder interleaved ``[2*I, H]`` rows into ``[gate(I); up(I)]``."""

    if expert_fp32.ndim != 2 or (expert_fp32.shape[0] % 2) != 0:
        raise ValueError(f"expected [2*I, H] w13 slice, got {expert_fp32.shape}")
    return np.ascontiguousarray(np.concatenate((expert_fp32[0::2], expert_fp32[1::2]), axis=0))


@dataclass(frozen=True, slots=True)
class CheckpointSources:
    model_dir: Path
    config_summary: dict[str, object]
    expert_quant_schemes: dict[str, str]
    weight_map: dict[str, str]


class Inkling2SourceReader:
    """Lazy shard-cached safetensors reader supporting slice-level expert streaming."""

    def __init__(self, model_dir: str | Path, weight_map: Mapping[str, str]) -> None:
        self.model_dir = Path(model_dir)
        self.weight_map = dict(weight_map)
        self._current_shard: str | None = None
        self._context = None
        self._handle = None
        self._cached_globals: dict[str, torch.Tensor] = {}

    def _open_shard(self, shard: str):
        if shard == self._current_shard:
            return self._handle
        self.close()
        self._context = safe_open(
            str(self.model_dir / shard),
            framework="pt",
            device="cpu",
        )
        self._handle = self._context.__enter__()
        self._current_shard = shard
        return self._handle

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        if name in self._cached_globals:
            return self._cached_globals[name]
        handle = self._open_shard(self.weight_map[name])
        return handle.get_tensor(name)

    def get_cached(self, name: str) -> torch.Tensor:
        if name not in self._cached_globals:
            self._cached_globals[name] = self.get(name)
        return self._cached_globals[name]

    def get_slice(self, name: str):
        handle = self._open_shard(self.weight_map[name])
        return handle.get_slice(name)

    def shape_dtype(self, name: str) -> tuple[tuple[int, ...], str]:
        sl = self.get_slice(name)
        return tuple(sl.get_shape()), str(sl.get_dtype())

    def decode_expert(
        self,
        prefix: str,
        *,
        is_w13: bool,
        expert: int,
        scheme: str,
    ) -> torch.Tensor:
        w_name = f"{prefix}.{'w13_weight' if is_w13 else 'w2_weight'}"
        s_name = f"{w_name}_scale"
        w_slice = self.get_slice(w_name)
        s_slice = self.get_slice(s_name)
        packed_u8 = w_slice[expert].view(torch.uint8).numpy()
        scale_u8 = s_slice[expert].view(torch.uint8).numpy()

        if scheme == "nvfp4_nearest":
            amax_val = float(self.get_cached(f"{w_name}_amax")[expert].item())
            denom_val = float(self.get_cached(f"{w_name}_outer_scale_denominator")[0].item())
            decoded = decode_nvfp4_expert_np(
                packed_u8, scale_u8, amax_val, denom_val, name=w_name, expert=expert
            )
        elif scheme == "mxfp8_nearest":
            decoded = decode_mxfp8_expert_np(packed_u8, scale_u8, name=w_name, expert=expert)
        else:
            raise ValueError(f"unsupported expert scheme {scheme!r} for {w_name}")

        if is_w13:
            decoded = deinterleave_w13_expert(decoded)
        return torch.from_numpy(decoded)

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
        self._current_shard = None
        self._context = None
        self._handle = None

    def __enter__(self) -> "Inkling2SourceReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def _resolve_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        return dict(data["weight_map"])
    single = model_dir / "model-00001-of-00001.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return {k: single.name for k in handle.keys()}
    raise FileNotFoundError(f"no safetensors index or weights found in {model_dir}")


def validate_hf_checkpoint(model_dir: str | Path) -> CheckpointSources:
    root = Path(model_dir)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "inkling2_loop":
        raise ValueError(f"expected model_type='inkling2_loop', got {config.get('model_type')!r}")
    if config.get("eos_token_id") != 200006:
        raise ValueError(f"expected eos_token_id=200006, got {config.get('eos_token_id')!r}")
    text = config.get("text_config", {})
    for key, expected in _EXPECTED_TEXT_CONFIG.items():
        actual = text.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-5:
                raise ValueError(f"text_config.{key}: expected {expected!r}, got {actual!r}")
        elif actual != expected:
            raise ValueError(f"text_config.{key}: expected {expected!r}, got {actual!r}")
    if text.get("loop_config") != _EXPECTED_LOOP_CONFIG:
        raise ValueError(f"text_config.loop_config mismatch: {text.get('loop_config')!r}")
    expected_local = sorted(inventory.SWA_LAYERS)
    if sorted(text.get("local_layer_ids", ())) != expected_local:
        raise ValueError("text_config.local_layer_ids does not match 5:1 SWA schedule")

    quant_cfg = json.loads((root / "hf_quant_config.json").read_text(encoding="utf-8"))
    expert_schemes: dict[str, str] = {}
    for group in quant_cfg.get("model", {}).get("parameter_groups", []):
        scheme = group.get("weight_transform")
        for tensor_name in group.get("tensors", []):
            if tensor_name.endswith((".w13_weight", ".w2_weight")):
                if any(p in tensor_name for p in ("visual.", "vision_tower.", "mtp.")):
                    continue
                expert_schemes[tensor_name] = scheme

    for layer in range(inventory.PHYS_LAYERS):
        expected_scheme = "nvfp4_nearest" if layer in inventory.Q3_MOE_LAYERS else "mxfp8_nearest"
        for suffix in ("w13_weight", "w2_weight"):
            tname = f"model.llm.layers.{layer}.mlp.experts.{suffix}"
            if expert_schemes.get(tname) != expected_scheme:
                raise ValueError(
                    f"{tname}: expected hf_quant_config scheme {expected_scheme!r}, "
                    f"got {expert_schemes.get(tname)!r}"
                )

    weight_map = _resolve_weight_map(root)
    return CheckpointSources(
        model_dir=root,
        config_summary={"model_type": config["model_type"], "text_config": text},
        expert_quant_schemes=expert_schemes,
        weight_map=weight_map,
    )


def load_frontend_resources(model_dir: str | Path) -> dict[str, bytes]:
    root = Path(model_dir)
    resources: dict[str, bytes] = {}
    for spec in inventory.RESOURCE_SPECS:
        filename = spec.name.removeprefix("frontend/")
        path = root / filename
        if path.is_file():
            data = path.read_bytes()
        elif filename == "chat_template.jinja":
            tc = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
            tmpl = tc.get("chat_template")
            if not isinstance(tmpl, str) or not tmpl:
                raise FileNotFoundError("neither chat_template.jinja nor tokenizer_config.chat_template exists")
            data = tmpl.encode("utf-8")
        else:
            raise FileNotFoundError(f"missing required frontend resource: {path}")
        if not data:
            raise ValueError(f"empty frontend resource: {spec.name}")
        resources[spec.name] = data
    return resources


def load_non_expert_tensor(spec: inventory.TensorSpec, reader: Inkling2SourceReader) -> torch.Tensor:
    """Load and transform one non-MoE-expert tensor from the HF checkpoint."""

    name = spec.name
    if name == "text/token_embedding":
        return reader.get("model.llm.embed.weight")
    if name == "text/token_norm":
        return reader.get("model.llm.embed_norm.weight").to(torch.bfloat16).contiguous()
    if name == "text/output_res_query":
        return reader.get("model.llm.attn_res.output_query").to(torch.float32).contiguous()
    if name == "text/final_norm":
        return reader.get("model.llm.norm.weight").to(torch.bfloat16).contiguous()
    if name == "text/output_head":
        return reader.get("model.llm.unembed.weight")

    parts = name.split("/")
    if len(parts) < 4 or parts[0] != "text" or parts[1] != "layers":
        raise ValueError(f"unexpected tensor name: {name}")
    layer = int(parts[2])
    suffix = "/".join(parts[3:])
    hp = f"model.llm.layers.{layer}."
    full = inventory.is_full_attention_layer(layer)

    if suffix == "attention/res_query":
        raw = reader.get(f"model.llm.attn_res.attn_queries.{layer - 1}").to(torch.float32)
        return raw.reshape(1, inventory.HIDDEN_SIZE).contiguous() if not full else raw.contiguous()
    if suffix == "attention/input_norm":
        return reader.get(hp + "attn_norm.weight").to(torch.bfloat16).contiguous()
    if suffix == "attention/q_input_norm":
        return reader.get(hp + "q_input_norm.weight").to(torch.bfloat16).contiguous()
    if suffix == "attention/k_input_norm":
        return reader.get(hp + "k_input_norm.weight").to(torch.bfloat16).contiguous()
    if suffix == "attention/v_input_norm":
        return reader.get(hp + "v_input_norm.weight").to(torch.bfloat16).contiguous()
    if suffix == "attention/qgr":
        wq = reader.get(hp + "attn.wq_du.weight")
        gate = reader.get(hp + "attn.gate_attn.weight")
        wr = reader.get(hp + "attn.wr_du.weight")
        return torch.cat((wq, gate, wr), dim=0).contiguous()
    if suffix == "attention/k":
        return reader.get(hp + "attn.wk_dv.weight")
    if suffix == "attention/v":
        return reader.get(hp + "attn.wv_dv.weight")
    if suffix == "attention/output":
        return reader.get(hp + "attn.wo_ud.weight")
    if suffix == "attention/sink_proj":
        return reader.get(hp + "attn.attn_sink_proj.weight").to(torch.bfloat16).contiguous()
    if suffix == "attention/q_clip":
        return reader.get(hp + "attn.q_l2_clip.scale").to(torch.float32).contiguous()
    if suffix == "attention/rel_clip":
        return reader.get(hp + "attn.rel_l2_clip.scale").to(torch.float32).contiguous()
    if suffix == "attention/rel_bias":
        # HF stores [d_rel=16, rel_extent=1024]; NInfer stores row-major [1024, 16].
        return reader.get(hp + "attn.rel_logits_proj.proj").T.contiguous().to(torch.float32)
    if suffix in ("attention/k_conv", "attention/v_conv"):
        kind = "k" if suffix.endswith("k_conv") else "v"
        static_key = hp + f"attn.{kind}_sconv.static_w"
        weight_key = hp + f"attn.{kind}_sconv.weight"
        if reader.has(static_key):
            # HF static_w is [4, C] in lag order -> flip(0).T yields chronological [C, 4].
            t = reader.get(static_key).flip(0).T.contiguous()
        else:
            t = reader.get(weight_key)
            if t.ndim == 3:
                t = t.squeeze(1)
        return t.to(torch.float32).contiguous()
    if suffix == "attention/k_conv_proj":
        return reader.get(hp + "attn.k_sconv.weight_proj").to(torch.bfloat16).contiguous()
    if suffix == "attention/v_conv_proj":
        return reader.get(hp + "attn.v_sconv.weight_proj").to(torch.bfloat16).contiguous()
    if suffix == "attention/post_conv":
        t = reader.get(hp + "attn_sconv.weight")
        if t.ndim == 3:
            t = t.squeeze(1)
        return t.to(torch.float32).contiguous()
    if suffix == "loop/msg_query":
        idx = layer - inventory.LOOP_START
        q = reader.get_cached("model.llm.middle_cycle_cross_pass_queries")
        kv = reader.get_cached("model.llm.middle_cycle_cross_pass_kv_queries")
        # 4-way stack: [Q, K, V, FFN]
        return torch.stack((q[2 * idx], kv[2 * idx], kv[2 * idx + 1], q[2 * idx + 1]), dim=0).to(
            torch.float32
        ).contiguous()
    if suffix == "loop/ctrl_down":
        parts_list = [
            reader.get(hp + f"controllers.{c}.down_weight").to(torch.bfloat16)
            for c in ("attn", "k", "v", "mlp")
        ]
        return torch.stack(parts_list, dim=0).contiguous()
    if suffix == "loop/ctrl_gain_forget":
        parts_list = [
            torch.stack(
                (
                    reader.get(hp + f"controllers.{c}.gain_weight").to(torch.bfloat16),
                    reader.get(hp + f"controllers.{c}.forget_weight").to(torch.bfloat16),
                ),
                dim=0,
            )
            for c in ("attn", "k", "v", "mlp")
        ]
        return torch.stack(parts_list, dim=0).contiguous()
    if suffix == "ffn/res_query":
        return reader.get(f"model.llm.attn_res.mlp_queries.{layer}").to(torch.float32).contiguous()
    if suffix == "ffn/input_norm":
        return reader.get(hp + "mlp_norm.weight").to(torch.bfloat16).contiguous()
    if suffix == "ffn/router":
        return reader.get(hp + "mlp.gate.weight").to(torch.bfloat16).contiguous()
    if suffix == "ffn/exp_probs_bias":
        bkey = hp + "mlp.gate.bias"
        if reader.has(bkey):
            return reader.get(bkey).to(torch.float32).contiguous()
        return torch.zeros((inventory.NUM_EXPERTS,), dtype=torch.float32)
    if suffix == "ffn/post_conv":
        t = reader.get(hp + "mlp_sconv.weight")
        if t.ndim == 3:
            t = t.squeeze(1)
        return t.to(torch.float32).contiguous()

    raise ValueError(f"unhandled non-expert tensor {name}")


DEFAULT_WORKSPACE_MTP = Path(r"D:\Users\yaldi\gits\ninfer-4080\models\mtp.safetensors")


def resolve_mtp_path(
    model_dir: str | Path,
    mtp_path: str | Path | None = None,
    *,
    allow_missing: bool = False,
) -> Path | None:
    if mtp_path is not None:
        resolved = Path(mtp_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"MTP safetensors file not found: {resolved}")
        return resolved
    for candidate in (DEFAULT_WORKSPACE_MTP, Path(model_dir) / "mtp.safetensors"):
        if candidate.is_file():
            return candidate
    if allow_missing:
        return None
    raise FileNotFoundError(
        f"mtp.safetensors not found in {DEFAULT_WORKSPACE_MTP} or {Path(model_dir) / 'mtp.safetensors'}"
    )


class MtpSourceReader:
    """Safetensors reader for DSpark mtp.safetensors."""

    def __init__(self, mtp_path: str | Path) -> None:
        self.mtp_path = Path(mtp_path)
        self._context = safe_open(str(self.mtp_path), framework="pt", device="cpu")
        self._handle = self._context.__enter__()

    def get(self, name: str) -> torch.Tensor:
        return self._handle.get_tensor(name)

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None
            self._handle = None

    def __enter__(self) -> "MtpSourceReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def load_mtp_tensor(spec: inventory.TensorSpec, reader: MtpSourceReader) -> torch.Tensor:
    name = spec.name
    if name == "mtp/fuse_weight":
        return reader.get("fuse_weight").to(torch.float32).contiguous()
    if name == "mtp/mask_embed":
        return reader.get("mask_embed").to(torch.bfloat16).contiguous()
    if name == "mtp/final_norm":
        return reader.get("norm.weight").to(torch.bfloat16).contiguous()
    if name == "mtp/markov/dflare_norm":
        return reader.get("markov_head.dflare_norm.weight").to(torch.bfloat16).contiguous()
    if name == "mtp/markov/w1":
        return reader.get("markov_head.markov_w1.weight").to(torch.bfloat16).contiguous()
    if name == "mtp/markov/rnn_proj_weight":
        return reader.get("markov_head.rnn_proj.weight").to(torch.bfloat16).contiguous()
    if name == "mtp/markov/rnn_proj_bias":
        return reader.get("markov_head.rnn_proj.bias").to(torch.float32).contiguous()
    if name == "mtp/markov/hidden_out_weight":
        return reader.get("markov_head.hidden_out.weight").to(torch.bfloat16).contiguous()
    if name == "mtp/markov/hidden_out_bias":
        return reader.get("markov_head.hidden_out.bias").to(torch.float32).contiguous()

    parts = name.split("/")
    if len(parts) < 4 or parts[0] != "mtp" or parts[1] != "layers":
        raise ValueError(f"unexpected MTP tensor name: {name}")
    layer = int(parts[2])
    suffix = "/".join(parts[3:])
    hp = f"layers.{layer}."
    mapping = {
        "dflare_norm": ("dflare_norm.weight", torch.bfloat16),
        "input_norm": ("input_layernorm.weight", torch.bfloat16),
        "post_attention_norm": ("post_attention_layernorm.weight", torch.bfloat16),
        "attention/q_norm": ("self_attn.q_norm.weight", torch.bfloat16),
        "attention/k_norm": ("self_attn.k_norm.weight", torch.bfloat16),
        "attention/k_norm_ctx": ("self_attn.k_norm_ctx.weight", torch.bfloat16),
        "attention/q": ("self_attn.q_proj.weight", None),
        "attention/k": ("self_attn.k_proj.weight", None),
        "attention/v": ("self_attn.v_proj.weight", None),
        "attention/k_ctx": ("self_attn.k_proj_ctx.weight", None),
        "attention/v_ctx": ("self_attn.v_proj_ctx.weight", None),
        "attention/output": ("self_attn.o_proj.weight", None),
        "mlp/down": ("mlp.down_proj.weight", None),
    }
    if suffix == "mlp/gate_up":
        gate = reader.get(hp + "mlp.gate_proj.weight")
        up = reader.get(hp + "mlp.up_proj.weight")
        return torch.cat((gate, up), dim=0).contiguous()
    if suffix in mapping:
        key, dtype = mapping[suffix]
        t = reader.get(hp + key)
        return t.to(dtype).contiguous() if dtype is not None else t
    raise ValueError(f"unhandled MTP tensor {name}")


def compute_dspark_draft_head_ids() -> torch.Tensor:
    """Deterministic 32,768-token shortlist: special tokens [200000..200058) + top BPE IDs."""

    specials = list(range(200000, inventory.UNPADDED_VOCAB_SIZE))
    bpe_count = inventory.MTP_DRAFT_HEAD_ROWS - len(specials)
    ids = list(range(bpe_count)) + specials
    assert len(ids) == inventory.MTP_DRAFT_HEAD_ROWS
    return torch.tensor(ids, dtype=torch.int32)


def load_mtp_draft_head(reader: Inkling2SourceReader, ids: torch.Tensor) -> torch.Tensor:
    unembed = reader.get("model.llm.unembed.weight")
    return unembed.index_select(0, ids.to(torch.long)).contiguous()


DEFAULT_WORKSPACE_DFLASH2 = Path(r"D:\Users\yaldi\gits\ninfer-4080\models\Inkling-2-Mini-DFlash2")

_EXPECTED_DFLASH2_CONFIG = {
    "hidden_size": inventory.HIDDEN_SIZE,
    "intermediate_size": inventory.DFLASH2_INTERMEDIATE,
    "num_hidden_layers": inventory.DFLASH2_LAYERS,
    "num_attention_heads": inventory.DFLASH2_Q_HEADS,
    "num_key_value_heads": inventory.NUM_KV_HEADS,
    "head_dim": inventory.HEAD_DIM,
    "sliding_window": 2048,
    "rms_norm_eps": 1e-6,
    "vocab_size": inventory.VOCAB_SIZE,
}

_EXPECTED_DFLASH2_SUBCONFIG = {
    "target_layer_ids": [5, 23, 41, 48],
    "mask_token_id": 200056,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "selector_rank": inventory.DFLASH2_SELECTOR_RANK,
    "selector_top_k": 16,
    "selector_softcap_mode": "after_correction",
    "final_logit_softcapping": 30.0,
}


def resolve_dflash2_dir(
    dflash2_dir: str | Path | None = None,
    *,
    allow_missing: bool = False,
) -> Path | None:
    if dflash2_dir is not None:
        resolved = Path(dflash2_dir)
        if not resolved.is_dir():
            raise FileNotFoundError(f"DFlash2 directory not found: {resolved}")
        return resolved
    if DEFAULT_WORKSPACE_DFLASH2.is_dir():
        return DEFAULT_WORKSPACE_DFLASH2
    if allow_missing:
        return None
    raise FileNotFoundError(f"DFlash2 directory not found: {DEFAULT_WORKSPACE_DFLASH2}")


def validate_dflash2_checkpoint(dflash2_dir: str | Path) -> Path:
    root = Path(dflash2_dir)
    config_path = root / "config.json"
    weights_path = root / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing DFlash2 config.json: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"missing DFlash2 model.safetensors: {weights_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if "DFlash2DraftModel" not in architectures:
        raise ValueError(f"expected DFlash2DraftModel in architectures, got {architectures!r}")
    for key, expected in _EXPECTED_DFLASH2_CONFIG.items():
        actual = config.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-5:
                raise ValueError(f"DFlash2 config.{key}: expected {expected!r}, got {actual!r}")
        elif actual != expected:
            raise ValueError(f"DFlash2 config.{key}: expected {expected!r}, got {actual!r}")

    rope_theta = config.get("rope_theta")
    if rope_theta is None:
        rope_theta = (config.get("rope_parameters") or {}).get("rope_theta")
    if rope_theta is None or abs(float(rope_theta) - 10000.0) > 1e-5:
        raise ValueError(f"DFlash2 config.rope_theta: expected 10000.0, got {rope_theta!r}")

    dflash_cfg = config.get("dflash_config") or {}
    for key, expected in _EXPECTED_DFLASH2_SUBCONFIG.items():
        actual = dflash_cfg.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-5:
                raise ValueError(
                    f"DFlash2 config.dflash_config.{key}: expected {expected!r}, got {actual!r}"
                )
        elif actual != expected:
            raise ValueError(
                f"DFlash2 config.dflash_config.{key}: expected {expected!r}, got {actual!r}"
            )

    expected_shapes: dict[str, tuple[int, ...]] = {
        "fc.weight": (inventory.HIDDEN_SIZE, inventory.DFLASH2_FEATURE_WIDTH),
        "hidden_norm.weight": (inventory.HIDDEN_SIZE,),
        "norm.weight": (inventory.HIDDEN_SIZE,),
        "candidate_selector.hidden_projection.weight": (
            inventory.DFLASH2_SELECTOR_RANK,
            inventory.HIDDEN_SIZE,
        ),
        "candidate_selector.predecessor_codebook": (
            inventory.VOCAB_SIZE,
            inventory.DFLASH2_SELECTOR_RANK,
        ),
        "candidate_selector.successor_codebook": (
            inventory.VOCAB_SIZE,
            inventory.DFLASH2_SELECTOR_RANK,
        ),
    }
    for layer in range(inventory.DFLASH2_LAYERS):
        lp = f"layers.{layer}."
        expected_shapes.update(
            {
                lp + "input_layernorm.weight": (inventory.HIDDEN_SIZE,),
                lp + "post_attention_layernorm.weight": (inventory.HIDDEN_SIZE,),
                lp + "attention_conv.base_kernel": (2, 2, inventory.HIDDEN_SIZE),
                lp + "attention_conv.kernel_projection.weight": (
                    inventory.DFLASH2_CONV_PROJ_ROWS,
                    inventory.HIDDEN_SIZE,
                ),
                lp + "self_attn.q_norm.weight": (inventory.HEAD_DIM,),
                lp + "self_attn.k_norm.weight": (inventory.HEAD_DIM,),
                lp + "self_attn.q_proj.weight": (inventory.DFLASH2_Q_DIM, inventory.HIDDEN_SIZE),
                lp + "self_attn.k_proj.weight": (inventory.DFLASH2_KV_DIM, inventory.HIDDEN_SIZE),
                lp + "self_attn.v_proj.weight": (inventory.DFLASH2_KV_DIM, inventory.HIDDEN_SIZE),
                lp + "self_attn.o_proj.weight": (inventory.HIDDEN_SIZE, inventory.DFLASH2_Q_DIM),
                lp + "mlp_conv.base_kernel": (2, 2, inventory.HIDDEN_SIZE),
                lp + "mlp_conv.kernel_projection.weight": (
                    inventory.DFLASH2_CONV_PROJ_ROWS,
                    inventory.HIDDEN_SIZE,
                ),
                lp + "mlp.gate_proj.weight": (inventory.DFLASH2_INTERMEDIATE, inventory.HIDDEN_SIZE),
                lp + "mlp.up_proj.weight": (inventory.DFLASH2_INTERMEDIATE, inventory.HIDDEN_SIZE),
                lp + "mlp.down_proj.weight": (inventory.HIDDEN_SIZE, inventory.DFLASH2_INTERMEDIATE),
            }
        )

    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != set(expected_shapes):
            missing = sorted(set(expected_shapes) - keys)
            extra = sorted(keys - set(expected_shapes))
            raise ValueError(
                f"DFlash2 safetensors inventory mismatch: missing={missing[:4]}, extra={extra[:4]}"
            )
        for name, shape in expected_shapes.items():
            actual_shape = tuple(handle.get_slice(name).get_shape())
            if actual_shape != shape:
                raise ValueError(f"DFlash2 {name}: shape {actual_shape} != {shape}")
    return weights_path


class DFlash2SourceReader:
    """Safetensors reader for Inkling-2-Mini-DFlash2 model.safetensors."""

    def __init__(self, dflash2_dir: str | Path) -> None:
        self.dflash2_dir = Path(dflash2_dir)
        self.weights_path = validate_dflash2_checkpoint(self.dflash2_dir)
        self._context = safe_open(str(self.weights_path), framework="pt", device="cpu")
        self._handle = self._context.__enter__()

    def get(self, name: str) -> torch.Tensor:
        return self._handle.get_tensor(name)

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None
            self._handle = None

    def __enter__(self) -> "DFlash2SourceReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def load_dflash2_tensor(spec: inventory.TensorSpec, reader: DFlash2SourceReader) -> torch.Tensor:
    name = spec.name
    direct_top = {
        "dflash2/fc": "fc.weight",
        "dflash2/hidden_norm": "hidden_norm.weight",
        "dflash2/final_norm": "norm.weight",
        "dflash2/selector/hidden_projection": "candidate_selector.hidden_projection.weight",
        "dflash2/selector/predecessor_codebook": "candidate_selector.predecessor_codebook",
        "dflash2/selector/successor_codebook": "candidate_selector.successor_codebook",
    }
    if name in direct_top:
        t = reader.get(direct_top[name]).to(torch.bfloat16).contiguous()
        if tuple(t.shape) != spec.shape:
            raise ValueError(f"{name}: shape {tuple(t.shape)} != {spec.shape}")
        return t

    parts = name.split("/")
    if len(parts) < 4 or parts[0] != "dflash2" or parts[1] != "layers":
        raise ValueError(f"unexpected DFlash2 tensor name: {name}")
    layer = int(parts[2])
    suffix = "/".join(parts[3:])
    hp = f"layers.{layer}."
    mapping = {
        "input_norm": "input_layernorm.weight",
        "post_attention_norm": "post_attention_layernorm.weight",
        "attention_conv/base_kernel": "attention_conv.base_kernel",
        "attention_conv/kernel_projection": "attention_conv.kernel_projection.weight",
        "attention/q_norm": "self_attn.q_norm.weight",
        "attention/k_norm": "self_attn.k_norm.weight",
        "attention/q": "self_attn.q_proj.weight",
        "attention/k": "self_attn.k_proj.weight",
        "attention/v": "self_attn.v_proj.weight",
        "attention/output": "self_attn.o_proj.weight",
        "mlp_conv/base_kernel": "mlp_conv.base_kernel",
        "mlp_conv/kernel_projection": "mlp_conv.kernel_projection.weight",
        "mlp/down": "mlp.down_proj.weight",
    }
    if suffix == "mlp/gate_up":
        gate = reader.get(hp + "mlp.gate_proj.weight")
        up = reader.get(hp + "mlp.up_proj.weight")
        t = torch.cat((gate, up), dim=0).to(torch.bfloat16).contiguous()
    elif suffix in mapping:
        t = reader.get(hp + mapping[suffix]).to(torch.bfloat16).contiguous()
    else:
        raise ValueError(f"unhandled DFlash2 tensor {name}")

    if tuple(t.shape) != spec.shape:
        raise ValueError(f"{name}: shape {tuple(t.shape)} != {spec.shape}")
    return t
