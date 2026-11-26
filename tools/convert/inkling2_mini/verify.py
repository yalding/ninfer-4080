"""Verify C:\\LLM\\inkling2_mini_q3g128.ninfer structure, direct bytes, and quantized SNR."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
    object_alignment,
)
from tools.artifact.layouts import (
    align_up,
    decode_direct,
    dequantize_row_split,
    encoded_size,
    row_split_geometry,
    split_row_planes,
)

from . import inventory, source


class VerificationError(ValueError):
    """Raised when an Inkling-2-Mini artifact violates its contract."""


@dataclass(frozen=True, slots=True)
class VerifySummary:
    model_id: str
    weights_id: str
    file_bytes: int
    objects: int
    tensors: int
    resources: int
    direct_exact_checked: int
    quantized_checked: int
    min_q3_cosine: float
    min_q4_cosine: float
    min_q6_cosine: float


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = float(af.norm().item() * bf.norm().item())
    if denom == 0.0:
        return 1.0 if float(af.norm().item()) == float(bf.norm().item()) else 0.0
    return float(torch.dot(af, bf).item() / denom)


def _verify_q3_canonical(stored: torch.Tensor, reference: torch.Tensor, label: str) -> None:
    """Check sampled Q3 rows against an independent host arithmetic oracle.

    MAXABS_F16_RECIP_RNE_V1 has a source-distribution-dependent cosine. In particular,
    real upstream NVFP4 expert rows can have cosine below .970 even when their Q3
    conversion is exact. Verify the locked codec instead of treating that arbitrary
    cosine floor as an encoding invariant. Model quality remains a perplexity gate.
    This intentionally does not call the converter quantizer or encoder.
    """
    values = reference.detach().cpu().float()
    if values.ndim != 2 or values.shape[1] % 128 or not torch.isfinite(values).all():
        raise VerificationError(f"{label}: invalid canonical Q3 source")
    grouped = values.reshape(values.shape[0], -1, 128)
    maximum = grouped.abs().amax(-1)
    scale = (maximum.double() / 3.0).float().half()
    underflow = (scale == 0) & (maximum > 0)
    scale[underflow] = 2.0 ** -24
    if not torch.isfinite(scale).all() or ((scale <= 0) & (maximum > 0)).any():
        raise VerificationError(f"{label}: invalid canonical Q3 scale")
    reciprocal = torch.zeros_like(maximum)
    positive = scale > 0
    reciprocal[positive] = (1.0 / scale[positive].double()).float()
    code = torch.round(grouped * reciprocal.unsqueeze(-1)).clamp(-4, 3)
    expected = (code * scale.float().unsqueeze(-1)).reshape_as(values)
    if not torch.equal(stored.detach().cpu().float(), expected):
        raise VerificationError(f"{label}: sampled Q3 differs from canonical maxabs/RNE conversion")


def _dequantize_rows(
    payload: memoryview,
    fmt: str,
    shape: tuple[int, int],
    rows: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    geom = row_split_geometry(fmt, shape)
    planes = split_row_planes(payload, geom)
    out_rows: list[torch.Tensor] = []
    try:
        for row in rows:
            sub_base = bytes(planes.base[row * geom.base_row_bytes : (row + 1) * geom.base_row_bytes])
            sub_high = bytes(planes.high[row * geom.high_row_bytes : (row + 1) * geom.high_row_bytes])
            sub_scale = bytes(planes.scale[row * geom.scale_row_bytes : (row + 1) * geom.scale_row_bytes])
            from tools.artifact.layouts import RowPlanes

            sub_planes = RowPlanes(sub_base, sub_high, sub_scale, 1)
            deq = dequantize_row_split(sub_planes, fmt, (1, shape[1]), device=device, dtype=torch.float32)
            out_rows.append(deq[0])
    finally:
        for plane in (planes.base, planes.high, planes.scale):
            if isinstance(plane, memoryview):
                plane.release()
        payload.release()
    return torch.stack(out_rows, dim=0)


def validate_structure(artifact: Artifact) -> tuple[bool, bool, bool]:
    expected_identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    if artifact.identity != expected_identity:
        raise VerificationError(f"identity {artifact.identity!r} != {expected_identity!r}")
    has_mtp = any(obj.name.startswith("mtp/") for obj in artifact.objects)
    has_draft_head = any(obj.name == "mtp/draft_head" for obj in artifact.objects)
    has_dflash2 = any(obj.name.startswith("dflash2/") for obj in artifact.objects)
    expected_objects = inventory.object_specs(
        include_mtp=has_mtp, include_draft_head=has_draft_head, include_dflash2=has_dflash2
    )
    expected_tensors = inventory.tensor_specs(
        include_mtp=has_mtp, include_draft_head=has_draft_head, include_dflash2=has_dflash2
    )
    if len(artifact.objects) != len(expected_objects):
        raise VerificationError(
            f"object count {len(artifact.objects)} != {len(expected_objects)}"
        )

    cursor = 0
    formats: Counter[str] = Counter()
    layouts: Counter[str] = Counter()
    for i, (actual, expected) in enumerate(zip(artifact.objects, expected_objects)):
        if actual.name != expected.name:
            raise VerificationError(f"object {i}: {actual.name!r} != {expected.name!r}")
        expected_offset = align_up(cursor, object_alignment(actual))
        if actual.offset != expected_offset:
            raise VerificationError(f"{actual.name}: offset {actual.offset} != {expected_offset}")
        if isinstance(expected, inventory.TensorSpec):
            if not isinstance(actual, TensorObject):
                raise VerificationError(f"{actual.name}: expected tensor")
            if (actual.shape, actual.format, actual.layout) != (
                expected.shape,
                expected.format,
                expected.layout,
            ):
                raise VerificationError(f"{actual.name}: signature mismatch")
            if actual.bytes != encoded_size(actual.layout, actual.format, actual.shape):
                raise VerificationError(f"{actual.name}: byte length mismatch")
            formats[actual.format] += 1
            layouts[actual.layout] += 1
        else:
            if not isinstance(actual, ResourceObject):
                raise VerificationError(f"{actual.name}: expected resource")
            if actual.encoding != expected.encoding:
                raise VerificationError(f"{actual.name}: encoding mismatch")
        cursor = actual.offset + actual.bytes

    expected_format_counts = {
        k: v for k, v in inventory.format_counts(expected_tensors).items() if v > 0
    }
    if dict(formats) != expected_format_counts:
        raise VerificationError(f"format counts mismatch: {dict(formats)}")
    if dict(layouts) != inventory.layout_counts(expected_tensors):
        raise VerificationError(f"layout counts mismatch: {dict(layouts)}")
    if cursor != artifact.file_bytes - artifact.payload_offset:
        raise VerificationError("artifact trailing padding / size mismatch")
    return has_mtp, has_draft_head, has_dflash2


def verify_artifact(
    artifact_path: str | Path,
    model_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    mtp_path: str | Path | None = None,
    dflash2_dir: str | Path | None = None,
) -> VerifySummary:
    sources = source.validate_hf_checkpoint(model_dir)
    expected_resources = source.load_frontend_resources(sources.model_dir)
    target = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")

    probe_layers = (0, 5, 6, 11, 17, 18, 20, 23)
    direct_checked = 0
    quantized_checked = 0
    min_q3 = 1.0
    min_q4 = 1.0
    min_q6 = 1.0

    with Artifact.open(artifact_path) as artifact:
        has_mtp, has_draft_head, has_dflash2 = validate_structure(artifact)
        t_specs = inventory.tensor_specs(
            include_mtp=has_mtp, include_draft_head=has_draft_head, include_dflash2=has_dflash2
        )
        draft_head_ids = source.compute_dspark_draft_head_ids() if has_draft_head else None
        mtp_reader = (
            source.MtpSourceReader(source.resolve_mtp_path(sources.model_dir, mtp_path))
            if has_mtp
            else None
        )
        dflash2_reader = (
            source.DFlash2SourceReader(source.resolve_dflash2_dir(dflash2_dir))
            if has_dflash2
            else None
        )

        for name, expected_bytes in expected_resources.items():
            actual_bytes = bytes(artifact.payload(name))
            if actual_bytes != expected_bytes:
                raise VerificationError(f"resource mismatch: {name}")

        try:
            with source.Inkling2SourceReader(sources.model_dir, sources.weight_map) as reader:
                # Verify ALL direct tensors (BF16/FP32) word-for-word against source.
                for spec in t_specs:
                    if spec.format not in inventory.DIRECT_FORMATS:
                        continue
                    stored = decode_direct(artifact.payload(spec.name), spec.format, spec.shape)
                    if spec.name == "mtp/draft_head_token_ids":
                        assert draft_head_ids is not None
                        expected = draft_head_ids
                    elif spec.name.startswith("mtp/"):
                        assert mtp_reader is not None
                        expected = source.load_mtp_tensor(spec, mtp_reader)
                    elif spec.name.startswith("dflash2/"):
                        assert dflash2_reader is not None
                        expected = source.load_dflash2_tensor(spec, dflash2_reader)
                    else:
                        expected = source.load_non_expert_tensor(spec, reader)
                    if not torch.isfinite(stored).all():
                        raise VerificationError(f"{spec.name}: stored non-finite value")
                    view_dtype = torch.int16 if spec.format == inventory.BF16 else torch.int32
                    if not torch.equal(stored.view(view_dtype), expected.view(view_dtype)):
                        raise VerificationError(f"{spec.name}: exact word mismatch vs source")
                    direct_checked += 1

                if mtp_reader is not None:
                    for spec in inventory.MTP_TENSOR_SPECS:
                        if spec.format in inventory.DIRECT_FORMATS:
                            continue
                        obj = artifact.find(spec.name)
                        assert isinstance(obj, TensorObject)
                        rows = (0, obj.shape[0] // 2, obj.shape[0] - 1)
                        deq = _dequantize_rows(
                            artifact.payload(obj), obj.format, obj.shape, rows, target
                        ).cpu()
                        src_t = source.load_mtp_tensor(spec, mtp_reader)
                        ref = src_t.index_select(0, torch.tensor(rows, dtype=torch.long)).float()
                        cos = _cosine_similarity(deq, ref)
                        if cos < 0.985:
                            raise VerificationError(f"{spec.name}: cosine {cos:.5f} < 0.985")
                        min_q4 = min(min_q4, cos)
                        quantized_checked += 1

                if dflash2_reader is not None:
                    for spec in inventory.DFLASH2_TENSOR_SPECS:
                        if spec.format in inventory.DIRECT_FORMATS:
                            continue
                        obj = artifact.find(spec.name)
                        assert isinstance(obj, TensorObject)
                        rows = (0, obj.shape[0] // 2, obj.shape[0] - 1)
                        deq = _dequantize_rows(
                            artifact.payload(obj), obj.format, obj.shape, rows, target
                        ).cpu()
                        src_t = source.load_dflash2_tensor(spec, dflash2_reader)
                        ref = src_t.index_select(0, torch.tensor(rows, dtype=torch.long)).float()
                        cos = _cosine_similarity(deq, ref)
                        if cos < 0.985:
                            raise VerificationError(f"{spec.name}: cosine {cos:.5f} < 0.985")
                        min_q4 = min(min_q4, cos)
                        quantized_checked += 1

                if has_draft_head:
                    assert draft_head_ids is not None
                    obj = artifact.find("mtp/draft_head")
                    assert isinstance(obj, TensorObject)
                    rows = (0, obj.shape[0] // 2, obj.shape[0] - 1)
                    deq = _dequantize_rows(
                        artifact.payload(obj), obj.format, obj.shape, rows, target
                    ).cpu()
                    src_t = source.load_mtp_draft_head(reader, draft_head_ids)
                    ref = src_t.index_select(0, torch.tensor(rows, dtype=torch.long)).float()
                    cos = _cosine_similarity(deq, ref)
                    if cos < 0.985:
                        raise VerificationError(f"mtp/draft_head: cosine {cos:.5f} < 0.985")
                    min_q4 = min(min_q4, cos)
                    quantized_checked += 1

                # Sampled verification on embedding + output_head
                for name, min_cos_attr in (
                    ("text/token_embedding", "q4"),
                    ("text/output_head", "q6"),
                ):
                    obj = artifact.find(name)
                    assert isinstance(obj, TensorObject)
                    rows = (0, 1, 200000, 200057, 201023)
                    deq = _dequantize_rows(artifact.payload(obj), obj.format, obj.shape, rows, target).cpu()
                    src_tensor = source.load_non_expert_tensor(
                        inventory.tensor_spec(name, obj.shape, obj.format), reader
                    )
                    ref = src_tensor.index_select(0, torch.tensor(rows, dtype=torch.long)).float()
                    cos = _cosine_similarity(deq, ref)
                    if cos < 0.990:
                        raise VerificationError(f"{name}: cosine {cos:.5f} below threshold")
                    if min_cos_attr == "q4":
                        min_q4 = min(min_q4, cos)
                    else:
                        min_q6 = min(min_q6, cos)
                    quantized_checked += 1

                # Sampled verification on attention + MoE projections across prelude/loop/epilogue
                for layer in probe_layers:
                    prefix = f"text/layers/{layer}/"
                    for attn_name in (
                        prefix + "attention/qgr",
                        prefix + "attention/k",
                        prefix + "attention/v",
                        prefix + "attention/output",
                    ):
                        obj = artifact.find(attn_name)
                        assert isinstance(obj, TensorObject)
                        rows = (0, obj.shape[0] // 2, obj.shape[0] - 1)
                        deq = _dequantize_rows(artifact.payload(obj), obj.format, obj.shape, rows, target).cpu()
                        src_t = source.load_non_expert_tensor(
                            inventory.tensor_spec(attn_name, obj.shape, obj.format), reader
                        )
                        ref = src_t.index_select(0, torch.tensor(rows, dtype=torch.long)).float()
                        cos = _cosine_similarity(deq, ref)
                        if cos < 0.990:
                            raise VerificationError(f"{attn_name}: cosine {cos:.5f} < 0.990")
                        min_q4 = min(min_q4, cos)
                        quantized_checked += 1

                if has_draft_head:
                    assert draft_head_ids is not None
                    obj = artifact.find("mtp/draft_head")
                    assert isinstance(obj, TensorObject)
                    rows = (0, obj.shape[0] // 2, obj.shape[0] - 1)
                    deq = _dequantize_rows(
                        artifact.payload(obj), obj.format, obj.shape, rows, target
                    ).cpu()
                    src_t = source.load_mtp_draft_head(reader, draft_head_ids)
                    ref = src_t.index_select(0, torch.tensor(rows, dtype=torch.long)).float()
                    cos = _cosine_similarity(deq, ref)
                    if cos < 0.985:
                        raise VerificationError(f"mtp/draft_head: cosine {cos:.5f} < 0.985")
                    min_q4 = min(min_q4, cos)
                    quantized_checked += 1

                    for is_w13, obj_suffix in (
                        (True, "ffn/routed_gate_up"),
                        (False, "ffn/routed_down"),
                    ):
                        obj_name = prefix + obj_suffix
                        obj = artifact.find(obj_name)
                        assert isinstance(obj, TensorObject)
                        rows_per_exp = 2 * inventory.MOE_INTERMEDIATE if is_w13 else inventory.HIDDEN_SIZE
                        wkey = f"model.llm.layers.{layer}.mlp.experts.{'w13_weight' if is_w13 else 'w2_weight'}"
                        scheme = sources.expert_quant_schemes[wkey]
                        for expert in (0, 63, 127):
                            local_rows = (0, rows_per_exp // 2, rows_per_exp - 1)
                            global_rows = tuple(expert * rows_per_exp + r for r in local_rows)
                            deq = _dequantize_rows(
                                artifact.payload(obj), obj.format, obj.shape, global_rows, target
                            ).cpu()
                            src_exp = reader.decode_expert(
                                f"model.llm.layers.{layer}.mlp.experts",
                                is_w13=is_w13,
                                expert=expert,
                                scheme=scheme,
                            )
                            ref = src_exp.index_select(0, torch.tensor(local_rows, dtype=torch.long)).float()
                            cos = _cosine_similarity(deq, ref)
                            if obj.format == inventory.Q3:
                                _verify_q3_canonical(deq, ref, f"{obj_name} e={expert}")
                                min_q3 = min(min_q3, cos)
                            else:
                                if cos < 0.990:
                                    raise VerificationError(f"{obj_name} e={expert}: cosine {cos:.5f} < 0.990")
                                min_q4 = min(min_q4, cos)
                        quantized_checked += 1

                if has_draft_head:
                    assert draft_head_ids is not None
                    obj = artifact.find("mtp/draft_head")
                    assert isinstance(obj, TensorObject)
                    rows = (0, obj.shape[0] // 2, obj.shape[0] - 1)
                    deq = _dequantize_rows(
                        artifact.payload(obj), obj.format, obj.shape, rows, target
                    ).cpu()
                    src_t = source.load_mtp_draft_head(reader, draft_head_ids)
                    ref = src_t.index_select(0, torch.tensor(rows, dtype=torch.long)).float()
                    cos = _cosine_similarity(deq, ref)
                    if cos < 0.985:
                        raise VerificationError(f"mtp/draft_head: cosine {cos:.5f} < 0.985")
                    min_q4 = min(min_q4, cos)
                    quantized_checked += 1
        finally:
            if dflash2_reader is not None:
                dflash2_reader.close()
            if mtp_reader is not None:
                mtp_reader.close()

        return VerifySummary(
            model_id=artifact.identity.model_id,
            weights_id=artifact.identity.weights_id,
            file_bytes=artifact.file_bytes,
            objects=len(artifact.objects),
            tensors=len(t_specs),
            resources=len(inventory.RESOURCE_SPECS),
            direct_exact_checked=direct_checked,
            quantized_checked=quantized_checked,
            min_q3_cosine=min_q3,
            min_q4_cosine=min_q4,
            min_q6_cosine=min_q6,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mtp", type=Path, default=None)
    parser.add_argument("--dflash2", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = verify_artifact(
        args.artifact,
        args.model,
        device=args.device,
        mtp_path=args.mtp,
        dflash2_dir=args.dflash2,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
