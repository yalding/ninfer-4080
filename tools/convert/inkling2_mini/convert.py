r"""Stream thinkingmachines/Inkling-2-Mini safetensors into a NInfer v2 artifact.

Canonical invocation::

    python -m tools.convert.inkling2_mini.convert \
      --model C:\LLM\thinkingmachines\Inkling-2-Mini \
      --out C:\LLM\inkling2_mini_q3g128.ninfer \
      --device cuda
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
    ResourceSpec as ArtifactResourceSpec,
    TensorSpec as ArtifactTensorSpec,
)
from tools.artifact.layouts import (
    encode_direct,
    row_split_geometry,
    split_row_planes,
)
from tools.convert.common.quantize import pick_device, quantize_and_encode

from . import inventory, source


RECIPE_ID = "inkling2_mini_q3g128-v1"
ENCODER_PROFILE = "MAXABS_F16_RECIP_RNE_V1"


def _build_writer_specs(resources: dict[str, bytes]) -> list[ArtifactResourceSpec | ArtifactTensorSpec]:
    specs: list[ArtifactResourceSpec | ArtifactTensorSpec] = []
    for item in inventory.OBJECT_SPECS:
        if isinstance(item, inventory.ResourceSpec):
            specs.append(ArtifactResourceSpec(item.name, item.encoding, len(resources[item.name])))
        else:
            specs.append(ArtifactTensorSpec(item.name, item.shape, item.format, item.layout))
    return specs


def encode_routed_experts_streaming(
    spec: inventory.TensorSpec,
    reader: source.Inkling2SourceReader,
    layer: int,
    *,
    is_w13: bool,
    scheme: str,
    device: torch.device,
) -> bytearray:
    """Decode and quantize one 128-expert tensor one expert at a time to bound host RAM."""

    rows_per_expert = 2 * inventory.MOE_INTERMEDIATE if is_w13 else inventory.HIDDEN_SIZE
    cols = inventory.HIDDEN_SIZE if is_w13 else inventory.MOE_INTERMEDIATE
    if spec.shape != (inventory.NUM_EXPERTS * rows_per_expert, cols):
        raise ValueError(f"{spec.name}: unexpected shape {spec.shape}")

    full_geom = row_split_geometry(spec.format, spec.shape)
    exp_geom = row_split_geometry(spec.format, (rows_per_expert, cols))
    if full_geom.high_bytes != 0 or exp_geom.high_bytes != 0:
        raise ValueError(f"{spec.name}: expected zero high-bit plane for {spec.format}")
    if exp_geom.base_bytes * inventory.NUM_EXPERTS != full_geom.base_bytes:
        raise ValueError(f"{spec.name}: base plane size mismatch")
    if exp_geom.scale_bytes * inventory.NUM_EXPERTS != full_geom.scale_bytes:
        raise ValueError(f"{spec.name}: scale plane size mismatch")

    payload = bytearray(full_geom.payload_bytes)
    prefix = f"model.llm.layers.{layer}.mlp.experts"
    base_stride = exp_geom.base_bytes
    scale_stride = exp_geom.scale_bytes

    for expert in range(inventory.NUM_EXPERTS):
        expert_fp32 = reader.decode_expert(
            prefix, is_w13=is_w13, expert=expert, scheme=scheme
        )
        if tuple(expert_fp32.shape) != (rows_per_expert, cols):
            raise ValueError(
                f"{spec.name} expert {expert}: decoded shape {tuple(expert_fp32.shape)} != "
                f"{(rows_per_expert, cols)}"
            )
        exp_payload = quantize_and_encode(expert_fp32, spec.format, device=device)
        planes = split_row_planes(exp_payload, exp_geom)
        payload[expert * base_stride : (expert + 1) * base_stride] = planes.base
        s_off = full_geom.scale_offset + expert * scale_stride
        payload[s_off : s_off + scale_stride] = planes.scale
        del expert_fp32, exp_payload, planes

    return payload


def _encode_non_expert_tensor(
    spec: inventory.TensorSpec,
    tensor: torch.Tensor,
    device: torch.device,
) -> bytes:
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}")
    if spec.format in inventory.DIRECT_FORMATS:
        return encode_direct(tensor, spec.format)
    return quantize_and_encode(tensor, spec.format, device=device)


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    started = time.perf_counter()
    resolved_device = pick_device(device)
    sources = source.validate_hf_checkpoint(model_dir)
    resources = source.load_frontend_resources(sources.model_dir)
    writer_specs = _build_writer_specs(resources)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")

    print(
        f"preflight complete: {len(writer_specs)} objects, "
        f"tensor_bytes={inventory.total_tensor_payload_bytes()}, device={resolved_device}",
        flush=True,
    )

    written_objects: tuple[ArtifactObject, ...]
    with source.Inkling2SourceReader(sources.model_dir, sources.weight_map) as reader:
        with ArtifactWriter(
            tmp_output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            writer_specs,
        ) as writer:
            written_objects = writer.objects
            total = len(inventory.OBJECT_SPECS)
            done = 0

            for res_spec in inventory.RESOURCE_SPECS:
                writer.write(res_spec.name, resources[res_spec.name])
                done += 1
                print(f"[{done}/{total}] {res_spec.name}", flush=True)

            for t_spec in inventory.TENSOR_SPECS:
                if t_spec.name.endswith(("ffn/routed_gate_up", "ffn/routed_down")):
                    layer = int(t_spec.name.split("/")[2])
                    is_w13 = t_spec.name.endswith("ffn/routed_gate_up")
                    wkey = f"model.llm.layers.{layer}.mlp.experts.{'w13_weight' if is_w13 else 'w2_weight'}"
                    scheme = sources.expert_quant_schemes[wkey]
                    payload_buf = encode_routed_experts_streaming(
                        t_spec,
                        reader,
                        layer,
                        is_w13=is_w13,
                        scheme=scheme,
                        device=resolved_device,
                    )
                    writer.write(t_spec.name, memoryview(payload_buf))
                    del payload_buf
                else:
                    tensor = source.load_non_expert_tensor(t_spec, reader)
                    payload = _encode_non_expert_tensor(t_spec, tensor, resolved_device)
                    del tensor
                    writer.write(t_spec.name, payload)
                    del payload
                done += 1
                if done % 25 == 0 or done == total or t_spec.name.endswith(("routed_gate_up", "routed_down", "output_head")):
                    print(f"[{done}/{total}] {t_spec.name}", flush=True)

    tmp_output.replace(output)
    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size

    report = {
        "identity": {"model_id": inventory.MODEL_ID, "weights_id": inventory.WEIGHTS_ID},
        "target_key": inventory.TARGET_KEY,
        "recipe_id": RECIPE_ID,
        "encoder_profile": ENCODER_PROFILE,
        "source": {"model_path": str(sources.model_dir.resolve())},
        "config_summary": sources.config_summary,
        "objects": {
            "count": len(written_objects),
            "tensors": len(inventory.TENSOR_SPECS),
            "resources": len(inventory.RESOURCE_SPECS),
            "formats": dict(sorted(Counter(s.format for s in inventory.TENSOR_SPECS).items())),
            "layouts": dict(sorted(Counter(s.layout for s in inventory.TENSOR_SPECS).items())),
            "tensor_bytes": inventory.total_tensor_payload_bytes(),
            "device_arena_bytes": inventory.total_device_arena_bytes(),
        },
        "elapsed_seconds": elapsed,
        "artifact": {"path": str(output), "bytes": final_bytes},
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
