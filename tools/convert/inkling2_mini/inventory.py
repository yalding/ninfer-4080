"""Persistent-object inventory contract for the Inkling-2-Mini target."""

from __future__ import annotations

from dataclasses import dataclass

from tools.artifact.layouts import align_up, encoded_size


MODEL_ID = "inkling2-mini"
WEIGHTS_ID = "q3g128"
TARGET_KEY = "inkling2_mini"

CONTIGUOUS_LAYOUT = "contiguous-le-v1"
ROW_SPLIT_LAYOUT = "row-split-k128-v1"
RESOURCE_ENCODING = "raw-bytes-v1"

BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
Q3 = "Q3G128_F16S"
Q4 = "Q4G64_F16S"
Q6 = "Q6G64_F16S"

DIRECT_FORMATS = frozenset((BF16, FP32, I32))
FORMAT_NAMES = (BF16, FP32, Q3, Q4, Q6)
LAYOUT_NAMES = (CONTIGUOUS_LAYOUT, ROW_SPLIT_LAYOUT)

HIDDEN_SIZE = 2304
VOCAB_SIZE = 201024
UNPADDED_VOCAB_SIZE = 200058
PHYS_LAYERS = 24
LOGICAL_LAYERS = 48
LOOP_START = 6
LOOP_END = 18
LOOP_PASSES = 3
LOOP_GATE_RANK = 64
ATTN_RES_BLOCK_SIZE = 6
NUM_Q_HEADS = 24
NUM_KV_HEADS = 8
HEAD_DIM = 128
D_REL = 16
REL_EXTENT = 1024
SLIDING_WINDOW = 1024
SCONV_KERNEL = 4
NUM_EXPERTS = 128
TOP_K = 8
MOE_INTERMEDIATE = 1152

FULL_ATTENTION_LAYERS = frozenset((5, 11, 17, 23))
SWA_LAYERS = frozenset(i for i in range(PHYS_LAYERS) if i not in FULL_ATTENTION_LAYERS)
LOOP_LAYERS = tuple(range(LOOP_START, LOOP_END))
Q3_MOE_LAYERS = frozenset(range(20))
Q4_MOE_LAYERS = frozenset(range(20, PHYS_LAYERS))


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    format: str
    layout: str

    @property
    def kind(self) -> str:
        return "tensor"


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    name: str
    encoding: str = RESOURCE_ENCODING

    @property
    def kind(self) -> str:
        return "resource"


StoredObjectSpec = TensorSpec | ResourceSpec


def tensor_spec(name: str, shape: tuple[int, ...], numeric_format: str) -> TensorSpec:
    layout = CONTIGUOUS_LAYOUT if numeric_format in DIRECT_FORMATS else ROW_SPLIT_LAYOUT
    return TensorSpec(name=name, shape=shape, format=numeric_format, layout=layout)


RESOURCE_SPECS: tuple[ResourceSpec, ...] = (
    ResourceSpec("frontend/tokenizer.json"),
    ResourceSpec("frontend/tokenizer_config.json"),
    ResourceSpec("frontend/chat_template.jinja"),
    ResourceSpec("frontend/generation_config.json"),
)


def is_full_attention_layer(layer: int) -> bool:
    return layer in FULL_ATTENTION_LAYERS


def is_loop_layer(layer: int) -> bool:
    return LOOP_START <= layer < LOOP_END


def moe_format_for_layer(layer: int) -> str:
    return Q3 if layer in Q3_MOE_LAYERS else Q4


def _build_tensor_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("text/token_embedding", (VOCAB_SIZE, HIDDEN_SIZE), Q4),
        tensor_spec("text/token_norm", (HIDDEN_SIZE,), BF16),
    ]

    for layer in range(PHYS_LAYERS):
        prefix = f"text/layers/{layer}/"
        full = is_full_attention_layer(layer)
        moe_fmt = moe_format_for_layer(layer)

        if layer > 0:
            res_rows = 3 if full else 1
            specs.append(tensor_spec(prefix + "attention/res_query", (res_rows, HIDDEN_SIZE), FP32))

        if full:
            specs.extend(
                (
                    tensor_spec(prefix + "attention/q_input_norm", (HIDDEN_SIZE,), BF16),
                    tensor_spec(prefix + "attention/k_input_norm", (HIDDEN_SIZE,), BF16),
                    tensor_spec(prefix + "attention/v_input_norm", (HIDDEN_SIZE,), BF16),
                )
            )
        else:
            specs.append(tensor_spec(prefix + "attention/input_norm", (HIDDEN_SIZE,), BF16))

        specs.extend(
            (
                tensor_spec(prefix + "attention/qgr", (6528, HIDDEN_SIZE), Q4),
                tensor_spec(prefix + "attention/k", (1024, HIDDEN_SIZE), Q4),
                tensor_spec(prefix + "attention/v", (1024, HIDDEN_SIZE), Q4),
                tensor_spec(prefix + "attention/output", (HIDDEN_SIZE, 3072), Q4),
                tensor_spec(prefix + "attention/sink_proj", (NUM_Q_HEADS, HIDDEN_SIZE), BF16),
                tensor_spec(prefix + "attention/q_clip", (NUM_Q_HEADS, HEAD_DIM), FP32),
                tensor_spec(prefix + "attention/rel_clip", (NUM_Q_HEADS, D_REL), FP32),
                tensor_spec(prefix + "attention/rel_bias", (REL_EXTENT, D_REL), FP32),
                tensor_spec(prefix + "attention/k_conv", (1024, SCONV_KERNEL), FP32),
                tensor_spec(prefix + "attention/v_conv", (1024, SCONV_KERNEL), FP32),
            )
        )

        if full:
            specs.extend(
                (
                    tensor_spec(prefix + "attention/k_conv_proj", (32, HIDDEN_SIZE), BF16),
                    tensor_spec(prefix + "attention/v_conv_proj", (32, HIDDEN_SIZE), BF16),
                )
            )

        specs.append(tensor_spec(prefix + "attention/post_conv", (HIDDEN_SIZE, SCONV_KERNEL), FP32))

        if is_loop_layer(layer):
            specs.extend(
                (
                    tensor_spec(prefix + "loop/msg_query", (4, HIDDEN_SIZE), FP32),
                    tensor_spec(
                        prefix + "loop/ctrl_down", (4, LOOP_GATE_RANK, 2 * HIDDEN_SIZE), BF16
                    ),
                    tensor_spec(
                        prefix + "loop/ctrl_gain_forget",
                        (4, 2, HIDDEN_SIZE, LOOP_GATE_RANK),
                        BF16,
                    ),
                )
            )

        specs.extend(
            (
                tensor_spec(prefix + "ffn/res_query", (HIDDEN_SIZE,), FP32),
                tensor_spec(prefix + "ffn/input_norm", (HIDDEN_SIZE,), BF16),
                tensor_spec(prefix + "ffn/router", (NUM_EXPERTS, HIDDEN_SIZE), BF16),
                tensor_spec(prefix + "ffn/exp_probs_bias", (NUM_EXPERTS,), FP32),
                tensor_spec(
                    prefix + "ffn/routed_gate_up",
                    (NUM_EXPERTS * 2 * MOE_INTERMEDIATE, HIDDEN_SIZE),
                    moe_fmt,
                ),
                tensor_spec(
                    prefix + "ffn/routed_down",
                    (NUM_EXPERTS * HIDDEN_SIZE, MOE_INTERMEDIATE),
                    moe_fmt,
                ),
                tensor_spec(prefix + "ffn/post_conv", (HIDDEN_SIZE, SCONV_KERNEL), FP32),
            )
        )

    specs.extend(
        (
            tensor_spec("text/output_res_query", (HIDDEN_SIZE,), FP32),
            tensor_spec("text/final_norm", (HIDDEN_SIZE,), BF16),
            tensor_spec("text/output_head", (VOCAB_SIZE, HIDDEN_SIZE), Q6),
        )
    )
    return tuple(specs)


TENSOR_SPECS: tuple[TensorSpec, ...] = _build_tensor_specs()
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    fmt: sum(1 for spec in TENSOR_SPECS if spec.format == fmt) for fmt in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(1 for spec in TENSOR_SPECS if spec.layout == layout) for layout in LAYOUT_NAMES
}


def total_tensor_payload_bytes() -> int:
    return sum(encoded_size(spec.layout, spec.format, spec.shape) for spec in TENSOR_SPECS)


def total_device_arena_bytes(alignment: int = 256) -> int:
    cursor = 0
    for spec in TENSOR_SPECS:
        cursor = align_up(cursor, alignment)
        cursor += encoded_size(spec.layout, spec.format, spec.shape)
    return cursor
