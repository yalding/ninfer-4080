#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct Inkling2MoeWeights {
    Weight router;         // BF16_CTRL [128, 2304]
    Weight exp_probs_bias; // FP32_CTRL [128]
    Weight routed_gate_up; // Q3G128_F16S or Q4G64_F16S RowSplit [128*2304, 2304] ([gate; up] per expert)
    Weight routed_down;    // Q3G128_F16S or Q4G64_F16S RowSplit [128*2304, 1152]
};

/**
 * Returns caller-owned transient workspace capacity in bytes required by `inkling2_moe` for
 * every token count in `[min_tokens, max_tokens]`. Supported expert formats are Q3G128_F16S and
 * Q4G64_F16S (matching across `routed_gate_up` and `routed_down`).
 */
[[nodiscard]] std::size_t inkling2_moe_workspace_capacity_bytes(QType routed_qtype,
                                                                std::int32_t min_tokens,
                                                                std::int32_t max_tokens);

/**
 * Closed Sparse-MoE Op for Inkling-2-Mini (`H=2304, E=128, top_k=8, I=1152`, 0 shared experts).
 *
 * Math per token column `t in [0, T)` (`T > 0`):
 *   - `probs[e] = sigmoid(sum_k FP32(router[e, k]) * FP32(x[k, t]))`
 *   - Select top-8 experts by `probs[e] + bias[e]` (lower expert id breaks exact ties)
 *   - `alpha[i] = (probs[id_i] / (sum_{j=0..7} probs[id_j] + 1e-20f)) * 8.0f`
 *   - `SiTU(g, u) = (4.0f * tanh(g / 4.0f) * sigmoid(g)) * (25.0f * tanh(u / 25.0f))`
 *   - `out[:, t] = BF16(sum_{i=0..7} alpha[i] * W_down[id_i] * SiTU(W_gate[id_i]*x, W_up[id_i]*x))`
 *
 * Numeric:
 *   - For `Q3G128_F16S` with `T >= 64`, expert projections apply the documented `A8G64` activation
 *     quantization (`amax / 127`, `rint(x * (127 / amax))` in `[-127, 127]`,
 *     `src/ops/common/rowsplit_a8_quantize.h`) to `x` before `W_gate`/`W_up` and to
 *     `BF16(SiTU(...))` before `W_down`; the router projection remains BF16.
 *
 * `x` and `out` are contiguous BF16 `[2304, T]` (no residual add; `out` is completely overwritten).
 */
void inkling2_moe(const Tensor& x, const Inkling2MoeWeights& weights, Tensor& out,
                  WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Fused Inkling-2-Mini Sparse-MoE + width-4 `ffn_post_conv` + residual add + 3-tap state update.
 * Preserves exact BF16 intermediate rounding between MoE reduction and `ffn_post_conv`.
 * When `moe_trace` is non-empty (BF16 `[2304, T]`), writes the pre-conv reduced MoE output there.
 */
void inkling2_moe_post_sconv(const Tensor& x, const Inkling2MoeWeights& weights,
                             const Tensor& post_conv_weight, const Tensor& state_in,
                             Tensor& state_out, const Tensor& residual_in, Tensor& out,
                             Tensor moe_trace, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
