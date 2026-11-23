#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

/**
 * Dense small-N projection used by Inkling-2-Mini attention sublayers:
 *   sink_proj   ([2304, 24] BF16 -> [24, T] BF16 or FP32)
 *   *_conv_proj ([2304, 32] BF16 -> [32, T] BF16 or FP32)
 *
 * `x` is contiguous BF16 [2304, T] (`T > 0`), `weight` is contiguous BF16 [2304, N] with
 * `1 <= N <= 64`, and `out` is contiguous BF16 or FP32 [N, T].
 *
 * Oracle evaluates `out[n, t] = sum_{k=0..2303} FP64(weight[k, n]) * FP64(x[k, t])`.
 * Uses zero caller workspace, preserves inputs, and overwrites `out` completely.
 */
void inkling2_small_bf16_proj(const Tensor& x, const Tensor& weight, Tensor& out,
                              cudaStream_t stream);

/**
 * Depthwise causal width-4 short convolution with optional input-dependent dynamic head taps,
 * scaled residual `(x + conv_out) / sqrt(5 or 9)`, optional outer residual add, and width-3 state
 * update.
 *
 * Shapes (`C in {1024, 2304}`, `T > 0`):
 *   - `x`, `out`: contiguous BF16 [C, T] (`out` may NOT overlap `x`)
 *   - `static_weight`: contiguous FP32 [4, C] in chronological order [w_{t-3}, w_{t-2}, w_{t-1}, w_t]
 *   - `dynamic_taps`: empty Tensor{}, OR when C == 1024 contiguous FP32/BF16 [32, T] interpreted as
 *     [4, 8, T] in lag order (lag 0..3 multiplies u[:, t - lag] for kv_head = c / 128)
 *   - `state_in`, `state_out`: contiguous BF16 [C, 3] chronological [t-3, t-2, t-1]; may alias
 *     identically (`state_in.data == state_out.data`) or be disjoint
 *   - `residual_in`: empty Tensor{}, OR contiguous BF16 [C, T] added after convolution scaling
 *
 * Scale is `1 / 3` (`1 / sqrt(9)`) when `dynamic_taps` is provided, and `1 / sqrt(5)` otherwise.
 */
void inkling2_fused_sconv(const Tensor& x, const Tensor& static_weight,
                          const Tensor& dynamic_taps, const Tensor& state_in, Tensor& state_out,
                          const Tensor& residual_in, Tensor& out, cudaStream_t stream);

/**
 * Inkling-2 Block Attention Residuals (AttnRes) mix + optional per-branch RMSNorm.
 *
 * `states` holds `M` (`1 <= M <= 8`) contiguous BF16 [2304, T] tensors (`T > 0`).
 * `query` is contiguous FP32 [2304, R] with `R in {1, 3, 4}`.
 * `norm_weights` is either empty (`{}`) or `R` contiguous BF16 [2304] tensors applied per branch
 * with `eps = 1e-6f` after the AttnRes weighted combination.
 * `out` is contiguous BF16 [2304, T, R] (or [2304, T] when `R == 1`).
 *
 * Both attention-over-depth keys AND mixed values are parameter-free `rms_norm(state_s, 1e-6f)`.
 * Softmax score is `dot(query[:, r], norm_s) / sqrt(2304)`.
 */
void inkling2_attn_res_mix(std::span<const Tensor> states, const Tensor& query,
                           std::span<const Tensor> norm_weights, Tensor& out,
                           cudaStream_t stream);

/**
 * Returns caller-owned transient workspace capacity in bytes for `inkling2_loop_control` over
 * `[min_tokens, max_tokens]` and controller batch size `controllers in {1, 3}`.
 */
[[nodiscard]] std::size_t inkling2_loop_control_workspace_capacity_bytes(std::int32_t controllers,
                                                                         std::int32_t min_tokens,
                                                                         std::int32_t max_tokens);

/**
 * Rank-64 Inkling-2 loop residual gate + optional fused post-RMSNorm (`eps = 1e-6f`), batched across
 * `C_ctrl in {1, 3}` controllers (`1` for FFN, `3` for Q/K/V).
 *
 * Shapes (`T > 0`):
 *   - `cur`, `msg`, `out`: contiguous BF16 `[2304, T, C_ctrl]` (or `[2304, T]` when `C_ctrl == 1`)
 *   - `ctrl_down`: contiguous BF16 `[4608, 64, C_ctrl]` (or `[4608, 64]` when `C_ctrl == 1`)
 *   - `ctrl_gain_forget`: contiguous BF16 `[64, 2304, 2, C_ctrl]` (`[gain, forget]`, or
 *     `[64, 2304, 2]` when `C_ctrl == 1`)
 *   - `norm_weights`: empty `{}` or `C_ctrl` contiguous BF16 `[2304]` tensors
 */
void inkling2_loop_control(const Tensor& cur, const Tensor& msg, const Tensor& ctrl_down,
                           const Tensor& ctrl_gain_forget, std::span<const Tensor> norm_weights,
                           Tensor& out, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Fused per-head L2-clip (`threshold / (||v||_2 + 1e-6)`), V RMSNorm (`1e-6`), and optional
 * full-attention log-length scaling `tau(pos) = 1 + 0.1 * ln(max(1, (pos + 1) / 8192))` applied
 * to `Q` and `R`.
 *
 * Inputs:
 *   - `qgr`: contiguous BF16 `[6528, T]` (`Q` `[0..3072)`, `Gate` `[3072..6144)`, `R` `[6144..6528)`)
 *   - `k_in`, `v_in`: contiguous BF16 `[1024, T]`
 *   - `q_clip`: contiguous FP32 `[128, 24]`
 *   - `rel_clip`: contiguous FP32 `[16, 24]`
 *   - `positions`: contiguous I32 `[T]` when `apply_tau == true` (may be empty if `apply_tau == false`)
 *
 * Outputs:
 *   - `q_out`: contiguous BF16 `[128, 24, T]` (`qk_clip = 16 * sqrt(2)`)
 *   - `k_out`: contiguous BF16 `[128, 8, T]` (`qk_clip = 16 * sqrt(2)`, may alias `k_in` if shape matches)
 *   - `v_out`: contiguous BF16 `[128, 8, T]` (parameter-free RMSNorm `1e-6`, may alias `v_in`)
 *   - `r_out`: contiguous FP32 `[16, 24, T]` (`rel_clip = 8.0`)
 */
void inkling2_qkr_clip_v_norm(const Tensor& qgr, const Tensor& k_in, const Tensor& v_in,
                              const Tensor& q_clip, const Tensor& rel_clip, const Tensor& positions,
                              bool apply_tau, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                              Tensor& r_out, cudaStream_t stream);

/**
 * Fused logit softcap (`cap * tanh(x / cap)`) and valid-vocabulary prefix projection.
 *
 * `logits_in` is contiguous BF16 `[padded_vocab, T]` (`padded_vocab >= valid_rows > 0`, `T > 0`).
 * `logits_out` is contiguous BF16 `[valid_rows, T]` OR `[padded_vocab, T]` (updating `[0..valid_rows)`
 * when aliasing `logits_in` or targeting padded sampling buffers).
 */
void inkling2_softcap_slice(const Tensor& logits_in, float cap, std::int32_t valid_rows,
                            Tensor& logits_out, cudaStream_t stream);

/**
 * DSpark target feature mixing:
 *   c_r(t) = RMSNorm(sum_{j=0..3} softmax(fuse_weight[r, :])[j] * features[:, t, j], dflare_norms[r])
 * `features` and `out` are contiguous BF16 `[2304, T, 4]` (`T > 0`), `fuse_weight` is FP32 `[4, 4]`,
 * and `dflare_norms` holds 4 contiguous BF16 `[2304]` tensors.
 */
void inkling2_dspark_feature_mix(const Tensor& features, const Tensor& fuse_weight,
                                 std::span<const Tensor> dflare_norms, Tensor& out,
                                 cudaStream_t stream);

/**
 * DSpark per-head RMSNorm (`eps = 1e-6`) + split-half 1-D RoPE (`theta = 1e6` by default).
 * `positions` is I32 `[T]`, `q` is BF16 `[128, Hq, T]`, `k` is BF16 `[128, Hkv, T]` (or empty for
 * single-tensor mode on `q`), `q_norm`/`k_norm` are BF16 `[128]`.
 */
void inkling2_dspark_rmsnorm_rope(const Tensor& positions, const Tensor& q_norm,
                                  const Tensor& k_norm, Tensor& q, Tensor& k, float rope_theta,
                                  cudaStream_t stream);

/**
 * Workspace capacity in bytes for `inkling2_dspark_rnn_step`.
 */
[[nodiscard]] std::size_t inkling2_dspark_rnn_step_workspace_bytes();

/**
 * DSpark fused Markov RNN-hidden step for a single draft step:
 *   e = markov_w1[prev_token]
 *   [g, u, z] = rnn_proj * [state; e; h_in; c_head] + rnn_proj_bias
 *   state_next = tanh(u) + sigmoid(g) * (state - tanh(u))
 *   m = tanh(z)
 *   h_out = h_in + hidden_out * m + hidden_out_bias
 */
void inkling2_dspark_rnn_step(const Tensor& prev_token, const Tensor& markov_w1,
                              const Tensor& rnn_proj_weight, const Tensor& rnn_proj_bias,
                              const Tensor& hidden_out_weight, const Tensor& hidden_out_bias,
                              const Tensor& h_in, const Tensor& c_head, Tensor& state,
                              Tensor& h_out, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Restores and advances a slot's 48-visit 3-tap short-conv state `[6656*3, 48]` to `accepted`
 * committed tokens (`1 <= accepted < width`) using `pre_state` and `conv_trace [6656*width, 48]`.
 */
void inkling2_dspark_commit_conv_state(const Tensor& pre_state, const Tensor& conv_trace,
                                       Tensor& slot_conv_state, std::int32_t width,
                                       std::int32_t accepted, cudaStream_t stream);

} // namespace ninfer::ops
