#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <array>
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
                          const Tensor& residual_in, Tensor& out, cudaStream_t stream,
                          Tensor trace_out = {});

/**
 * Paired depthwise causal width-4 short convolution for K and V (`C = 1024`, `T > 0`) in a
 * single kernel launch.
 */
void inkling2_fused_sconv_kv(const Tensor& kx, const Tensor& k_weight, const Tensor& k_dynamic,
                             const Tensor& k_state_in, Tensor& k_state_out, Tensor& k_out,
                             const Tensor& vx, const Tensor& v_weight, const Tensor& v_dynamic,
                             const Tensor& v_state_in, Tensor& v_state_out, Tensor& v_out,
                             cudaStream_t stream, Tensor k_trace_out = {},
                             Tensor v_trace_out = {});

/**
 * Multi-slot batched (`1 <= B <= 8`, 1 token/slot) `inkling2_fused_sconv` indexing
 * `all_conv_states [6656*3, 48, total_slots]` via device `slots [B]` for `component in {2, 3}`.
 */
void inkling2_fused_sconv_batched(const Tensor& x, const Tensor& static_weight,
                                  Tensor& all_conv_states, std::int32_t visit,
                                  std::int32_t component, const Tensor& slots,
                                  const Tensor& residual_in, Tensor& out, cudaStream_t stream);

/**
 * Multi-slot batched (`1 <= B <= 8`, 1 token/slot) `inkling2_fused_sconv_kv` indexing
 * `all_conv_states [6656*3, 48, total_slots]` via device `slots [B]` for components `0` (K)
 * and `1` (V).
 */
void inkling2_fused_sconv_kv_batched(const Tensor& kx, const Tensor& k_weight,
                                     const Tensor& k_dynamic, Tensor& k_out, const Tensor& vx,
                                     const Tensor& v_weight, const Tensor& v_dynamic, Tensor& v_out,
                                     Tensor& all_conv_states, std::int32_t visit,
                                     const Tensor& slots, cudaStream_t stream);

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
 * Pre-folds `output_head` (`Q6G64_F16S [P, 2304]`) with `markov_hidden_out_{weight,bias}` into
 * `w_fold_q` (`I8 [256, P]`), `w_fold_scale` (`FP16 [4, P]`), and `b_fold` (`FP32 [P]`).
 */
void inkling2_dspark_fold_head(const Weight& output_head, const Tensor& hidden_out_weight,
                               const Tensor& hidden_out_bias, Tensor& w_fold_q,
                               Tensor& w_fold_scale, Tensor& b_fold, cudaStream_t stream);

/**
 * Workspace capacity in bytes for `inkling2_dspark_rnn_step_folded`.
 */
[[nodiscard]] std::size_t inkling2_dspark_rnn_step_folded_workspace_bytes();

/**
 * DSpark Markov RNN latent step + folded rank-256 head + full-vocab argmax:
 *   state_next = tanh(u) + sigmoid(g) * (state - tanh(u))
 *   m = tanh(z)
 *   next_token = argmax_{v < valid_vocab} (base_logits_col[v] + b_fold[v] + dot(W_fold[v], m))
 */
void inkling2_dspark_rnn_step_folded(const Tensor& prev_token, const Tensor& markov_w1,
                                     const Tensor& rnn_proj_weight, const Tensor& rnn_proj_bias,
                                     const Tensor& h_in, const Tensor& c_head,
                                     const Tensor& base_logits_col, const Tensor& w_fold_q,
                                     const Tensor& w_fold_scale, const Tensor& b_fold,
                                     std::int32_t valid_vocab, Tensor& state, Tensor& next_token,
                                     WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Restores and advances a slot's 48-visit 3-tap short-conv state `[6656*3, 48]` to `accepted`
 * committed tokens (`1 <= accepted < width`) using `pre_state` and `conv_trace [6656*width, 48]`.
 */
void inkling2_dspark_commit_conv_state(const Tensor& pre_state, const Tensor& conv_trace,
                                       Tensor& slot_conv_state, std::int32_t width,
                                       std::int32_t accepted, cudaStream_t stream);

/**
 * Workspace capacity in bytes for `inkling2_dflash2_kv_materialize` up to `max_width` (`1..2048`).
 */
[[nodiscard]] std::size_t inkling2_dflash2_kv_materialize_workspace_bytes(std::int32_t max_width);

/**
 * Materialize normalized DFlash2 context features `BF16 [2304, W, 1]` (`W in 1..2048`) into 5
 * capacity-2048 cyclic KV layers (`Hq=32, Hkv=8, D=128`, `rope_theta = 10000.0`, `eps = 1e-6`).
 */
void inkling2_dflash2_kv_materialize(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Workspace capacity in bytes for `inkling2_dflash2_conv_prepare` up to `max_width` (`1..16`).
 */
[[nodiscard]] std::size_t inkling2_dflash2_conv_prepare_workspace_bytes(std::int32_t max_width);

/**
 * Fused RMSNorm + dynamic grouped width-2 convolution prepare (`H = 2304`, `G = 144`, `W in 1..16`):
 * normalizes `residual [2304, W]` with `norm_weight [2304]` and `eps`, projects through
 * `kernel_proj [2304, 576]`, emits `prepared [2304, W]` (side 0) and `finish_delta [144, 2, W]`
 * (side 1).
 */
void inkling2_dflash2_conv_prepare(const Tensor& residual, const Tensor& norm_weight, float eps,
                                   const Tensor& base_kernel, const Tensor& kernel_proj,
                                   Tensor& prepared, Tensor& finish_delta,
                                   WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Finish side of DFlash2 dynamic grouped width-2 convolution + residual add (`H = 2304`, `G = 144`):
 * `residual += (base_kernel[:, 0, 1] + finish_delta[:, 0, :]) * proj_out[:, i]`
 * `          + (base_kernel[:, 1, 1] + finish_delta[:, 1, :]) * proj_out[:, i - 1]`.
 */
void inkling2_dflash2_conv_finish_add(const Tensor& proj_out, const Tensor& base_kernel,
                                      const Tensor& finish_delta, Tensor& residual,
                                      cudaStream_t stream);

/**
 * Workspace capacity in bytes for `inkling2_dflash2_topk16_and_select` up to `max_k` (`1..15`).
 */
[[nodiscard]] std::size_t inkling2_dflash2_topk16_and_select_workspace_bytes(std::int32_t max_k);

/**
 * Fused hidden projection (`[2304, 256]`), exact stable top-16 over `[0, 200058)` from
 * un-softcapped `unary_logits [201024, K]`, and rank-256 bilinear path selection with post-correction
 * `30 * tanh((u + corr) / 30)` softcap. Outputs `candidate_ids [16, K, 1]`, `drafts [K, 1]`, and
 * `proposal_q [16, K, 1]`.
 */
void inkling2_dflash2_topk16_and_select(const Tensor& mask_hidden, const Tensor& unary_logits,
                                        const Tensor& hidden_projection, const Tensor& anchors,
                                        const Tensor& predecessor_codebook,
                                        const Tensor& successor_codebook,
                                        const Tensor& base_positions,
                                        const SamplingConfig* configs, Tensor& candidate_ids,
                                        Tensor& drafts, Tensor& proposal_q,
                                        WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Inkling-2-Mini ViT patch embedding, CLS prepend, bilinear position-embedding addition, and
 * pre-LayerNorm (`D = 1024`, `K = 1200`, `P >= 1`).
 */
void inkling2_vision_patch_embed(const Tensor& patches, const Tensor& patch_weight,
                                 const Tensor& class_embedding,
                                 const Tensor& cls_position_embedding, const Tensor& position_table,
                                 const Tensor& pos_indices, const Tensor& pos_weights,
                                 const Tensor& ln_pre_weight, const Tensor& ln_pre_bias, float eps,
                                 Tensor& out, cudaStream_t stream);

/**
 * In-place interleaved 2D axial RoPE on Q (`[0..1024)`) and K (`[1024..2048)`) of `qkv [3072, T]`,
 * `H = 16`, `head_dim = 64`. Skips tokens with `coords[0, t] < 0 || coords[1, t] < 0` bit-exactly.
 */
void inkling2_vision_rope_2d(const Tensor& coords, float rope_theta, Tensor& qkv,
                             cudaStream_t stream);

/**
 * Fused LayerScale residual add: `hidden[:, t] += gamma * (proj[:, t] + bias)` (`D = 1024`).
 */
void inkling2_vision_layerscale_add(const Tensor& proj, const Tensor& bias, const Tensor& gamma,
                                    Tensor& hidden, cudaStream_t stream);

/**
 * Gathers `2x2` spatial sub-patches from `src [1024, T_vis]` via `merge_indices [4, V]`
 * (`-1` -> zero-fill) into `out [4096, V]`.
 */
void inkling2_vision_merge_2x2(const Tensor& src, const Tensor& merge_indices, Tensor& out,
                               cudaStream_t stream);

} // namespace ninfer::ops
