#include "ninfer/ops/inkling2_fused.h"
#include "ops/inkling2/launch.h"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
void require(bool valid, const char* reason) {
    if (!valid) throw std::invalid_argument(std::string("inkling2: ") + reason);
}
void shape(const Tensor& t, DType dtype, std::initializer_list<int> dims) {
    require(t.data && t.dtype == dtype && t.is_contiguous(), "invalid tensor storage");
    int i = 0;
    for (int d : dims) { require(d > 0 && t.ne[i++] == d, "invalid tensor shape"); }
    while (i < 4) require(t.ne[i++] == 1, "invalid tensor rank");
}
bool overlap(const Tensor& a, const Tensor& b) {
    if (!a.data || !b.data) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a.data);
    const auto y = reinterpret_cast<std::uintptr_t>(b.data);
    return x <= y ? y - x < a.bytes() : x - y < b.bytes();
}
void disjoint(const Tensor& a, const Tensor& b) {
    require(!overlap(a, b), "unsupported tensor overlap");
}
void identical_or_disjoint(const Tensor& a, const Tensor& b) {
    require(a.data == b.data || !overlap(a, b), "partial tensor overlap");
}
detail::Inkling2Branches norms(std::span<const Tensor> weights, int count) {
    require(weights.empty() || weights.size() == static_cast<std::size_t>(count),
            "invalid normalization branch count");
    detail::Inkling2Branches result;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        shape(weights[i], DType::BF16, {2304});
        result.norm[i] = weights[i].data;
    }
    return result;
}
} // namespace

void inkling2_small_bf16_proj(const Tensor& x, const Tensor& weight, Tensor& out, cudaStream_t s) {
    const int t = x.ne[1], n = weight.ne[1];
    require(n >= 1 && n <= 64, "projection output width must be 1..64");
    shape(x, DType::BF16, {2304, t});
    shape(weight, DType::BF16, {2304, n});
    require(out.dtype == DType::BF16 || out.dtype == DType::FP32, "projection output dtype");
    shape(out, out.dtype, {n, t});
    disjoint(out, x); disjoint(out, weight);
    detail::inkling2_proj_launch(x, weight, out, s);
}

void inkling2_fused_sconv(const Tensor& x, const Tensor& w, const Tensor& dynamic,
                          const Tensor& state_in, Tensor& state_out, const Tensor& residual,
                          Tensor& out, cudaStream_t s, Tensor trace_out) {
    const int c = x.ne[0], t = x.ne[1];
    require(c == 1024 || c == 2304, "convolution channels must be 1024 or 2304");
    shape(x, DType::BF16, {c, t}); shape(out, DType::BF16, {c, t});
    shape(w, DType::FP32, {4, c});
    shape(state_in, DType::BF16, {c, 3}); shape(state_out, DType::BF16, {c, 3});
    identical_or_disjoint(state_in, state_out);
    disjoint(x, out); disjoint(x, state_out); disjoint(out, state_in); disjoint(out, state_out);
    disjoint(w, out); disjoint(w, state_out);
    if (dynamic.data) {
        require(c == 1024 && (dynamic.dtype == DType::FP32 || dynamic.dtype == DType::BF16),
                "dynamic convolution geometry/dtype");
        shape(dynamic, dynamic.dtype, {32, t});
        disjoint(dynamic, out); disjoint(dynamic, state_out);
    }
    if (residual.data) {
        shape(residual, DType::BF16, {c, t});
        identical_or_disjoint(residual, out); disjoint(residual, state_out);
    }
    if (trace_out.data) {
        shape(trace_out, DType::BF16, {c, t});
        disjoint(x, trace_out); disjoint(out, trace_out); disjoint(state_out, trace_out);
    }
    detail::inkling2_sconv_launch(x, w, dynamic, state_in, state_out, residual, out, trace_out, s);
}

void inkling2_fused_sconv_kv(const Tensor& kx, const Tensor& k_weight, const Tensor& k_dynamic,
                             const Tensor& k_state_in, Tensor& k_state_out, Tensor& k_out,
                             const Tensor& vx, const Tensor& v_weight, const Tensor& v_dynamic,
                             const Tensor& v_state_in, Tensor& v_state_out, Tensor& v_out,
                             cudaStream_t s, Tensor k_trace_out, Tensor v_trace_out) {
    const int t = kx.ne[1];
    auto check_branch = [&](const Tensor& x, const Tensor& w, const Tensor& dyn,
                            const Tensor& st_in, const Tensor& st_out, const Tensor& out,
                            const Tensor& trace) {
        shape(x, DType::BF16, {1024, t});
        shape(out, DType::BF16, {1024, t});
        shape(w, DType::FP32, {4, 1024});
        shape(st_in, DType::BF16, {1024, 3});
        shape(st_out, DType::BF16, {1024, 3});
        identical_or_disjoint(st_in, st_out);
        disjoint(x, out); disjoint(x, st_out); disjoint(out, st_in); disjoint(out, st_out);
        disjoint(w, out); disjoint(w, st_out);
        if (dyn.data) {
            require(dyn.dtype == DType::FP32 || dyn.dtype == DType::BF16,
                    "dynamic convolution geometry/dtype");
            shape(dyn, dyn.dtype, {32, t});
            disjoint(dyn, out); disjoint(dyn, st_out);
        }
        if (trace.data) {
            shape(trace, DType::BF16, {1024, t});
            disjoint(x, trace); disjoint(out, trace); disjoint(st_out, trace);
        }
    };
    check_branch(kx, k_weight, k_dynamic, k_state_in, k_state_out, k_out, k_trace_out);
    check_branch(vx, v_weight, v_dynamic, v_state_in, v_state_out, v_out, v_trace_out);
    require(k_dynamic.data != nullptr == (v_dynamic.data != nullptr),
            "paired sconv_kv dynamic taps mismatch");
    detail::inkling2_sconv_kv_launch(kx, k_weight, k_dynamic, k_state_in, k_state_out, k_out, vx,
                                     v_weight, v_dynamic, v_state_in, v_state_out, v_out,
                                     k_trace_out, v_trace_out, s);
}

void inkling2_fused_sconv_batched(const Tensor& x, const Tensor& static_weight,
                                  Tensor& all_conv_states, std::int32_t visit,
                                  std::int32_t component, const Tensor& slots,
                                  const Tensor& residual, Tensor& out, cudaStream_t s) {
    const int b = x.ne[1];
    require(b >= 1 && b <= 8 && visit >= 0 && visit < 48 && (component == 2 || component == 3),
            "inkling2_fused_sconv_batched: invalid B/visit/component");
    shape(x, DType::BF16, {2304, b});
    shape(static_weight, DType::FP32, {4, 2304});
    require(all_conv_states.data != nullptr && all_conv_states.dtype == DType::BF16 &&
                all_conv_states.is_contiguous() && all_conv_states.ne[0] == 6656 * 3 &&
                all_conv_states.ne[1] == 48 && all_conv_states.ne[2] >= 1 &&
                all_conv_states.ne[3] == 1,
            "inkling2_fused_sconv_batched: invalid all_conv_states");
    shape(slots, DType::I32, {b});
    shape(out, DType::BF16, {2304, b});
    disjoint(out, x);
    if (residual.data) {
        shape(residual, DType::BF16, {2304, b});
        disjoint(out, residual);
    }
    detail::inkling2_sconv_batched_launch(x, static_weight, all_conv_states, visit, component,
                                          slots, residual, out, s);
}

void inkling2_fused_sconv_kv_batched(const Tensor& kx, const Tensor& kw, const Tensor& kdyn,
                                     Tensor& kout, const Tensor& vx, const Tensor& vw,
                                     const Tensor& vdyn, Tensor& vout, Tensor& all_conv_states,
                                     std::int32_t visit, const Tensor& slots, cudaStream_t s) {
    const int b = kx.ne[1];
    require(b >= 1 && b <= 8 && visit >= 0 && visit < 48,
            "inkling2_fused_sconv_kv_batched: invalid B/visit");
    shape(kx, DType::BF16, {1024, b});
    shape(vx, DType::BF16, {1024, b});
    shape(kw, DType::FP32, {4, 1024});
    shape(vw, DType::FP32, {4, 1024});
    shape(kout, DType::BF16, {1024, b});
    shape(vout, DType::BF16, {1024, b});
    shape(slots, DType::I32, {b});
    require(all_conv_states.data != nullptr && all_conv_states.dtype == DType::BF16 &&
                all_conv_states.is_contiguous() && all_conv_states.ne[0] == 6656 * 3 &&
                all_conv_states.ne[1] == 48 && all_conv_states.ne[2] >= 1 &&
                all_conv_states.ne[3] == 1,
            "inkling2_fused_sconv_kv_batched: invalid all_conv_states");
    if (kdyn.data || vdyn.data) {
        require(kdyn.data && vdyn.data && (kdyn.dtype == DType::FP32 || kdyn.dtype == DType::BF16) &&
                    vdyn.dtype == kdyn.dtype,
                "inkling2_fused_sconv_kv_batched: dynamic taps");
        shape(kdyn, kdyn.dtype, {32, b});
        shape(vdyn, vdyn.dtype, {32, b});
    }
    disjoint(kout, kx);
    disjoint(vout, vx);
    disjoint(kout, vout);
    detail::inkling2_sconv_kv_batched_launch(kx, kw, kdyn, kout, vx, vw, vdyn, vout,
                                             all_conv_states, visit, slots, s);
}

void inkling2_attn_res_mix(std::span<const Tensor> states, const Tensor& q,
                           std::span<const Tensor> nw, Tensor& out, cudaStream_t s) {
    const int m = static_cast<int>(states.size()), r = q.ne[1], t = out.ne[1];
    require(m >= 1 && m <= 8 && (r == 1 || r == 3 || r == 4), "mix branch geometry");
    shape(q, DType::FP32, {2304, r}); shape(out, DType::BF16, {2304, t, r});
    auto branches = norms(nw, r); branches.count = m;
    for (int i = 0; i < m; ++i) {
        shape(states[i], DType::BF16, {2304, t}); disjoint(states[i], out);
        branches.states[i] = states[i].data;
    }
    for (const auto& weight : nw) disjoint(weight, out);
    disjoint(q, out);
    detail::inkling2_mix_launch(branches, q, out, s);
}

std::size_t inkling2_loop_control_workspace_capacity_bytes(int c, int lo, int hi) {
    require((c == 1 || c == 3) && lo > 0 && hi >= lo, "control workspace envelope");
    const auto n = static_cast<std::size_t>(hi) * c;
    if (n > (std::numeric_limits<std::size_t>::max() - 255) / (2304 * sizeof(float)))
        throw std::overflow_error("inkling2 control workspace overflow");
    const auto z_bytes   = (n * 64 * sizeof(float) + 255) & ~std::size_t(255);
    const auto inv_bytes = (n * 2 * sizeof(float) + 255) & ~std::size_t(255);
    const auto y_bytes   = (n * 2304 * sizeof(float) + 255) & ~std::size_t(255);
    return z_bytes + inv_bytes + y_bytes;
}

void inkling2_loop_control(const Tensor& cur, const Tensor& msg, const Tensor& down,
                           const Tensor& gf, std::span<const Tensor> nw, Tensor& out,
                           WorkspaceArena& workspace, cudaStream_t s) {
    const int t = msg.ne[1], c = msg.ne[2], cur_c = cur.ne[2];
    require((c == 1 || c == 3) && (cur_c == 1 || cur_c == c), "loop control branch count");
    const auto n = static_cast<std::size_t>(t) * c;
    const auto z_bytes   = (n * 64 * sizeof(float) + 255) & ~std::size_t(255);
    const auto inv_bytes = (n * 2 * sizeof(float) + 255) & ~std::size_t(255);
    const auto y_bytes =
        !nw.empty() ? ((n * 2304 * sizeof(float) + 255) & ~std::size_t(255)) : std::size_t(0);
    shape(cur, DType::BF16, {2304, t, cur_c}); shape(msg, DType::BF16, {2304, t, c});
    shape(out, DType::BF16, {2304, t, c}); shape(down, DType::BF16, {4608, 64, c});
    shape(gf, DType::BF16, {64, 2304, 2, c});
    disjoint(out, down); disjoint(out, gf);
    if (cur_c == c) identical_or_disjoint(cur, out); else disjoint(cur, out);
    identical_or_disjoint(msg, out);
    auto branches = norms(nw, c);
    for (const auto& weight : nw) disjoint(out, weight);
    auto scope = workspace.scope();
    auto z_buf = workspace.alloc_bytes(z_bytes);
    float* inv_ptr = inv_bytes ? static_cast<float*>(workspace.alloc_bytes(inv_bytes).data) : nullptr;
    float* y_ptr = y_bytes ? static_cast<float*>(workspace.alloc_bytes(y_bytes).data) : nullptr;
    detail::inkling2_control_launch(cur, msg, down, gf, branches, out,
                                    static_cast<float*>(z_buf.data), inv_ptr, y_ptr, s);
}

void inkling2_qkr_clip_v_norm(const Tensor& qgr, const Tensor& k, const Tensor& v,
                              const Tensor& qc, const Tensor& rc, const Tensor& positions,
                              bool tau, Tensor& qo, Tensor& ko, Tensor& vo, Tensor& ro,
                              cudaStream_t s) {
    const int t = qgr.ne[1];
    shape(qgr, DType::BF16, {6528, t}); shape(k, DType::BF16, {1024, t});
    shape(v, DType::BF16, {1024, t}); shape(qc, DType::FP32, {128, 24});
    shape(rc, DType::FP32, {16, 24}); shape(qo, DType::BF16, {128, 24, t});
    shape(ko, DType::BF16, {128, 8, t}); shape(vo, DType::BF16, {128, 8, t});
    shape(ro, DType::FP32, {16, 24, t});
    if (tau) shape(positions, DType::I32, {t});
    const Tensor* outputs[]{&qo, &ko, &vo, &ro};
    for (int i = 0; i < 4; ++i) {
        disjoint(*outputs[i], qgr); disjoint(*outputs[i], qc); disjoint(*outputs[i], rc);
        if (positions.data) disjoint(*outputs[i], positions);
        for (int j = i + 1; j < 4; ++j) disjoint(*outputs[i], *outputs[j]);
        if (i == 1) identical_or_disjoint(*outputs[i], k); else disjoint(*outputs[i], k);
        if (i == 2) identical_or_disjoint(*outputs[i], v); else disjoint(*outputs[i], v);
    }
    detail::inkling2_clip_launch(qgr, k, v, qc, rc, positions, tau, qo, ko, vo, ro, s);
}

void inkling2_softcap_slice(const Tensor& x, float cap, int valid, Tensor& out, cudaStream_t s) {
    const int p = x.ne[0], t = x.ne[1];
    require(std::isfinite(cap) && cap > 0 && valid > 0 && valid <= p, "softcap parameters");
    shape(x, DType::BF16, {p, t});
    require(out.ne[0] == p || out.ne[0] == valid, "softcap output width");
    shape(out, DType::BF16, {out.ne[0], t});
    identical_or_disjoint(x, out);
    require(x.data != out.data || out.ne[0] == p, "in-place softcap needs equal strides");
    detail::inkling2_softcap_launch(x, cap, valid, out, s);
}

void inkling2_dspark_feature_mix(const Tensor& features, const Tensor& fuse_weight,
                                 std::span<const Tensor> dflare_norms, Tensor& out,
                                 cudaStream_t s) {
    const int t = features.ne[1];
    require(t > 0, "dspark feature_mix T must be positive");
    shape(features, DType::BF16, {2304, t, 4});
    shape(fuse_weight, DType::FP32, {4, 4});
    shape(out, DType::BF16, {2304, t, 4});
    require(dflare_norms.size() == 4, "dspark feature_mix requires 4 dflare_norms");
    auto branches = norms(dflare_norms, 4);
    disjoint(features, out);
    disjoint(fuse_weight, out);
    for (const auto& w : dflare_norms) disjoint(w, out);
    detail::inkling2_dspark_feature_mix_launch(features, fuse_weight, branches, out, s);
}

void inkling2_dspark_rmsnorm_rope(const Tensor& positions, const Tensor& q_norm,
                                  const Tensor& k_norm, Tensor& q, Tensor& k, float rope_theta,
                                  cudaStream_t s) {
    const int t  = q.ne[2];
    const int hq = q.ne[1];
    require(t > 0 && hq > 0 && rope_theta > 0.0f, "dspark rmsnorm_rope geometry");
    shape(positions, DType::I32, {t});
    shape(q_norm, DType::BF16, {128});
    shape(q, DType::BF16, {128, hq, t});
    disjoint(q, positions);
    disjoint(q, q_norm);
    if (k.data) {
        const int hk = k.ne[1];
        require(hk > 0, "dspark rmsnorm_rope k heads");
        shape(k_norm, DType::BF16, {128});
        shape(k, DType::BF16, {128, hk, t});
        disjoint(q, k);
        disjoint(k, positions);
        disjoint(k, k_norm);
    }
    detail::inkling2_dspark_rmsnorm_rope_launch(positions, q_norm, k_norm, q, k, rope_theta, s);
}

std::size_t inkling2_dspark_rnn_step_workspace_bytes() {
    return ((5120 * 2 + 255) & ~std::size_t(255)) + ((768 * sizeof(float) + 255) & ~std::size_t(255));
}

void inkling2_dspark_rnn_step(const Tensor& prev_token, const Tensor& markov_w1,
                              const Tensor& rnn_proj_weight, const Tensor& rnn_proj_bias,
                              const Tensor& hidden_out_weight, const Tensor& hidden_out_bias,
                              const Tensor& h_in, const Tensor& c_head, Tensor& state,
                              Tensor& h_out, WorkspaceArena& workspace, cudaStream_t s) {
    shape(prev_token, DType::I32, {1});
    shape(markov_w1, DType::BF16, {256, 201024});
    shape(rnn_proj_weight, DType::BF16, {5120, 768});
    shape(rnn_proj_bias, DType::FP32, {768});
    shape(hidden_out_weight, DType::BF16, {256, 2304});
    shape(hidden_out_bias, DType::FP32, {2304});
    shape(h_in, DType::BF16, {2304});
    shape(c_head, DType::BF16, {2304});
    shape(state, DType::FP32, {256});
    shape(h_out, DType::BF16, {2304});
    disjoint(h_out, h_in);
    disjoint(h_out, state);
    auto scope = workspace.scope();
    void* x_buf = workspace.alloc_bytes(5120 * 2).data;
    auto* guz   = static_cast<float*>(workspace.alloc_bytes(768 * sizeof(float)).data);
    detail::inkling2_dspark_rnn_step_launch(prev_token, markov_w1, rnn_proj_weight, rnn_proj_bias,
                                            hidden_out_weight, hidden_out_bias, h_in, c_head, state,
                                            h_out, x_buf, guz, s);
}

void inkling2_dspark_fold_head(const Weight& output_head, const Tensor& hidden_out_weight,
                               const Tensor& hidden_out_bias, Tensor& w_fold_q,
                               Tensor& w_fold_scale, Tensor& b_fold, cudaStream_t s) {
    require(output_head.qtype == QType::Q6G64_F16S && output_head.layout == QuantLayout::RowSplit &&
                output_head.group_size == 64 && output_head.group == 64 &&
                output_head.scale_dtype == DType::FP16 && output_head.k == 2304 &&
                output_head.padded_shape[1] == 2304 && output_head.n > 0 &&
                (output_head.n % 64) == 0 && output_head.qdata != nullptr &&
                output_head.qhigh != nullptr && output_head.scales != nullptr,
            "dspark fold head: output_head must be Q6G64_F16S [N, 2304] with 64 | N");
    shape(hidden_out_weight, DType::BF16, {256, 2304});
    shape(hidden_out_bias, DType::FP32, {2304});
    shape(w_fold_q, DType::I8, {256, output_head.n});
    shape(w_fold_scale, DType::FP16, {4, output_head.n});
    shape(b_fold, DType::FP32, {output_head.n});
    disjoint(w_fold_q, hidden_out_weight);
    disjoint(w_fold_scale, hidden_out_weight);
    disjoint(b_fold, hidden_out_bias);
    detail::inkling2_dspark_fold_head_launch(output_head, hidden_out_weight, hidden_out_bias,
                                             w_fold_q, w_fold_scale, b_fold, s);
}

std::size_t inkling2_dspark_rnn_step_folded_workspace_bytes() {
    return ((5120 * 2 + 255) & ~std::size_t(255)) +
           ((768 * sizeof(float) + 255) & ~std::size_t(255)) +
           ((256 * sizeof(float) + 255) & ~std::size_t(255)) +
           ((272 * sizeof(float) + 255) & ~std::size_t(255)) +
           ((272 * sizeof(int) + 255) & ~std::size_t(255));
}

void inkling2_dspark_rnn_step_folded(const Tensor& prev_token, const Tensor& markov_w1,
                                     const Tensor& rnn_proj_weight, const Tensor& rnn_proj_bias,
                                     const Tensor& h_in, const Tensor& c_head,
                                     const Tensor& base_logits_col, const Tensor& w_fold_q,
                                     const Tensor& w_fold_scale, const Tensor& b_fold,
                                     std::int32_t valid_vocab, Tensor& state, Tensor& next_token,
                                     WorkspaceArena& workspace, cudaStream_t s) {
    shape(prev_token, DType::I32, {1});
    shape(markov_w1, DType::BF16, {256, 201024});
    shape(rnn_proj_weight, DType::BF16, {5120, 768});
    shape(rnn_proj_bias, DType::FP32, {768});
    shape(h_in, DType::BF16, {2304});
    shape(c_head, DType::BF16, {2304});
    const int n = b_fold.ne[0];
    require(n > 0 && (n % 64) == 0 && valid_vocab > 0 && valid_vocab <= n,
            "dspark folded step: invalid vocab");
    shape(base_logits_col, DType::BF16, {n});
    shape(w_fold_q, DType::I8, {256, n});
    shape(w_fold_scale, DType::FP16, {4, n});
    shape(b_fold, DType::FP32, {n});
    shape(state, DType::FP32, {256});
    shape(next_token, DType::I32, {1});
    disjoint(next_token, state);
    auto scope       = workspace.scope();
    void* x_buf      = workspace.alloc_bytes(5120 * 2).data;
    auto* guz        = static_cast<float*>(workspace.alloc_bytes(768 * sizeof(float)).data);
    auto* m_latent   = static_cast<float*>(workspace.alloc_bytes(256 * sizeof(float)).data);
    auto* part_vals  = static_cast<float*>(workspace.alloc_bytes(272 * sizeof(float)).data);
    auto* part_ids   = static_cast<int*>(workspace.alloc_bytes(272 * sizeof(int)).data);
    detail::inkling2_dspark_rnn_step_folded_launch(
        prev_token, markov_w1, rnn_proj_weight, rnn_proj_bias, h_in, c_head, base_logits_col,
        w_fold_q, w_fold_scale, b_fold, valid_vocab, state, next_token, x_buf, guz, m_latent,
        part_vals, part_ids, s);
}

void inkling2_dspark_commit_conv_state(const Tensor& pre_state, const Tensor& conv_trace,
                                       Tensor& slot_conv_state, std::int32_t width,
                                       std::int32_t accepted, cudaStream_t s) {
    require(width > 0 && accepted >= 1 && accepted <= width, "dspark commit conv range");
    shape(pre_state, DType::BF16, {6656 * 3, 48});
    shape(conv_trace, DType::BF16, {6656 * width, 48});
    shape(slot_conv_state, DType::BF16, {6656 * 3, 48});
    detail::inkling2_dspark_commit_conv_launch(pre_state, conv_trace, slot_conv_state, width,
                                               accepted, s);
}

namespace {
std::size_t align256(std::size_t bytes) {
    return (bytes + 255U) & ~std::size_t(255);
}

void require_q4_kv_weight(const Weight& weight, const char* name) {
    constexpr std::uint64_t kCodeBytes  = static_cast<std::uint64_t>(1024) * 2304U / 2U;
    constexpr std::uint64_t kScaleBytes = static_cast<std::uint64_t>(1024) * (2304U / 64U) * 2U;
    require(weight.qtype == QType::Q4G64_F16S && weight.layout == QuantLayout::RowSplit &&
                weight.scale_dtype == DType::FP16 && weight.group == 64 &&
                weight.group_size == 64 && weight.ndim == 2 && weight.n == 1024 &&
                weight.k == 2304 && weight.shape[0] == 1024 && weight.shape[1] == 2304 &&
                weight.padded_shape[0] == 1024 && weight.padded_shape[1] == 2304 &&
                weight.qhigh == nullptr && weight.high_plane_bytes == 0 &&
                weight.payload_bytes >= kCodeBytes + kScaleBytes && weight.qdata != nullptr &&
                weight.scales != nullptr,
            name);
}
} // namespace

std::size_t inkling2_dflash2_kv_materialize_workspace_bytes(std::int32_t max_width) {
    require(max_width >= 1 && max_width <= 2048, "dflash2 kv materialize max_width must be 1..2048");
    return align256(static_cast<std::size_t>(1024) * static_cast<std::size_t>(max_width) * 2U);
}

void inkling2_dflash2_kv_materialize(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, WorkspaceArena& workspace, cudaStream_t s) {
    const int w = context.ne[1];
    require(w >= 1 && w <= 2048, "dflash2 kv materialize W must be 1..2048");
    shape(context, DType::BF16, {2304, w, 1});
    shape(positions, DType::I32, {w, 1});
    shape(counts, DType::I32, {1});
    shape(state_slots, DType::I32, {1});
    require(envelope.min_count <= envelope.max_count &&
                envelope.max_count <= static_cast<std::uint32_t>(w),
            "dflash2 kv materialize invalid envelope");

    const std::int32_t padded        = layers[0].cache.padded_capacity;
    const std::int32_t lane_capacity = layers[0].cache.lane_capacity;
    require(padded >= 2048 && lane_capacity > 0, "dflash2 kv materialize invalid cyclic cache");
    for (std::size_t i = 0; i < kContextKVMaterializeLayers; ++i) {
        const auto& l = layers[i];
        require_q4_kv_weight(l.key_weight, "dflash2 kv materialize key_weight");
        require_q4_kv_weight(l.value_weight, "dflash2 kv materialize value_weight");
        shape(l.key_norm_weight, DType::BF16, {128});
        require(l.cache.capacity == 2048 && l.cache.padded_capacity == padded &&
                    l.cache.num_kv_heads == 8 && l.cache.head_dim == 128 &&
                    l.cache.lane_capacity == lane_capacity,
                "dflash2 kv materialize cyclic cache mismatch");
        shape(l.cache.k, DType::BF16, {128, padded, 8, lane_capacity});
        shape(l.cache.v, DType::FP16, {128, padded, 8, lane_capacity});
        disjoint(context, l.cache.k);
        disjoint(context, l.cache.v);
        disjoint(positions, l.cache.k);
        disjoint(positions, l.cache.v);
    }

    auto scope         = workspace.scope();
    Tensor scratch     = workspace.alloc(DType::BF16, {1024, w}, 256);
    detail::inkling2_dflash2_kv_materialize_launch(context, positions, counts, state_slots, layers,
                                                   envelope, scratch, s);
}

std::size_t inkling2_dflash2_conv_prepare_workspace_bytes(std::int32_t max_width) {
    require(max_width >= 1 && max_width <= 16, "dflash2 conv prepare max_width must be 1..16");
    return align256(static_cast<std::size_t>(2304) * static_cast<std::size_t>(max_width) * 2U) +
           align256(static_cast<std::size_t>(576) * static_cast<std::size_t>(max_width) *
                    sizeof(float));
}

void inkling2_dflash2_conv_prepare(const Tensor& residual, const Tensor& norm_weight, float eps,
                                   const Tensor& base_kernel, const Tensor& kernel_proj,
                                   Tensor& prepared, Tensor& finish_delta,
                                   WorkspaceArena& workspace, cudaStream_t s) {
    const int w = residual.ne[1];
    require(w >= 1 && w <= 16 && std::isfinite(eps) && eps > 0.0f,
            "dflash2 conv prepare geometry/eps");
    shape(residual, DType::BF16, {2304, w});
    shape(norm_weight, DType::BF16, {2304});
    shape(base_kernel, DType::BF16, {2304, 2, 2});
    shape(kernel_proj, DType::BF16, {2304, 576});
    shape(prepared, DType::BF16, {2304, w});
    shape(finish_delta, DType::BF16, {144, 2, w});
    disjoint(prepared, residual);
    disjoint(prepared, finish_delta);
    disjoint(finish_delta, residual);
    disjoint(prepared, base_kernel);
    disjoint(prepared, kernel_proj);

    auto scope      = workspace.scope();
    void* normed    = workspace.alloc_bytes(align256(2304ULL * w * 2ULL)).data;
    auto* delta_buf = static_cast<float*>(
        workspace.alloc_bytes(align256(576ULL * w * sizeof(float))).data);
    detail::inkling2_dflash2_conv_prepare_launch(residual, norm_weight, eps, base_kernel,
                                                 kernel_proj, prepared, finish_delta, normed,
                                                 delta_buf, s);
}

void inkling2_dflash2_conv_finish_add(const Tensor& proj_out, const Tensor& base_kernel,
                                      const Tensor& finish_delta, Tensor& residual,
                                      cudaStream_t s) {
    const int w = proj_out.ne[1];
    require(w >= 1 && w <= 16, "dflash2 conv finish W must be 1..16");
    shape(proj_out, DType::BF16, {2304, w});
    shape(base_kernel, DType::BF16, {2304, 2, 2});
    shape(finish_delta, DType::BF16, {144, 2, w});
    shape(residual, DType::BF16, {2304, w});
    disjoint(residual, proj_out);
    disjoint(residual, base_kernel);
    disjoint(residual, finish_delta);
    detail::inkling2_dflash2_conv_finish_add_launch(proj_out, base_kernel, finish_delta, residual,
                                                    s);
}

std::size_t inkling2_dflash2_topk16_and_select_workspace_bytes(std::int32_t max_k) {
    require(max_k >= 1 && max_k <= 15, "dflash2 topk16_and_select max_k must be 1..15");
    return align256(256ULL * static_cast<std::size_t>(max_k) * 2ULL) +
           align256(16ULL * 128ULL * static_cast<std::size_t>(max_k) * sizeof(std::int32_t)) +
           align256(16ULL * 128ULL * static_cast<std::size_t>(max_k) * sizeof(float)) +
           align256(16ULL * static_cast<std::size_t>(max_k) * sizeof(float));
}

void inkling2_dflash2_topk16_and_select(const Tensor& mask_hidden, const Tensor& unary_logits,
                                        const Tensor& hidden_projection, const Tensor& anchors,
                                        const Tensor& predecessor_codebook,
                                        const Tensor& successor_codebook,
                                        const Tensor& base_positions,
                                        const SamplingConfig* configs, Tensor& candidate_ids,
                                        Tensor& drafts, Tensor& proposal_q,
                                        WorkspaceArena& workspace, cudaStream_t s) {
    const int k = mask_hidden.ne[1];
    require(k >= 1 && k <= 15 && configs != nullptr, "dflash2 topk16_and_select K/configs");
    shape(mask_hidden, DType::BF16, {2304, k});
    shape(unary_logits, DType::BF16, {201024, k});
    shape(hidden_projection, DType::BF16, {2304, 256});
    shape(anchors, DType::I32, {1});
    shape(predecessor_codebook, DType::BF16, {256, 201024});
    shape(successor_codebook, DType::BF16, {256, 201024});
    shape(base_positions, DType::I32, {1});
    shape(candidate_ids, DType::I32, {16, k, 1});
    shape(drafts, DType::I32, {k, 1});
    shape(proposal_q, DType::FP32, {16, k, 1});

    auto scope               = workspace.scope();
    void* projected_hidden   = workspace.alloc_bytes(align256(256ULL * k * 2ULL)).data;
    auto* partial_ids        = static_cast<std::int32_t*>(
        workspace.alloc_bytes(align256(16ULL * 128ULL * k * sizeof(std::int32_t))).data);
    auto* partial_vals       = static_cast<float*>(
        workspace.alloc_bytes(align256(16ULL * 128ULL * k * sizeof(float))).data);
    auto* unary_scores       = static_cast<float*>(
        workspace.alloc_bytes(align256(16ULL * k * sizeof(float))).data);

    detail::inkling2_dflash2_topk16_and_select_launch(
        mask_hidden, unary_logits, hidden_projection, anchors, predecessor_codebook,
        successor_codebook, base_positions, configs, candidate_ids, drafts, proposal_q,
        projected_hidden, partial_ids, partial_vals, unary_scores, s);
}

} // namespace ninfer::ops
