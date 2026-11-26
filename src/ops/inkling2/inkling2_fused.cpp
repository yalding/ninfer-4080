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
                          Tensor& out, cudaStream_t s) {
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
    detail::inkling2_sconv_launch(x, w, dynamic, state_in, state_out, residual, out, s);
}

void inkling2_fused_sconv_kv(const Tensor& kx, const Tensor& k_weight, const Tensor& k_dynamic,
                             const Tensor& k_state_in, Tensor& k_state_out, Tensor& k_out,
                             const Tensor& vx, const Tensor& v_weight, const Tensor& v_dynamic,
                             const Tensor& v_state_in, Tensor& v_state_out, Tensor& v_out,
                             cudaStream_t s) {
    const int t = kx.ne[1];
    auto check_branch = [&](const Tensor& x, const Tensor& w, const Tensor& dyn,
                            const Tensor& st_in, const Tensor& st_out, const Tensor& out) {
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
    };
    check_branch(kx, k_weight, k_dynamic, k_state_in, k_state_out, k_out);
    check_branch(vx, v_weight, v_dynamic, v_state_in, v_state_out, v_out);
    require(k_dynamic.data != nullptr == (v_dynamic.data != nullptr),
            "paired sconv_kv dynamic taps mismatch");
    detail::inkling2_sconv_kv_launch(kx, k_weight, k_dynamic, k_state_in, k_state_out, k_out, vx,
                                     v_weight, v_dynamic, v_state_in, v_state_out, v_out, s);
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

} // namespace ninfer::ops
