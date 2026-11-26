#include "ninfer/ops/inkling2_moe.h"
#include "ninfer/ops/inkling2_fused.h"
#include "ops/inkling2_moe/launch.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {
std::size_t aligned(std::size_t n) { return (n + 255) & ~std::size_t(255); }

void quant(const Weight& w, int rows, int k, QType type) {
    const int group = type == QType::Q3G128_F16S ? 128 : 64;
    if (w.qtype != type || w.layout != QuantLayout::RowSplit || w.n != rows || w.k != k ||
        w.padded_shape[1] != k || w.group != group || w.scale_dtype != DType::FP16 || !w.qdata ||
        !w.scales)
        throw std::invalid_argument("inkling2_moe: invalid expert weight");
}

std::size_t small_t_workspace_bytes(int hi) {
    const int t = std::min(hi, inkling2_detail::kInkling2MoePrefillMinTokens - 1);
    return aligned(std::size_t(t) * 128 * sizeof(float)) +
           aligned(std::size_t(t) * 8 * sizeof(int)) +
           aligned(std::size_t(t) * 8 * sizeof(float)) +
           aligned(std::size_t(t) * 8 * 1152 * sizeof(float)) +
           aligned(std::size_t(t) * 2304 * 2);
}

std::size_t prefill_workspace_bytes(int hi) {
    const int t = std::min(hi, inkling2_detail::kInkling2MoeSliceMax);
    const int route_tiles = (t + 7) / 8;
    const int max_jobs = (t * 8) / inkling2_detail::kInkling2MoeVerifyExpertBN + 128;
    std::size_t bytes = aligned(std::size_t(t) * 128 * sizeof(float)) +
                        aligned(std::size_t(t) * 8 * sizeof(int)) +
                        aligned(std::size_t(t) * 8 * sizeof(float)) +
                        aligned(std::size_t(t) * 8 * sizeof(int)) +
                        aligned(std::size_t(t) * 8 * sizeof(int)) +
                        aligned(std::size_t(t) * 8 * sizeof(int)) +
                        aligned(std::size_t(route_tiles) * 128 * sizeof(int)) +
                        aligned(std::size_t(route_tiles) * 128 * sizeof(int)) +
                        aligned(129 * sizeof(int)) +
                        aligned(std::size_t(max_jobs) * sizeof(int)) +
                        aligned(std::size_t(max_jobs) * sizeof(int)) +
                        aligned(2 * sizeof(int)) +
                        aligned(std::size_t(t) * 8 * 1152 * 2) +
                        aligned(std::size_t(t) * 8 * 2304 * 2);
    if (hi >= 64) {
        bytes += aligned(std::size_t(t) * 2304) +
                 aligned(std::size_t(t) * (2304 / 64) * sizeof(float)) +
                 aligned(std::size_t(t) * 8 * 1152) +
                 aligned(std::size_t(t) * 8 * (1152 / 64) * sizeof(float));
    }
    return bytes;
}
} // namespace

std::size_t inkling2_moe_workspace_capacity_bytes(QType q, int lo, int hi) {
    if ((q != QType::Q3G128_F16S && q != QType::Q4G64_F16S) || lo <= 0 || hi < lo)
        throw std::invalid_argument("inkling2_moe: workspace envelope");
    std::size_t peak = 0;
    if (lo < inkling2_detail::kInkling2MoePrefillMinTokens)
        peak = std::max(peak, small_t_workspace_bytes(hi));
    if (hi >= inkling2_detail::kInkling2MoePrefillMinTokens)
        peak = std::max(peak, prefill_workspace_bytes(hi));
    return peak;
}

void inkling2_moe(const Tensor& x, const Inkling2MoeWeights& w, Tensor& out,
                  WorkspaceArena& workspace, cudaStream_t stream) {
    int t = x.ne[1];
    if (!x.data || !out.data || x.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        !x.is_contiguous() || !out.is_contiguous() || x.ne[0] != 2304 || out.ne[0] != 2304 ||
        t <= 0 || out.ne[1] != t || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[2] != 1 ||
        out.ne[3] != 1 || x.data == out.data)
        throw std::invalid_argument("inkling2_moe: input/output");
    (void)inkling2_moe_workspace_capacity_bytes(w.routed_gate_up.qtype, t, t);
    if (w.router.qtype != QType::BF16_CTRL || w.router.n != 128 || w.router.k != 2304 ||
        !w.router.qdata || w.exp_probs_bias.qtype != QType::FP32_CTRL ||
        w.exp_probs_bias.n != 128 || !w.exp_probs_bias.qdata)
        throw std::invalid_argument("inkling2_moe: router/bias");
    quant(w.routed_gate_up, 128 * 2304, 2304, w.routed_gate_up.qtype);
    quant(w.routed_down, 128 * 2304, 1152, w.routed_gate_up.qtype);
    auto scope = workspace.scope();
    if (t < inkling2_detail::kInkling2MoePrefillMinTokens) {
        auto* probs =
            static_cast<float*>(workspace.alloc_bytes(std::size_t(t) * 128 * sizeof(float)).data);
        auto* ids =
            static_cast<int*>(workspace.alloc_bytes(std::size_t(t) * 8 * sizeof(int)).data);
        auto* alpha =
            static_cast<float*>(workspace.alloc_bytes(std::size_t(t) * 8 * sizeof(float)).data);
        auto* act = static_cast<float*>(
            workspace.alloc_bytes(std::size_t(t) * 8 * 1152 * sizeof(float)).data);
        inkling2_detail::moe_launch(x, w, out, probs, ids, alpha, act, stream);
        return;
    }

    const int slice_max = std::min(t, inkling2_detail::kInkling2MoeSliceMax);
    const int route_tiles = (slice_max + 7) / 8;
    const int max_jobs = (slice_max * 8) / inkling2_detail::kInkling2MoeVerifyExpertBN + 128;
    inkling2_detail::Inkling2MoePrefillWorkspace ws{};
    ws.router_probs =
        static_cast<float*>(workspace.alloc_bytes(std::size_t(slice_max) * 128 * sizeof(float)).data);
    ws.token_ids =
        static_cast<int*>(workspace.alloc_bytes(std::size_t(slice_max) * 8 * sizeof(int)).data);
    ws.token_alpha =
        static_cast<float*>(workspace.alloc_bytes(std::size_t(slice_max) * 8 * sizeof(float)).data);
    ws.local_rank =
        static_cast<int*>(workspace.alloc_bytes(std::size_t(slice_max) * 8 * sizeof(int)).data);
    ws.packed_index =
        static_cast<int*>(workspace.alloc_bytes(std::size_t(slice_max) * 8 * sizeof(int)).data);
    ws.packed_token =
        static_cast<int*>(workspace.alloc_bytes(std::size_t(slice_max) * 8 * sizeof(int)).data);
    ws.tile_counts = static_cast<int*>(
        workspace.alloc_bytes(std::size_t(route_tiles) * 128 * sizeof(int)).data);
    ws.tile_bases = static_cast<int*>(
        workspace.alloc_bytes(std::size_t(route_tiles) * 128 * sizeof(int)).data);
    ws.expert_offsets = static_cast<int*>(workspace.alloc_bytes(129 * sizeof(int)).data);
    ws.route_job_experts =
        static_cast<int*>(workspace.alloc_bytes(std::size_t(max_jobs) * sizeof(int)).data);
    ws.route_job_columns =
        static_cast<int*>(workspace.alloc_bytes(std::size_t(max_jobs) * sizeof(int)).data);
    ws.route_job_count = static_cast<int*>(workspace.alloc_bytes(2 * sizeof(int)).data);
    ws.routed_activation = workspace.alloc_bytes(std::size_t(slice_max) * 8 * 1152 * 2).data;
    ws.grouped_output = workspace.alloc_bytes(std::size_t(slice_max) * 8 * 2304 * 2).data;
    if (t >= 64) {
        ws.x_a8_q = static_cast<std::int8_t*>(
            workspace.alloc_bytes(std::size_t(slice_max) * 2304).data);
        ws.x_a8_scale = static_cast<float*>(
            workspace.alloc_bytes(std::size_t(slice_max) * (2304 / 64) * sizeof(float)).data);
        ws.act_a8_q = static_cast<std::int8_t*>(
            workspace.alloc_bytes(std::size_t(slice_max) * 8 * 1152).data);
        ws.act_a8_scale = static_cast<float*>(
            workspace.alloc_bytes(std::size_t(slice_max) * 8 * (1152 / 64) * sizeof(float)).data);
    }

    const auto* x_bytes = static_cast<const std::byte*>(x.data);
    auto* out_bytes = static_cast<std::byte*>(out.data);
    for (int offset = 0; offset < t; offset += slice_max) {
        const int slice_t = std::min(slice_max, t - offset);
        inkling2_detail::moe_prefill_launch(
            x_bytes + std::size_t(offset) * 2304 * 2, w,
            out_bytes + std::size_t(offset) * 2304 * 2, slice_t, ws, stream);
    }
}

void inkling2_moe_post_sconv(const Tensor& x, const Inkling2MoeWeights& w,
                             const Tensor& post_conv_weight, const Tensor& state_in,
                             Tensor& state_out, const Tensor& residual_in, Tensor& out,
                             Tensor moe_trace, WorkspaceArena& workspace, cudaStream_t stream) {
    int t = x.ne[1];
    if (!x.data || !out.data || x.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        !x.is_contiguous() || !out.is_contiguous() || x.ne[0] != 2304 || out.ne[0] != 2304 ||
        t <= 0 || t > 16 || out.ne[1] != t || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[2] != 1 ||
        out.ne[3] != 1 || x.data == out.data)
        throw std::invalid_argument("inkling2_moe_post_sconv: input/output");
    (void)inkling2_moe_workspace_capacity_bytes(w.routed_gate_up.qtype, t, t);
    if (w.router.qtype != QType::BF16_CTRL || w.router.n != 128 || w.router.k != 2304 ||
        !w.router.qdata || w.exp_probs_bias.qtype != QType::FP32_CTRL ||
        w.exp_probs_bias.n != 128 || !w.exp_probs_bias.qdata)
        throw std::invalid_argument("inkling2_moe_post_sconv: router/bias");
    quant(w.routed_gate_up, 128 * 2304, 2304, w.routed_gate_up.qtype);
    quant(w.routed_down, 128 * 2304, 1152, w.routed_gate_up.qtype);
    if (moe_trace.data &&
        (moe_trace.dtype != DType::BF16 || !moe_trace.is_contiguous() || moe_trace.ne[0] != 2304 ||
         moe_trace.ne[1] != t || moe_trace.ne[2] != 1 || moe_trace.ne[3] != 1))
        throw std::invalid_argument("inkling2_moe_post_sconv: invalid moe_trace");

    auto scope = workspace.scope();
    if (t < inkling2_detail::kInkling2MoePrefillMinTokens) {
        auto* probs =
            static_cast<float*>(workspace.alloc_bytes(std::size_t(t) * 128 * sizeof(float)).data);
        auto* ids =
            static_cast<int*>(workspace.alloc_bytes(std::size_t(t) * 8 * sizeof(int)).data);
        auto* alpha =
            static_cast<float*>(workspace.alloc_bytes(std::size_t(t) * 8 * sizeof(float)).data);
        auto* act = static_cast<float*>(
            workspace.alloc_bytes(std::size_t(t) * 8 * 1152 * sizeof(float)).data);
        Tensor moe_tmp(workspace.alloc_bytes(std::size_t(t) * 2304 * 2).data, DType::BF16,
                       {2304, t});
        inkling2_detail::moe_launch(x, w, moe_tmp, probs, ids, alpha, act, stream);
        inkling2_fused_sconv(moe_tmp, post_conv_weight, {}, state_in, state_out, residual_in, out,
                             stream, moe_trace);
        return;
    }

    const int route_tiles = (t + 7) / 8;
    const int max_jobs    = (t * 8) / inkling2_detail::kInkling2MoeVerifyExpertBN + 128;
    inkling2_detail::Inkling2MoePrefillWorkspace ws{};
    ws.router_probs =
        static_cast<float*>(workspace.alloc_bytes(std::size_t(t) * 128 * sizeof(float)).data);
    ws.token_ids =
        static_cast<int*>(workspace.alloc_bytes(std::size_t(t) * 8 * sizeof(int)).data);
    ws.token_alpha =
        static_cast<float*>(workspace.alloc_bytes(std::size_t(t) * 8 * sizeof(float)).data);
    ws.local_rank =
        static_cast<int*>(workspace.alloc_bytes(std::size_t(t) * 8 * sizeof(int)).data);
    ws.packed_index =
        static_cast<int*>(workspace.alloc_bytes(std::size_t(t) * 8 * sizeof(int)).data);
    ws.packed_token =
        static_cast<int*>(workspace.alloc_bytes(std::size_t(t) * 8 * sizeof(int)).data);
    ws.tile_counts = static_cast<int*>(
        workspace.alloc_bytes(std::size_t(route_tiles) * 128 * sizeof(int)).data);
    ws.tile_bases = static_cast<int*>(
        workspace.alloc_bytes(std::size_t(route_tiles) * 128 * sizeof(int)).data);
    ws.expert_offsets = static_cast<int*>(workspace.alloc_bytes(129 * sizeof(int)).data);
    ws.route_job_experts =
        static_cast<int*>(workspace.alloc_bytes(std::size_t(max_jobs) * sizeof(int)).data);
    ws.route_job_columns =
        static_cast<int*>(workspace.alloc_bytes(std::size_t(max_jobs) * sizeof(int)).data);
    ws.route_job_count   = static_cast<int*>(workspace.alloc_bytes(2 * sizeof(int)).data);
    ws.routed_activation = workspace.alloc_bytes(std::size_t(t) * 8 * 1152 * 2).data;
    ws.grouped_output    = workspace.alloc_bytes(std::size_t(t) * 8 * 2304 * 2).data;

    inkling2_detail::moe_prefill_post_sconv_launch(
        x.data, w, post_conv_weight, state_in, state_out, residual_in, out.data, moe_trace.data,
        t, ws, stream);
}
} // namespace ninfer::ops
