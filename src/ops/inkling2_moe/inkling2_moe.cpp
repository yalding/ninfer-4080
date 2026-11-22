#include "ninfer/ops/inkling2_moe.h"
#include "ops/inkling2_moe/launch.h"
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
} // namespace

std::size_t inkling2_moe_workspace_capacity_bytes(QType q, int lo, int hi) {
    if ((q != QType::Q3G128_F16S && q != QType::Q4G64_F16S) || lo <= 0 || hi < lo)
        throw std::invalid_argument("inkling2_moe: workspace envelope");
    return aligned(std::size_t(hi) * 128 * sizeof(float)) +
           aligned(std::size_t(hi) * 8 * sizeof(int)) +
           aligned(std::size_t(hi) * 8 * sizeof(float)) +
           aligned(std::size_t(hi) * 8 * 1152 * sizeof(float));
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
    auto* probs =
        static_cast<float*>(workspace.alloc_bytes(std::size_t(t) * 128 * sizeof(float)).data);
    auto* ids = static_cast<int*>(workspace.alloc_bytes(std::size_t(t) * 8 * sizeof(int)).data);
    auto* alpha =
        static_cast<float*>(workspace.alloc_bytes(std::size_t(t) * 8 * sizeof(float)).data);
    auto* act =
        static_cast<float*>(workspace.alloc_bytes(std::size_t(t) * 8 * 1152 * sizeof(float)).data);
    inkling2_detail::moe_launch(x, w, out, probs, ids, alpha, act, stream);
}
} // namespace ninfer::ops
