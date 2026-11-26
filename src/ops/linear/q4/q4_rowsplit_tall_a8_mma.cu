#include "ops/linear/q4/q4_rowsplit_tall_a8_mma.cuh"

#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q4/q4_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
static_assert(q4_tall_a8::kRows == kQ4A8Rows, "A8 dispatch rows must match the tall engine");
static_assert(q4_tall_a8::kStepK == kQ4A8StepK, "A8 dispatch K step must match the tall engine");

namespace {

void require_q4_tall_a8_shape(const A8G64Activation& act, const Weight& w, const Tensor& out) {
    if (w.n <= 0 || (w.n % q4_tall_a8::kRows) != 0 || w.k <= 0 || w.padded_shape[1] != w.k ||
        (w.k % q4_tall_a8::kStepK) != 0) {
        throw std::invalid_argument("q4 tall A8 linear: unsupported weight shape");
    }
    if (out.ne[0] != w.n || out.ne[1] != act.q.ne[1] || act.q.ne[0] != w.k ||
        act.scale.ne[0] != act.q.ne[1] || act.scale.ne[1] != w.k / 64) {
        throw std::invalid_argument("q4 tall A8 linear: unsupported tensor shape");
    }
}

} // namespace

void launch_q4_mma_tall_a8_r128_c64(const A8G64Activation& act, const Weight& w, Tensor& out,
                                    cudaStream_t stream) {
    require_q4_tall_a8_shape(act, w, out);
    q4_tall_a8::launch<64>(static_cast<const std::int8_t*>(act.q.data),
                           static_cast<const float*>(act.scale.data),
                           static_cast<const std::uint8_t*>(w.qdata),
                           static_cast<const std::uint16_t*>(w.scales),
                           static_cast<__nv_bfloat16*>(out.data), w.n, w.k, act.q.ne[1], stream);
}

void launch_q4_mma_tall_a8_r128_c128(const A8G64Activation& act, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    require_q4_tall_a8_shape(act, w, out);
    q4_tall_a8::launch<128>(static_cast<const std::int8_t*>(act.q.data),
                            static_cast<const float*>(act.scale.data),
                            static_cast<const std::uint8_t*>(w.qdata),
                            static_cast<const std::uint16_t*>(w.scales),
                            static_cast<__nv_bfloat16*>(out.data), w.n, w.k, act.q.ne[1], stream);
}

void launch_q4_mma_tall_a8_r128_c256(const A8G64Activation& act, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    require_q4_tall_a8_shape(act, w, out);
    q4_tall_a8::launch<256>(static_cast<const std::int8_t*>(act.q.data),
                            static_cast<const float*>(act.scale.data),
                            static_cast<const std::uint8_t*>(w.qdata),
                            static_cast<const std::uint16_t*>(w.scales),
                            static_cast<__nv_bfloat16*>(out.data), w.n, w.k, act.q.ne[1], stream);
}

} // namespace ninfer::ops::detail
