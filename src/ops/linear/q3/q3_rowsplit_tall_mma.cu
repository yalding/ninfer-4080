#include "ops/linear/q3/q3_rowsplit_tall_mma.cuh"

#include "core/device.h"
#include "ops/linear/q3/q3_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

void require_tall_linear_shape(const Tensor& x, const Weight& w, const Tensor& out) {
    if (w.n <= 0 || w.n % q3_tall::kRows != 0 || w.padded_shape[1] <= 0 ||
        w.padded_shape[1] % q3_tall::kStepK != 0 || w.padded_shape[1] < w.k) {
        throw std::invalid_argument("q3 tall linear: unsupported weight shape");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("q3 tall linear: unsupported tensor shape");
    }
}

} // namespace

void launch_q3_mma_tall_r128_c64(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    require_tall_linear_shape(x, w, out);
    const q3_tall::Q3LinearProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), out.ne[0]};
    q3_tall::launch<64>(problem, w.n / q3_tall::kRows, static_cast<const __nv_bfloat16*>(x.data),
                        x.ne[0], static_cast<std::int32_t>(w.padded_shape[1]), x.ne[1], stream);
}

void launch_q3_mma_tall_r128_c128(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    require_tall_linear_shape(x, w, out);
    const q3_tall::Q3LinearProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), out.ne[0]};
    q3_tall::launch<128>(problem, w.n / q3_tall::kRows, static_cast<const __nv_bfloat16*>(x.data),
                         x.ne[0], static_cast<std::int32_t>(w.padded_shape[1]), x.ne[1], stream);
}

void launch_q3_mma_tall_swiglu_r64_c128(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    const std::int32_t intermediate = out.ne[0];
    if (w.n != 2 * intermediate || intermediate <= 0 || intermediate % 64 != 0 ||
        w.padded_shape[1] <= 0 || w.padded_shape[1] % q3_tall::kStepK != 0 ||
        w.padded_shape[1] < w.k || x.ne[0] != w.k || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("q3 tall swiglu: unsupported tensor shape");
    }
    const q3_tall::Q3SwiGluProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), intermediate};
    q3_tall::launch<128>(problem, intermediate / 64, static_cast<const __nv_bfloat16*>(x.data),
                         x.ne[0], static_cast<std::int32_t>(w.padded_shape[1]), x.ne[1], stream);
}

void launch_q3_mma_tall_r128_c64_pitched(const Tensor& x, const Weight& w, void* out,
                                         std::int32_t out_ld, cudaStream_t stream) {
    const q3_tall::Q3LinearProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out), out_ld};
    q3_tall::launch<64>(problem, w.n / q3_tall::kRows, static_cast<const __nv_bfloat16*>(x.data),
                        x.ne[0], static_cast<std::int32_t>(w.padded_shape[1]), x.ne[1], stream);
}

void launch_q3_mma_tall_r128_c128_pitched(const Tensor& x, const Weight& w, void* out,
                                          std::int32_t out_ld, cudaStream_t stream) {
    const q3_tall::Q3LinearProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out), out_ld};
    q3_tall::launch<128>(problem, w.n / q3_tall::kRows, static_cast<const __nv_bfloat16*>(x.data),
                         x.ne[0], static_cast<std::int32_t>(w.padded_shape[1]), x.ne[1], stream);
}

void launch_q3_mma_tall_r128_c64_split(const Tensor& x, const Weight& w, void* out0,
                                       std::int32_t out0_ld, void* out1, std::int32_t out1_ld,
                                       std::int32_t split_rows, cudaStream_t stream) {
    const q3_tall::Q3SplitLinearProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out0),         out0_ld,
        static_cast<__nv_bfloat16*>(out1),         out1_ld,
        split_rows / q3_tall::kRows};
    q3_tall::launch<64>(problem, w.n / q3_tall::kRows, static_cast<const __nv_bfloat16*>(x.data),
                        x.ne[0], static_cast<std::int32_t>(w.padded_shape[1]), x.ne[1], stream);
}

void launch_q3_mma_tall_r128_c128_split(const Tensor& x, const Weight& w, void* out0,
                                        std::int32_t out0_ld, void* out1, std::int32_t out1_ld,
                                        std::int32_t split_rows, cudaStream_t stream) {
    const q3_tall::Q3SplitLinearProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out0),         out0_ld,
        static_cast<__nv_bfloat16*>(out1),         out1_ld,
        split_rows / q3_tall::kRows};
    q3_tall::launch<128>(problem, w.n / q3_tall::kRows, static_cast<const __nv_bfloat16*>(x.data),
                         x.ne[0], static_cast<std::int32_t>(w.padded_shape[1]), x.ne[1], stream);
}

void launch_q3_mma_tall_r128_c64_add(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream) {
    require_tall_linear_shape(x, w, residual_out);
    const q3_tall::Q3LinearAddProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(residual_out.data), residual_out.ne[0]};
    q3_tall::launch<64>(problem, w.n / q3_tall::kRows, static_cast<const __nv_bfloat16*>(x.data),
                        x.ne[0], static_cast<std::int32_t>(w.padded_shape[1]), x.ne[1], stream);
}

void launch_q3_mma_tall_r128_c128_add(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    require_tall_linear_shape(x, w, residual_out);
    const q3_tall::Q3LinearAddProblem problem{
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat16*>(residual_out.data), residual_out.ne[0]};
    q3_tall::launch<128>(problem, w.n / q3_tall::kRows, static_cast<const __nv_bfloat16*>(x.data),
                         x.ne[0], static_cast<std::int32_t>(w.padded_shape[1]), x.ne[1], stream);
}

} // namespace ninfer::ops::detail
