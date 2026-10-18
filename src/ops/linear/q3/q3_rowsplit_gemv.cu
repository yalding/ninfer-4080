#include "ops/linear/q3/q3_rowsplit_gemv.cuh"

#include "ops/linear/q3/q3_rowsplit_gemv_staged.cuh"

#include "core/device.h"
#include "ops/linear/q3/q3_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kRowsPerCta = 8;

constexpr std::int32_t div_up(std::int32_t value, std::int32_t divisor) {
    return (value + divisor - 1) / divisor;
}

template <int MaxCols>
void launch_schedule(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = static_cast<std::int32_t>(w.padded_shape[1]);
    const dim3 grid(static_cast<unsigned>(div_up(rows, kRowsPerCta)), 1u, 1u);

    if (k == padded_k) {
        q3_rowsplit_gemv_kernel<MaxCols, true><<<grid, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows,
            k, cols, padded_k);
    } else {
        q3_rowsplit_gemv_kernel<MaxCols, false><<<grid, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows,
            k, cols, padded_k);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q3_gemv_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_schedule<8>(x, w, out, stream);
}

// Staged small-T schedule: a cp.async pipeline over shared-memory code windows, one weight row
// per warp, a whole 128-code window decoded per lane per iteration.
void launch_q3_gemv_r8_c8_staged(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = static_cast<std::int32_t>(w.padded_shape[1]);
    const dim3 grid(static_cast<unsigned>(div_up(rows, q3_gemv_staged::kWarpsPerCta)), 1u, 1u);
    constexpr int kMinBlocks = 6;

    if (k == padded_k) {
        q3_gemv_staged::q3_gemv_staged_kernel<8, true, false,
                                               q3_gemv_staged::kProductionGroupsPerStage,
                                               q3_gemv_staged::kProductionPipelineStages,
                                               kMinBlocks>
            <<<grid, q3_gemv_staged::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            rows, k, cols, padded_k, 1, rows);
    } else {
        q3_gemv_staged::q3_gemv_staged_kernel<8, false, false,
                                               q3_gemv_staged::kProductionGroupsPerStage,
                                               q3_gemv_staged::kProductionPipelineStages,
                                               kMinBlocks>
            <<<grid, q3_gemv_staged::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            rows, k, cols, padded_k, 1, rows);
    }
    CUDA_CHECK(cudaGetLastError());
}

namespace {

template <q3_gemv_staged::Q3GemvEpilogue Epilogue>
void launch_staged_epilogue(const Tensor& x, const Weight& w, void* out0, std::int32_t out0_ld,
                            void* out1, std::int32_t out1_ld, std::int32_t split_rows,
                            cudaStream_t stream) {
    const std::int32_t rows     = w.n;
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = static_cast<std::int32_t>(w.padded_shape[1]);
    const dim3 grid(static_cast<unsigned>(div_up(rows, q3_gemv_staged::kWarpsPerCta)), 1u, 1u);
    constexpr int kMinBlocks = 6;

    if (k == padded_k) {
        q3_gemv_staged::q3_gemv_staged_epilogue_kernel<
            8, true, q3_gemv_staged::kProductionGroupsPerStage,
            q3_gemv_staged::kProductionPipelineStages, kMinBlocks, Epilogue>
            <<<grid, q3_gemv_staged::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out0),
                out0_ld, static_cast<__nv_bfloat16*>(out1), out1_ld, split_rows, rows, k, cols,
                padded_k);
    } else {
        q3_gemv_staged::q3_gemv_staged_epilogue_kernel<
            8, false, q3_gemv_staged::kProductionGroupsPerStage,
            q3_gemv_staged::kProductionPipelineStages, kMinBlocks, Epilogue>
            <<<grid, q3_gemv_staged::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out0),
                out0_ld, static_cast<__nv_bfloat16*>(out1), out1_ld, split_rows, rows, k, cols,
                padded_k);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q3_gemv_r8_c8_staged_pitched(const Tensor& x, const Weight& w, void* out,
                                         std::int32_t out_ld, cudaStream_t stream) {
    launch_staged_epilogue<q3_gemv_staged::Q3GemvEpilogue::Plain>(x, w, out, out_ld, nullptr, 0,
                                                                  w.n, stream);
}

void launch_q3_gemv_r8_c8_staged_split(const Tensor& x, const Weight& w, void* out0,
                                       std::int32_t out0_ld, void* out1, std::int32_t out1_ld,
                                       std::int32_t split_rows, cudaStream_t stream) {
    launch_staged_epilogue<q3_gemv_staged::Q3GemvEpilogue::Split>(x, w, out0, out0_ld, out1,
                                                                  out1_ld, split_rows, stream);
}

void launch_q3_gemv_r8_c8_staged_add(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream) {
    launch_staged_epilogue<q3_gemv_staged::Q3GemvEpilogue::Add>(
        x, w, residual_out.data, residual_out.ne[0], nullptr, 0, w.n, stream);
}

} // namespace ninfer::ops::detail
