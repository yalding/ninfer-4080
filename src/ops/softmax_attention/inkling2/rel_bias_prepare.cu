#include "ops/softmax_attention/inkling2/launch.h"
#include "core/device.h"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::inkling2_detail {
template <bool OutBf16>
__global__ void rel_kernel(const float* __restrict__ r, const float* __restrict__ bias,
                           const void* __restrict__ sink, bool bf, void* __restrict__ out) {
    const int ht = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const float4* r4 = reinterpret_cast<const float4*>(r + static_cast<std::int64_t>(ht) * 16);
    const float4 r0 = r4[0], r1 = r4[1], r2 = r4[2], r3 = r4[3];
    const float s = bf ? __bfloat162float(static_cast<const __nv_bfloat16*>(sink)[ht])
                       : static_cast<const float*>(sink)[ht];
    const std::int64_t base = static_cast<std::int64_t>(ht) * 1025;
    for (int d = tid; d < 1024; d += 256) {
        const float4* b4 = reinterpret_cast<const float4*>(bias + static_cast<std::int64_t>(d) * 16);
        const float4 b0 = b4[0], b1 = b4[1], b2 = b4[2], b3 = b4[3];
        float sum = fmaf(r0.x, b0.x, 0.0f);
        sum = fmaf(r0.y, b0.y, sum);
        sum = fmaf(r0.z, b0.z, sum);
        sum = fmaf(r0.w, b0.w, sum);
        sum = fmaf(r1.x, b1.x, sum);
        sum = fmaf(r1.y, b1.y, sum);
        sum = fmaf(r1.z, b1.z, sum);
        sum = fmaf(r1.w, b1.w, sum);
        sum = fmaf(r2.x, b2.x, sum);
        sum = fmaf(r2.y, b2.y, sum);
        sum = fmaf(r2.z, b2.z, sum);
        sum = fmaf(r2.w, b2.w, sum);
        sum = fmaf(r3.x, b3.x, sum);
        sum = fmaf(r3.y, b3.y, sum);
        sum = fmaf(r3.z, b3.z, sum);
        sum = fmaf(r3.w, b3.w, sum);
        if constexpr (OutBf16) {
            static_cast<__nv_bfloat16*>(out)[base + d] = __float2bfloat16_rn(sum - s);
        } else {
            static_cast<float*>(out)[base + d] = sum - s;
        }
    }
    if (tid == 0) {
        if constexpr (OutBf16) {
            static_cast<__nv_bfloat16*>(out)[base + 1024] = __float2bfloat16_rn(-s);
        } else {
            static_cast<float*>(out)[base + 1024] = -s;
        }
    }
}

void rel_launch(const Tensor& r, const Tensor& bias, const Tensor& sink, Tensor& out,
                cudaStream_t s) {
    const int ht_count = r.ne[1] * r.ne[2];
    if (out.dtype == DType::BF16) {
        rel_kernel<true><<<ht_count, 256, 0, s>>>(
            static_cast<const float*>(r.data), static_cast<const float*>(bias.data), sink.data,
            sink.dtype == DType::BF16, out.data);
    } else {
        rel_kernel<false><<<ht_count, 256, 0, s>>>(
            static_cast<const float*>(r.data), static_cast<const float*>(bias.data), sink.data,
            sink.dtype == DType::BF16, out.data);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::inkling2_detail
