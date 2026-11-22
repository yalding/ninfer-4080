#include "ops/softmax_attention/inkling2/launch.h"
#include "core/device.h"
#include <cuda_bf16.h>

namespace ninfer::ops::inkling2_detail {
__global__ void rel_kernel(const float* r, const float* bias, const void* sink, bool bf, float* out,
                           std::int64_t n) {
    auto i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = i % 1025, ht = i / 1025;
    float sum = 0;
    if (d < 1024)
        for (int j = 0; j < 16; ++j) sum = fmaf(r[j + 16 * ht], bias[j + 16 * d], sum);
    float s = bf ? __bfloat162float(static_cast<const __nv_bfloat16*>(sink)[ht])
                 : static_cast<const float*>(sink)[ht];
    out[i]  = sum - s;
}

void rel_launch(const Tensor& r, const Tensor& bias, const Tensor& sink, Tensor& out,
                cudaStream_t s) {
    auto n = out.numel();
    rel_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(
        static_cast<const float*>(r.data), static_cast<const float*>(bias.data), sink.data,
        sink.dtype == DType::BF16, static_cast<float*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::inkling2_detail
