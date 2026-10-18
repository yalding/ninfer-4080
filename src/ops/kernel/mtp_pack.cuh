#pragma once

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kMtpAttnRows = 14336;
inline constexpr int kMtpQRows    = 6144;
inline constexpr int kMtpKvRows   = 1024;

template <int D, int Block = 512>
__launch_bounds__(Block)
__global__ void mtp_rmsnorm_pack_fc_input_kernel(
    const __nv_bfloat162* __restrict__ embedding,
    const __nv_bfloat162* __restrict__ embedding_weight,
    const __nv_bfloat162* __restrict__ hidden,
    const __nv_bfloat162* __restrict__ hidden_weight,
    float eps,
    __nv_bfloat162* __restrict__ out) {
    constexpr int kPairs = D / 2;
    constexpr int kPairsPerThread = kPairs / Block;
    static_assert(kPairs % Block == 0);

    const int token = static_cast<int>(blockIdx.x);
    const int half  = static_cast<int>(blockIdx.y);

    const __nv_bfloat162* in_vec = half == 0 ? embedding : hidden;
    const __nv_bfloat162* w_vec  = half == 0 ? embedding_weight : hidden_weight;

    const std::int64_t in_base  = static_cast<std::int64_t>(token) * kPairs;
    const std::int64_t out_base = static_cast<std::int64_t>(token) * (2 * kPairs) + half * kPairs;

    __nv_bfloat162 vals[kPairsPerThread];
    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        const int p = static_cast<int>(threadIdx.x) + i * Block;
        vals[i]     = in_vec[in_base + p];
        const float2 f = __bfloat1622float2(vals[i]);
        local_sum += f.x * f.x + f.y * f.y;
    }

    __shared__ float warp_sums[Block / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<Block>(local_sum, warp_sums);
    if (threadIdx.x == 0) {
        inv_shared = rsqrtf(block_sum / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        const int p = static_cast<int>(threadIdx.x) + i * Block;
        const float2 vf = __bfloat1622float2(vals[i]);
        const float2 wf = __bfloat1622float2(w_vec[p]);
        const float out_x = vf.x * inv * (wf.x + 1.0f);
        const float out_y = vf.y * inv * (wf.y + 1.0f);
        out[out_base + p] = __floats2bfloat162_rn(out_x, out_y);
    }
}

__launch_bounds__(256)
__global__ void mtp_rmsnorm_pack_fc_input_generic_kernel(
    const __nv_bfloat16* __restrict__ embedding,
    const __nv_bfloat16* __restrict__ embedding_weight,
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ hidden_weight,
    float eps,
    __nv_bfloat16* __restrict__ out,
    std::int32_t rows) {
    const int token = static_cast<int>(blockIdx.x);
    const int half  = static_cast<int>(blockIdx.y);

    const __nv_bfloat16* in_ptr = half == 0 ? embedding : hidden;
    const __nv_bfloat16* w_ptr  = half == 0 ? embedding_weight : hidden_weight;

    const std::int64_t in_base  = static_cast<std::int64_t>(token) * rows;
    const std::int64_t out_base = static_cast<std::int64_t>(token) * (2 * rows) + half * rows;

    float sum = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < rows; i += static_cast<int>(blockDim.x)) {
        const float v = __bfloat162float(in_ptr[in_base + i]);
        sum += v * v;
    }
    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / static_cast<float>(rows) + eps);
    for (int i = static_cast<int>(threadIdx.x); i < rows; i += static_cast<int>(blockDim.x)) {
        const float v = __bfloat162float(in_ptr[in_base + i]);
        const float w = __bfloat162float(w_ptr[i]);
        out[out_base + i] = __float2bfloat16_rn(v * inv * (w + 1.0f));
    }
}

__global__ void mtp_pack_fc_input_kernel(const __nv_bfloat16* embedding_norm,
                                         const __nv_bfloat16* hidden_norm, __nv_bfloat16* out,
                                         std::int32_t rows) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) { return; }

    const int token             = static_cast<int>(blockIdx.y);
    const std::int64_t in_idx   = static_cast<std::int64_t>(token) * rows + row;
    const std::int64_t out_base = static_cast<std::int64_t>(token) * (2 * rows);
    out[out_base + row]         = embedding_norm[in_idx];
    out[out_base + rows + row]  = hidden_norm[in_idx];
}

__global__ void mtp_split_attn_in_kernel(const __nv_bfloat16* attn_in, __nv_bfloat16* q,
                                         __nv_bfloat16* k, __nv_bfloat16* gate, __nv_bfloat16* v,
                                         std::int32_t tokens) {
    const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(kMtpAttnRows) * tokens;
    if (idx >= n) { return; }

    const int row             = static_cast<int>(idx % kMtpAttnRows);
    const int token           = static_cast<int>(idx / kMtpAttnRows);
    const __nv_bfloat16 value = attn_in[idx];

    if (row < kMtpQRows) {
        q[static_cast<std::int64_t>(token) * kMtpQRows + row] = value;
        return;
    }
    if (row < kMtpQRows + kMtpKvRows) {
        const int local                                          = row - kMtpQRows;
        k[static_cast<std::int64_t>(token) * kMtpKvRows + local] = value;
        return;
    }
    if (row < kMtpQRows + kMtpKvRows + kMtpQRows) {
        const int local                                            = row - kMtpQRows - kMtpKvRows;
        gate[static_cast<std::int64_t>(token) * kMtpQRows + local] = value;
        return;
    }

    const int local = row - kMtpQRows - kMtpKvRows - kMtpQRows;
    v[static_cast<std::int64_t>(token) * kMtpKvRows + local] = value;
}

} // namespace ninfer::ops
