#include "ops/inkling2/launch.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/kernel/layer_norm.cuh"

#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using BF = __nv_bfloat16;

constexpr int kVisHidden   = 1024;
constexpr int kVisHeads    = 16;
constexpr int kVisHeadDim  = 64;
constexpr int kVisPatchDim = 1200;
constexpr int kVisQkvRows  = 3072;

__global__ __launch_bounds__(256) void vision_cls_embed_kernel(const BF* __restrict__ cls_embed,
                                                               const BF* __restrict__ cls_pos,
                                                               BF* __restrict__ out) {
    const int base = static_cast<int>(threadIdx.x) * 4;
    if (base >= kVisHidden) { return; }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = base + i;
        out[d] = __float2bfloat16_rn(__bfloat162float(cls_embed[d]) + __bfloat162float(cls_pos[d]));
    }
}

__global__ __launch_bounds__(128) void vision_patch_embed_mma_kernel(
    const BF* __restrict__ patches, const BF* __restrict__ weight, const BF* __restrict__ pos_table,
    const std::int32_t* __restrict__ pos_indices, const float* __restrict__ pos_weights,
    std::int32_t patch_count, BF* __restrict__ out) {
    constexpr int BM     = 64;
    constexpr int BN     = 32;
    constexpr int BK     = 16;
    constexpr int Stride = 24;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int block_m = static_cast<int>(blockIdx.y) * BM;
    const int block_n = static_cast<int>(blockIdx.x) * BN;

    __shared__ __align__(16) BF w_s[2][BM * Stride];
    __shared__ __align__(16) BF x_s[2][BN * Stride];

    auto stage_tile = [&](int stage, int k0) {
        for (int idx = tid; idx < BM * (BK / 8); idx += 128) {
            const int r   = idx >> 1;
            const int c   = (idx & 1) << 3;
            const int n   = block_m + r;
            const BF* src = weight + static_cast<std::int64_t>(n) * kVisPatchDim + k0 + c;
            cp_async_zfill<16, Cache::ca>(&w_s[stage][r * Stride + c], src, 16);
        }
        for (int idx = tid; idx < BN * (BK / 8); idx += 128) {
            const int r      = idx >> 1;
            const int c      = (idx & 1) << 3;
            const int p      = block_n + r;
            const bool valid = p < patch_count;
            const BF* src =
                patches + static_cast<std::int64_t>(valid ? p : 0) * kVisPatchDim + k0 + c;
            cp_async_zfill<16, Cache::cg>(&x_s[stage][r * Stride + c], src, valid ? 16 : 0);
        }
        cp_commit();
    };

    float acc[4][4]{};
    constexpr int kSteps = kVisPatchDim / BK;
    stage_tile(0, 0);

    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_row = warp * 16 + a_rin + ((a_mat & 1) << 3);
    const int a_col = (a_mat >> 1) << 3;
    const int b_mat = (lane >> 3) & 1;
    const int b_rin = lane & 7;
    const int b_col = b_mat << 3;

    for (int ks = 0; ks < kSteps; ++ks) {
        const int cur = ks & 1;
        const int nxt = cur ^ 1;
        if (ks + 1 < kSteps) { stage_tile(nxt, (ks + 1) * BK); }
        if (ks + 1 < kSteps) {
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        unsigned a0, a1, a2, a3;
        ldmatrix_x4(a0, a1, a2, a3, smem_addr(&w_s[cur][a_row * Stride + a_col]));
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            unsigned b0, b1;
            ldmatrix_x2(b0, b1, smem_addr(&x_s[cur][(nt * 8 + b_rin) * Stride + b_col]));
            mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a0, a1, a2, a3, b0, b1);
        }
        __syncthreads();
    }

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int n0  = block_m + warp * 16 + gid;
    const int n1  = n0 + 8;

    auto add_pos = [&](int n, int p, float val) -> BF {
        const std::int32_t* idx = pos_indices + static_cast<std::int64_t>(p) * 4;
        const float* w          = pos_weights + static_cast<std::int64_t>(p) * 4;
        float pos_val           = 0.0f;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int row_idx = idx[c];
            pos_val           = fmaf(
                __bfloat162float(pos_table[static_cast<std::int64_t>(row_idx) * kVisHidden + n]),
                w[c], pos_val);
        }
        return __float2bfloat16_rn(val + pos_val);
    };

#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int p0 = block_n + nt * 8 + 2 * lid;
        const int p1 = p0 + 1;
        if (p0 < patch_count) {
            out[static_cast<std::int64_t>(p0 + 1) * kVisHidden + n0] = add_pos(n0, p0, acc[nt][0]);
            out[static_cast<std::int64_t>(p0 + 1) * kVisHidden + n1] = add_pos(n1, p0, acc[nt][2]);
        }
        if (p1 < patch_count) {
            out[static_cast<std::int64_t>(p1 + 1) * kVisHidden + n0] = add_pos(n0, p1, acc[nt][1]);
            out[static_cast<std::int64_t>(p1 + 1) * kVisHidden + n1] = add_pos(n1, p1, acc[nt][3]);
        }
    }
}

__global__ __launch_bounds__(32) void vision_rope_2d_kernel(const std::int32_t* __restrict__ coords,
                                                            float rope_theta,
                                                            BF* __restrict__ qkv) {
    const int token = static_cast<int>(blockIdx.x);
    const int head  = static_cast<int>(blockIdx.y);
    const int lane  = static_cast<int>(threadIdx.x);

    const std::int32_t y = coords[static_cast<std::int64_t>(token) * 4 + 0];
    const std::int32_t x = coords[static_cast<std::int64_t>(token) * 4 + 1];
    if (y < 0 || x < 0) { return; }

    const int pair  = lane & 15;
    const bool is_x = (lane >= 16);
    const double inv_freq =
        pow(static_cast<double>(rope_theta), -static_cast<double>(2 * pair) / 32.0);
    const double angle = static_cast<double>(is_x ? x : y) * inv_freq;
    float s, c;
    __sincosf(static_cast<float>(angle), &s, &c);

    const std::int64_t base =
        static_cast<std::int64_t>(token) * kVisQkvRows + head * kVisHeadDim + lane * 2;
    for (int qk = 0; qk < 2; ++qk) {
        auto* ptr      = reinterpret_cast<__nv_bfloat162*>(qkv + base + qk * kVisHidden);
        const float2 v = __bfloat1622float2(*ptr);
        *ptr           = __floats2bfloat162_rn(v.x * c - v.y * s, v.x * s + v.y * c);
    }
}

__global__ __launch_bounds__(256) void vision_layerscale_add_kernel(const BF* __restrict__ proj,
                                                                    const BF* __restrict__ bias,
                                                                    const BF* __restrict__ gamma,
                                                                    BF* __restrict__ hidden) {
    const std::int64_t col_base = static_cast<std::int64_t>(blockIdx.x) * (kVisHidden / 2);
    const int pair_base         = static_cast<int>(threadIdx.x) * 2;
    const auto* p2              = reinterpret_cast<const __nv_bfloat162*>(proj) + col_base;
    const auto* b2              = reinterpret_cast<const __nv_bfloat162*>(bias);
    const auto* g2              = reinterpret_cast<const __nv_bfloat162*>(gamma);
    auto* h2                    = reinterpret_cast<__nv_bfloat162*>(hidden) + col_base;

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx   = pair_base + i;
        const float2 pv = __bfloat1622float2(p2[idx]);
        const float2 bv = __bfloat1622float2(b2[idx]);
        const float2 gv = __bfloat1622float2(g2[idx]);
        const float2 hv = __bfloat1622float2(h2[idx]);
        h2[idx] =
            __floats2bfloat162_rn(fmaf(gv.x, pv.x + bv.x, hv.x), fmaf(gv.y, pv.y + bv.y, hv.y));
    }
}

__global__
__launch_bounds__(128) void vision_merge_2x2_kernel(const BF* __restrict__ src,
                                                    const std::int32_t* __restrict__ merge_indices,
                                                    BF* __restrict__ out) {
    const int v   = static_cast<int>(blockIdx.x);
    const int q   = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);

    const std::int32_t src_col = merge_indices[static_cast<std::int64_t>(v) * 4 + q];
    auto* dst =
        reinterpret_cast<uint4*>(out + (static_cast<std::int64_t>(v) * 4 + q) * kVisHidden) + tid;
    if (src_col < 0) {
        *dst = make_uint4(0u, 0u, 0u, 0u);
    } else {
        const auto* src_ptr =
            reinterpret_cast<const uint4*>(src + static_cast<std::int64_t>(src_col) * kVisHidden) +
            tid;
        *dst = *src_ptr;
    }
}

} // namespace

void inkling2_vision_patch_embed_launch(const Tensor& patches, const Tensor& patch_weight,
                                        const Tensor& class_embedding,
                                        const Tensor& cls_position_embedding,
                                        const Tensor& position_table, const Tensor& pos_indices,
                                        const Tensor& pos_weights, const Tensor& ln_pre_weight,
                                        const Tensor& ln_pre_bias, float eps, Tensor& out,
                                        cudaStream_t stream) {
    const std::int32_t patch_count = patches.ne[1];
    vision_cls_embed_kernel<<<1, 256, 0, stream>>>(
        static_cast<const BF*>(class_embedding.data),
        static_cast<const BF*>(cls_position_embedding.data), static_cast<BF*>(out.data));
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid(static_cast<unsigned>((patch_count + 31) / 32),
                    static_cast<unsigned>(kVisHidden / 64), 1u);
    vision_patch_embed_mma_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const BF*>(patches.data), static_cast<const BF*>(patch_weight.data),
        static_cast<const BF*>(position_table.data),
        static_cast<const std::int32_t*>(pos_indices.data),
        static_cast<const float*>(pos_weights.data), patch_count, static_cast<BF*>(out.data));
    CUDA_CHECK(cudaGetLastError());

    constexpr int block     = 128;
    constexpr int warps     = block / 32;
    const std::int64_t rows = static_cast<std::int64_t>(patch_count) + 1;
    const unsigned ln_grid  = static_cast<unsigned>((rows + warps - 1) / warps);
    layer_norm_warp_kernel<kVisHidden, block>
        <<<ln_grid, block, 0, stream>>>(static_cast<const __nv_bfloat162*>(out.data),
                                        static_cast<const __nv_bfloat162*>(ln_pre_weight.data),
                                        static_cast<const __nv_bfloat162*>(ln_pre_bias.data),
                                        static_cast<__nv_bfloat162*>(out.data), rows, eps);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_vision_rope_2d_launch(const Tensor& coords, float rope_theta, Tensor& qkv,
                                    cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(qkv.ne[1]), static_cast<unsigned>(kVisHeads), 1u);
    vision_rope_2d_kernel<<<grid, 32, 0, stream>>>(static_cast<const std::int32_t*>(coords.data),
                                                   rope_theta, static_cast<BF*>(qkv.data));
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_vision_layerscale_add_launch(const Tensor& proj, const Tensor& bias,
                                           const Tensor& gamma, Tensor& hidden,
                                           cudaStream_t stream) {
    vision_layerscale_add_kernel<<<static_cast<unsigned>(hidden.ne[1]), 256, 0, stream>>>(
        static_cast<const BF*>(proj.data), static_cast<const BF*>(bias.data),
        static_cast<const BF*>(gamma.data), static_cast<BF*>(hidden.data));
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_vision_merge_2x2_launch(const Tensor& src, const Tensor& merge_indices, Tensor& out,
                                      cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(out.ne[1]), 4u, 1u);
    vision_merge_2x2_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const BF*>(src.data), static_cast<const std::int32_t*>(merge_indices.data),
        static_cast<BF*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
