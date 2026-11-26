#include <math_constants.h>
#include "ops/inkling2_moe/launch.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_a8_quantize.h"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/linear/q3/q3_rowsplit_a8_codec.cuh"
#include "ops/linear/q3/q3_rowsplit_storage.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "core/device.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops::inkling2_detail {
using BF = __nv_bfloat16;

struct Quant {
    const unsigned char* code;
    const __half* scale;
    int k;
};

template <bool Q3>
__device__ float weight(Quant w, int row, int k) {
    constexpr int group =
        Q3 ? detail::Q3RowSplitStorage::kGroupK : detail::Q4RowSplitStorage::kGroupK;
    int value;
    if constexpr (Q3) {
        auto bit        = std::int64_t(row) * w.k * 3 + k * 3;
        auto byte       = bit / 8;
        int shift       = bit % 8;
        unsigned packed = w.code[byte];
        if (shift > 5) packed |= unsigned(w.code[byte + 1]) << 8;
        value = detail::q3_signed_code((packed >> shift) & 7u);
    } else {
        unsigned b = w.code[(std::int64_t(row) * w.k + k) / 2];
        int nibble = (k & 1) ? b >> 4 : b & 15;
        value      = (nibble ^ 8) - 8;
    }
    return value * __half2float(w.scale[std::int64_t(row) * (w.k / group) + k / group]);
}

__device__ float sum_warp(float x) {
    for (int d = 16; d; d /= 2) x += __shfl_down_sync(0xffffffffu, x, d);
    return x;
}

__device__ __forceinline__ float moe_situ(float g, float u) {
    return (4.0f * tanhf(g * 0.25f) / (1.0f + expf(-g))) * (25.0f * tanhf(u * 0.04f));
}

__device__ __forceinline__ float dot_bf16x8(const uint4& a, const uint4& b) {
    const auto* ap = reinterpret_cast<const __nv_bfloat162*>(&a);
    const auto* bp = reinterpret_cast<const __nv_bfloat162*>(&b);
    float s = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 x = __bfloat1622float2(ap[i]);
        const float2 y = __bfloat1622float2(bp[i]);
        s = fmaf(x.x, y.x, s);
        s = fmaf(x.y, y.y, s);
    }
    return s;
}

// 8 warps/CTA, one expert row per warp; each lane streams uint4 (8 BF16) over 9 iterations.
__global__ void router_kernel(const BF* x, const BF* router, float* probs, int tokens) {
    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int expert = blockIdx.x * 8 + warp;
    const int t      = blockIdx.y;
    const BF* xp     = x + static_cast<std::int64_t>(t) * 2304;
    const BF* rp     = router + static_cast<std::int64_t>(expert) * 2304;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        dot += dot_bf16x8(load_vec<uint4>(xp + (i * 32 + lane) * 8),
                          load_vec<uint4>(rp + (i * 32 + lane) * 8));
    }
    dot = sum_warp(dot);
    if (lane == 0) probs[expert + 128 * t] = 1.0f / (1.0f + expf(-dot));
}

// One warp per token: parallel top-8 selection over 128 experts (4 experts per lane).
__global__ void route_kernel(const float* probs, const float* bias, int* ids, float* alpha) {
    const int t    = blockIdx.x;
    const int lane = threadIdx.x & 31;
    float p[4];
    float s[4];
    bool used[4] = {};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = lane + 32 * j;
        p[j] = probs[128 * t + e];
        s[j] = p[j] + bias[e];
    }
    float sum = 0.0f;
    int sel_ids[8];
    float sel_p[8];
    for (int i = 0; i < 8; ++i) {
        float best_s = -CUDART_INF_F;
        int best_e   = 128;
        float best_p = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int e = lane + 32 * j;
            if (!used[j] && (s[j] > best_s || (s[j] == best_s && e < best_e))) {
                best_s = s[j];
                best_e = e;
                best_p = p[j];
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float os = __shfl_xor_sync(0xffffffffu, best_s, off);
            const int oe   = __shfl_xor_sync(0xffffffffu, best_e, off);
            const float op = __shfl_xor_sync(0xffffffffu, best_p, off);
            if (os > best_s || (os == best_s && oe < best_e)) {
                best_s = os;
                best_e = oe;
                best_p = op;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (lane + 32 * j == best_e) used[j] = true;
        }
        sel_ids[i] = best_e;
        sel_p[i]   = best_p;
        sum += best_p;
    }
    if (lane == 0) {
        const float inv = 8.0f / (sum + 1.0e-20f);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            ids[8 * t + i]   = sel_ids[i];
            alpha[8 * t + i] = sel_p[i] * inv;
        }
    }
}

// 8 warps/CTA = 8 routed experts for RowsPerCta intermediate rows. x[t] is staged into shared
// memory once, then each warp decodes 3-bit/4-bit windows against it without re-reading x.
template <bool Q3, int RowsPerCta>
__global__ void gate_up_kernel(const BF* x, Quant w, const int* ids, float* act) {
    __shared__ __align__(16) BF x_shared[2304];

    const int tid      = threadIdx.x;
    const int t        = blockIdx.y;
    const int row_base = blockIdx.x * RowsPerCta;
    const int slot     = tid >> 5;
    const int lane     = tid & 31;

    {
        // 2304 BF16 = 288 uint4: every thread takes one, the first 32 take a second.
        const BF* xp  = x + static_cast<std::int64_t>(t) * 2304;
        const uint4 a = load_vec<uint4>(xp + tid * 8);
        store_vec(&x_shared[tid * 8], a);
        if (tid < 32) {
            const uint4 b = load_vec<uint4>(xp + 2048 + tid * 8);
            store_vec(&x_shared[2048 + tid * 8], b);
        }
    }
    __syncthreads();

    const int expert_base = ids[8 * t + slot] * 2304 + row_base;
    float g[RowsPerCta]   = {};
    float u[RowsPerCta]   = {};

    if constexpr (Q3) {
        // 18 groups of 128; group_lane 0..1, each lane owns 8 codes (24-bit window).
        // Each 48-byte group is 16-byte aligned; lanes local < 12 load 12 aligned uint32_t words.
        const int group_lane = lane >> 4;
        const int local      = lane & 15;
        const int half_base  = group_lane << 4;
        const int w0_lane    = half_base + ((local * 3) >> 2);
        const int w1_lane    = min(w0_lane + 1, half_base + 11);
        const int shift24    = ((local & 3) * 24) & 31;
#pragma unroll
        for (int grp = 0; grp < 18; grp += 2) {
            const int gi   = grp + group_lane;
            const uint4 xv = load_vec<uint4>(&x_shared[gi * 128 + 8 * local]);
            const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
            float xf[8];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float2 xa = __bfloat1622float2(xb[c]);
                xf[2 * c]       = xa.x;
                xf[2 * c + 1]   = xa.y;
            }
#pragma unroll
            for (int r = 0; r < RowsPerCta; ++r) {
                const int row_gate = expert_base + r;
                const int row_up   = row_gate + 1152;
                const std::uint32_t* cg = reinterpret_cast<const std::uint32_t*>(
                    w.code + static_cast<std::int64_t>(row_gate) * 864 +
                    static_cast<std::int64_t>(gi) * 48);
                const std::uint32_t* cu = reinterpret_cast<const std::uint32_t*>(
                    w.code + static_cast<std::int64_t>(row_up) * 864 +
                    static_cast<std::int64_t>(gi) * 48);
                const std::uint32_t rg_raw = local < 12 ? __ldg(cg + local) : 0u;
                const std::uint32_t ru_raw = local < 12 ? __ldg(cu + local) : 0u;
                const std::uint32_t wg =
                    __funnelshift_r(__shfl_sync(0xffffffffu, rg_raw, w0_lane),
                                    __shfl_sync(0xffffffffu, rg_raw, w1_lane), shift24) &
                    0x00ffffffu;
                const std::uint32_t wu =
                    __funnelshift_r(__shfl_sync(0xffffffffu, ru_raw, w0_lane),
                                    __shfl_sync(0xffffffffu, ru_raw, w1_lane), shift24) &
                    0x00ffffffu;
                const float sg = __half2float(__ushort_as_half(
                    *reinterpret_cast<const std::uint16_t*>(&w.scale[static_cast<std::int64_t>(row_gate) * 18 + gi])));
                const float su = __half2float(__ushort_as_half(
                    *reinterpret_cast<const std::uint16_t*>(&w.scale[static_cast<std::int64_t>(row_up) * 18 + gi])));
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    const float gw =
                        static_cast<float>(detail::q3_signed_code((wg >> (3 * c)) & 7u)) * sg;
                    const float uw =
                        static_cast<float>(detail::q3_signed_code((wu >> (3 * c)) & 7u)) * su;
                    g[r] = fmaf(gw, xf[c], g[r]);
                    u[r] = fmaf(uw, xf[c], u[r]);
                }
            }
        }
    } else {
        // 36 groups of 64; lane_group 0..3, each lane owns 8 codes (uint32 window).
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
#pragma unroll
        for (int grp = 0; grp < 36; grp += 4) {
            const int gi   = grp + lane_group;
            const uint4 xv = load_vec<uint4>(&x_shared[gi * 64 + 8 * lane_in_group]);
            const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
            float xf[8];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float2 xa = __bfloat1622float2(xb[c]);
                xf[2 * c]       = xa.x;
                xf[2 * c + 1]   = xa.y;
            }
#pragma unroll
            for (int r = 0; r < RowsPerCta; ++r) {
                const int row_gate = expert_base + r;
                const int row_up   = row_gate + 1152;
                const std::uint32_t wg = *reinterpret_cast<const std::uint32_t*>(
                    w.code + static_cast<std::int64_t>(row_gate) * 1152 +
                    static_cast<std::int64_t>(gi) * 32 + 4 * lane_in_group);
                const std::uint32_t wu = *reinterpret_cast<const std::uint32_t*>(
                    w.code + static_cast<std::int64_t>(row_up) * 1152 +
                    static_cast<std::int64_t>(gi) * 32 + 4 * lane_in_group);
                const std::uint16_t sg =
                    *reinterpret_cast<const std::uint16_t*>(&w.scale[static_cast<std::int64_t>(row_gate) * 36 + gi]);
                const std::uint16_t su =
                    *reinterpret_cast<const std::uint16_t*>(&w.scale[static_cast<std::int64_t>(row_up) * 36 + gi]);
                float gw[8], uw[8];
                detail::Q4SimtDecodeAtom::decode_eight(wg, sg, gw);
                detail::Q4SimtDecodeAtom::decode_eight(wu, su, uw);
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    g[r] = fmaf(gw[c], xf[c], g[r]);
                    u[r] = fmaf(uw[c], xf[c], u[r]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RowsPerCta; ++r) {
        const float gs = sum_warp(g[r]);
        const float us = sum_warp(u[r]);
        if (lane == 0)
            act[row_base + r + 1152 * (slot + 8 * t)] = moe_situ(gs, us);
    }
}

// 8 warps/CTA = 8 routed experts; each warp owns RowsPerCta adjacent output rows and reuses
// its act window across all of them. Final 8-slot combine happens in shared memory.
template <bool Q3, int RowsPerCta>
__global__ void down_kernel(Quant w, const int* ids, const float* alpha, const float* act,
                            BF* out) {
    __shared__ float paths[8][RowsPerCta];

    const int tid     = threadIdx.x;
    const int t       = blockIdx.y;
    const int slot    = tid >> 5;
    const int lane    = tid & 31;
    const int expert  = ids[8 * t + slot];
    const float a     = alpha[8 * t + slot];
    const float* act_s = act + 1152 * (slot + 8 * t);
    float dot[RowsPerCta] = {};

    if constexpr (Q3) {
        // K = 1152 = 9 groups of 128: paired steps g = 0,2,4,6 plus solo group 8.
        const int group_lane = lane >> 4;
        const int local      = lane & 15;
        const int half_base  = group_lane << 4;
        const int w0_lane    = half_base + ((local * 3) >> 2);
        const int w1_lane    = min(w0_lane + 1, half_base + 11);
        const int shift24    = ((local & 3) * 24) & 31;
#pragma unroll
        for (int grp = 0; grp < 9; grp += 2) {
            const int gi = grp + group_lane;
            float xv[8];
            if (gi < 9) {
                const float4 f0 = *reinterpret_cast<const float4*>(act_s + gi * 128 + 8 * local);
                const float4 f1 = *reinterpret_cast<const float4*>(act_s + gi * 128 + 8 * local + 4);
                xv[0] = f0.x; xv[1] = f0.y; xv[2] = f0.z; xv[3] = f0.w;
                xv[4] = f1.x; xv[5] = f1.y; xv[6] = f1.z; xv[7] = f1.w;
            }
#pragma unroll
            for (int r = 0; r < RowsPerCta; ++r) {
                const int erow = expert * 2304 + blockIdx.x * RowsPerCta + r;
                const std::uint32_t* c = reinterpret_cast<const std::uint32_t*>(
                    w.code + static_cast<std::int64_t>(erow) * 432 +
                    static_cast<std::int64_t>(gi < 9 ? gi : 0) * 48);
                const std::uint32_t r_raw = (gi < 9 && local < 12) ? __ldg(c + local) : 0u;
                const std::uint32_t w3 =
                    __funnelshift_r(__shfl_sync(0xffffffffu, r_raw, w0_lane),
                                    __shfl_sync(0xffffffffu, r_raw, w1_lane), shift24) &
                    0x00ffffffu;
                if (gi < 9) {
                    const float sc = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                        &w.scale[static_cast<std::int64_t>(erow) * 9 + gi])));
#pragma unroll
                    for (int c2 = 0; c2 < 8; ++c2)
                        dot[r] = fmaf(static_cast<float>(detail::q3_signed_code((w3 >> (3 * c2)) & 7u)) * sc,
                                      xv[c2], dot[r]);
                }
            }
        }
    } else {
        // K = 1152 = 18 groups of 64: quad steps g = 0,4,8,12 plus final pair 16,17.
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
#pragma unroll
        for (int grp = 0; grp < 18; grp += 4) {
            const int gi = grp + lane_group;
            if (gi < 18) {
                float xv[8];
                const float4 f0 = *reinterpret_cast<const float4*>(act_s + gi * 64 + 8 * lane_in_group);
                const float4 f1 = *reinterpret_cast<const float4*>(act_s + gi * 64 + 8 * lane_in_group + 4);
                xv[0] = f0.x; xv[1] = f0.y; xv[2] = f0.z; xv[3] = f0.w;
                xv[4] = f1.x; xv[5] = f1.y; xv[6] = f1.z; xv[7] = f1.w;
#pragma unroll
                for (int r = 0; r < RowsPerCta; ++r) {
                    const int erow = expert * 2304 + blockIdx.x * RowsPerCta + r;
                    const std::uint32_t w4 = *reinterpret_cast<const std::uint32_t*>(
                        w.code + static_cast<std::int64_t>(erow) * 576 +
                        static_cast<std::int64_t>(gi) * 32 + 4 * lane_in_group);
                    const std::uint16_t sb = *reinterpret_cast<const std::uint16_t*>(
                        &w.scale[static_cast<std::int64_t>(erow) * 18 + gi]);
                    float ww[8];
                    detail::Q4SimtDecodeAtom::decode_eight(w4, sb, ww);
#pragma unroll
                    for (int c2 = 0; c2 < 8; ++c2)
                        dot[r] = fmaf(ww[c2], xv[c2], dot[r]);
                }
            }
        }
    }

    float dot_r[RowsPerCta];
#pragma unroll
    for (int r = 0; r < RowsPerCta; ++r)
        dot_r[r] = sum_warp(dot[r]);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RowsPerCta; ++r)
            paths[slot][r] = a * dot_r[r];
    }
    __syncthreads();
    if (tid < RowsPerCta) {
        float s = 0;
#pragma unroll
        for (int s2 = 0; s2 < 8; ++s2)
            s += paths[s2][tid];
        out[blockIdx.x * RowsPerCta + tid + 2304 * t] = __float2bfloat16_rn(s);
    }
}

Quant quant(const Weight& w) {
    return {static_cast<const unsigned char*>(w.qdata), static_cast<const __half*>(w.scales), w.k};
}

void moe_launch(const Tensor& x, const Inkling2MoeWeights& w, Tensor& out, float* probs, int* ids,
                float* alpha, float* act, cudaStream_t s) {
    int t      = x.ne[1];
    auto input = static_cast<const BF*>(x.data);
    router_kernel<<<dim3(16, t), 256, 0, s>>>(input, static_cast<const BF*>(w.router.qdata), probs,
                                              t);
    CUDA_CHECK(cudaGetLastError());
    route_kernel<<<t, 32, 0, s>>>(probs, static_cast<const float*>(w.exp_probs_bias.qdata), ids,
                                  alpha);
    CUDA_CHECK(cudaGetLastError());
    if (w.routed_gate_up.qtype == QType::Q3G128_F16S) {
        gate_up_kernel<true, 2><<<dim3(576, t), 256, 0, s>>>(input, quant(w.routed_gate_up), ids,
                                                             act);
        down_kernel<true, 4><<<dim3(576, t), 256, 0, s>>>(quant(w.routed_down), ids, alpha, act,
                                                          static_cast<BF*>(out.data));
    } else {
        gate_up_kernel<false, 4><<<dim3(288, t), 256, 0, s>>>(input, quant(w.routed_gate_up), ids,
                                                              act);
        down_kernel<false, 4><<<dim3(576, t), 256, 0, s>>>(quant(w.routed_down), ids, alpha, act,
                                                           static_cast<BF*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

__device__ __forceinline__ int moe_swz128(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__global__ __launch_bounds__(256, 2) void inkling2_moe_router_mma_kernel(
    const BF* __restrict__ x, const BF* __restrict__ router, float* __restrict__ probs,
    int tokens) {
    constexpr int BM = 16;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int STAGES = 2;
    __shared__ __align__(16) BF Ws[STAGES][BM * BK];
    __shared__ __align__(16) BF Xs[STAGES][BN * BK];

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row0 = static_cast<int>(blockIdx.x) * BM;
    const int token0 = static_cast<int>(blockIdx.y) * BN;

    float acc[4] = {};

    auto stage_inputs = [&](int stage, int kt) {
        const int k0 = kt * BK;
        for (int item = tid; item < BM * (BK / 8); item += 256) {
            const int row = item >> 3;
            const int k8 = item & 7;
            const int rr = row0 + row;
            cp_async<16, Cache::cg>(
                &Ws[stage][row * BK + detail::gemm_swz64(row, k8 * 8)],
                router + static_cast<std::int64_t>(rr) * 2304 + k0 + k8 * 8);
        }
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int item = tid + it * 256;
            const int col = item >> 3;
            const int k8 = item & 7;
            const int tt = token0 + col;
            cp_async_zfill<16, Cache::cg>(
                &Xs[stage][col * BK + detail::gemm_swz64(col, k8 * 8)],
                x + static_cast<std::int64_t>(tt < tokens ? tt : 0) * 2304 + k0 + k8 * 8,
                tt < tokens ? 16 : 0);
        }
    };

#pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int kt = 0; kt < 2304 / BK; ++kt) {
        const int stage = kt & 1;
        cp_wait<STAGES - 1>();
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < BK / 16; ++ki) {
            unsigned af[4];
            unsigned bf[2];
            const int arow = a_rowoff;
            const int acol = ki * 16 + a_coloff;
            const int brow = warp * 8 + b_rin;
            const int bcol = ki * 16 + b_koff;
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(&Ws[stage][arow * BK + detail::gemm_swz64(arow, acol)]));
            ldmatrix_x2(bf[0], bf[1],
                        smem_addr(&Xs[stage][brow * BK + detail::gemm_swz64(brow, bcol)]));
            mma_bf16(acc[0], acc[1], acc[2], acc[3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
        }

        __syncthreads();
        const int next = kt + STAGES;
        if (next < 2304 / BK) stage_inputs(stage, next);
        cp_commit();
    }
    cp_wait<0>();

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int r0 = row0 + gid;
    const int r1 = r0 + 8;
    const int c0 = token0 + warp * 8 + 2 * lid;
    const int c1 = c0 + 1;
    if (c0 < tokens) {
        probs[static_cast<std::int64_t>(c0) * 128 + r0] = 1.0f / (1.0f + expf(-acc[0]));
        probs[static_cast<std::int64_t>(c0) * 128 + r1] = 1.0f / (1.0f + expf(-acc[2]));
    }
    if (c1 < tokens) {
        probs[static_cast<std::int64_t>(c1) * 128 + r0] = 1.0f / (1.0f + expf(-acc[1]));
        probs[static_cast<std::int64_t>(c1) * 128 + r1] = 1.0f / (1.0f + expf(-acc[3]));
    }
}

__global__ void inkling2_moe_select_count_kernel(const float* __restrict__ probs,
                                                 const float* __restrict__ bias,
                                                 int* __restrict__ ids, float* __restrict__ alpha,
                                                 int* __restrict__ local_rank,
                                                 int* __restrict__ tile_counts, int tokens) {
    __shared__ int counts[128];
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid < 128) counts[tid] = 0;
    __syncthreads();

    const int token = static_cast<int>(blockIdx.x) * 8 + warp;
    if (token < tokens) {
        float p[4];
        float s[4];
        bool used[4] = {};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int e = lane + 32 * j;
            p[j] = probs[static_cast<std::int64_t>(token) * 128 + e];
            s[j] = p[j] + bias[e];
        }
        float sum = 0.0f;
        int sel_ids[8];
        float sel_p[8];
        for (int i = 0; i < 8; ++i) {
            float best_s = -CUDART_INF_F;
            int best_e = 128;
            float best_p = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int e = lane + 32 * j;
                if (!used[j] && (s[j] > best_s || (s[j] == best_s && e < best_e))) {
                    best_s = s[j];
                    best_e = e;
                    best_p = p[j];
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float os = __shfl_xor_sync(0xffffffffu, best_s, off);
                const int oe = __shfl_xor_sync(0xffffffffu, best_e, off);
                const float op = __shfl_xor_sync(0xffffffffu, best_p, off);
                if (os > best_s || (os == best_s && oe < best_e)) {
                    best_s = os;
                    best_e = oe;
                    best_p = op;
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (lane + 32 * j == best_e) used[j] = true;
            }
            sel_ids[i] = best_e;
            sel_p[i] = best_p;
            sum += best_p;
        }
        if (lane == 0) {
            const float inv = 8.0f / (sum + 1e-20f);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int assignment = token * 8 + i;
                ids[assignment] = sel_ids[i];
                alpha[assignment] = sel_p[i] * inv;
                local_rank[assignment] = atomicAdd(&counts[sel_ids[i]], 1);
            }
        }
    }
    __syncthreads();
    if (tid < 128) tile_counts[static_cast<std::int64_t>(blockIdx.x) * 128 + tid] = counts[tid];
}

template <int ExpertBN>
__global__ void inkling2_moe_scan_kernel(const int* __restrict__ tile_counts,
                                         int* __restrict__ tile_bases,
                                         int* __restrict__ expert_offsets,
                                         int* __restrict__ route_job_experts,
                                         int* __restrict__ route_job_columns,
                                         int* __restrict__ route_job_count, int route_tiles) {
    __shared__ int scan[128];
    const int expert = static_cast<int>(threadIdx.x);
    int count = 0;
    for (int tile = 0; tile < route_tiles; ++tile) {
        count += tile_counts[static_cast<std::int64_t>(tile) * 128 + expert];
    }
    scan[expert] = count;
    __syncthreads();
#pragma unroll
    for (int offset = 1; offset < 128; offset <<= 1) {
        const int add = expert >= offset ? scan[expert - offset] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }
    const int base = expert == 0 ? 0 : scan[expert - 1];
    expert_offsets[expert] = base;
    if (expert == 127) expert_offsets[128] = scan[127];

    int cursor = base;
    for (int tile = 0; tile < route_tiles; ++tile) {
        const std::int64_t idx = static_cast<std::int64_t>(tile) * 128 + expert;
        tile_bases[idx] = cursor;
        cursor += tile_counts[idx];
    }

    __syncthreads();
    const int jobs = (count + ExpertBN - 1) / ExpertBN;
    scan[expert] = jobs;
    __syncthreads();
#pragma unroll
    for (int offset = 1; offset < 128; offset <<= 1) {
        const int add = expert >= offset ? scan[expert - offset] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }
    const int job_begin = expert == 0 ? 0 : scan[expert - 1];
    for (int j = 0; j < jobs; ++j) {
        route_job_experts[job_begin + j] = expert;
        route_job_columns[job_begin + j] = j * ExpertBN;
    }
    if (expert == 127) route_job_count[0] = scan[127];
}

__global__ void inkling2_moe_index_kernel(const int* __restrict__ ids,
                                          const int* __restrict__ local_rank,
                                          int* __restrict__ packed_index,
                                          const int* __restrict__ tile_bases,
                                          int* __restrict__ packed_token, int assignments) {
    const int assignment = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (assignment >= assignments) return;
    const int token = assignment >> 3;
    const int expert = ids[assignment];
    const int tile = token >> 3;
    const int packed = tile_bases[static_cast<std::int64_t>(tile) * 128 + expert] + local_rank[assignment];
    packed_index[assignment] = packed;
    packed_token[packed] = token;
}

template <bool Q3, bool GateUp, int ExpertBN>
__global__ __launch_bounds__(256, 2) void inkling2_moe_grouped_mma_kernel(
    const BF* __restrict__ x, const int* __restrict__ packed_token,
    const int* __restrict__ expert_offsets, const int* __restrict__ route_job_experts,
    const int* __restrict__ route_job_columns, const int* __restrict__ route_job_count,
    const std::uint8_t* __restrict__ codes, const std::uint16_t* __restrict__ scales,
    BF* __restrict__ out) {
    constexpr int BM = 64;
    constexpr int BN = ExpertBN;
    constexpr int BK = Q3 ? 128 : 64;
    constexpr int K = GateUp ? 2304 : 1152;
    constexpr int KTiles = K / BK;
    constexpr int GroupsTotal = K / (Q3 ? 128 : 64);
    constexpr int CodeBytes = Q3 ? 48 : 32;
    constexpr int Stages = 2;
    constexpr int RowBlocks = GateUp ? (1152 / 32) : (2304 / 64); // 36
    constexpr int StageChunks = BK / 8;                            // 16 (Q3) or 8 (Q4)
    constexpr int StageItems  = BN * StageChunks;
    constexpr int StageIters  = (StageItems + 255) / 256;
    constexpr int ActiveWarpN = BN / 8;                            // 2 (BN=16) or 4 (BN=32)

    __shared__ __align__(16) BF As[BM * BK];
    __shared__ __align__(16) BF Bs[Stages][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[Stages][BM * CodeBytes];
    __shared__ __align__(16) std::uint16_t Sr[Stages][BM];

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int total_work = route_job_count[0] * RowBlocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job = work / RowBlocks;
        const int row_block = work - route_job * RowBlocks;
        const int expert = route_job_experts[route_job];
        const int begin = expert_offsets[expert];
        const int count = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols = count - column_base < BN ? count - column_base : BN;

        const BF* src_row[StageIters];
#pragma unroll
        for (int i = 0; i < StageIters; ++i) {
            const int col = (tid + i * 256) / StageChunks;
            const int packed_col = col < cols ? (begin + column_base + col) : begin;
            const int src_col = GateUp ? packed_token[packed_col] : packed_col;
            src_row[i] = x + static_cast<std::int64_t>(src_col) * K;
        }

        auto global_row = [&](int local_row) {
            if constexpr (GateUp) {
                const int logical = row_block * 32 + ((local_row >> 5) << 4) + (local_row & 15);
                const int is_up = (local_row >> 4) & 1;
                return expert * 2304 + logical + (is_up ? 1152 : 0);
            } else {
                return expert * 2304 + row_block * 64 + local_row;
            }
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * BK;
#pragma unroll
            for (int i = 0; i < StageIters; ++i) {
                const int item = tid + i * 256;
                if (item < StageItems) {
                    const int col = item / StageChunks;
                    const int k8 = item - col * StageChunks;
                    const int swz_col = Q3 ? moe_swz128(col, k8 * 8) : detail::gemm_swz64(col, k8 * 8);
                    cp_async_zfill<16, Cache::cg>(&Bs[stage][col * BK + swz_col],
                                                  src_row[i] + k0 + k8 * 8, col < cols ? 16 : 0);
                }
            }
            constexpr int CodeChunksPerRow = CodeBytes / 16; // 3 (Q3) or 2 (Q4)
            if (tid < BM * CodeChunksPerRow) {
                const int row = tid / CodeChunksPerRow;
                const int chunk = tid - row * CodeChunksPerRow;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * GroupsTotal + kt;
                cp_async<16, Cache::cg>(&Cr[stage][row * CodeBytes + chunk * 16],
                                        &codes[gi * CodeBytes + chunk * 16]);
            }
            if (tid < BM) {
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(tid)) * GroupsTotal + kt;
                Sr[stage][tid] = scales[gi];
            }
        };

#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            if (stage < KTiles) stage_inputs(stage, stage);
            cp_commit();
        }

        float acc[2][1][4] = {};

        for (int kt = 0; kt < KTiles; ++kt) {
            const int stage = kt & 1;
            cp_wait<Stages - 1>();
            __syncthreads();

            if constexpr (Q3) {
                const int half = lane >> 4;
                const int local = lane & 15;
                const int byte0 = 3 * local;
                for (int row_pair = warp * 2; row_pair < BM; row_pair += 16) {
                    const int row = row_pair + half;
                    const float scale = __half2float(__ushort_as_half(Sr[stage][row]));
                    const std::uint8_t* gc = &Cr[stage][row * 48];
                    const std::uint32_t word = static_cast<std::uint32_t>(gc[byte0]) |
                                               (static_cast<std::uint32_t>(gc[byte0 + 1]) << 8) |
                                               (static_cast<std::uint32_t>(gc[byte0 + 2]) << 16);
                    float vals[8];
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        vals[c] =
                            static_cast<float>(detail::q3_signed_code((word >> (3 * c)) & 0x7u)) *
                            scale;
                    }
                    const __nv_bfloat162 p0 = __floats2bfloat162_rn(vals[0], vals[1]);
                    const __nv_bfloat162 p1 = __floats2bfloat162_rn(vals[2], vals[3]);
                    const __nv_bfloat162 p2 = __floats2bfloat162_rn(vals[4], vals[5]);
                    const __nv_bfloat162 p3 = __floats2bfloat162_rn(vals[6], vals[7]);
                    store_vec(&As[row * 128 + moe_swz128(row, 8 * local)],
                              make_uint4(*reinterpret_cast<const unsigned*>(&p0),
                                         *reinterpret_cast<const unsigned*>(&p1),
                                         *reinterpret_cast<const unsigned*>(&p2),
                                         *reinterpret_cast<const unsigned*>(&p3)));
                }
            } else {
#pragma unroll
                for (int pass = 0; pass < 2; ++pass) {
                    const int row = pass * 32 + (tid >> 3);
                    const int local = tid & 7;
                    const float scale = __half2float(__ushort_as_half(Sr[stage][row]));
                    const std::uint32_t word =
                        *reinterpret_cast<const std::uint32_t*>(&Cr[stage][row * 32 + local * 4]);
                    float vals[8];
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        const int nib = static_cast<int>((word >> (4 * c)) & 0xfu);
                        vals[c] = static_cast<float>((nib ^ 8) - 8) * scale;
                    }
                    const __nv_bfloat162 p0 = __floats2bfloat162_rn(vals[0], vals[1]);
                    const __nv_bfloat162 p1 = __floats2bfloat162_rn(vals[2], vals[3]);
                    const __nv_bfloat162 p2 = __floats2bfloat162_rn(vals[4], vals[5]);
                    const __nv_bfloat162 p3 = __floats2bfloat162_rn(vals[6], vals[7]);
                    store_vec(&As[row * 64 + detail::gemm_swz64(row, 8 * local)],
                              make_uint4(*reinterpret_cast<const unsigned*>(&p0),
                                         *reinterpret_cast<const unsigned*>(&p1),
                                         *reinterpret_cast<const unsigned*>(&p2),
                                         *reinterpret_cast<const unsigned*>(&p3)));
                }
            }
            __syncthreads();

#pragma unroll
            for (int k_step = 0; k_step < BK; k_step += 16) {
                unsigned af[2][4];
                unsigned bf[2];
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    const int arow = warp_m * 32 + mi * 16 + a_rowoff;
                    const int acol = k_step + a_coloff;
                    const int swz_a = Q3 ? moe_swz128(arow, acol) : detail::gemm_swz64(arow, acol);
                    ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                smem_addr(&As[arow * BK + swz_a]));
                }
                if (warp_n < ActiveWarpN) {
                    const int brow = warp_n * 8 + b_rin;
                    const int bcol = k_step + b_koff;
                    const int swz_b = Q3 ? moe_swz128(brow, bcol) : detail::gemm_swz64(brow, bcol);
                    ldmatrix_x2(bf[0], bf[1], smem_addr(&Bs[stage][brow * BK + swz_b]));
#pragma unroll
                    for (int mi = 0; mi < 2; ++mi) {
                        mma_bf16(acc[mi][0][0], acc[mi][0][1], acc[mi][0][2], acc[mi][0][3],
                                 af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf[0], bf[1]);
                    }
                }
            }

            __syncthreads();
            const int next = kt + Stages;
            if (next < KTiles) stage_inputs(stage, next);
            cp_commit();
        }
        cp_wait<0>();

        const int c0 = warp_n * 8 + 2 * lid;
        const int c1 = c0 + 1;
        const int p0 = begin + column_base + c0;
        const int p1 = begin + column_base + c1;
        if (warp_n < ActiveWarpN) {
            if constexpr (GateUp) {
                const int r0 = row_block * 32 + warp_m * 16 + gid;
                const int r1 = r0 + 8;
                if (c0 < cols) {
                    out[static_cast<std::int64_t>(p0) * 1152 + r0] =
                        __float2bfloat16_rn(moe_situ(acc[0][0][0], acc[1][0][0]));
                    out[static_cast<std::int64_t>(p0) * 1152 + r1] =
                        __float2bfloat16_rn(moe_situ(acc[0][0][2], acc[1][0][2]));
                }
                if (c1 < cols) {
                    out[static_cast<std::int64_t>(p1) * 1152 + r0] =
                        __float2bfloat16_rn(moe_situ(acc[0][0][1], acc[1][0][1]));
                    out[static_cast<std::int64_t>(p1) * 1152 + r1] =
                        __float2bfloat16_rn(moe_situ(acc[0][0][3], acc[1][0][3]));
                }
            } else {
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    const int r0 = row_block * 64 + warp_m * 32 + mi * 16 + gid;
                    const int r1 = r0 + 8;
                    if (c0 < cols) {
                        out[static_cast<std::int64_t>(p0) * 2304 + r0] =
                            __float2bfloat16_rn(acc[mi][0][0]);
                        out[static_cast<std::int64_t>(p0) * 2304 + r1] =
                            __float2bfloat16_rn(acc[mi][0][2]);
                    }
                    if (c1 < cols) {
                        out[static_cast<std::int64_t>(p1) * 2304 + r0] =
                            __float2bfloat16_rn(acc[mi][0][1]);
                        out[static_cast<std::int64_t>(p1) * 2304 + r1] =
                            __float2bfloat16_rn(acc[mi][0][3]);
                    }
                }
            }
        }
    }
}

__global__ void inkling2_moe_reduce_kernel(const BF* __restrict__ grouped_output,
                                           const int* __restrict__ packed_index,
                                           const float* __restrict__ alpha, BF* __restrict__ out) {
    const int t = static_cast<int>(blockIdx.x);
    const int d0 = static_cast<int>(threadIdx.x) * 8;
    float acc[8] = {};
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int p = packed_index[t * 8 + r];
        const float a = alpha[t * 8 + r];
        const uint4 v = load_vec<uint4>(grouped_output + static_cast<std::int64_t>(p) * 2304 + d0);
        const auto* p2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(p2[i]);
            acc[i * 2] = fmaf(f.x, a, acc[i * 2]);
            acc[i * 2 + 1] = fmaf(f.y, a, acc[i * 2 + 1]);
        }
    }
    const __nv_bfloat162 r0 = __floats2bfloat162_rn(acc[0], acc[1]);
    const __nv_bfloat162 r1 = __floats2bfloat162_rn(acc[2], acc[3]);
    const __nv_bfloat162 r2 = __floats2bfloat162_rn(acc[4], acc[5]);
    const __nv_bfloat162 r3 = __floats2bfloat162_rn(acc[6], acc[7]);
    store_vec(out + static_cast<std::int64_t>(t) * 2304 + d0,
              make_uint4(*reinterpret_cast<const unsigned*>(&r0),
                         *reinterpret_cast<const unsigned*>(&r1),
                         *reinterpret_cast<const unsigned*>(&r2),
                         *reinterpret_cast<const unsigned*>(&r3)));
}

template <bool GateUp, int ExpertBN>
__global__ __launch_bounds__(256, 2) void inkling2_moe_grouped_a8_mma_kernel(
    const std::int8_t* __restrict__ qx, const float* __restrict__ x_scale, int x_stride_tokens,
    const int* __restrict__ packed_token, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ codes,
    const std::uint16_t* __restrict__ scales, BF* __restrict__ out);

void moe_prefill_launch(const void* x, const Inkling2MoeWeights& w, void* out, int tokens,
                        const Inkling2MoePrefillWorkspace& ws, cudaStream_t s) {
    const auto* x_bf = static_cast<const BF*>(x);
    auto* out_bf = static_cast<BF*>(out);
    auto* act_bf = static_cast<BF*>(ws.routed_activation);
    auto* grouped_bf = static_cast<BF*>(ws.grouped_output);
    const int route_tiles = (tokens + 7) / 8;

    inkling2_moe_router_mma_kernel<<<dim3(128 / 16, (tokens + 63) / 64), 256, 0, s>>>(
        x_bf, static_cast<const BF*>(w.router.qdata), ws.router_probs, tokens);
    inkling2_moe_select_count_kernel<<<route_tiles, 256, 0, s>>>(
        ws.router_probs, static_cast<const float*>(w.exp_probs_bias.qdata), ws.token_ids,
        ws.token_alpha, ws.local_rank, ws.tile_counts, tokens);
    const auto* gu_codes  = static_cast<const std::uint8_t*>(w.routed_gate_up.qdata);
    const auto* gu_scales = static_cast<const std::uint16_t*>(w.routed_gate_up.scales);
    const auto* dn_codes  = static_cast<const std::uint8_t*>(w.routed_down.qdata);
    const auto* dn_scales = static_cast<const std::uint16_t*>(w.routed_down.scales);

    const auto dispatch_bn = [&]<int ExpertBN>() {
        inkling2_moe_scan_kernel<ExpertBN><<<1, 128, 0, s>>>(
            ws.tile_counts, ws.tile_bases, ws.expert_offsets, ws.route_job_experts,
            ws.route_job_columns, ws.route_job_count, route_tiles);
        inkling2_moe_index_kernel<<<(tokens * 8 + 255) / 256, 256, 0, s>>>(
            ws.token_ids, ws.local_rank, ws.packed_index, ws.tile_bases, ws.packed_token,
            tokens * 8);
        const int max_jobs    = (tokens * 8) / ExpertBN + 128;
        const int grid_blocks = max_jobs * 36 < 2048 ? max_jobs * 36 : 2048;
        if (w.routed_gate_up.qtype == QType::Q3G128_F16S) {
            if (tokens >= 64) {
                Tensor x_t(const_cast<void*>(x), DType::BF16, {2304, tokens});
                detail::A8G64Activation x_a8{
                    Tensor(ws.x_a8_q, DType::I8, {2304, tokens}),
                    Tensor(ws.x_a8_scale, DType::FP32, {tokens, 36})};
                detail::a8_g64_quantize(x_t, x_a8, s);
                inkling2_moe_grouped_a8_mma_kernel<true, ExpertBN><<<grid_blocks, 256, 0, s>>>(
                    ws.x_a8_q, ws.x_a8_scale, tokens, ws.packed_token, ws.expert_offsets,
                    ws.route_job_experts, ws.route_job_columns, ws.route_job_count, gu_codes,
                    gu_scales, act_bf);

                Tensor act_t(ws.routed_activation, DType::BF16, {1152, tokens * 8});
                detail::A8G64Activation act_a8{
                    Tensor(ws.act_a8_q, DType::I8, {1152, tokens * 8}),
                    Tensor(ws.act_a8_scale, DType::FP32, {tokens * 8, 18})};
                detail::a8_g64_quantize(act_t, act_a8, s);
                inkling2_moe_grouped_a8_mma_kernel<false, ExpertBN><<<grid_blocks, 256, 0, s>>>(
                    ws.act_a8_q, ws.act_a8_scale, tokens * 8, ws.packed_token, ws.expert_offsets,
                    ws.route_job_experts, ws.route_job_columns, ws.route_job_count, dn_codes,
                    dn_scales, grouped_bf);
            } else {
                inkling2_moe_grouped_mma_kernel<true, true, ExpertBN><<<grid_blocks, 256, 0, s>>>(
                    x_bf, ws.packed_token, ws.expert_offsets, ws.route_job_experts,
                    ws.route_job_columns, ws.route_job_count, gu_codes, gu_scales, act_bf);
                inkling2_moe_grouped_mma_kernel<true, false, ExpertBN><<<grid_blocks, 256, 0, s>>>(
                    act_bf, ws.packed_token, ws.expert_offsets, ws.route_job_experts,
                    ws.route_job_columns, ws.route_job_count, dn_codes, dn_scales, grouped_bf);
            }
        } else {
            inkling2_moe_grouped_mma_kernel<false, true, ExpertBN><<<grid_blocks, 256, 0, s>>>(
                x_bf, ws.packed_token, ws.expert_offsets, ws.route_job_experts,
                ws.route_job_columns, ws.route_job_count, gu_codes, gu_scales, act_bf);
            inkling2_moe_grouped_mma_kernel<false, false, ExpertBN><<<grid_blocks, 256, 0, s>>>(
                act_bf, ws.packed_token, ws.expert_offsets, ws.route_job_experts,
                ws.route_job_columns, ws.route_job_count, dn_codes, dn_scales, grouped_bf);
        }
    };
    if (tokens <= 16) {
        dispatch_bn.template operator()<kInkling2MoeSmallExpertBN>();
    } else {
        dispatch_bn.template operator()<kInkling2MoeExpertBN>();
    }
    inkling2_moe_reduce_kernel<<<tokens, 288, 0, s>>>(grouped_bf, ws.packed_index, ws.token_alpha,
                                                      out_bf);
    CUDA_CHECK(cudaGetLastError());
}

__device__ __forceinline__ unsigned moe_a8_swz(int line, int chunk) {
    return static_cast<unsigned>(line * 64 + ((chunk ^ ((line >> 1) & 3)) << 4));
}

__device__ __forceinline__ std::uint32_t moe_q3_window(std::uint32_t w0, std::uint32_t w1,
                                                       std::uint32_t w2, int j) {
    if (j == 0) return w0 & 0x00ffffffu;
    if (j == 1) return (w0 >> 24) | ((w1 & 0x0000ffffu) << 8);
    if (j == 2) return (w1 >> 16) | ((w2 & 0x000000ffu) << 16);
    return w2 >> 8;
}

template <bool GateUp, int ExpertBN>
__global__ __launch_bounds__(256, 2) void inkling2_moe_grouped_a8_mma_kernel(
    const std::int8_t* __restrict__ qx, const float* __restrict__ x_scale, int x_stride_tokens,
    const int* __restrict__ packed_token, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ codes,
    const std::uint16_t* __restrict__ scales, BF* __restrict__ out) {
    constexpr int BM              = 64;
    constexpr int BN              = ExpertBN;
    constexpr int BK              = 64;
    constexpr int K               = GateUp ? 2304 : 1152;
    constexpr int KTiles          = K / BK;
    constexpr int GroupsTotal     = K / 128;
    constexpr int Stages          = 2;
    constexpr int RowBlocks       = 36;
    constexpr int kFloatMagic     = 0x4B400000;
    constexpr float kFloatMagicValue = 12582912.0f;

    struct Stage {
        std::uint8_t a[BM * BK];
        std::uint8_t x[BN * BK];
        float row_scale[BM];
        float token_scale[BN];
    };

    __shared__ __align__(128) Stage stages[Stages];
    __shared__ int block_src_cols[BN];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;

    unsigned a_off[2][2];
    unsigned b_off[2];
    {
        const int a_row_base = warp_m * 32 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int b_tok      = warp_n * 8 + (lane & 7);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            a_off[0][ks] = moe_a8_swz(a_row_base, 2 * ks + (lane >> 4));
            a_off[1][ks] = moe_a8_swz(a_row_base + 16, 2 * ks + (lane >> 4));
            b_off[ks]    = moe_a8_swz(b_tok, 2 * ks + ((lane >> 3) & 1));
        }
    }

    const int my_row  = tid >> 1;
    const int my_half = tid & 1;

    const int total_work = route_job_count[0] * RowBlocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / RowBlocks;
        const int row_block   = work - route_job * RowBlocks;
        const int expert      = route_job_experts[route_job];
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = count - column_base < BN ? count - column_base : BN;

        if (tid < BN) {
            const int packed_col = tid < cols ? (begin + column_base + tid) : begin;
            block_src_cols[tid]  = GateUp ? packed_token[packed_col] : packed_col;
        }
        __syncthreads();

        int my_global_row = 0;
        if (tid < 128) {
            if constexpr (GateUp) {
                const int logical = row_block * 32 + ((my_row >> 5) << 4) + (my_row & 15);
                const int is_up   = (my_row >> 4) & 1;
                my_global_row     = expert * 2304 + logical + (is_up ? 1152 : 0);
            } else {
                my_global_row = expert * 2304 + row_block * 64 + my_row;
            }
        }
        const std::uint8_t* my_codes =
            codes + static_cast<std::int64_t>(my_global_row) * GroupsTotal * 48;
        const std::uint16_t* my_scales =
            scales + static_cast<std::int64_t>(my_global_row) * GroupsTotal;

        std::uint32_t raw0 = 0, raw1 = 0, raw2 = 0;
        std::uint16_t raw_scale = 0;
        auto load_q3            = [&](int kt) {
            if (tid < 128) {
                const std::uint8_t* ptr =
                    my_codes + static_cast<std::int64_t>(kt >> 1) * 48 + (kt & 1) * 24 +
                    my_half * 12;
                asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n"
                             : "=r"(raw0)
                             : "l"(ptr));
                asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n"
                             : "=r"(raw1)
                             : "l"(ptr + 4));
                asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n"
                             : "=r"(raw2)
                             : "l"(ptr + 8));
                if (my_half == 0) raw_scale = __ldg(my_scales + (kt >> 1));
            }
        };
        auto decode_q3 = [&](Stage& s) {
            if (tid < 128) {
                unsigned out[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const std::uint32_t win = moe_q3_window(raw0, raw1, raw2, j);
                    out[2 * j]              = detail::q3_a8::spread4(win & 0x0fffu);
                    out[2 * j + 1]          = detail::q3_a8::spread4(win >> 12);
                }
                *reinterpret_cast<uint4*>(s.a + moe_a8_swz(my_row, 2 * my_half)) =
                    make_uint4(out[0], out[1], out[2], out[3]);
                *reinterpret_cast<uint4*>(s.a + moe_a8_swz(my_row, 2 * my_half + 1)) =
                    make_uint4(out[4], out[5], out[6], out[7]);
                if (my_half == 0) s.row_scale[my_row] = __half2float(__ushort_as_half(raw_scale));
            }
        };
        auto stage_x = [&](int stage, int kt) {
            if (tid < 128) {
                const int col   = tid >> 2;
                const int chunk = tid & 3;
                const int sc    = block_src_cols[col];
                cp_async_zfill<16, Cache::cg>(
                    stages[stage].x + moe_a8_swz(col, chunk),
                    qx + static_cast<std::int64_t>(sc) * K + kt * 64 + chunk * 16,
                    col < cols ? 16 : 0);
            }
            if (tid < BN) {
                stages[stage].token_scale[tid] =
                    tid < cols
                        ? x_scale[static_cast<std::int64_t>(kt) * x_stride_tokens +
                                  block_src_cols[tid]]
                        : 0.0f;
            }
        };

        load_q3(0);
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            if (stage < KTiles) stage_x(stage, stage);
            cp_commit();
        }

        float acc[2][4] = {};
        for (int kt = 0; kt < KTiles; ++kt) {
            const int stage = kt & 1;
            cp_wait<Stages - 1>();
            __syncthreads();
            decode_q3(stages[stage]);
            if (kt + 1 < KTiles) load_q3(kt + 1);
            __syncthreads();

            int g[2][4];
            const Stage& s     = stages[stage];
            const unsigned base_a = smem_addr(s.a);
            const unsigned base_x = smem_addr(s.x);
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                unsigned bf0, bf1;
                ldmatrix_x2(bf0, bf1, base_x + b_off[ks]);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    unsigned af0, af1, af2, af3;
                    ldmatrix_x4(af0, af1, af2, af3, base_a + a_off[mi][ks]);
                    int* d = g[mi];
                    if (ks == 0) {
                        mma_s8_from(d[0], d[1], d[2], d[3], af0, af1, af2, af3, bf0, bf1,
                                    kFloatMagic, kFloatMagic, kFloatMagic, kFloatMagic);
                    } else {
                        mma_s8(d[0], d[1], d[2], d[3], af0, af1, af2, af3, bf0, bf1);
                    }
                }
            }
            const float ts0 = s.token_scale[warp_n * 8 + 2 * lid];
            const float ts1 = s.token_scale[warp_n * 8 + 2 * lid + 1];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const float rs0 = s.row_scale[warp_m * 32 + mi * 16 + gid];
                const float rs1 = s.row_scale[warp_m * 32 + mi * 16 + gid + 8];
                acc[mi][0] = fmaf(rs0 * ts0, __int_as_float(g[mi][0]) - kFloatMagicValue, acc[mi][0]);
                acc[mi][1] = fmaf(rs0 * ts1, __int_as_float(g[mi][1]) - kFloatMagicValue, acc[mi][1]);
                acc[mi][2] = fmaf(rs1 * ts0, __int_as_float(g[mi][2]) - kFloatMagicValue, acc[mi][2]);
                acc[mi][3] = fmaf(rs1 * ts1, __int_as_float(g[mi][3]) - kFloatMagicValue, acc[mi][3]);
            }

            __syncthreads();
            const int next = kt + Stages;
            if (next < KTiles) stage_x(stage, next);
            cp_commit();
        }
        cp_wait<0>();

        const int c0 = warp_n * 8 + 2 * lid;
        const int c1 = c0 + 1;
        const int p0 = begin + column_base + c0;
        const int p1 = begin + column_base + c1;
        if constexpr (GateUp) {
            const int r0 = row_block * 32 + warp_m * 16 + gid;
            const int r1 = r0 + 8;
            if (c0 < cols) {
                out[static_cast<std::int64_t>(p0) * 1152 + r0] =
                    __float2bfloat16_rn(moe_situ(acc[0][0], acc[1][0]));
                out[static_cast<std::int64_t>(p0) * 1152 + r1] =
                    __float2bfloat16_rn(moe_situ(acc[0][2], acc[1][2]));
            }
            if (c1 < cols) {
                out[static_cast<std::int64_t>(p1) * 1152 + r0] =
                    __float2bfloat16_rn(moe_situ(acc[0][1], acc[1][1]));
                out[static_cast<std::int64_t>(p1) * 1152 + r1] =
                    __float2bfloat16_rn(moe_situ(acc[0][3], acc[1][3]));
            }
        } else {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r0 = row_block * 64 + warp_m * 32 + mi * 16 + gid;
                const int r1 = r0 + 8;
                if (c0 < cols) {
                    out[static_cast<std::int64_t>(p0) * 2304 + r0] = __float2bfloat16_rn(acc[mi][0]);
                    out[static_cast<std::int64_t>(p0) * 2304 + r1] = __float2bfloat16_rn(acc[mi][2]);
                }
                if (c1 < cols) {
                    out[static_cast<std::int64_t>(p1) * 2304 + r0] = __float2bfloat16_rn(acc[mi][1]);
                    out[static_cast<std::int64_t>(p1) * 2304 + r1] = __float2bfloat16_rn(acc[mi][3]);
                }
            }
        }
    }
}
} // namespace ninfer::ops::inkling2_detail
