#include <math_constants.h>
#include "ops/inkling2_moe/launch.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
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

__global__ void router_kernel(const BF* x, const BF* router, float* probs, int tokens) {
    int lane = threadIdx.x % 32, expert = blockIdx.x * 8 + threadIdx.x / 32, t = blockIdx.y;
    float dot = 0;
    for (int k = lane; k < 2304; k += 32)
        dot = fmaf(__bfloat162float(x[k + 2304 * t]), __bfloat162float(router[k + 2304 * expert]),
                   dot);
    dot = sum_warp(dot);
    if (lane == 0) probs[expert + 128 * t] = 1 / (1 + expf(-dot));
}

__global__ void route_kernel(const float* probs, const float* bias, int* ids, float* alpha) {
    int t = blockIdx.x; // finite E=128, top=8; serialized selection gives deterministic tie order.
    if (threadIdx.x) return;
    float sum = 0;
    for (int i = 0; i < 8; ++i) {
        int best    = -1;
        float score = -CUDART_INF_F;
        for (int e = 0; e < 128; ++e) {
            bool used = false;
            for (int j = 0; j < i; ++j) used |= ids[8 * t + j] == e;
            float s = probs[128 * t + e] + bias[e];
            if (!used && (best < 0 || s > score)) {
                best  = e;
                score = s;
            }
        }
        ids[8 * t + i] = best;
        sum += probs[128 * t + best];
    }
    for (int i = 0; i < 8; ++i)
        alpha[8 * t + i] = 8 * probs[128 * t + ids[8 * t + i]] / (sum + 1e-20f);
}

template <bool Q3>
__global__ void gate_up_kernel(const BF* x, Quant w, const int* ids, float* act) {
    int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32, slot = blockIdx.y % 8,
        t = blockIdx.y / 8;
    if (row >= 1152) return;
    int erow = ids[8 * t + slot] * 2304 + row;
    float g = 0, u = 0;
    for (int k = lane; k < 2304; k += 32) {
        float a = __bfloat162float(x[k + 2304 * t]);
        g       = fmaf(weight<Q3>(w, erow, k), a, g);
        u       = fmaf(weight<Q3>(w, erow + 1152, k), a, u);
    }
    g = sum_warp(g);
    u = sum_warp(u);
    if (lane == 0)
        act[row + 1152 * (slot + 8 * t)] =
            (4 * tanhf(g * 0.25f) / (1 + expf(-g))) * (25 * tanhf(u * 0.04f));
}

template <bool Q3>
__global__ void down_kernel(Quant w, const int* ids, const float* alpha, const float* act,
                            BF* out) {
    int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32, t = blockIdx.y;
    if (row >= 2304) return;
    float total = 0;
    for (int slot = 0; slot < 8; ++slot) {
        int erow  = ids[8 * t + slot] * 2304 + row;
        float dot = 0;
        for (int k = lane; k < 1152; k += 32)
            dot = fmaf(weight<Q3>(w, erow, k), act[k + 1152 * (slot + 8 * t)], dot);
        dot = sum_warp(dot);
        if (lane == 0) total = fmaf(dot, alpha[8 * t + slot], total);
    }
    if (lane == 0) out[row + 2304 * t] = __float2bfloat16_rn(total);
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
    route_kernel<<<t, 1, 0, s>>>(probs, static_cast<const float*>(w.exp_probs_bias.qdata), ids,
                                 alpha);
    CUDA_CHECK(cudaGetLastError());
    if (w.routed_gate_up.qtype == QType::Q3G128_F16S) {
        gate_up_kernel<true>
            <<<dim3(144, t * 8), 256, 0, s>>>(input, quant(w.routed_gate_up), ids, act);
        down_kernel<true><<<dim3(288, t), 256, 0, s>>>(quant(w.routed_down), ids, alpha, act,
                                                       static_cast<BF*>(out.data));
    } else {
        gate_up_kernel<false>
            <<<dim3(144, t * 8), 256, 0, s>>>(input, quant(w.routed_gate_up), ids, act);
        down_kernel<false><<<dim3(288, t), 256, 0, s>>>(quant(w.routed_down), ids, alpha, act,
                                                        static_cast<BF*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

__device__ __forceinline__ int moe_swz128(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ float moe_situ(float g, float u) {
    return (4.0f * tanhf(g * 0.25f) / (1.0f + expf(-g))) * (25.0f * tanhf(u * 0.04f));
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
    const int jobs = (count + kInkling2MoeExpertBN - 1) / kInkling2MoeExpertBN;
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
        route_job_columns[job_begin + j] = j * kInkling2MoeExpertBN;
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

template <bool Q3, bool GateUp>
__global__ __launch_bounds__(256, 2) void inkling2_moe_grouped_mma_kernel(
    const BF* __restrict__ x, const int* __restrict__ packed_token,
    const int* __restrict__ expert_offsets, const int* __restrict__ route_job_experts,
    const int* __restrict__ route_job_columns, const int* __restrict__ route_job_count,
    const std::uint8_t* __restrict__ codes, const std::uint16_t* __restrict__ scales,
    BF* __restrict__ out) {
    constexpr int BM = 64;
    constexpr int BN = kInkling2MoeExpertBN; // 32
    constexpr int BK = Q3 ? 128 : 64;
    constexpr int K = GateUp ? 2304 : 1152;
    constexpr int KTiles = K / BK;
    constexpr int GroupsTotal = K / (Q3 ? 128 : 64);
    constexpr int CodeBytes = Q3 ? 48 : 32;
    constexpr int Stages = 2;
    constexpr int RowBlocks = GateUp ? (1152 / 32) : (2304 / 64); // 36
    constexpr int StageChunks = BK / 8;                            // 16 (Q3) or 8 (Q4)
    constexpr int StageIters = BN * StageChunks / 256;             // 2 (Q3) or 1 (Q4)

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
                const int col = item / StageChunks;
                const int k8 = item - col * StageChunks;
                const int swz_col = Q3 ? moe_swz128(col, k8 * 8) : detail::gemm_swz64(col, k8 * 8);
                cp_async_zfill<16, Cache::cg>(&Bs[stage][col * BK + swz_col],
                                              src_row[i] + k0 + k8 * 8, col < cols ? 16 : 0);
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
    inkling2_moe_scan_kernel<<<1, 128, 0, s>>>(
        ws.tile_counts, ws.tile_bases, ws.expert_offsets, ws.route_job_experts,
        ws.route_job_columns, ws.route_job_count, route_tiles);
    inkling2_moe_index_kernel<<<(tokens * 8 + 255) / 256, 256, 0, s>>>(
        ws.token_ids, ws.local_rank, ws.packed_index, ws.tile_bases, ws.packed_token, tokens * 8);

    const int max_jobs = (tokens * 8) / kInkling2MoeExpertBN + 128;
    const int grid_blocks = max_jobs * 36 < 2048 ? max_jobs * 36 : 2048;
    const auto* gu_codes = static_cast<const std::uint8_t*>(w.routed_gate_up.qdata);
    const auto* gu_scales = static_cast<const std::uint16_t*>(w.routed_gate_up.scales);
    const auto* dn_codes = static_cast<const std::uint8_t*>(w.routed_down.qdata);
    const auto* dn_scales = static_cast<const std::uint16_t*>(w.routed_down.scales);

    if (w.routed_gate_up.qtype == QType::Q3G128_F16S) {
        inkling2_moe_grouped_mma_kernel<true, true><<<grid_blocks, 256, 0, s>>>(
            x_bf, ws.packed_token, ws.expert_offsets, ws.route_job_experts, ws.route_job_columns,
            ws.route_job_count, gu_codes, gu_scales, act_bf);
        inkling2_moe_grouped_mma_kernel<true, false><<<grid_blocks, 256, 0, s>>>(
            act_bf, ws.packed_token, ws.expert_offsets, ws.route_job_experts, ws.route_job_columns,
            ws.route_job_count, dn_codes, dn_scales, grouped_bf);
    } else {
        inkling2_moe_grouped_mma_kernel<false, true><<<grid_blocks, 256, 0, s>>>(
            x_bf, ws.packed_token, ws.expert_offsets, ws.route_job_experts, ws.route_job_columns,
            ws.route_job_count, gu_codes, gu_scales, act_bf);
        inkling2_moe_grouped_mma_kernel<false, false><<<grid_blocks, 256, 0, s>>>(
            act_bf, ws.packed_token, ws.expert_offsets, ws.route_job_experts, ws.route_job_columns,
            ws.route_job_count, dn_codes, dn_scales, grouped_bf);
    }
    inkling2_moe_reduce_kernel<<<tokens, 288, 0, s>>>(grouped_bf, ws.packed_index, ws.token_alpha,
                                                      out_bf);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::inkling2_detail
