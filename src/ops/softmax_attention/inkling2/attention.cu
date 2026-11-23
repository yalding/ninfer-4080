#include <math_constants.h>
#include "ops/softmax_attention/inkling2/launch.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "core/device.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::inkling2_detail {
using BF = __nv_bfloat16;

struct Plane {
    const char* p;
    std::int64_t b1, b2, b3;

    __device__ const char* at(int page, int h, int off) const {
        return p + page * b3 + h * b2 + off * b1;
    }
};

Plane plane(const Tensor& x) {
    return {static_cast<const char*>(x.data), x.nb[1], x.nb[2], x.nb[3]};
}

struct Cache {
    Plane k, v, ks, vs;
    const int* table;
    int capacity;
    int lane;
};

// Barrier-free warp-parallel FlashDecoding for single-token / small-T decode.
// Each lane owns four channels (32 * 4 = 128 = D) and each warp strides the key
// range with its own register online-softmax state, so the key loop has no
// barriers. A single end-of-CTA merge combines the warp partials (and the
// zero-valued sink when no split reduction follows).
template <bool Swa, bool I8, int NumWarps>
__global__ __launch_bounds__(NumWarps * 32)
    void attention_decode_kernel(const BF* __restrict__ q, const BF* __restrict__ ck,
                                 const BF* __restrict__ cv, const BF* __restrict__ gate,
                                 const float* __restrict__ rel, const int* __restrict__ pos,
                                 Cache cache, int nsplits, int partition, float* __restrict__ parts,
                                 BF* __restrict__ output) {
    constexpr int D = 128;
    constexpr float Scale = 0.08838834764831844f;

    const int h = blockIdx.x % 24, t = blockIdx.x / 24, split = Swa ? 0 : blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int d0   = lane * 4, kh = h / 3;
    const int qp   = pos[t], first = pos[0];
    const int lo   = Swa ? max(0, qp - 1023) : split * partition;
    const int hi   = Swa ? qp + 1 : min(qp + 1, lo + partition);

    const float* rel_row = rel + 1025 * (h + 24 * t);
    const float far_rel   = rel_row[1024];

    const uint2 qb = load_vec<uint2>(q + d0 + 128 * (h + 24 * t));
    const auto* q2  = reinterpret_cast<const __nv_bfloat162*>(&qb);
    const float2 qa = __bfloat1622float2(q2[0]);
    const float2 qbb = __bfloat1622float2(q2[1]);
    const float qf[4] = {qa.x, qa.y, qbb.x, qbb.y};

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float den = 0.0f, maximum = -CUDART_INF_F;

    for (int key = lo + warp; key < hi; key += NumWarps) {
        float dot = 0.0f;
        float v[4];
        if constexpr (Swa) {
            if (key >= first) {
                const int off  = 128 * (kh + 8 * (key - first));
                const uint2 kb = load_vec<uint2>(ck + d0 + off);
                const auto* k2 = reinterpret_cast<const __nv_bfloat162*>(&kb);
                const float2 k0 = __bfloat1622float2(k2[0]);
                const float2 k1 = __bfloat1622float2(k2[1]);
                dot = fmaf(qf[3], k1.y, fmaf(qf[2], k1.x, fmaf(qf[1], k0.y, qf[0] * k0.x)));
                const uint2 vb = load_vec<uint2>(cv + d0 + off);
                const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(&vb);
                const float2 v0 = __bfloat1622float2(v2[0]);
                const float2 v1 = __bfloat1622float2(v2[1]);
                v[0] = v0.x;
                v[1] = v0.y;
                v[2] = v1.x;
                v[3] = v1.y;
            } else {
                const int slot = key & 1023;
                const uint2 kb = load_vec<uint2>(cache.k.at(cache.lane, kh, slot) + 2 * d0);
                const auto* k2 = reinterpret_cast<const __nv_bfloat162*>(&kb);
                const float2 k0 = __bfloat1622float2(k2[0]);
                const float2 k1 = __bfloat1622float2(k2[1]);
                dot = fmaf(qf[3], k1.y, fmaf(qf[2], k1.x, fmaf(qf[1], k0.y, qf[0] * k0.x)));
                const uint2 vb = load_vec<uint2>(cache.v.at(cache.lane, kh, slot) + 2 * d0);
                const auto* v2 = reinterpret_cast<const __half2*>(&vb);
                const float2 v0 = __half22float2(v2[0]);
                const float2 v1 = __half22float2(v2[1]);
                v[0] = v0.x;
                v[1] = v0.y;
                v[2] = v1.x;
                v[3] = v1.y;
            }
        } else {
            const int page = cache.table[key >> 6], off = key & 63;
            if constexpr (I8) {
                const int32_t ki  = *reinterpret_cast<const int32_t*>(cache.k.at(page, kh, off) + d0);
                const auto* k8    = reinterpret_cast<const signed char*>(&ki);
                const float ksc   = __half2float(
                    reinterpret_cast<const __half*>(cache.ks.at(page, kh, off))[lane >> 4]);
                dot = fmaf(qf[3], k8[3], fmaf(qf[2], k8[2], fmaf(qf[1], k8[1], qf[0] * k8[0]))) * ksc;
                const int32_t vi  = *reinterpret_cast<const int32_t*>(cache.v.at(page, kh, off) + d0);
                const auto* v8    = reinterpret_cast<const signed char*>(&vi);
                const float vsc   = __half2float(
                    reinterpret_cast<const __half*>(cache.vs.at(page, kh, off))[lane >> 4]);
                v[0] = v8[0] * vsc;
                v[1] = v8[1] * vsc;
                v[2] = v8[2] * vsc;
                v[3] = v8[3] * vsc;
            } else {
                const uint2 kb = load_vec<uint2>(cache.k.at(page, kh, off) + 2 * d0);
                const auto* k2 = reinterpret_cast<const __nv_bfloat162*>(&kb);
                const float2 k0 = __bfloat1622float2(k2[0]);
                const float2 k1 = __bfloat1622float2(k2[1]);
                dot = fmaf(qf[3], k1.y, fmaf(qf[2], k1.x, fmaf(qf[1], k0.y, qf[0] * k0.x)));
                const uint2 vb = load_vec<uint2>(cache.v.at(page, kh, off) + 2 * d0);
                const auto* v2 = reinterpret_cast<const __half2*>(&vb);
                const float2 v0 = __half22float2(v2[0]);
                const float2 v1 = __half22float2(v2[1]);
                v[0] = v0.x;
                v[1] = v0.y;
                v[2] = v1.x;
                v[3] = v1.y;
            }
        }

#pragma unroll
        for (int m = 16; m; m >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, m);
        const int dist = qp - key;
        const float score = fmaf(dot, Scale, (!Swa && dist >= 1024) ? far_rel : rel_row[min(dist, 1024)]);
        const float next    = fmaxf(maximum, score);
        const float rescale = maximum == -CUDART_INF_F ? 0.0f : expf(maximum - next);
        const float p       = expf(score - next);
        den = fmaf(den, rescale, p);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = fmaf(acc[i], rescale, p * v[i]);
        maximum = next;
    }

    __shared__ float s_max[NumWarps];
    __shared__ float s_den[NumWarps];
    __shared__ float s_acc[NumWarps][D];

    s_max[warp] = maximum;
    s_den[warp] = den;
#pragma unroll
    for (int i = 0; i < 4; ++i) s_acc[warp][d0 + i] = acc[i];
    __syncthreads();
    if (threadIdx.x >= D) return;

    const int d = threadIdx.x;
    float m = -CUDART_INF_F;
#pragma unroll
    for (int w = 0; w < NumWarps; ++w) m = fmaxf(m, s_max[w]);
    const bool sink = Swa || nsplits == 1;
    if (sink) m = fmaxf(0.0f, m);
    float acc_total = 0.0f, den_total = 0.0f;
#pragma unroll
    for (int w = 0; w < NumWarps; ++w) {
        const float scale = s_max[w] == -CUDART_INF_F ? 0.0f : expf(s_max[w] - m);
        acc_total = fmaf(scale, s_acc[w][d], acc_total);
        den_total = fmaf(scale, s_den[w], den_total);
    }
    if (sink) {
        den_total += expf(-m);
        const float g = __bfloat162float(gate[3072 + d + 128 * h + 6528 * t]);
        output[d + 128 * (h + 24 * t)] = __float2bfloat16_rn(acc_total / den_total / (1.0f + expf(-g)));
    } else {
        const std::int64_t base = (std::int64_t(t) * 24 * nsplits + h * nsplits + split) * 130;
        parts[base + d] = acc_total;
        if (d == 0) {
            parts[base + 128] = m;
            parts[base + 129] = den_total;
        }
    }
}

__global__ void reduce_kernel(const float* parts, const BF* gate, int nsplits, BF* out) {
    int h = blockIdx.x % 24, t = blockIdx.x / 24, d = threadIdx.x;
    float m   = 0; // zero-valued sink is included exactly once, here.
    auto base = (std::int64_t(t) * 24 + h) * nsplits * 130;
    for (int s = 0; s < nsplits; ++s) m = fmaxf(m, parts[base + s * 130 + 128]);
    float acc = 0, den = expf(-m);
    for (int s = 0; s < nsplits; ++s) {
        float maximum = parts[base + s * 130 + 128];
        float scale   = maximum == -CUDART_INF_F ? 0 : expf(maximum - m);
        acc           = fmaf(scale, parts[base + s * 130 + d], acc);
        den           = fmaf(scale, parts[base + s * 130 + 129], den);
    }
    float sigmoid = 1 / (1 + expf(-__bfloat162float(gate[3072 + d + 128 * h + 6528 * t])));
    out[d + 128 * (h + 24 * t)] = __float2bfloat16_rn(acc / den * sigmoid);
}

__device__ __forceinline__ int inkling2_swz128(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned inkling2_swz_addr(unsigned lane_base, unsigned ck, unsigned as,
                                                     unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

template <bool Swa, bool I8>
__device__ __forceinline__ void inkling2_stage_kv(BF* k_s, __half* v_s, Cache cache, int kv_head,
                                                  int k0, int base_pos, int max_query_abs,
                                                  int physical_page, const BF* ck, const BF* cv,
                                                  int tid) {
    constexpr int D = 128;
    constexpr int Bc = 64;
    constexpr int Threads = 128;
    constexpr int VecPerRow = D / 8; // 16
#pragma unroll
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key_l = chunk >> 4;
        const int d = (chunk & 15) << 3;
        const int swz_col = inkling2_swz128(key_l, d);
        BF* pk = &k_s[key_l * D + swz_col];
        __half* pv = &v_s[key_l * D + swz_col];
        const int key_pos = k0 + key_l;
        if (key_pos > max_query_abs) {
            store_vec(pk, make_int4(0, 0, 0, 0));
            store_vec(pv, make_int4(0, 0, 0, 0));
        } else if constexpr (Swa) {
            if (key_pos < base_pos) {
                cp_async<16, ninfer::ops::Cache::cg>(pk, cache.k.at(cache.lane, kv_head, key_pos & 1023) + d * 2);
                cp_async<16, ninfer::ops::Cache::cg>(pv, cache.v.at(cache.lane, kv_head, key_pos & 1023) + d * 2);
            } else {
                const int local_t = key_pos - base_pos;
                const std::int64_t off = (static_cast<std::int64_t>(local_t) * 8 + kv_head) * D + d;
                cp_async<16, ninfer::ops::Cache::cg>(pk, ck + off);
                const int4 vb = load_vec<int4>(cv + off);
                store_vec(pv, bf16x8_bits_to_f16x8_bits(vb));
            }
        } else if constexpr (!I8) {
            cp_async<16, ninfer::ops::Cache::cg>(pk, cache.k.at(physical_page, kv_head, key_l) + d * 2);
            cp_async<16, ninfer::ops::Cache::cg>(pv, cache.v.at(physical_page, kv_head, key_l) + d * 2);
        } else {
            const int2 rk = load_vec<int2>(cache.k.at(physical_page, kv_head, key_l) + d);
            const int2 rv = load_vec<int2>(cache.v.at(physical_page, kv_head, key_l) + d);
            const float ksc = __half2float(
                reinterpret_cast<const __half*>(cache.ks.at(physical_page, kv_head, key_l))[d >> 6]);
            const float vsc = __half2float(
                reinterpret_cast<const __half*>(cache.vs.at(physical_page, kv_head, key_l))[d >> 6]);
            const auto* bk = reinterpret_cast<const signed char*>(&rk);
            const auto* bv = reinterpret_cast<const signed char*>(&rv);
            const __nv_bfloat162 kb0 = __floats2bfloat162_rn(bk[0] * ksc, bk[1] * ksc);
            const __nv_bfloat162 kb1 = __floats2bfloat162_rn(bk[2] * ksc, bk[3] * ksc);
            const __nv_bfloat162 kb2 = __floats2bfloat162_rn(bk[4] * ksc, bk[5] * ksc);
            const __nv_bfloat162 kb3 = __floats2bfloat162_rn(bk[6] * ksc, bk[7] * ksc);
            store_vec(pk, make_uint4(*reinterpret_cast<const unsigned*>(&kb0),
                                     *reinterpret_cast<const unsigned*>(&kb1),
                                     *reinterpret_cast<const unsigned*>(&kb2),
                                     *reinterpret_cast<const unsigned*>(&kb3)));
            const __half2 vh0 = __floats2half2_rn(bv[0] * vsc, bv[1] * vsc);
            const __half2 vh1 = __floats2half2_rn(bv[2] * vsc, bv[3] * vsc);
            const __half2 vh2 = __floats2half2_rn(bv[4] * vsc, bv[5] * vsc);
            const __half2 vh3 = __floats2half2_rn(bv[6] * vsc, bv[7] * vsc);
            store_vec(pv, make_uint4(*reinterpret_cast<const unsigned*>(&vh0),
                                     *reinterpret_cast<const unsigned*>(&vh1),
                                     *reinterpret_cast<const unsigned*>(&vh2),
                                     *reinterpret_cast<const unsigned*>(&vh3)));
        }
    }
}

template <bool Swa, bool I8>
__launch_bounds__(128, 1) __global__
    void inkling2_prompt_kernel(const BF* __restrict__ q, const BF* __restrict__ ck,
                                const BF* __restrict__ cv, const BF* __restrict__ gate,
                                const float* __restrict__ rel, const int* __restrict__ pos,
                                Cache cache, int tokens, BF* __restrict__ out) {
    constexpr int D = 128;
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int Threads = 128;
    constexpr int QKNt = Bc / 8;  // 8
    constexpr int QKKs = D / 16;  // 8
    constexpr int PVNt = D / 8;   // 16
    constexpr int PVKs = Bc / 16; // 4
    constexpr float Scale = 0.08838834764831844f;
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    __shared__ __align__(16) BF q_s[Br * D];
    __shared__ __align__(16) BF k_s[Bc * D];
    __shared__ __align__(16) __half v_s[Bc * D];

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int q0 = q_block * Br;
    const int kv_head = q_head / 3;
    if (q0 >= tokens) return;

    const int base_pos = pos[0];
    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16;

    const unsigned q_sbase = smem_addr(q_s);
    const unsigned k_sbase = smem_addr(k_s);
    const unsigned v_sbase = smem_addr(v_s);
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * 256);
    const unsigned q_as = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base =
        k_sbase + static_cast<unsigned>(b_rin * 256) + (static_cast<unsigned>(lane >> 4) << 11);
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base =
        v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 2048) + static_cast<unsigned>(b_rin * 256);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r = static_cast<unsigned>(b_rin << 4);

    // Stage Q once.
    {
        constexpr int VecPerRow = D / 8; // 16
        constexpr int QRowStride = D * 24;
        const BF* q_block_ptr = q + (static_cast<std::int64_t>(q0) * 24 + q_head) * D;
#pragma unroll
        for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
            const int row = chunk >> 4;
            const int d = (chunk & 15) << 3;
            BF* p = &q_s[row * D + inkling2_swz128(row, d)];
            if (q0 + row < tokens) {
                cp_async<16, ninfer::ops::Cache::cg>(p, &q_block_ptr[row * QRowStride + d]);
            } else {
                store_vec(p, make_int4(0, 0, 0, 0));
            }
        }
    }
    cp_commit();

    const int qrow0 = q0 + warp_row0 + gid;
    const int qrow1 = q0 + warp_row0 + gid + 8;
    const int qabs0 = (qrow0 < tokens) ? base_pos + qrow0 : -1;
    const int qabs1 = (qrow1 < tokens) ? base_pos + qrow1 : -1;
    const float* rel_row0 =
        (qrow0 < tokens) ? rel + static_cast<std::int64_t>(q_head + 24 * qrow0) * 1025 : nullptr;
    const float* rel_row1 =
        (qrow1 < tokens) ? rel + static_cast<std::int64_t>(q_head + 24 * qrow1) * 1025 : nullptr;
    const float far_rel0 = rel_row0 ? rel_row0[1024] : 0.0f;
    const float far_rel1 = rel_row1 ? rel_row1[1024] : 0.0f;

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[n][i] = 0.0f;
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int tile_rows = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int kb_lo = Swa ? max(0, base_pos + q0 - 1023) / Bc : 0;
    const int kb_hi = max_query_abs / Bc;

    for (int kb = kb_lo; kb <= kb_hi; ++kb) {
        const int k0 = kb * Bc;
        const int physical_page = Swa ? 0 : cache.table[kb];
        inkling2_stage_kv<Swa, I8>(k_s, v_s, cache, kv_head, k0, base_pos, max_query_abs,
                                   physical_page, ck, cv, tid);
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }

        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                    inkling2_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
        for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
            ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                        inkling2_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 2048), 0u, k_as,
                                          k_r));
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck_off = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            inkling2_swz_addr(q_lane_base, ck_off, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0],
                                bf[nxt][nt2 + 1][1],
                                inkling2_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 2048),
                                                  ck_off, k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        const bool far_field = (!Swa) && (q0 + Br <= tokens) && ((base_pos + q0) - (k0 + 63) >= 1024);
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if (far_field) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = fmaf(score[nt][0], Scale, far_rel0);
                score[nt][1] = fmaf(score[nt][1], Scale, far_rel0);
                score[nt][2] = fmaf(score[nt][2], Scale, far_rel1);
                score[nt][3] = fmaf(score[nt][3], Scale, far_rel1);
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
            const int swa_min0 = Swa ? max(0, qabs0 - 1023) : 0;
            const int swa_min1 = Swa ? max(0, qabs1 - 1023) : 0;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                const bool v00 = (qrow0 < tokens) && (key0 <= qabs0) && (!Swa || key0 >= swa_min0);
                const bool v01 = (qrow0 < tokens) && (key1 <= qabs0) && (!Swa || key1 >= swa_min0);
                const bool v10 = (qrow1 < tokens) && (key0 <= qabs1) && (!Swa || key0 >= swa_min1);
                const bool v11 = (qrow1 < tokens) && (key1 <= qabs1) && (!Swa || key1 >= swa_min1);
                score[nt][0] = v00 ? fmaf(score[nt][0], Scale, rel_row0[min(qabs0 - key0, 1024)])
                                   : -CUDART_INF_F;
                score[nt][1] = v01 ? fmaf(score[nt][1], Scale, rel_row0[min(qabs0 - key1, 1024)])
                                   : -CUDART_INF_F;
                score[nt][2] = v10 ? fmaf(score[nt][2], Scale, rel_row1[min(qabs1 - key0, 1024)])
                                   : -CUDART_INF_F;
                score[nt][3] = v11 ? fmaf(score[nt][3], Scale, rel_row1[min(qabs1 - key1, 1024)])
                                   : -CUDART_INF_F;
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0 = fmaxf(m0, bm0);
        const float nm1 = fmaxf(m1, bm1);
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 =
                (score[nt][0] > -CUDART_INF_F) ? exp2_approx((score[nt][0] - nm0) * Log2E) : 0.0f;
            const float p01 =
                (score[nt][1] > -CUDART_INF_F) ? exp2_approx((score[nt][1] - nm0) * Log2E) : 0.0f;
            const float p10 =
                (score[nt][2] > -CUDART_INF_F) ? exp2_approx((score[nt][2] - nm1) * Log2E) : 0.0f;
            const float p11 =
                (score[nt][3] > -CUDART_INF_F) ? exp2_approx((score[nt][3] - nm1) * Log2E) : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            const int pk = nt >> 1;
            if ((nt & 1) == 0) {
                p_frag[pk][0] = pack_f16x2(p00, p01);
                p_frag[pk][1] = pack_f16x2(p10, p11);
            } else {
                p_frag[pk][2] = pack_f16x2(p00, p01);
                p_frag[pk][3] = pack_f16x2(p10, p11);
            }
        }

        l0 = fmaf(l0, alpha0, bl0);
        l1 = fmaf(l1, alpha1, bl1);
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        constexpr int PVHalf = PVNt / 2;       // 8
        constexpr int PVLoads = PVKs * PVHalf; // 32
        unsigned vf[2][4];
        ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                      inkling2_swz_addr(v_lane_base, 0u, v_as, v_r));
#pragma unroll
        for (int li = 0; li < PVLoads; ++li) {
            const int k = li / PVHalf;
            const int n2 = (li % PVHalf) * 2;
            const int cur = li & 1;
            const int nxt = cur ^ 1;
            if (li + 1 < PVLoads) {
                const int k2 = (li + 1) / PVHalf;
                const int n2b = ((li + 1) % PVHalf) * 2;
                const unsigned ckv = static_cast<unsigned>(n2b << 4);
                ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                              inkling2_swz_addr(v_lane_base + static_cast<unsigned>(k2 * 4096), ckv,
                                                v_as, v_r));
            }
            mma_f16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                    p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_f16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                    p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
        __syncthreads();
    }

    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);
    const float m_final0 = fmaxf(0.0f, m0);
    const float m_final1 = fmaxf(0.0f, m1);
    const float z0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - m_final0) * Log2E);
    const float z1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - m_final1) * Log2E);
    const float inv_l0 = z0 / (l0 * z0 + exp2_approx(-m_final0 * Log2E));
    const float inv_l1 = z1 / (l1 * z1 + exp2_approx(-m_final1 * Log2E));

#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0 = n * 8 + 2 * lid;
        if (qrow0 < tokens) {
            const float2 g0 = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(
                    &gate[3072 + d0 + 128 * q_head + static_cast<std::int64_t>(6528) * qrow0]));
            const float s0 = 1.0f / (1.0f + expf(-g0.x));
            const float s1 = 1.0f / (1.0f + expf(-g0.y));
            *reinterpret_cast<unsigned*>(
                &out[d0 + 128 * (q_head + static_cast<std::int64_t>(24) * qrow0)]) =
                pack_bf16x2(acc[n][0] * inv_l0 * s0, acc[n][1] * inv_l0 * s1);
        }
        if (qrow1 < tokens) {
            const float2 g1 = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(
                    &gate[3072 + d0 + 128 * q_head + static_cast<std::int64_t>(6528) * qrow1]));
            const float s0 = 1.0f / (1.0f + expf(-g1.x));
            const float s1 = 1.0f / (1.0f + expf(-g1.y));
            *reinterpret_cast<unsigned*>(
                &out[d0 + 128 * (q_head + static_cast<std::int64_t>(24) * qrow1)]) =
                pack_bf16x2(acc[n][2] * inv_l1 * s0, acc[n][3] * inv_l1 * s1);
        }
    }
}

void swa_launch(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& gate,
                const Tensor& rel, const Tensor& pos, const CyclicKVCacheLayerView& c, int lane,
                Tensor& out, cudaStream_t s) {
    Cache cache{plane(c.k), plane(c.v), {}, {}, nullptr, 1024, lane};
    const int tokens = q.ne[2];
    if (tokens >= 16) {
        dim3 grid((tokens + 63) / 64, 24);
        inkling2_prompt_kernel<true, false><<<grid, 128, 0, s>>>(
            static_cast<const BF*>(q.data), static_cast<const BF*>(k.data),
            static_cast<const BF*>(v.data), static_cast<const BF*>(gate.data),
            static_cast<const float*>(rel.data), static_cast<const int*>(pos.data), cache, tokens,
            static_cast<BF*>(out.data));
    } else {
        attention_decode_kernel<true, false, 8><<<dim3(tokens * 24), 8 * 32, 0, s>>>(
            static_cast<const BF*>(q.data), static_cast<const BF*>(k.data),
            static_cast<const BF*>(v.data), static_cast<const BF*>(gate.data),
            static_cast<const float*>(rel.data), static_cast<const int*>(pos.data), cache, 1, 0,
            nullptr, static_cast<BF*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

void full_launch(const Tensor& q, const Tensor& gate, const Tensor& rel, const Tensor& pos,
                 const PagedKVLayerView& c, unsigned keys, float* scratch, Tensor& out,
                 cudaStream_t s) {
    const int tokens = q.ne[2];
    Cache cache{plane(c.k_pages),
                plane(c.v_pages),
                plane(c.k_scale_pages),
                plane(c.v_scale_pages),
                static_cast<const int*>(c.block_table.data),
                0,
                0};
    if (tokens >= 16) {
        dim3 grid((tokens + 63) / 64, 24);
        if (c.storage == KvCacheStorage::Int8Group64) {
            inkling2_prompt_kernel<false, true><<<grid, 128, 0, s>>>(
                static_cast<const BF*>(q.data), nullptr, nullptr, static_cast<const BF*>(gate.data),
                static_cast<const float*>(rel.data), static_cast<const int*>(pos.data), cache,
                tokens, static_cast<BF*>(out.data));
        } else {
            inkling2_prompt_kernel<false, false><<<grid, 128, 0, s>>>(
                static_cast<const BF*>(q.data), nullptr, nullptr, static_cast<const BF*>(gate.data),
                static_cast<const float*>(rel.data), static_cast<const int*>(pos.data), cache,
                tokens, static_cast<BF*>(out.data));
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    int ns = splits(keys, tokens), partition = (keys + ns - 1) / ns;
    dim3 grid(tokens * 24, ns);
    // With one split the kernel applies the zero-valued sink directly and writes
    // the final result to `out`; with several splits it writes per-split partials
    // to `scratch` and reduce_kernel produces the final result.
    BF* out_arg = ns == 1 ? static_cast<BF*>(out.data) : nullptr;
    if (c.storage == KvCacheStorage::Int8Group64)
        attention_decode_kernel<false, true, 4><<<grid, 4 * 32, 0, s>>>(
            static_cast<const BF*>(q.data), nullptr, nullptr, static_cast<const BF*>(gate.data),
            static_cast<const float*>(rel.data), static_cast<const int*>(pos.data), cache, ns,
            partition, scratch, out_arg);
    else
        attention_decode_kernel<false, false, 4><<<grid, 4 * 32, 0, s>>>(
            static_cast<const BF*>(q.data), nullptr, nullptr, static_cast<const BF*>(gate.data),
            static_cast<const float*>(rel.data), static_cast<const int*>(pos.data), cache, ns,
            partition, scratch, out_arg);
    CUDA_CHECK(cudaGetLastError());
    if (ns > 1) {
        reduce_kernel<<<tokens * 24, 128, 0, s>>>(scratch, static_cast<const BF*>(gate.data), ns,
                                                  static_cast<BF*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ninfer::ops::inkling2_detail
