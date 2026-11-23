#include "ops/inkling2/launch.h"
#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {
using BF = __nv_bfloat16;
__device__ float bf(const void* p, std::int64_t i) {
    return __bfloat162float(static_cast<const BF*>(p)[i]);
}
__device__ float sum256(float value, float* scratch) {
    value = block_reduce_sum<256>(value, scratch);
    if (threadIdx.x == 0) scratch[0] = value;
    __syncthreads();
    const float result = scratch[0];
    __syncthreads();
    return result;
}
__device__ __forceinline__ float dot_uint4_bf16(uint4 a, uint4 b) {
    const auto* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
    const auto* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 fa = __bfloat1622float2(pa[i]);
        const float2 fb = __bfloat1622float2(pb[i]);
        acc = fmaf(fa.x, fb.x, acc);
        acc = fmaf(fa.y, fb.y, acc);
    }
    return acc;
}
__global__ void proj_kernel(const BF* __restrict__ x, const BF* __restrict__ w,
                            void* __restrict__ out, int n, bool fp32) {
    __shared__ __align__(16) BF x_s[2304];
    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int token = static_cast<int>(blockIdx.y);
    const int row   = static_cast<int>(blockIdx.x) * 8 + warp;

    const BF* xp = x + static_cast<std::int64_t>(token) * 2304;
    store_vec(&x_s[tid * 8], load_vec<uint4>(xp + tid * 8));
    if (tid < 32) store_vec(&x_s[2048 + tid * 8], load_vec<uint4>(xp + 2048 + tid * 8));
    __syncthreads();

    if (row >= n) return;
    const BF* wp = w + static_cast<std::int64_t>(row) * 2304;
    float acc    = 0.0f;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int k = (i * 32 + lane) * 8;
        acc += dot_uint4_bf16(load_vec<uint4>(x_s + k), load_vec<uint4>(wp + k));
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        const auto offset = static_cast<std::int64_t>(token) * n + row;
        if (fp32) static_cast<float*>(out)[offset] = acc;
        else static_cast<BF*>(out)[offset] = __float2bfloat16_rn(acc);
    }
}

__global__ __launch_bounds__(256, 2) void proj_mma_kernel(const BF* __restrict__ x,
                                                          const BF* __restrict__ w,
                                                          void* __restrict__ out, int n,
                                                          int tokens, bool fp32) {
    constexpr int BM = 32;
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

    float acc[2][4] = {};

    auto stage_inputs = [&](int stage, int kt) {
        const int k0 = kt * BK;
        {
            const int row = tid >> 3;
            const int k8 = tid & 7;
            const int rr = row0 + row;
            auto* dst = &Ws[stage][row * BK + gemm_swz64(row, k8 * 8)];
            const BF* src = w + static_cast<std::int64_t>(rr < n ? rr : 0) * 2304 + k0 + k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, rr < n ? 16 : 0);
        }
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int item = tid + it * 256;
            const int col = item >> 3;
            const int k8 = item & 7;
            const int tt = token0 + col;
            auto* dst = &Xs[stage][col * BK + gemm_swz64(col, k8 * 8)];
            const BF* src = x + static_cast<std::int64_t>(tt < tokens ? tt : 0) * 2304 + k0 + k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, tt < tokens ? 16 : 0);
        }
    };

#pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
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
            const int arow = warp_m * 16 + a_rowoff;
            const int acol = ki * 16 + a_coloff;
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(&Ws[stage][arow * BK + gemm_swz64(arow, acol)]));
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                unsigned bf[2];
                const int brow = warp_n * 16 + ni * 8 + b_rin;
                const int bcol = ki * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&Xs[stage][brow * BK + gemm_swz64(brow, bcol)]));
                mma_bf16(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                         af[0], af[1], af[2], af[3], bf[0], bf[1]);
            }
        }

        __syncthreads();
        const int next = kt + STAGES;
        if (next < 2304 / BK) { stage_inputs(stage, next); }
        cp_commit();
    }
    cp_wait<0>();

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int r0 = row0 + warp_m * 16 + gid;
    const int r1 = r0 + 8;
#pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
        const int c0 = token0 + warp_n * 16 + ni * 8 + 2 * lid;
        const int c1 = c0 + 1;
        auto write_elem = [&](int r, int c, float val) {
            if (r < n && c < tokens) {
                const auto offset = static_cast<std::int64_t>(c) * n + r;
                if (fp32) static_cast<float*>(out)[offset] = val;
                else static_cast<BF*>(out)[offset] = __float2bfloat16_rn(val);
            }
        };
        write_elem(r0, c0, acc[ni][0]);
        write_elem(r0, c1, acc[ni][1]);
        write_elem(r1, c0, acc[ni][2]);
        write_elem(r1, c1, acc[ni][3]);
    }
}

__device__ float conv_input(const BF* x, const BF* old, int c, int channels, int token) {
    return __bfloat162float(token >= 0 ? x[static_cast<std::int64_t>(token) * channels + c]
                                      : old[(token + 3) * channels + c]);
}
template<bool Single>
__global__ void sconv_kernel(const BF* x, const float* weight, const void* taps, bool taps_fp32,
                             const BF* old, BF* state, const BF* residual, BF* out, int channels,
                             int tokens) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<std::int64_t>(channels) * tokens) return;
    const int c = i % channels, t = i / channels;
    const float* wc = weight + 4 * c;
    const float4 w = make_float4(wc[0], wc[1], wc[2], wc[3]);
    float u[4];
    #pragma unroll
    for (int lag = 0; lag < 4; ++lag) u[lag] = conv_input(x, old, c, channels, t - lag);
    float result = u[0] + w.x * u[3] + w.y * u[2] + w.z * u[1] + w.w * u[0];
    if (taps) {
        #pragma unroll
        for (int lag = 0; lag < 4; ++lag) {
            const auto index = static_cast<std::int64_t>(t) * 32 + 4 * (c / 128) + lag;
            const float coefficient = taps_fp32 ? static_cast<const float*>(taps)[index] : bf(taps, index);
            result = fmaf(coefficient, u[lag], result);
        }
    }
    result *= taps ? (1.0f / 3.0f) : 0.44721359549995794f;
    if (residual) result += __bfloat162float(residual[i]);
    out[i] = __float2bfloat16_rn(result);
    if constexpr (Single) {
        state[c] = __float2bfloat16_rn(u[2]);
        state[channels + c] = __float2bfloat16_rn(u[1]);
        state[2 * channels + c] = __float2bfloat16_rn(u[0]);
    }
}
__global__ void sconv_state_kernel(const BF* x, const BF* old, BF* state, int channels, int tokens) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= channels) return;
    // Stage all three old-state reads before writing to permit identical in/out storage at T=2.
    const float a = conv_input(x, old, c, channels, tokens - 3);
    const float b = conv_input(x, old, c, channels, tokens - 2);
    const float d = conv_input(x, old, c, channels, tokens - 1);
    state[c] = __float2bfloat16_rn(a);
    state[channels + c] = __float2bfloat16_rn(b);
    state[2 * channels + c] = __float2bfloat16_rn(d);
}
__global__ void sconv_small_t_kernel(const BF* __restrict__ x, const float* __restrict__ weight,
                                     const void* __restrict__ taps, bool taps_fp32,
                                     const BF* __restrict__ old, BF* __restrict__ state,
                                     const BF* __restrict__ residual, BF* __restrict__ out,
                                     int channels, int tokens) {
    const int c = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (c >= channels) return;
    const float* wc   = weight + 4 * c;
    const float4 w    = make_float4(wc[0], wc[1], wc[2], wc[3]);
    float u3          = __bfloat162float(old[c]);
    float u2          = __bfloat162float(old[channels + c]);
    float u1          = __bfloat162float(old[2 * channels + c]);
    const float scale = taps ? (1.0f / 3.0f) : 0.44721359549995794f;
    for (int t = 0; t < tokens; ++t) {
        const auto idx = static_cast<std::int64_t>(t) * channels + c;
        const float u0 = __bfloat162float(x[idx]);
        float result   = u0 + w.x * u3 + w.y * u2 + w.z * u1 + w.w * u0;
        if (taps) {
            const auto tbase = static_cast<std::int64_t>(t) * 32 + 4 * (c / 128);
            auto tap_at      = [&](int lag) {
                return taps_fp32 ? static_cast<const float*>(taps)[tbase + lag]
                                 : bf(taps, tbase + lag);
            };
            result = fmaf(tap_at(0), u0, result);
            result = fmaf(tap_at(1), u1, result);
            result = fmaf(tap_at(2), u2, result);
            result = fmaf(tap_at(3), u3, result);
        }
        result *= scale;
        if (residual) result += __bfloat162float(residual[idx]);
        out[idx] = __float2bfloat16_rn(result);
        u3       = u2;
        u2       = u1;
        u1       = u0;
    }
    state[c]                = __float2bfloat16_rn(u3);
    state[channels + c]     = __float2bfloat16_rn(u2);
    state[2 * channels + c] = __float2bfloat16_rn(u1);
}

__global__ void mix_kernel(Inkling2Branches inputs, const float* __restrict__ query,
                           BF* __restrict__ out, int tokens) {
    const int t = blockIdx.x % tokens, r = blockIdx.x / tokens, tid = threadIdx.x;
    __shared__ float scratch[8], inverse[8], probability[8];
    const int kv0 = tid;
    const int kv1 = 256 + tid;
    const bool has_kv1 = tid < 32;
    const float* qr = query + static_cast<std::int64_t>(r) * 2304;
    const float4 q0a = load_vec<float4>(qr + kv0 * 8);
    const float4 q0b = load_vec<float4>(qr + kv0 * 8 + 4);
    float4 q1a{}, q1b{};
    if (has_kv1) {
        q1a = load_vec<float4>(qr + kv1 * 8);
        q1b = load_vec<float4>(qr + kv1 * 8 + 4);
    }

    auto accum_state = [&](const void* state_ptr, const float4& qa, const float4& qb, int kv,
                           float& ss, float& dot) {
        const uint4 vx = load_vec<uint4>(static_cast<const BF*>(state_ptr) +
                                         static_cast<std::int64_t>(t) * 2304 + kv * 8);
        const auto* px = reinterpret_cast<const __nv_bfloat162*>(&vx);
        const float2 f0 = __bfloat1622float2(px[0]);
        const float2 f1 = __bfloat1622float2(px[1]);
        const float2 f2 = __bfloat1622float2(px[2]);
        const float2 f3 = __bfloat1622float2(px[3]);
        ss = fmaf(f0.x, f0.x, ss); ss = fmaf(f0.y, f0.y, ss);
        ss = fmaf(f1.x, f1.x, ss); ss = fmaf(f1.y, f1.y, ss);
        ss = fmaf(f2.x, f2.x, ss); ss = fmaf(f2.y, f2.y, ss);
        ss = fmaf(f3.x, f3.x, ss); ss = fmaf(f3.y, f3.y, ss);
        dot = fmaf(f0.x, qa.x, dot); dot = fmaf(f0.y, qa.y, dot);
        dot = fmaf(f1.x, qa.z, dot); dot = fmaf(f1.y, qa.w, dot);
        dot = fmaf(f2.x, qb.x, dot); dot = fmaf(f2.y, qb.y, dot);
        dot = fmaf(f3.x, qb.z, dot); dot = fmaf(f3.y, qb.w, dot);
    };

    for (int state = 0; state < inputs.count; ++state) {
        float ss = 0.0f, dot = 0.0f;
        accum_state(inputs.states[state], q0a, q0b, kv0, ss, dot);
        if (has_kv1) accum_state(inputs.states[state], q1a, q1b, kv1, ss, dot);
        const float inv   = rsqrtf(sum256(ss, scratch) / 2304.0f + 1e-6f);
        const float score = sum256(dot, scratch) * inv / 48.0f;
        if (tid == 0) { inverse[state] = inv; probability[state] = score; }
        __syncthreads();
    }
    if (tid == 0) {
        float maximum = -CUDART_INF_F, denominator = 0.0f;
        for (int state = 0; state < inputs.count; ++state) maximum = fmaxf(maximum, probability[state]);
        for (int state = 0; state < inputs.count; ++state) {
            probability[state] = expf(probability[state] - maximum);
            denominator += probability[state];
        }
        for (int state = 0; state < inputs.count; ++state) probability[state] /= denominator;
    }
    __syncthreads();

    float v0[8] = {}, v1[8] = {};
    for (int state = 0; state < inputs.count; ++state) {
        const float w = probability[state] * inverse[state];
        const auto base = static_cast<std::int64_t>(t) * 2304;
        const uint4 vx0 = load_vec<uint4>(static_cast<const BF*>(inputs.states[state]) + base + kv0 * 8);
        const auto* px0 = reinterpret_cast<const __nv_bfloat162*>(&vx0);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(px0[i]);
            v0[2 * i]     = fmaf(w, f.x, v0[2 * i]);
            v0[2 * i + 1] = fmaf(w, f.y, v0[2 * i + 1]);
        }
        if (has_kv1) {
            const uint4 vx1 = load_vec<uint4>(static_cast<const BF*>(inputs.states[state]) + base + kv1 * 8);
            const auto* px1 = reinterpret_cast<const __nv_bfloat162*>(&vx1);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __bfloat1622float2(px1[i]);
                v1[2 * i]     = fmaf(w, f.x, v1[2 * i]);
                v1[2 * i + 1] = fmaf(w, f.y, v1[2 * i + 1]);
            }
        }
    }
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) ss = fmaf(v0[i], v0[i], ss);
    if (has_kv1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = fmaf(v1[i], v1[i], ss);
    }
    const float post = inputs.norm[r] ? rsqrtf(sum256(ss, scratch) / 2304.0f + 1e-6f) : 1.0f;
    auto store_vec8 = [&](const float* v, int kv) {
        __nv_bfloat162 p[4];
        if (inputs.norm[r]) {
            const uint4 wn = load_vec<uint4>(static_cast<const BF*>(inputs.norm[r]) + kv * 8);
            const auto* pw = reinterpret_cast<const __nv_bfloat162*>(&wn);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 fw = __bfloat1622float2(pw[i]);
                p[i] = __floats2bfloat162_rn(v[2 * i] * post * fw.x, v[2 * i + 1] * post * fw.y);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                p[i] = __floats2bfloat162_rn(v[2 * i] * post, v[2 * i + 1] * post);
            }
        }
        store_vec(out + (static_cast<std::int64_t>(r) * tokens + t) * 2304 + kv * 8,
                  make_uint4(*reinterpret_cast<const unsigned*>(&p[0]),
                             *reinterpret_cast<const unsigned*>(&p[1]),
                             *reinterpret_cast<const unsigned*>(&p[2]),
                             *reinterpret_cast<const unsigned*>(&p[3])));
    };
    store_vec8(v0, kv0);
    if (has_kv1) store_vec8(v1, kv1);
}

__global__ void control_down_small_t_kernel(const BF* __restrict__ cur, const BF* __restrict__ msg,
                                            const BF* __restrict__ down, float* __restrict__ z,
                                            int tokens, int cur_c) {
    __shared__ __align__(16) BF cur_s[2304];
    __shared__ __align__(16) BF msg_s[2304];
    __shared__ float scratch[8];

    const int tid    = static_cast<int>(threadIdx.x);
    const int tok    = static_cast<int>(blockIdx.y);
    const int ch     = static_cast<int>(blockIdx.z);
    const int cur_ch = cur_c == 1 ? 0 : ch;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const auto cur_base = (static_cast<std::int64_t>(cur_ch) * tokens + tok) * 2304;
    const auto msg_base = (static_cast<std::int64_t>(ch) * tokens + tok) * 2304;

    float sx = 0.0f, sm = 0.0f;
    for (int kv = tid; kv < 288; kv += 256) {
        const uint4 vx = load_vec<uint4>(cur + cur_base + kv * 8);
        const uint4 vm = load_vec<uint4>(msg + msg_base + kv * 8);
        store_vec(&cur_s[kv * 8], vx);
        store_vec(&msg_s[kv * 8], vm);
        const auto* px = reinterpret_cast<const __nv_bfloat162*>(&vx);
        const auto* pm = reinterpret_cast<const __nv_bfloat162*>(&vm);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 fx = __bfloat1622float2(px[i]);
            const float2 fm = __bfloat1622float2(pm[i]);
            sx              = fmaf(fx.x, fx.x, sx);
            sx              = fmaf(fx.y, fx.y, sx);
            sm              = fmaf(fm.x, fm.x, sm);
            sm              = fmaf(fm.y, fm.y, sm);
        }
    }
    const float ix = rsqrtf(sum256(sx, scratch) / 2304.0f + 1e-6f);
    const float im = rsqrtf(sum256(sm, scratch) / 2304.0f + 1e-6f);

    const int r_base = static_cast<int>(blockIdx.x) * 32 + warp * 4;
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int r   = r_base + rr;
        const BF* dp  = down + static_cast<std::int64_t>(ch * 64 + r) * 4608;
        float acc_cur = 0.0f, acc_msg = 0.0f;
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            const int k = (i * 32 + lane) * 8;
            acc_cur += dot_uint4_bf16(load_vec<uint4>(cur_s + k), load_vec<uint4>(dp + k));
            acc_msg += dot_uint4_bf16(load_vec<uint4>(msg_s + k), load_vec<uint4>(dp + 2304 + k));
        }
        acc_cur = warp_sum(acc_cur);
        acc_msg = warp_sum(acc_msg);
        if (lane == 0) {
            const auto idx  = static_cast<std::int64_t>(ch) * tokens + tok;
            z[idx * 64 + r] = fmaf(acc_cur, ix, acc_msg * im);
        }
    }
}

__global__ void control_out_small_t_kernel(const BF* __restrict__ cur, const BF* __restrict__ msg,
                                           const BF* __restrict__ gf, const float* __restrict__ z,
                                           float* __restrict__ y_fp32, BF* __restrict__ out,
                                           int tokens, int cur_c) {
    const int tok     = static_cast<int>(blockIdx.y);
    const int ch      = static_cast<int>(blockIdx.z);
    const int cur_ch  = cur_c == 1 ? 0 : ch;
    const int tid     = static_cast<int>(threadIdx.x);
    const int d_group = tid >> 3;
    const int k8      = tid & 7;
    const int k0      = k8 * 8;
    const int d       = static_cast<int>(blockIdx.x) * 32 + d_group;
    __shared__ float zs[64];
    const float* latent = z + (static_cast<std::int64_t>(ch) * tokens + tok) * 64;
    if (tid < 64) zs[tid] = latent[tid];
    __syncthreads();
    const uint4 vg =
        load_vec<uint4>(&gf[((static_cast<std::int64_t>(ch) * 2) * 2304 + d) * 64 + k0]);
    const uint4 vf =
        load_vec<uint4>(&gf[((static_cast<std::int64_t>(ch) * 2 + 1) * 2304 + d) * 64 + k0]);
    const auto* pg = reinterpret_cast<const __nv_bfloat162*>(&vg);
    const auto* pf = reinterpret_cast<const __nv_bfloat162*>(&vf);
    float gain = 0.0f, forget = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 fg = __bfloat1622float2(pg[i]);
        const float2 ff = __bfloat1622float2(pf[i]);
        const float z0  = zs[k0 + i * 2];
        const float z1  = zs[k0 + i * 2 + 1];
        gain   = fmaf(fg.x, z0, gain);
        gain   = fmaf(fg.y, z1, gain);
        forget = fmaf(ff.x, z0, forget);
        forget = fmaf(ff.y, z1, forget);
    }
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        gain += __shfl_down_sync(0xffffffffu, gain, off, 8);
        forget += __shfl_down_sync(0xffffffffu, forget, off, 8);
    }
    if (k8 == 0) {
        const auto cur_idx = (static_cast<std::int64_t>(cur_ch) * tokens + tok) * 2304 + d;
        const auto idx     = (static_cast<std::int64_t>(ch) * tokens + tok) * 2304 + d;
        const float x      = __bfloat162float(cur[cur_idx]);
        const float y      = (x - tanhf(forget) * x) + tanhf(gain) * __bfloat162float(msg[idx]);
        if (y_fp32) y_fp32[idx] = y;
        else out[idx] = __float2bfloat16_rn(y);
    }
}

__global__ void control_inv_rms_kernel(const BF* __restrict__ cur, const BF* __restrict__ msg,
                                       float* __restrict__ inv_x, float* __restrict__ inv_m,
                                       int tokens, int cur_c) {
    const int t        = static_cast<int>(blockIdx.x);
    const int c        = static_cast<int>(blockIdx.y);
    const int cur_c_id = cur_c == 1 ? 0 : c;
    const int tid      = static_cast<int>(threadIdx.x);
    const auto cur_base = (static_cast<std::int64_t>(cur_c_id) * tokens + t) * 2304;
    const auto msg_base = (static_cast<std::int64_t>(c) * tokens + t) * 2304;
    float sx = 0.0f, sm = 0.0f;
    for (int kv = tid; kv < 288; kv += 256) {
        const uint4 vx = load_vec<uint4>(cur + cur_base + kv * 8);
        const uint4 vm = load_vec<uint4>(msg + msg_base + kv * 8);
        const auto* px = reinterpret_cast<const __nv_bfloat162*>(&vx);
        const auto* pm = reinterpret_cast<const __nv_bfloat162*>(&vm);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 fx = __bfloat1622float2(px[i]);
            const float2 fm = __bfloat1622float2(pm[i]);
            sx = fmaf(fx.x, fx.x, sx);
            sx = fmaf(fx.y, fx.y, sx);
            sm = fmaf(fm.x, fm.x, sm);
            sm = fmaf(fm.y, fm.y, sm);
        }
    }
    __shared__ float scratch[8];
    const float ix = rsqrtf(sum256(sx, scratch) / 2304.0f + 1e-6f);
    const float im = rsqrtf(sum256(sm, scratch) / 2304.0f + 1e-6f);
    if (tid == 0) {
        const auto idx = static_cast<std::int64_t>(c) * tokens + t;
        inv_x[idx] = ix;
        inv_m[idx] = im;
    }
}

__global__ __launch_bounds__(256, 2) void control_down_mma_kernel(
    const BF* __restrict__ cur, const BF* __restrict__ msg, const BF* __restrict__ down,
    const float* __restrict__ inv_x, const float* __restrict__ inv_m, float* __restrict__ z,
    int tokens, int cur_c) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int STAGES = 2;
    __shared__ __align__(16) BF Ws[STAGES][BM * BK];
    __shared__ __align__(16) BF Xs[STAGES][BN * BK];
    __shared__ float s_inv_x[BN];
    __shared__ float s_inv_m[BN];

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int token0 = static_cast<int>(blockIdx.y) * BN;
    const int c = static_cast<int>(blockIdx.z);
    const int cur_ch = cur_c == 1 ? 0 : c;

    if (tid < BN) {
        const int tt = token0 + tid;
        s_inv_x[tid] = tt < tokens ? inv_x[static_cast<std::int64_t>(c) * tokens + tt] : 0.0f;
        s_inv_m[tid] = tt < tokens ? inv_m[static_cast<std::int64_t>(c) * tokens + tt] : 0.0f;
    }
    __syncthreads();

    float acc_cur[2][2][4] = {};
    float acc_msg[2][2][4] = {};

    auto stage_inputs = [&](int stage, int kt) {
        const bool is_msg = kt >= 36;
        const int k0 = (kt - (is_msg ? 36 : 0)) * BK;
        const int down_k0 = kt * BK;
        const BF* x_base =
            is_msg ? (msg + static_cast<std::int64_t>(c) * tokens * 2304)
                   : (cur + static_cast<std::int64_t>(cur_ch) * tokens * 2304);
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int item = tid + it * 256;
            const int row = item >> 3;
            const int k8 = item & 7;
            cp_async<16, Cache::cg>(
                &Ws[stage][row * BK + gemm_swz64(row, k8 * 8)],
                down + (static_cast<std::int64_t>(c) * 64 + row) * 4608 + down_k0 + k8 * 8);
            const int tt = token0 + row;
            cp_async_zfill<16, Cache::cg>(
                &Xs[stage][row * BK + gemm_swz64(row, k8 * 8)],
                x_base + static_cast<std::int64_t>(tt < tokens ? tt : 0) * 2304 + k0 + k8 * 8,
                tt < tokens ? 16 : 0);
        }
    };

#pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int kt = 0; kt < 72; ++kt) {
        const int stage = kt & 1;
        cp_wait<STAGES - 1>();
        __syncthreads();

        auto& acc = (kt < 36) ? acc_cur : acc_msg;
#pragma unroll
        for (int ki = 0; ki < BK / 16; ++ki) {
            unsigned af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int arow = warp_m * 32 + mi * 16 + a_rowoff;
                const int acol = ki * 16 + a_coloff;
                ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                            smem_addr(&Ws[stage][arow * BK + gemm_swz64(arow, acol)]));
            }
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                unsigned bf[2];
                const int brow = warp_n * 16 + ni * 8 + b_rin;
                const int bcol = ki * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&Xs[stage][brow * BK + gemm_swz64(brow, bcol)]));
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                             af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf[0], bf[1]);
                }
            }
        }

        __syncthreads();
        const int next = kt + STAGES;
        if (next < 72) { stage_inputs(stage, next); }
        cp_commit();
    }
    cp_wait<0>();

    const int gid = lane >> 2;
    const int lid = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int r0 = warp_m * 32 + mi * 16 + gid;
        const int r1 = r0 + 8;
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const int c0_local = warp_n * 16 + ni * 8 + 2 * lid;
            const int c1_local = c0_local + 1;
            const int c0 = token0 + c0_local;
            const int c1 = token0 + c1_local;
            if (c0 < tokens) {
                const float ix = s_inv_x[c0_local];
                const float im = s_inv_m[c0_local];
                z[(static_cast<std::int64_t>(c) * tokens + c0) * 64 + r0] =
                    fmaf(acc_cur[mi][ni][0], ix, acc_msg[mi][ni][0] * im);
                z[(static_cast<std::int64_t>(c) * tokens + c0) * 64 + r1] =
                    fmaf(acc_cur[mi][ni][2], ix, acc_msg[mi][ni][2] * im);
            }
            if (c1 < tokens) {
                const float ix = s_inv_x[c1_local];
                const float im = s_inv_m[c1_local];
                z[(static_cast<std::int64_t>(c) * tokens + c1) * 64 + r0] =
                    fmaf(acc_cur[mi][ni][1], ix, acc_msg[mi][ni][1] * im);
                z[(static_cast<std::int64_t>(c) * tokens + c1) * 64 + r1] =
                    fmaf(acc_cur[mi][ni][3], ix, acc_msg[mi][ni][3] * im);
            }
        }
    }
}

__global__ __launch_bounds__(256, 2) void control_out_mma_kernel(
    const BF* __restrict__ cur, const BF* __restrict__ msg, const BF* __restrict__ gf,
    const float* __restrict__ z, float* __restrict__ y_fp32, BF* __restrict__ out, int tokens,
    int cur_c) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    __shared__ __align__(16) BF Ws[BM * BK];
    __shared__ __align__(16) BF Xs_hi[BN * BK];
    __shared__ __align__(16) BF Xs_lo[BN * BK];

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int d0 = static_cast<int>(blockIdx.x) * 32;
    const int token0 = static_cast<int>(blockIdx.y) * BN;
    const int c = static_cast<int>(blockIdx.z);
    const int cur_ch = cur_c == 1 ? 0 : c;

#pragma unroll
    for (int it = 0; it < 2; ++it) {
        const int item = tid + it * 256;
        const int row = item >> 3;
        const int k8 = item & 7;
        const int half_d = ((row >> 5) << 4) + (row & 15);
        const int is_forget = (row >> 4) & 1;
        const int d_abs = d0 + half_d;
        cp_async<16, Cache::cg>(
            &Ws[row * BK + gemm_swz64(row, k8 * 8)],
            gf + ((static_cast<std::int64_t>(c) * 2 + is_forget) * 2304 + d_abs) * 64 + k8 * 8);

        const int tt = token0 + row;
        uint4 zv_hi = make_uint4(0, 0, 0, 0);
        uint4 zv_lo = make_uint4(0, 0, 0, 0);
        if (tt < tokens) {
            const float* zp = z + (static_cast<std::int64_t>(c) * tokens + tt) * 64 + k8 * 8;
            const float4 z0 = load_vec<float4>(zp);
            const float4 z1 = load_vec<float4>(zp + 4);
            auto split_pair = [](float a, float b, __nv_bfloat162& hi, __nv_bfloat162& lo) {
                const __nv_bfloat16 ha = __float2bfloat16_rn(a);
                const __nv_bfloat16 hb = __float2bfloat16_rn(b);
                hi = __halves2bfloat162(ha, hb);
                lo = __floats2bfloat162_rn(a - __bfloat162float(ha), b - __bfloat162float(hb));
            };
            auto* p_hi = reinterpret_cast<__nv_bfloat162*>(&zv_hi);
            auto* p_lo = reinterpret_cast<__nv_bfloat162*>(&zv_lo);
            split_pair(z0.x, z0.y, p_hi[0], p_lo[0]);
            split_pair(z0.z, z0.w, p_hi[1], p_lo[1]);
            split_pair(z1.x, z1.y, p_hi[2], p_lo[2]);
            split_pair(z1.z, z1.w, p_hi[3], p_lo[3]);
        }
        const int x_off = row * BK + gemm_swz64(row, k8 * 8);
        store_vec(&Xs_hi[x_off], zv_hi);
        store_vec(&Xs_lo[x_off], zv_lo);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;

    float acc[2][2][4] = {};
#pragma unroll
    for (int ki = 0; ki < BK / 16; ++ki) {
        unsigned af[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int arow = warp_m * 32 + mi * 16 + a_rowoff;
            const int acol = ki * 16 + a_coloff;
            ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                        smem_addr(&Ws[arow * BK + gemm_swz64(arow, acol)]));
        }
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            unsigned bf_hi[2], bf_lo[2];
            const int brow = warp_n * 16 + ni * 8 + b_rin;
            const int bcol = ki * 16 + b_koff;
            const int x_off = brow * BK + gemm_swz64(brow, bcol);
            ldmatrix_x2(bf_hi[0], bf_hi[1], smem_addr(&Xs_hi[x_off]));
            ldmatrix_x2(bf_lo[0], bf_lo[1], smem_addr(&Xs_lo[x_off]));
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                         af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf_hi[0], bf_hi[1]);
                mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                         af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf_lo[0], bf_lo[1]);
            }
        }
    }

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int d_lo = d0 + warp_m * 16 + gid;
    const int d_hi = d_lo + 8;
#pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
        const int c0 = token0 + warp_n * 16 + ni * 8 + 2 * lid;
        const int c1 = c0 + 1;
        auto write_elem = [&](int d, int tok, float gain, float forget) {
            if (tok < tokens) {
                const auto cur_idx = (static_cast<std::int64_t>(cur_ch) * tokens + tok) * 2304 + d;
                const auto idx = (static_cast<std::int64_t>(c) * tokens + tok) * 2304 + d;
                const float x = __bfloat162float(cur[cur_idx]);
                const float m = __bfloat162float(msg[idx]);
                const float y = (x - tanhf(forget) * x) + tanhf(gain) * m;
                if (y_fp32) y_fp32[idx] = y;
                else out[idx] = __float2bfloat16_rn(y);
            }
        };
        write_elem(d_lo, c0, acc[0][ni][0], acc[1][ni][0]);
        write_elem(d_lo, c1, acc[0][ni][1], acc[1][ni][1]);
        write_elem(d_hi, c0, acc[0][ni][2], acc[1][ni][2]);
        write_elem(d_hi, c1, acc[0][ni][3], acc[1][ni][3]);
    }
}

__global__ void control_post_rmsnorm_kernel(const float* __restrict__ y_fp32,
                                            BF* __restrict__ out, Inkling2Branches branches,
                                            int tokens) {
    const int t = static_cast<int>(blockIdx.x);
    const int c = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const auto base = (static_cast<std::int64_t>(c) * tokens + t) * 2304;
    const BF* nw = static_cast<const BF*>(branches.norm[c]);
    float ss = 0.0f;
    for (int kv = tid; kv < 576; kv += 256) {
        const float4 v = load_vec<float4>(y_fp32 + base + kv * 4);
        ss = fmaf(v.x, v.x, ss);
        ss = fmaf(v.y, v.y, ss);
        ss = fmaf(v.z, v.z, ss);
        ss = fmaf(v.w, v.w, ss);
    }
    __shared__ float scratch[8];
    const float inv = rsqrtf(sum256(ss, scratch) / 2304.0f + 1e-6f);
    for (int kv = tid; kv < 288; kv += 256) {
        const float4 v0 = load_vec<float4>(y_fp32 + base + kv * 8);
        const float4 v1 = load_vec<float4>(y_fp32 + base + kv * 8 + 4);
        const uint4 w = load_vec<uint4>(nw + kv * 8);
        const auto* pw = reinterpret_cast<const __nv_bfloat162*>(&w);
        const float2 fw0 = __bfloat1622float2(pw[0]);
        const float2 fw1 = __bfloat1622float2(pw[1]);
        const float2 fw2 = __bfloat1622float2(pw[2]);
        const float2 fw3 = __bfloat1622float2(pw[3]);
        const __nv_bfloat162 r0 = __floats2bfloat162_rn(v0.x * inv * fw0.x, v0.y * inv * fw0.y);
        const __nv_bfloat162 r1 = __floats2bfloat162_rn(v0.z * inv * fw1.x, v0.w * inv * fw1.y);
        const __nv_bfloat162 r2 = __floats2bfloat162_rn(v1.x * inv * fw2.x, v1.y * inv * fw2.y);
        const __nv_bfloat162 r3 = __floats2bfloat162_rn(v1.z * inv * fw3.x, v1.w * inv * fw3.y);
        store_vec(out + base + kv * 8,
                  make_uint4(*reinterpret_cast<const unsigned*>(&r0),
                             *reinterpret_cast<const unsigned*>(&r1),
                             *reinterpret_cast<const unsigned*>(&r2),
                             *reinterpret_cast<const unsigned*>(&r3)));
    }
}

__global__ void clip_kernel(const BF* qgr, const BF* k, const BF* v, const float* qc,
                            const float* rc, const int* positions, bool tau, BF* qo, BF* ko,
                            BF* vo, float* ro) {
    const int group = blockIdx.x % 64, t = blockIdx.x / 64, d = threadIdx.x;
    const bool isq = group < 24, isk = group >= 24 && group < 32;
    const bool isv = group >= 32 && group < 40;
    const int h = isq ? group : isk ? group - 24 : isv ? group - 32 : group - 40;
    const int width = (isq || isk || isv) ? 128 : 16;
    float value = 0;
    if (d < width) {
        if (isq) value = __bfloat162float(qgr[static_cast<std::int64_t>(t) * 6528 + h * 128 + d]) * qc[h * 128 + d];
        else if (isk) value = __bfloat162float(k[static_cast<std::int64_t>(t) * 1024 + h * 128 + d]);
        else if (isv) value = __bfloat162float(v[static_cast<std::int64_t>(t) * 1024 + h * 128 + d]);
        else value = __bfloat162float(qgr[static_cast<std::int64_t>(t) * 6528 + 6144 + h * 16 + d]) * rc[h * 16 + d];
    }
    __shared__ float scratch[8];
    const float ss = sum256(value * value, scratch);
    float factor = isv ? rsqrtf(ss / 128.0f + 1e-6f)
                      : fminf(1.0f, ((isq || isk) ? 22.62741699796952f : 8.0f) / (sqrtf(ss) + 1e-6f));
    if (tau && !isk && !isv)
        factor *= 1.0f + 0.1f * logf(fmaxf(1.0f, (static_cast<float>(positions[t]) + 1.0f) / 8192.0f));
    if (d < width) {
        const auto index = (static_cast<std::int64_t>(t) * (isq ? 24 : 8) + h) * 128 + d;
        if (isq) qo[index] = __float2bfloat16_rn(value * factor);
        else if (isk) ko[index] = __float2bfloat16_rn(value * factor);
        else if (isv) vo[index] = __float2bfloat16_rn(value * factor);
        else ro[(static_cast<std::int64_t>(t) * 24 + h) * 16 + d] = value * factor;
    }
}
__global__ void softcap_kernel(const BF* x, BF* out, float cap, int valid, int in_stride,
                               int out_stride, std::int64_t total) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= total) return;
    const auto t = i / valid, row = i % valid;
    out[t * out_stride + row] = __float2bfloat16_rn(cap * tanhf(__bfloat162float(x[t * in_stride + row]) / cap));
}

__global__ void dspark_feature_mix_kernel(const BF* __restrict__ features,
                                          const float* __restrict__ fuse_weight,
                                          Inkling2Branches norms, BF* __restrict__ out,
                                          int tokens) {
    const int t   = static_cast<int>(blockIdx.x);
    const int r   = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);

    __shared__ float probs[4], scratch[8];
    if (tid == 0) {
        const float* row = fuse_weight + r * 4;
        float mx         = fmaxf(fmaxf(row[0], row[1]), fmaxf(row[2], row[3]));
        float sum        = 0.0f;
        for (int j = 0; j < 4; ++j) {
            probs[j] = expf(row[j] - mx);
            sum += probs[j];
        }
        const float inv_sum = 1.0f / sum;
        for (int j = 0; j < 4; ++j) probs[j] *= inv_sum;
    }
    __syncthreads();

    auto mix_vec8 = [&](int kv, float v[8]) {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = 0.0f;
        for (int j = 0; j < 4; ++j) {
            const float p   = probs[j];
            const auto base = (static_cast<std::int64_t>(j) * tokens + t) * 2304 + kv * 8;
            const uint4 vx  = load_vec<uint4>(features + base);
            const auto* px  = reinterpret_cast<const __nv_bfloat162*>(&vx);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __bfloat1622float2(px[i]);
                v[2 * i]       = fmaf(p, f.x, v[2 * i]);
                v[2 * i + 1]   = fmaf(p, f.y, v[2 * i + 1]);
            }
        }
    };

    float v0[8], v1[8]{};
    mix_vec8(tid, v0);
    const bool has_kv1 = tid < 32;
    if (has_kv1) mix_vec8(256 + tid, v1);

    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) ss = fmaf(v0[i], v0[i], ss);
    if (has_kv1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = fmaf(v1[i], v1[i], ss);
    }
    const float inv = rsqrtf(sum256(ss, scratch) / 2304.0f + 1e-6f);
    const BF* nw    = static_cast<const BF*>(norms.norm[r]);

    auto store_vec8 = [&](const float v[8], int kv) {
        const uint4 wn = load_vec<uint4>(nw + kv * 8);
        const auto* pw = reinterpret_cast<const __nv_bfloat162*>(&wn);
        __nv_bfloat162 p[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 fw = __bfloat1622float2(pw[i]);
            p[i] = __floats2bfloat162_rn(v[2 * i] * inv * fw.x, v[2 * i + 1] * inv * fw.y);
        }
        store_vec(out + (static_cast<std::int64_t>(r) * tokens + t) * 2304 + kv * 8,
                  make_uint4(*reinterpret_cast<const unsigned*>(&p[0]),
                             *reinterpret_cast<const unsigned*>(&p[1]),
                             *reinterpret_cast<const unsigned*>(&p[2]),
                             *reinterpret_cast<const unsigned*>(&p[3])));
    };
    store_vec8(v0, tid);
    if (has_kv1) store_vec8(v1, 256 + tid);
}

__global__ void dspark_rmsnorm_rope_kernel(const int* __restrict__ positions,
                                           const BF* __restrict__ norm, BF* __restrict__ x,
                                           int heads, int tokens, float theta) {
    const int unit = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (unit >= heads * tokens) return;

    const int t   = unit / heads;
    const int pos = positions[t];
    BF* xp        = x + static_cast<std::int64_t>(unit) * 128;

    const int i0 = lane;
    const int i1 = lane + 32;
    const float x0a = __bfloat162float(xp[i0]);
    const float x0b = __bfloat162float(xp[i0 + 64]);
    const float x1a = __bfloat162float(xp[i1]);
    const float x1b = __bfloat162float(xp[i1 + 64]);

    float ss = x0a * x0a + x0b * x0b + x1a * x1a + x1b * x1b;
    ss       = warp_sum(ss);
    const float inv = rsqrtf(__shfl_sync(0xffffffffu, ss, 0) / 128.0f + 1e-6f);

    const float n0a = x0a * inv * __bfloat162float(norm[i0]);
    const float n0b = x0b * inv * __bfloat162float(norm[i0 + 64]);
    const float n1a = x1a * inv * __bfloat162float(norm[i1]);
    const float n1b = x1b * inv * __bfloat162float(norm[i1 + 64]);

    const float fpos = static_cast<float>(pos);
    const float a0   = fpos * powf(theta, -static_cast<float>(2 * i0) / 128.0f);
    const float a1   = fpos * powf(theta, -static_cast<float>(2 * i1) / 128.0f);
    float s0, c0, s1, c1;
    __sincosf(a0, &s0, &c0);
    __sincosf(a1, &s1, &c1);

    xp[i0]      = __float2bfloat16_rn(n0a * c0 - n0b * s0);
    xp[i0 + 64] = __float2bfloat16_rn(n0b * c0 + n0a * s0);
    xp[i1]      = __float2bfloat16_rn(n1a * c1 - n1b * s1);
    xp[i1 + 64] = __float2bfloat16_rn(n1b * c1 + n1a * s1);
}

__global__ void dspark_rnn_pack_kernel(const int* __restrict__ prev_token,
                                       const BF* __restrict__ w1, const float* __restrict__ state,
                                       const BF* __restrict__ h_in, const BF* __restrict__ c_head,
                                       BF* __restrict__ x_buf) {
    const int tid = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (tid >= 5120) return;
    if (tid < 256) {
        x_buf[tid] = __float2bfloat16_rn(state[tid]);
    } else if (tid < 512) {
        const int tok = prev_token[0];
        x_buf[tid]    = w1[static_cast<std::int64_t>(tok) * 256 + (tid - 256)];
    } else if (tid < 2816) {
        x_buf[tid] = h_in[tid - 512];
    } else {
        x_buf[tid] = c_head[tid - 2816];
    }
}

__global__ void dspark_rnn_proj_kernel(const BF* __restrict__ x_buf, const BF* __restrict__ w,
                                       const float* __restrict__ bias, float* __restrict__ guz) {
    __shared__ __align__(16) BF x_s[5120];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row  = static_cast<int>(blockIdx.x) * 8 + warp;

    for (int kv = tid; kv < 640; kv += 256) {
        store_vec(&x_s[kv * 8], load_vec<uint4>(x_buf + kv * 8));
    }
    __syncthreads();

    const BF* wp = w + static_cast<std::int64_t>(row) * 5120;
    float acc    = 0.0f;
    for (int kv = lane; kv < 640; kv += 32) {
        acc += dot_uint4_bf16(load_vec<uint4>(x_s + kv * 8), load_vec<uint4>(wp + kv * 8));
    }
    acc = warp_sum(acc);
    if (lane == 0) guz[row] = acc + bias[row];
}

__global__ void dspark_rnn_out_kernel(const float* __restrict__ guz,
                                      const BF* __restrict__ hidden_out_w,
                                      const float* __restrict__ hidden_out_b,
                                      const BF* __restrict__ h_in, float* __restrict__ state,
                                      BF* __restrict__ h_out) {
    __shared__ __align__(16) float m_s[256];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row  = static_cast<int>(blockIdx.x) * 8 + warp;

    const float g      = guz[tid];
    const float u      = guz[256 + tid];
    const float z      = guz[512 + tid];
    const float u_tanh = tanhf(u);
    const float g_sig  = 1.0f / (1.0f + expf(-g));
    const float s_prev = state[tid];
    const float s_next = u_tanh + g_sig * (s_prev - u_tanh);
    m_s[tid]           = tanhf(z);
    if (blockIdx.x == 0) state[tid] = s_next;
    __syncthreads();

    const BF* wp = hidden_out_w + static_cast<std::int64_t>(row) * 256;
    const uint4 wv = load_vec<uint4>(wp + lane * 8);
    const auto* pw = reinterpret_cast<const __nv_bfloat162*>(&wv);
    const float4 m0 = load_vec<float4>(m_s + lane * 8);
    const float4 m1 = load_vec<float4>(m_s + lane * 8 + 4);
    const float2 fw0 = __bfloat1622float2(pw[0]);
    const float2 fw1 = __bfloat1622float2(pw[1]);
    const float2 fw2 = __bfloat1622float2(pw[2]);
    const float2 fw3 = __bfloat1622float2(pw[3]);
    float acc = fw0.x * m0.x + fw0.y * m0.y + fw1.x * m0.z + fw1.y * m0.w +
                fw2.x * m1.x + fw2.y * m1.y + fw3.x * m1.z + fw3.y * m1.w;
    acc = warp_sum(acc);
    if (lane == 0) {
        h_out[row] = __float2bfloat16_rn(__bfloat162float(h_in[row]) + acc + hidden_out_b[row]);
    }
}

__global__ void dspark_commit_conv_kernel(const BF* __restrict__ pre,
                                          const BF* __restrict__ trace, BF* __restrict__ dst,
                                          int width, int accepted) {
    const int c     = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    const int visit = static_cast<int>(blockIdx.y);
    if (c >= 6656) return;
    int comp_off = 0, c_comp = 1024, local_c = c;
    if (c >= 4352) {
        comp_off = 4352;
        c_comp   = 2304;
        local_c  = c - 4352;
    } else if (c >= 2048) {
        comp_off = 2048;
        c_comp   = 2304;
        local_c  = c - 2048;
    } else if (c >= 1024) {
        comp_off = 1024;
        c_comp   = 1024;
        local_c  = c - 1024;
    }
    const std::int64_t state_base = (static_cast<std::int64_t>(visit) * 6656 + comp_off) * 3;
    const std::int64_t trace_base = (static_cast<std::int64_t>(visit) * 6656 + comp_off) * width;
#pragma unroll
    for (int lag = 0; lag < 3; ++lag) {
        const int idx = accepted - 3 + lag;
        const BF val  = idx >= 0 ? trace[trace_base + static_cast<std::int64_t>(idx) * c_comp + local_c]
                                 : pre[state_base + static_cast<std::int64_t>(idx + 3) * c_comp + local_c];
        dst[state_base + static_cast<std::int64_t>(lag) * c_comp + local_c] = val;
    }
}
} // namespace

void inkling2_proj_launch(const Tensor& x, const Tensor& w, Tensor& out, cudaStream_t s) {
    const int t = x.ne[1], n = out.ne[0];
    if (t >= 16) {
        const dim3 grid((n + 31) / 32, (t + 63) / 64);
        proj_mma_kernel<<<grid, 256, 0, s>>>(static_cast<const BF*>(x.data),
                                             static_cast<const BF*>(w.data), out.data, n, t,
                                             out.dtype == DType::FP32);
    } else {
        proj_kernel<<<dim3((n + 7) / 8, t), 256, 0, s>>>(
            static_cast<const BF*>(x.data), static_cast<const BF*>(w.data), out.data, n,
            out.dtype == DType::FP32);
    }
    CUDA_CHECK(cudaGetLastError());
}
void inkling2_sconv_launch(const Tensor& x, const Tensor& w, const Tensor& dynamic,
                          const Tensor& old, Tensor& state, const Tensor& residual, Tensor& out,
                          cudaStream_t s) {
    const int c = x.ne[0], t = x.ne[1];
    if (t == 1) {
        const auto grid = (x.numel() + 255) / 256;
        sconv_kernel<true><<<grid, 256, 0, s>>>(
            static_cast<const BF*>(x.data), static_cast<const float*>(w.data), dynamic.data,
            dynamic.dtype == DType::FP32, static_cast<const BF*>(old.data),
            static_cast<BF*>(state.data), static_cast<const BF*>(residual.data),
            static_cast<BF*>(out.data), c, t);
    } else if (t <= 15) {
        sconv_small_t_kernel<<<(c + 255) / 256, 256, 0, s>>>(
            static_cast<const BF*>(x.data), static_cast<const float*>(w.data), dynamic.data,
            dynamic.dtype == DType::FP32, static_cast<const BF*>(old.data),
            static_cast<BF*>(state.data), static_cast<const BF*>(residual.data),
            static_cast<BF*>(out.data), c, t);
    } else {
        const auto grid = (x.numel() + 255) / 256;
        sconv_kernel<false><<<grid, 256, 0, s>>>(
            static_cast<const BF*>(x.data), static_cast<const float*>(w.data), dynamic.data,
            dynamic.dtype == DType::FP32, static_cast<const BF*>(old.data),
            static_cast<BF*>(state.data), static_cast<const BF*>(residual.data),
            static_cast<BF*>(out.data), c, t);
        sconv_state_kernel<<<(c + 255) / 256, 256, 0, s>>>(
            static_cast<const BF*>(x.data), static_cast<const BF*>(old.data),
            static_cast<BF*>(state.data), c, t);
    }
    CUDA_CHECK(cudaGetLastError());
}
void inkling2_mix_launch(Inkling2Branches branches, const Tensor& q, Tensor& out, cudaStream_t s) {
    mix_kernel<<<static_cast<std::uint64_t>(out.ne[1]) * out.ne[2], 256, 0, s>>>(branches, static_cast<const float*>(q.data), static_cast<BF*>(out.data), out.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
void inkling2_control_launch(const Tensor& cur, const Tensor& msg, const Tensor& down,
                            const Tensor& gf, Inkling2Branches branches, Tensor& out, float* z,
                            float* inv_rms, float* y_fp32, cudaStream_t s) {
    const int t = msg.ne[1], c = msg.ne[2], cur_c = cur.ne[2];
    float* inv_x = inv_rms;
    float* inv_m = inv_rms + static_cast<std::size_t>(t) * c;
    if (t >= 16) {
        control_inv_rms_kernel<<<dim3(t, c), 256, 0, s>>>(
            static_cast<const BF*>(cur.data), static_cast<const BF*>(msg.data), inv_x, inv_m, t,
            cur_c);
        control_down_mma_kernel<<<dim3(1, (t + 63) / 64, c), 256, 0, s>>>(
            static_cast<const BF*>(cur.data), static_cast<const BF*>(msg.data),
            static_cast<const BF*>(down.data), inv_x, inv_m, z, t, cur_c);
        control_out_mma_kernel<<<dim3(2304 / 32, (t + 63) / 64, c), 256, 0, s>>>(
            static_cast<const BF*>(cur.data), static_cast<const BF*>(msg.data),
            static_cast<const BF*>(gf.data), z, y_fp32, static_cast<BF*>(out.data), t, cur_c);
    } else {
        control_down_small_t_kernel<<<dim3(2, t, c), 256, 0, s>>>(
            static_cast<const BF*>(cur.data), static_cast<const BF*>(msg.data),
            static_cast<const BF*>(down.data), z, t, cur_c);
        control_out_small_t_kernel<<<dim3(72, t, c), 256, 0, s>>>(
            static_cast<const BF*>(cur.data), static_cast<const BF*>(msg.data),
            static_cast<const BF*>(gf.data), z, y_fp32, static_cast<BF*>(out.data), t, cur_c);
    }
    if (y_fp32) {
        control_post_rmsnorm_kernel<<<dim3(t, c), 256, 0, s>>>(
            y_fp32, static_cast<BF*>(out.data), branches, t);
    }
    CUDA_CHECK(cudaGetLastError());
}
void inkling2_clip_launch(const Tensor& qgr, const Tensor& k, const Tensor& v, const Tensor& qc,
                         const Tensor& rc, const Tensor& positions, bool tau, Tensor& qo,
                         Tensor& ko, Tensor& vo, Tensor& ro, cudaStream_t s) {
    clip_kernel<<<static_cast<std::uint64_t>(qgr.ne[1]) * 64, 256, 0, s>>>(static_cast<const BF*>(qgr.data), static_cast<const BF*>(k.data), static_cast<const BF*>(v.data), static_cast<const float*>(qc.data), static_cast<const float*>(rc.data), static_cast<const int*>(positions.data), tau, static_cast<BF*>(qo.data), static_cast<BF*>(ko.data), static_cast<BF*>(vo.data), static_cast<float*>(ro.data));
    CUDA_CHECK(cudaGetLastError());
}
void inkling2_softcap_launch(const Tensor& x, float cap, int valid, Tensor& out, cudaStream_t s) {
    const auto total = static_cast<std::int64_t>(valid) * x.ne[1];
    softcap_kernel<<<(total + 255) / 256, 256, 0, s>>>(static_cast<const BF*>(x.data), static_cast<BF*>(out.data), cap, valid, x.ne[0], out.ne[0], total);
    CUDA_CHECK(cudaGetLastError());
}
void inkling2_dspark_feature_mix_launch(const Tensor& features, const Tensor& fuse_weight,
                                        Inkling2Branches norms, Tensor& out, cudaStream_t s) {
    const int t = features.ne[1];
    dspark_feature_mix_kernel<<<dim3(t, 4), 256, 0, s>>>(
        static_cast<const BF*>(features.data), static_cast<const float*>(fuse_weight.data), norms,
        static_cast<BF*>(out.data), t);
    CUDA_CHECK(cudaGetLastError());
}
void inkling2_dspark_rmsnorm_rope_launch(const Tensor& positions, const Tensor& q_norm,
                                         const Tensor& k_norm, Tensor& q, Tensor& k, float theta,
                                         cudaStream_t s) {
    const int t = q.ne[2];
    const auto* pos_ptr = static_cast<const int*>(positions.data);
    const int q_units = q.ne[1] * t;
    dspark_rmsnorm_rope_kernel<<<(q_units + 7) / 8, 256, 0, s>>>(
        pos_ptr, static_cast<const BF*>(q_norm.data), static_cast<BF*>(q.data), q.ne[1], t, theta);
    if (k.data) {
        const int k_units = k.ne[1] * t;
        dspark_rmsnorm_rope_kernel<<<(k_units + 7) / 8, 256, 0, s>>>(
            pos_ptr, static_cast<const BF*>(k_norm.data), static_cast<BF*>(k.data), k.ne[1], t,
            theta);
    }
    CUDA_CHECK(cudaGetLastError());
}
void inkling2_dspark_rnn_step_launch(const Tensor& prev_token, const Tensor& w1,
                                     const Tensor& rnn_proj_w, const Tensor& rnn_proj_b,
                                     const Tensor& hidden_out_w, const Tensor& hidden_out_b,
                                     const Tensor& h_in, const Tensor& c_head, Tensor& state,
                                     Tensor& h_out, void* x_buf, float* guz, cudaStream_t s) {
    dspark_rnn_pack_kernel<<<5120 / 256, 256, 0, s>>>(
        static_cast<const int*>(prev_token.data), static_cast<const BF*>(w1.data),
        static_cast<const float*>(state.data), static_cast<const BF*>(h_in.data),
        static_cast<const BF*>(c_head.data), static_cast<BF*>(x_buf));
    dspark_rnn_proj_kernel<<<768 / 8, 256, 0, s>>>(
        static_cast<const BF*>(x_buf), static_cast<const BF*>(rnn_proj_w.data),
        static_cast<const float*>(rnn_proj_b.data), guz);
    dspark_rnn_out_kernel<<<2304 / 8, 256, 0, s>>>(
        guz, static_cast<const BF*>(hidden_out_w.data),
        static_cast<const float*>(hidden_out_b.data), static_cast<const BF*>(h_in.data),
        static_cast<float*>(state.data), static_cast<BF*>(h_out.data));
    CUDA_CHECK(cudaGetLastError());
}
void inkling2_dspark_commit_conv_launch(const Tensor& pre, const Tensor& trace, Tensor& dst,
                                        int width, int accepted, cudaStream_t s) {
    dspark_commit_conv_kernel<<<dim3((6656 + 255) / 256, 48), 256, 0, s>>>(
        static_cast<const BF*>(pre.data), static_cast<const BF*>(trace.data),
        static_cast<BF*>(dst.data), width, accepted);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
