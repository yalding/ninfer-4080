#include "ops/inkling2/launch.h"
#include "core/device.h"
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
__global__ void proj_kernel(const BF* x, const BF* w, void* out, int n, bool fp32) {
    const int row = blockIdx.x % n, token = blockIdx.x / n;
    float acc = 0;
    for (int k = threadIdx.x; k < 2304; k += 256)
        acc = fmaf(__bfloat162float(x[token * 2304 + k]), __bfloat162float(w[row * 2304 + k]), acc);
    __shared__ float sums[8];
    acc = block_reduce_sum<256>(acc, sums);
    if (threadIdx.x == 0) {
        const auto offset = static_cast<std::int64_t>(token) * n + row;
        if (fp32) static_cast<float*>(out)[offset] = acc;
        else static_cast<BF*>(out)[offset] = __float2bfloat16_rn(acc);
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

__global__ void mix_kernel(Inkling2Branches inputs, const float* query, BF* out, int tokens) {
    const int t = blockIdx.x % tokens, r = blockIdx.x / tokens, tid = threadIdx.x;
    __shared__ float scratch[8], inverse[8], probability[8], values[2304];
    for (int state = 0; state < inputs.count; ++state) {
        float ss = 0, dot = 0;
        for (int d = tid; d < 2304; d += 256) {
            const float x = bf(inputs.states[state], static_cast<std::int64_t>(t) * 2304 + d);
            ss = fmaf(x, x, ss);
            dot = fmaf(x, query[r * 2304 + d], dot);
        }
        const float inv = rsqrtf(sum256(ss, scratch) / 2304.0f + 1e-6f);
        const float score = sum256(dot, scratch) * inv / 48.0f;
        if (tid == 0) { inverse[state] = inv; probability[state] = score; }
        __syncthreads();
    }
    if (tid == 0) {
        float maximum = -CUDART_INF_F, denominator = 0;
        for (int state = 0; state < inputs.count; ++state) maximum = fmaxf(maximum, probability[state]);
        for (int state = 0; state < inputs.count; ++state) {
            probability[state] = expf(probability[state] - maximum);
            denominator += probability[state];
        }
        for (int state = 0; state < inputs.count; ++state) probability[state] /= denominator;
    }
    __syncthreads();
    float ss = 0;
    for (int d = tid; d < 2304; d += 256) {
        float value = 0;
        for (int state = 0; state < inputs.count; ++state)
            value = fmaf(probability[state] * inverse[state],
                         bf(inputs.states[state], static_cast<std::int64_t>(t) * 2304 + d), value);
        values[d] = value; ss = fmaf(value, value, ss);
    }
    const float post = inputs.norm[r] ? rsqrtf(sum256(ss, scratch) / 2304.0f + 1e-6f) : 1.0f;
    __syncthreads();
    for (int d = tid; d < 2304; d += 256) {
        float value = values[d] * post;
        if (inputs.norm[r]) value *= bf(inputs.norm[r], d);
        out[(static_cast<std::int64_t>(r) * tokens + t) * 2304 + d] = __float2bfloat16_rn(value);
    }
}

__global__ void control_down_kernel(const BF* cur, const BF* msg, const BF* down, float* z,
                                    int tokens) {
    const int t = blockIdx.x % tokens, c = blockIdx.x / tokens, tid = threadIdx.x;
    const auto base = (static_cast<std::int64_t>(c) * tokens + t) * 2304;
    __shared__ float normalized[4608], scratch[8];
    float sx = 0, sm = 0;
    for (int d = tid; d < 2304; d += 256) {
        const float x = __bfloat162float(cur[base + d]), m = __bfloat162float(msg[base + d]);
        sx = fmaf(x, x, sx); sm = fmaf(m, m, sm);
    }
    const float ix = rsqrtf(sum256(sx, scratch) / 2304.0f + 1e-6f);
    const float im = rsqrtf(sum256(sm, scratch) / 2304.0f + 1e-6f);
    for (int d = tid; d < 2304; d += 256) {
        normalized[d] = __bfloat162float(cur[base + d]) * ix;
        normalized[2304 + d] = __bfloat162float(msg[base + d]) * im;
    }
    __syncthreads();
    const int rank = tid / 32, lane = tid % 32;
    for (int r = rank; r < 64; r += 8) {
        float value = 0;
        for (int k = lane; k < 4608; k += 32)
            value = fmaf(normalized[k], __bfloat162float(down[(c * 64 + r) * 4608 + k]), value);
        value = warp_sum(value);
        if (lane == 0) z[(static_cast<std::int64_t>(c) * tokens + t) * 64 + r] = value;
    }
}
__global__ void control_out_kernel(const BF* cur, const BF* msg, const BF* gf,
                                   Inkling2Branches branches, const float* z, BF* out, int tokens) {
    const int t = blockIdx.x % tokens, c = blockIdx.x / tokens, tid = threadIdx.x;
    const auto base = (static_cast<std::int64_t>(c) * tokens + t) * 2304;
    const float* latent = z + (static_cast<std::int64_t>(c) * tokens + t) * 64;
    __shared__ float values[2304], scratch[8], zs[64];
    if (tid < 64) zs[tid] = latent[tid];
    __syncthreads();
    float ss = 0;
    for (int d = tid; d < 2304; d += 256) {
        float gain = 0, forget = 0;
        for (int k = 0; k < 64; ++k) {
            gain = fmaf(__bfloat162float(gf[((c * 2) * 2304 + d) * 64 + k]), zs[k], gain);
            forget = fmaf(__bfloat162float(gf[((c * 2 + 1) * 2304 + d) * 64 + k]), zs[k], forget);
        }
        const float x = __bfloat162float(cur[base + d]);
        const float value = (x - tanhf(forget) * x) + tanhf(gain) * __bfloat162float(msg[base + d]);
        values[d] = value; ss = fmaf(value, value, ss);
    }
    const float inverse = branches.norm[c] ? rsqrtf(sum256(ss, scratch) / 2304.0f + 1e-6f) : 1.0f;
    __syncthreads();
    for (int d = tid; d < 2304; d += 256) {
        float value = values[d] * inverse;
        if (branches.norm[c]) value *= bf(branches.norm[c], d);
        out[base + d] = __float2bfloat16_rn(value);
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
} // namespace

void inkling2_proj_launch(const Tensor& x, const Tensor& w, Tensor& out, cudaStream_t s) {
    proj_kernel<<<static_cast<std::uint64_t>(out.ne[0]) * out.ne[1], 256, 0, s>>>(
        static_cast<const BF*>(x.data), static_cast<const BF*>(w.data), out.data, out.ne[0], out.dtype == DType::FP32);
    CUDA_CHECK(cudaGetLastError());
}
void inkling2_sconv_launch(const Tensor& x, const Tensor& w, const Tensor& dynamic,
                          const Tensor& old, Tensor& state, const Tensor& residual, Tensor& out,
                          cudaStream_t s) {
    const int c = x.ne[0], t = x.ne[1];
    const auto grid = (x.numel() + 255) / 256;
    if (t == 1)
        sconv_kernel<true><<<grid, 256, 0, s>>>(static_cast<const BF*>(x.data), static_cast<const float*>(w.data), dynamic.data, dynamic.dtype == DType::FP32, static_cast<const BF*>(old.data), static_cast<BF*>(state.data), static_cast<const BF*>(residual.data), static_cast<BF*>(out.data), c, t);
    else {
        sconv_kernel<false><<<grid, 256, 0, s>>>(static_cast<const BF*>(x.data), static_cast<const float*>(w.data), dynamic.data, dynamic.dtype == DType::FP32, static_cast<const BF*>(old.data), static_cast<BF*>(state.data), static_cast<const BF*>(residual.data), static_cast<BF*>(out.data), c, t);
        sconv_state_kernel<<<(c + 255) / 256, 256, 0, s>>>(static_cast<const BF*>(x.data), static_cast<const BF*>(old.data), static_cast<BF*>(state.data), c, t);
    }
    CUDA_CHECK(cudaGetLastError());
}
void inkling2_mix_launch(Inkling2Branches branches, const Tensor& q, Tensor& out, cudaStream_t s) {
    mix_kernel<<<static_cast<std::uint64_t>(out.ne[1]) * out.ne[2], 256, 0, s>>>(branches, static_cast<const float*>(q.data), static_cast<BF*>(out.data), out.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
void inkling2_control_launch(const Tensor& cur, const Tensor& msg, const Tensor& down,
                            const Tensor& gf, Inkling2Branches branches, Tensor& out, float* z,
                            cudaStream_t s) {
    const auto grid = static_cast<std::uint64_t>(cur.ne[1]) * cur.ne[2];
    control_down_kernel<<<grid, 256, 0, s>>>(static_cast<const BF*>(cur.data), static_cast<const BF*>(msg.data), static_cast<const BF*>(down.data), z, cur.ne[1]);
    control_out_kernel<<<grid, 256, 0, s>>>(static_cast<const BF*>(cur.data), static_cast<const BF*>(msg.data), static_cast<const BF*>(gf.data), branches, z, static_cast<BF*>(out.data), cur.ne[1]);
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
} // namespace ninfer::ops::detail
