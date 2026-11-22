#include <math_constants.h>
#include "ops/inkling2_moe/launch.h"
#include "ops/linear/q3/q3_rowsplit_storage.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "core/device.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

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
} // namespace ninfer::ops::inkling2_detail
