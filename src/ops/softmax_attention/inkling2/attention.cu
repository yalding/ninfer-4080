#include <math_constants.h>
#include "ops/softmax_attention/inkling2/launch.h"
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

template <bool Swa, bool I8>
__device__ float load(Cache c, bool value, int position, int head, int d, int first,
                      const BF* current_k, const BF* current_v) {
    if constexpr (Swa) {
        if (position >= first)
            return __bfloat162float(
                (value ? current_v : current_k)[d + 128 * (head + 8 * (position - first))]);
        auto ptr = (value ? c.v : c.k).at(c.lane, head, position & 1023);
        return value ? __half2float(reinterpret_cast<const __half*>(ptr)[d])
                     : __bfloat162float(reinterpret_cast<const BF*>(ptr)[d]);
    } else {
        int page = c.table[position / 64], off = position % 64;
        auto ptr = (value ? c.v : c.k).at(page, head, off);
        if constexpr (I8) {
            auto scale = (value ? c.vs : c.ks).at(page, head, off);
            return static_cast<float>(reinterpret_cast<const signed char*>(ptr)[d]) *
                   __half2float(reinterpret_cast<const __half*>(scale)[d / 64]);
        } else
            return value ? __half2float(reinterpret_cast<const __half*>(ptr)[d])
                         : __bfloat162float(reinterpret_cast<const BF*>(ptr)[d]);
    }
}

// Four warps cooperate on four keys. Each warp contracts a key across D; every thread
// retains one output channel and folds the four probabilities with online softmax.
template <bool Swa, bool I8>
__global__ void attention_kernel(const BF* q, const BF* ck, const BF* cv, const BF* gate,
                                 const float* rel, const int* pos, Cache cache, int nsplits,
                                 int partition, float* parts, BF* output) {
    int h = blockIdx.x % 24, t = blockIdx.x / 24, split = blockIdx.y, d = threadIdx.x;
    int qp = pos[t], first = pos[0], lo = Swa ? max(0, qp - 1023) : split * partition;
    int hi = Swa ? qp + 1 : min(qp + 1, lo + partition);
    int kh = h / 3, warp = d / 32, lane = d % 32;
    __shared__ float scores[4];
    float acc = 0, den = 0, maximum = -CUDART_INF_F;
    for (int base = lo; base < hi; base += 4) {
        int key     = base + warp;
        float score = 0;
        if (key < hi) {
            for (int j = 0; j < 4; ++j) {
                int dd   = lane + 32 * j;
                float qq = __bfloat162float(q[dd + 128 * (h + 24 * t)]);
                score    = fmaf(qq, load<Swa, I8>(cache, false, key, kh, dd, first, ck, cv), score);
            }
            for (int offset = 16; offset; offset /= 2)
                score += __shfl_down_sync(0xffffffffu, score, offset);
            if (lane == 0)
                scores[warp] =
                    score * 0.08838834764831844f + rel[min(qp - key, 1024) + 1025 * (h + 24 * t)];
        } else if (lane == 0)
            scores[warp] = -CUDART_INF_F;
        __syncthreads();
        float next    = fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
        next          = fmaxf(maximum, next);
        float rescale = maximum == -CUDART_INF_F ? 0 : expf(maximum - next);
        den *= rescale;
        acc *= rescale;
        for (int j = 0; j < 4 && base + j < hi; ++j) {
            float p = expf(scores[j] - next);
            den += p;
            acc = fmaf(p, load<Swa, I8>(cache, true, base + j, kh, d, first, ck, cv), acc);
        }
        maximum = next;
        __syncthreads();
    }
    if constexpr (Swa) {
        float m = fmaxf(0, maximum), z = maximum == -CUDART_INF_F ? 0 : expf(maximum - m);
        float y                        = acc * z / (den * z + expf(-m));
        output[d + 128 * (h + 24 * t)] = __float2bfloat16_rn(
            y / (1 + expf(-__bfloat162float(gate[3072 + d + 128 * h + 6528 * t]))));
    } else {
        auto base       = (std::int64_t(t) * 24 * nsplits + h * nsplits + split) * 130;
        parts[base + d] = acc;
        if (d == 0) {
            parts[base + 128] = maximum;
            parts[base + 129] = den;
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

void swa_launch(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& gate,
                const Tensor& rel, const Tensor& pos, const CyclicKVCacheLayerView& c, int lane,
                Tensor& out, cudaStream_t s) {
    Cache cache{plane(c.k), plane(c.v), {}, {}, nullptr, 1024, lane};
    attention_kernel<true, false><<<q.ne[2] * 24, 128, 0, s>>>(
        static_cast<const BF*>(q.data), static_cast<const BF*>(k.data),
        static_cast<const BF*>(v.data), static_cast<const BF*>(gate.data),
        static_cast<const float*>(rel.data), static_cast<const int*>(pos.data), cache, 1, 0,
        nullptr, static_cast<BF*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void full_launch(const Tensor& q, const Tensor& gate, const Tensor& rel, const Tensor& pos,
                 const PagedKVLayerView& c, unsigned keys, float* scratch, Tensor& out,
                 cudaStream_t s) {
    int ns = splits(keys, q.ne[2]), partition = (keys + ns - 1) / ns;
    Cache cache{plane(c.k_pages),
                plane(c.v_pages),
                plane(c.k_scale_pages),
                plane(c.v_scale_pages),
                static_cast<const int*>(c.block_table.data),
                0,
                0};
    dim3 grid(q.ne[2] * 24, ns);
    if (c.storage == KvCacheStorage::Int8Group64)
        attention_kernel<false, true><<<grid, 128, 0, s>>>(
            static_cast<const BF*>(q.data), nullptr, nullptr, static_cast<const BF*>(gate.data),
            static_cast<const float*>(rel.data), static_cast<const int*>(pos.data), cache, ns,
            partition, scratch, nullptr);
    else
        attention_kernel<false, false><<<grid, 128, 0, s>>>(
            static_cast<const BF*>(q.data), nullptr, nullptr, static_cast<const BF*>(gate.data),
            static_cast<const float*>(rel.data), static_cast<const int*>(pos.data), cache, ns,
            partition, scratch, nullptr);
    CUDA_CHECK(cudaGetLastError());
    reduce_kernel<<<q.ne[2] * 24, 128, 0, s>>>(scratch, static_cast<const BF*>(gate.data), ns,
                                               static_cast<BF*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::inkling2_detail
