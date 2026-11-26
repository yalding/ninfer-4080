#include "ops/inkling2/launch.h"

#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using BF = __nv_bfloat16;

constexpr int kDFlash2Hidden     = 2304;
constexpr int kDFlash2KVRows     = 1024;
constexpr int kDFlash2ConvGroups = 144;
constexpr int kDFlash2ProjRows   = 576;
constexpr int kDFlash2ValidVocab = 200058;
constexpr int kDFlash2PhysVocab  = 201024;
constexpr int kDFlash2Rank       = 256;
constexpr int kDFlash2Candidates = 16;
constexpr int kTopkStage1Blocks  = 128;

__device__ __forceinline__ float warp_sum(float v) {
    for (int m = 16; m >= 1; m >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, m); }
    return v;
}

__device__ __forceinline__ float warp_max_val(float v) {
    for (int m = 16; m >= 1; m >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m)); }
    return v;
}

struct DFlash2StoreLayerView {
    const BF* key_norm;
    BF* cache_k;
    __half* cache_v;
    std::int32_t padded_capacity;
};

__device__ __forceinline__ void dflash2_rope_sincos(int pos, int j, float* s, float* c) {
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    const double inv_freq   = pow(10000.0, -static_cast<double>(2 * j) / 128.0);
    const double angle      = fmod(static_cast<double>(pos) * inv_freq, kTwoPi);
    __sincosf(static_cast<float>(angle), s, c);
}

__global__ __launch_bounds__(256) void dflash2_kv_key_store_kernel(
    const BF* __restrict__ key, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ slots,
    DFlash2StoreLayerView layer, int min_count, int max_count) {
    const int col   = static_cast<int>(blockIdx.x);
    const int count = counts[0];
    if (count < min_count || count > max_count || col >= count) return;

    const int head      = static_cast<int>(threadIdx.x) >> 5;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int j         = lane * 2;
    const BF* input     = key + static_cast<std::int64_t>(col) * kDFlash2KVRows + head * 128;
    float x0            = __bfloat162float(input[j]);
    float x1            = __bfloat162float(input[j + 1]);
    float y0            = __bfloat162float(input[j + 64]);
    float y1            = __bfloat162float(input[j + 65]);
    const float sum     = warp_sum(x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1);
    const float inv     = rsqrtf(__shfl_sync(0xffffffffu, sum, 0) / 128.0f + 1.0e-6f);
    x0 *= inv * __bfloat162float(layer.key_norm[j]);
    x1 *= inv * __bfloat162float(layer.key_norm[j + 1]);
    y0 *= inv * __bfloat162float(layer.key_norm[j + 64]);
    y1 *= inv * __bfloat162float(layer.key_norm[j + 65]);

    const int pos = positions[col];
    float s0, c0, s1, c1;
    dflash2_rope_sincos(pos, j, &s0, &c0);
    dflash2_rope_sincos(pos, j + 1, &s1, &c1);

    const std::int64_t dst =
        128LL * ((pos & 2047) +
                 static_cast<std::int64_t>(layer.padded_capacity) * (head + 8 * slots[0]));
    auto* out      = reinterpret_cast<__nv_bfloat162*>(layer.cache_k + dst);
    out[lane]      = __floats2bfloat162_rn(x0 * c0 - y0 * s0, x1 * c1 - y1 * s1);
    out[lane + 32] = __floats2bfloat162_rn(y0 * c0 + x0 * s0, y1 * c1 + x1 * s1);
}

__global__ __launch_bounds__(256) void dflash2_kv_value_store_kernel(
    const BF* __restrict__ value, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ slots,
    DFlash2StoreLayerView layer, int min_count, int max_count) {
    const int col   = static_cast<int>(blockIdx.x);
    const int count = counts[0];
    if (count < min_count || count > max_count || col >= count) return;

    const int pos   = positions[col];
    const int ring  = pos & 2047;
    const int slot  = slots[0];
    const BF* input = value + static_cast<std::int64_t>(col) * kDFlash2KVRows;
    for (int row = static_cast<int>(threadIdx.x); row < kDFlash2KVRows; row += blockDim.x) {
        const std::int64_t dst =
            (row & 127) +
            128LL * (ring + static_cast<std::int64_t>(layer.padded_capacity) *
                                ((row >> 7) + 8 * slot));
        layer.cache_v[dst] = __float2half_rn(__bfloat162float(input[row]));
    }
}

__global__ __launch_bounds__(256) void dflash2_rmsnorm_2304_kernel(const BF* __restrict__ in,
                                                                   const BF* __restrict__ weight,
                                                                   float eps, BF* __restrict__ out) {
    const int col = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const BF* x   = in + static_cast<std::int64_t>(col) * kDFlash2Hidden;
    BF* y         = out + static_cast<std::int64_t>(col) * kDFlash2Hidden;

    float vals[9];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int idx = tid + i * 256;
        const float v = __bfloat162float(x[idx]);
        vals[i]       = v;
        ss            = fmaf(v, v, ss);
    }
    __shared__ float s_warp[8];
    const float wsum = warp_sum(ss);
    if ((tid & 31) == 0) s_warp[tid >> 5] = wsum;
    __syncthreads();
    float total = tid < 8 ? s_warp[tid] : 0.0f;
    if (tid < 32) {
        total = warp_sum(total);
        if (tid == 0) s_warp[0] = rsqrtf(total / static_cast<float>(kDFlash2Hidden) + eps);
    }
    __syncthreads();
    const float inv = s_warp[0];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int idx = tid + i * 256;
        y[idx]        = __float2bfloat16_rn(vals[i] * inv * __bfloat162float(weight[idx]));
    }
}

__global__ __launch_bounds__(256) void dflash2_conv_proj_kernel(const BF* __restrict__ normed,
                                                                  const BF* __restrict__ proj,
                                                                  float* __restrict__ delta,
                                                                  int width) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * 8 + warp;
    if (row >= kDFlash2ProjRows) return;

    const BF* w_row = proj + static_cast<std::int64_t>(row) * kDFlash2Hidden;
    float acc[16]{};
    for (int h = lane * 8; h < kDFlash2Hidden; h += 256) {
        const uint4 wv    = *reinterpret_cast<const uint4*>(w_row + h);
        const auto* wb    = reinterpret_cast<const __nv_bfloat162*>(&wv);
        for (int col = 0; col < width; ++col) {
            const uint4 xv =
                *reinterpret_cast<const uint4*>(normed + static_cast<std::int64_t>(col) * kDFlash2Hidden + h);
            const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
            float s        = acc[col];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 wf = __bfloat1622float2(wb[j]);
                const float2 xf = __bfloat1622float2(xb[j]);
                s               = fmaf(wf.x, xf.x, s);
                s               = fmaf(wf.y, xf.y, s);
            }
            acc[col] = s;
        }
    }
    for (int col = 0; col < width; ++col) {
        const float sum = warp_sum(acc[col]);
        if (lane == 0) { delta[static_cast<std::int64_t>(col) * kDFlash2ProjRows + row] = sum; }
    }
}

__global__ __launch_bounds__(256) void dflash2_conv_prepare_epilogue_kernel(
    const BF* __restrict__ normed, const BF* __restrict__ base, const float* __restrict__ delta,
    BF* __restrict__ prepared, BF* __restrict__ finish_delta, int width) {
    const int group = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const int col   = tid >> 4;
    const int chan  = tid & 15;
    const int h     = group * 16 + chan;

    if (col < width) {
        const std::int64_t idx = static_cast<std::int64_t>(col) * kDFlash2Hidden + h;
        const float* d         = delta + static_cast<std::int64_t>(col) * kDFlash2ProjRows;
        float val = (__bfloat162float(base[h]) + d[0 * kDFlash2ConvGroups + group]) *
                    __bfloat162float(normed[idx]);
        if (col > 0) {
            val = fmaf(__bfloat162float(base[kDFlash2Hidden + h]) + d[1 * kDFlash2ConvGroups + group],
                       __bfloat162float(normed[idx - kDFlash2Hidden]), val);
        }
        prepared[idx] = __float2bfloat16_rn(val);
    }
    if (tid < width * 2) {
        const int c   = tid >> 1;
        const int tap = tid & 1;
        const float* d = delta + static_cast<std::int64_t>(c) * kDFlash2ProjRows;
        finish_delta[static_cast<std::int64_t>(c) * 2 * kDFlash2ConvGroups +
                     tap * kDFlash2ConvGroups + group] =
            __float2bfloat16_rn(d[(2 + tap) * kDFlash2ConvGroups + group]);
    }
}

__global__ __launch_bounds__(256) void dflash2_conv_finish_add_kernel(
    const BF* __restrict__ proj_out, const BF* __restrict__ base, const BF* __restrict__ delta,
    BF* __restrict__ residual, int width) {
    const int idx = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (idx >= width * kDFlash2Hidden) return;
    const int col   = idx / kDFlash2Hidden;
    const int h     = idx % kDFlash2Hidden;
    const int group = h >> 4;
    const int di    = col * 2 * kDFlash2ConvGroups + group;

    float val = fmaf(__bfloat162float(base[2 * kDFlash2Hidden + h]) + __bfloat162float(delta[di]),
                     __bfloat162float(proj_out[idx]), __bfloat162float(residual[idx]));
    if (col > 0) {
        val = fmaf(__bfloat162float(base[3 * kDFlash2Hidden + h]) +
                       __bfloat162float(delta[di + kDFlash2ConvGroups]),
                   __bfloat162float(proj_out[idx - kDFlash2Hidden]), val);
    }
    residual[idx] = __float2bfloat16_rn(val);
}

__global__ __launch_bounds__(256) void dflash2_hidden_proj_kernel(
    const BF* __restrict__ mask_hidden, const BF* __restrict__ weight, BF* __restrict__ out,
    int k) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int r    = static_cast<int>(blockIdx.x) * 8 + warp;
    const int col  = static_cast<int>(blockIdx.y);
    if (r >= kDFlash2Rank || col >= k) return;

    const BF* w_row = weight + static_cast<std::int64_t>(r) * kDFlash2Hidden;
    const BF* x_col = mask_hidden + static_cast<std::int64_t>(col) * kDFlash2Hidden;
    float acc       = 0.0f;
    for (int h = lane * 8; h < kDFlash2Hidden; h += 256) {
        const uint4 wv = *reinterpret_cast<const uint4*>(w_row + h);
        const uint4 xv = *reinterpret_cast<const uint4*>(x_col + h);
        const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
        const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 wf = __bfloat1622float2(wb[j]);
            const float2 xf = __bfloat1622float2(xb[j]);
            acc             = fmaf(wf.x, xf.x, acc);
            acc             = fmaf(wf.y, xf.y, acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) { out[static_cast<std::int64_t>(col) * kDFlash2Rank + r] = __float2bfloat16_rn(acc); }
}

__device__ __forceinline__ bool is_better(float v_a, int id_a, float v_b, int id_b) {
    if (id_b < 0) return id_a >= 0;
    if (id_a < 0) return false;
    return v_a > v_b || (v_a == v_b && id_a < id_b);
}

__global__ __launch_bounds__(256) void dflash2_topk16_stage1_kernel(
    const BF* __restrict__ logits, std::int32_t* __restrict__ partial_ids,
    float* __restrict__ partial_vals) {
    const int tid    = static_cast<int>(threadIdx.x);
    const int chunk  = static_cast<int>(blockIdx.x);
    const int col    = static_cast<int>(blockIdx.y);
    const int start  = (chunk * kDFlash2ValidVocab) / kTopkStage1Blocks;
    const int end    = ((chunk + 1) * kDFlash2ValidVocab) / kTopkStage1Blocks;
    const BF* column = logits + static_cast<std::int64_t>(col) * kDFlash2PhysVocab;

    float lv[16];
    int lid[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        lv[m]  = -CUDART_INF_F;
        lid[m] = -1;
    }

    for (int v = start + tid; v < end; v += 256) {
        const float val = __bfloat162float(column[v]);
        if (is_better(val, v, lv[15], lid[15])) {
            int pos = 15;
            while (pos > 0 && is_better(val, v, lv[pos - 1], lid[pos - 1])) {
                lv[pos]  = lv[pos - 1];
                lid[pos] = lid[pos - 1];
                --pos;
            }
            lv[pos]  = val;
            lid[pos] = v;
        }
    }

    __shared__ float s_v[256];
    __shared__ int s_id[256];
    __shared__ int s_who[256];

    int cursor = 0;
    const std::int64_t out_base =
        (static_cast<std::int64_t>(col) * kTopkStage1Blocks + chunk) * kDFlash2Candidates;

    for (int m = 0; m < 16; ++m) {
        s_v[tid]   = lv[cursor];
        s_id[tid]  = lid[cursor];
        s_who[tid] = tid;
        __syncthreads();
        for (int s = 128; s >= 1; s >>= 1) {
            if (tid < s) {
                if (is_better(s_v[tid + s], s_id[tid + s], s_v[tid], s_id[tid])) {
                    s_v[tid]   = s_v[tid + s];
                    s_id[tid]  = s_id[tid + s];
                    s_who[tid] = s_who[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            partial_vals[out_base + m] = s_v[0];
            partial_ids[out_base + m]  = s_id[0];
        }
        if (tid == s_who[0] && cursor < 15) { ++cursor; }
        __syncthreads();
    }
}

__global__ __launch_bounds__(256) void dflash2_topk16_stage2_kernel(
    const std::int32_t* __restrict__ partial_ids, const float* __restrict__ partial_vals,
    std::int32_t* __restrict__ candidate_ids, float* __restrict__ unary_scores) {
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.x);
    const std::int64_t in_base =
        static_cast<std::int64_t>(col) * kTopkStage1Blocks * kDFlash2Candidates;

    float lv[8];
    int lid[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int idx = tid + j * 256; // 256 * 8 == 2048 == 128 * 16
        lv[j]         = partial_vals[in_base + idx];
        lid[j]        = partial_ids[in_base + idx];
    }
    // Sort local 8 items descending.
    for (int a = 0; a < 8; ++a) {
        for (int b = a + 1; b < 8; ++b) {
            if (is_better(lv[b], lid[b], lv[a], lid[a])) {
                const float tv = lv[a];
                lv[a]          = lv[b];
                lv[b]          = tv;
                const int ti   = lid[a];
                lid[a]         = lid[b];
                lid[b]         = ti;
            }
        }
    }

    __shared__ float s_v[256];
    __shared__ int s_id[256];
    __shared__ int s_who[256];

    int cursor                  = 0;
    const std::int64_t out_base = static_cast<std::int64_t>(col) * kDFlash2Candidates;
    for (int m = 0; m < 16; ++m) {
        s_v[tid]   = lv[cursor];
        s_id[tid]  = lid[cursor];
        s_who[tid] = tid;
        __syncthreads();
        for (int s = 128; s >= 1; s >>= 1) {
            if (tid < s) {
                if (is_better(s_v[tid + s], s_id[tid + s], s_v[tid], s_id[tid])) {
                    s_v[tid]   = s_v[tid + s];
                    s_id[tid]  = s_id[tid + s];
                    s_who[tid] = s_who[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            unary_scores[out_base + m]  = s_v[0];
            candidate_ids[out_base + m] = s_id[0];
        }
        if (tid == s_who[0] && cursor < 7) { ++cursor; }
        __syncthreads();
    }
}

struct alignas(16) DFlash2SelectorShared {
    BF successors[kDFlash2Candidates * kDFlash2Rank];
    float product[kDFlash2Rank];
    float edge[kDFlash2Candidates];
    int predecessor, base_position;
    float temperature;
    unsigned long long seed;
};

__device__ int dflash2_draw_rank(float edge, float temperature, unsigned long long seed,
                                 int position, float* q) {
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const float maximum = warp_max_val(edge);
    if (temperature <= 0.0f) {
        const unsigned winners =
            __ballot_sync(kFullWarpMask, lane < kDFlash2Candidates && edge == maximum);
        const int selected = winners == 0 ? 0 : __ffs(winners) - 1;
        if (lane < kDFlash2Candidates) q[lane] = lane == selected ? 1.0f : 0.0f;
        return selected;
    }
    const float weight = lane < kDFlash2Candidates ? __expf((edge - maximum) / temperature) : 0.0f;
    const float probability = weight / warp_sum(weight);
    if (lane < kDFlash2Candidates) q[lane] = probability;
    float uniform =
        lane == 0 ? sampling_uniform(seed, position, kSamplePurposeDFlash2Proposal, 0U) : 0.0f;
    uniform          = __shfl_sync(kFullWarpMask, uniform, 0);
    float cumulative = probability;
#pragma unroll
    for (int offset = 1; offset < kDFlash2Candidates; offset *= 2) {
        const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
        if (lane >= offset) cumulative += previous;
    }
    const unsigned hits =
        __ballot_sync(kFullWarpMask, lane < kDFlash2Candidates && uniform < cumulative);
    return hits ? __ffs(hits) - 1 : kDFlash2Candidates - 1;
}

__global__ __launch_bounds__(512, 1) void dflash2_selector_walk_kernel(
    const std::int32_t* __restrict__ ids, const float* __restrict__ unary,
    const BF* __restrict__ hidden, const std::int32_t* __restrict__ anchors,
    const BF* __restrict__ predecessor_cb, const BF* __restrict__ successor_cb,
    const std::int32_t* __restrict__ base_positions, const SamplingConfig* __restrict__ configs,
    std::int32_t* __restrict__ drafts, float* __restrict__ q, int steps) {
    __shared__ DFlash2SelectorShared shared;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) {
        shared.predecessor   = anchors[0];
        shared.base_position = base_positions[0];
        shared.temperature   = configs[0].temperature;
        shared.seed          = configs[0].seed;
    }
#pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        int token = lane == 0 ? ids[step * kDFlash2Candidates + warp] : 0;
        token     = __shfl_sync(kFullWarpMask, token, 0);
        cp_async<16, Cache::cg>(&shared.successors[warp * kDFlash2Rank + lane * 8],
                                successor_cb + static_cast<std::int64_t>(token) * kDFlash2Rank +
                                    lane * 8);
        cp_commit();
        __syncthreads();
        if (tid < kDFlash2Rank) {
            shared.product[tid] =
                __bfloat162float(
                    predecessor_cb[static_cast<std::int64_t>(shared.predecessor) * kDFlash2Rank +
                                   tid]) *
                __bfloat162float(hidden[static_cast<std::int64_t>(step) * kDFlash2Rank + tid]);
        }
        cp_wait<0>();
        __syncthreads();

        float sum = 0.0f;
#pragma unroll
        for (int r = lane; r < kDFlash2Rank; r += 32) {
            sum = fmaf(shared.product[r], __bfloat162float(shared.successors[warp * kDFlash2Rank + r]),
                       sum);
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            const float raw_edge = unary[step * kDFlash2Candidates + warp] + sum;
            shared.edge[warp]    = 30.0f * tanhf(raw_edge / 30.0f);
        }
        __syncthreads();

        if (warp == 0) {
            const float edge = lane < kDFlash2Candidates ? shared.edge[lane] : -CUDART_INF_F;
            const int selected =
                dflash2_draw_rank(edge, shared.temperature, shared.seed, shared.base_position + step,
                                  q + step * kDFlash2Candidates);
            if (lane == 0) {
                shared.predecessor = ids[step * kDFlash2Candidates + selected];
                drafts[step]       = shared.predecessor;
            }
        }
    }
}

} // namespace

void inkling2_dflash2_kv_materialize_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, const Tensor& scratch, cudaStream_t stream) {
    if (envelope.max_count == 0) return;
    const int w               = context.ne[1];
    const Tensor context_flat = context.view({kDFlash2Hidden, w});
    Tensor scratch_flat       = scratch.view({kDFlash2KVRows, w});
    const dim3 store_grid(static_cast<unsigned>(envelope.max_count));
    const auto* pos_ptr    = static_cast<const std::int32_t*>(positions.data);
    const auto* cnt_ptr    = static_cast<const std::int32_t*>(counts.data);
    const auto* slots_ptr  = static_cast<const std::int32_t*>(state_slots.data);

    for (std::size_t i = 0; i < kContextKVMaterializeLayers; ++i) {
        const auto& l = layers[i];
        const DFlash2StoreLayerView view{
            static_cast<const BF*>(l.key_norm_weight.data),
            static_cast<BF*>(l.cache.k.data),
            static_cast<__half*>(l.cache.v.data),
            static_cast<std::int32_t>(l.cache.padded_capacity),
        };
        linear(context_flat, l.key_weight, scratch_flat, stream);
        dflash2_kv_key_store_kernel<<<store_grid, 256, 0, stream>>>(
            static_cast<const BF*>(scratch.data), pos_ptr, cnt_ptr, slots_ptr, view,
            static_cast<int>(envelope.min_count), static_cast<int>(envelope.max_count));
        linear(context_flat, l.value_weight, scratch_flat, stream);
        dflash2_kv_value_store_kernel<<<store_grid, 256, 0, stream>>>(
            static_cast<const BF*>(scratch.data), pos_ptr, cnt_ptr, slots_ptr, view,
            static_cast<int>(envelope.min_count), static_cast<int>(envelope.max_count));
    }
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_dflash2_conv_prepare_launch(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const Tensor& base_kernel,
                                          const Tensor& kernel_proj, Tensor& prepared,
                                          Tensor& finish_delta, void* normed_buf, float* delta_buf,
                                          cudaStream_t stream) {
    const int w   = residual.ne[1];
    auto* normed  = static_cast<BF*>(normed_buf);
    dflash2_rmsnorm_2304_kernel<<<static_cast<unsigned>(w), 256, 0, stream>>>(
        static_cast<const BF*>(residual.data), static_cast<const BF*>(norm_weight.data), eps,
        normed);
    dflash2_conv_proj_kernel<<<kDFlash2ProjRows / 8, 256, 0, stream>>>(
        normed, static_cast<const BF*>(kernel_proj.data), delta_buf, w);
    dflash2_conv_prepare_epilogue_kernel<<<kDFlash2ConvGroups, 256, 0, stream>>>(
        normed, static_cast<const BF*>(base_kernel.data), delta_buf,
        static_cast<BF*>(prepared.data), static_cast<BF*>(finish_delta.data), w);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_dflash2_conv_finish_add_launch(const Tensor& proj_out, const Tensor& base_kernel,
                                             const Tensor& finish_delta, Tensor& residual,
                                             cudaStream_t stream) {
    const int w     = proj_out.ne[1];
    const int total = w * kDFlash2Hidden;
    dflash2_conv_finish_add_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        static_cast<const BF*>(proj_out.data), static_cast<const BF*>(base_kernel.data),
        static_cast<const BF*>(finish_delta.data), static_cast<BF*>(residual.data), w);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_dflash2_topk16_and_select_launch(
    const Tensor& mask_hidden, const Tensor& unary_logits, const Tensor& hidden_projection,
    const Tensor& anchors, const Tensor& predecessor_codebook, const Tensor& successor_codebook,
    const Tensor& base_positions, const SamplingConfig* configs, Tensor& candidate_ids,
    Tensor& drafts, Tensor& proposal_q, void* projected_hidden_buf, std::int32_t* partial_ids,
    float* partial_vals, float* unary_scores, cudaStream_t stream) {
    const int k   = mask_hidden.ne[1];
    auto* proj_bf = static_cast<BF*>(projected_hidden_buf);
    dflash2_hidden_proj_kernel<<<dim3(kDFlash2Rank / 8, static_cast<unsigned>(k)), 256, 0,
                                 stream>>>(
        static_cast<const BF*>(mask_hidden.data), static_cast<const BF*>(hidden_projection.data),
        proj_bf, k);
    dflash2_topk16_stage1_kernel<<<dim3(kTopkStage1Blocks, static_cast<unsigned>(k)), 256, 0,
                                   stream>>>(
        static_cast<const BF*>(unary_logits.data), partial_ids, partial_vals);
    dflash2_topk16_stage2_kernel<<<static_cast<unsigned>(k), 256, 0, stream>>>(
        partial_ids, partial_vals, static_cast<std::int32_t*>(candidate_ids.data), unary_scores);
    dflash2_selector_walk_kernel<<<1, 512, 0, stream>>>(
        static_cast<const std::int32_t*>(candidate_ids.data), unary_scores, proj_bf,
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const BF*>(predecessor_codebook.data),
        static_cast<const BF*>(successor_codebook.data),
        static_cast<const std::int32_t*>(base_positions.data), configs,
        static_cast<std::int32_t*>(drafts.data), static_cast<float*>(proposal_q.data), k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
