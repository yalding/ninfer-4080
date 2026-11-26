#pragma once

// Q4G64_F16S RowSplit x int8 activation (A8) pipelined tensor-core GEMM for widths >= 64 columns
// (one CTA per SM).

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::q4_tall_a8 {

constexpr int kRows    = 128;
constexpr int kThreads = 256;
constexpr int kStepK   = 64;

constexpr int kFloatMagic        = 0x4B400000;
constexpr float kFloatMagicValue = 12582912.0f;
constexpr int kOutLd             = kRows + 8;

__device__ __forceinline__ unsigned swizzled(int line, int chunk) {
    return static_cast<unsigned>(line * 64 + ((chunk ^ ((line >> 1) & 3)) << 4));
}

// Unpack 4 packed Q4 bytes (8 signed 4-bit codes `(n ^ 8) - 8` in [-8, 7]) into two `unsigned`
// registers each holding 4 signed int8 bytes.
__device__ __forceinline__ void unpack8_s8(std::uint32_t word, unsigned& out0, unsigned& out1) {
    word ^= 0x88888888u;
    const unsigned lo = word & 0x0f0f0f0fu;
    const unsigned hi = (word >> 4) & 0x0f0f0f0fu;
    out0 = (__byte_perm(lo, hi, 0x5140) + 0x78787878u) ^ 0x80808080u;
    out1 = (__byte_perm(lo, hi, 0x7362) + 0x78787878u) ^ 0x80808080u;
}

template <int Tokens>
struct Config {
    static_assert(Tokens == 256 || Tokens == 128 || Tokens == 64);
    static constexpr int kHalves     = Tokens == 256 ? 2 : 1;
    static constexpr int kHalfTokens = Tokens / kHalves;
    static constexpr int kWarpsN     = kHalfTokens / 32;
    static constexpr int kWarpsM     = 8 / kWarpsN;
    static constexpr int kWarpRows   = kRows / kWarpsM;
    static constexpr int MT          = kWarpRows / 16;

    struct Stage {
        std::uint8_t a[kRows * kStepK];
        std::uint8_t x[Tokens * kStepK];
        float row_scale[kRows];
        float token_scale[Tokens];
    };

    static constexpr std::size_t shared_bytes() {
        return std::max(2 * sizeof(Stage),
                        static_cast<std::size_t>(kHalfTokens) * kOutLd * sizeof(__nv_bfloat16));
    }
};

template <int Tokens>
__global__ void __launch_bounds__(kThreads, 1)
    q4_rowsplit_tall_a8_kernel(const std::int8_t* __restrict__ qx,
                               const float* __restrict__ x_scale,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint16_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, std::int32_t rows, std::int32_t k,
                               std::int32_t tokens, std::int32_t token_tiles) {
    using Cfg                 = Config<Tokens>;
    using Stage               = typename Cfg::Stage;
    constexpr int kHalves     = Cfg::kHalves;
    constexpr int kHalfTokens = Cfg::kHalfTokens;
    constexpr int MT          = Cfg::MT;
    extern __shared__ __align__(128) std::uint8_t q4_tall_a8_smem[];
    Stage* stages = reinterpret_cast<Stage*>(q4_tall_a8_smem);

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int wm        = warp % Cfg::kWarpsM;
    const int wn        = warp / Cfg::kWarpsM;
    const bool pong     = warp >= 4;
    const int tile      = static_cast<int>(blockIdx.x);
    const int row_block = tile / token_tiles;
    const int token0    = tile % token_tiles * Tokens;
    const int live      = min(Tokens, tokens - token0);
    const int steps     = k / kStepK;
    const int groups    = k / Q4RowSplitStorage::kGroupK;

    const int my_row     = tid >> 1;
    const int my_half    = tid & 1;
    const int global_row = row_block * kRows + my_row;
    const std::uint8_t* my_codes =
        codes + (static_cast<std::int64_t>(global_row) * groups) * 32 + my_half * 16;
    const std::uint16_t* my_scales = scales + static_cast<std::int64_t>(global_row) * groups;

    std::uint32_t raw0 = 0, raw1 = 0, raw2 = 0, raw3 = 0;
    std::uint16_t raw_scale = 0;
    const auto load         = [&](int step) {
        const std::uint8_t* ptr = my_codes + static_cast<std::int64_t>(step) * 32;
        asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(raw0) : "l"(ptr));
        asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(raw1) : "l"(ptr + 4));
        asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(raw2) : "l"(ptr + 8));
        asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(raw3) : "l"(ptr + 12));
        raw_scale = __ldg(my_scales + step);
    };
    const auto decode = [&](Stage& s) {
        unsigned out_words[8];
        unpack8_s8(raw0, out_words[0], out_words[1]);
        unpack8_s8(raw1, out_words[2], out_words[3]);
        unpack8_s8(raw2, out_words[4], out_words[5]);
        unpack8_s8(raw3, out_words[6], out_words[7]);
        *reinterpret_cast<uint4*>(s.a + swizzled(my_row, 2 * my_half)) =
            make_uint4(out_words[0], out_words[1], out_words[2], out_words[3]);
        *reinterpret_cast<uint4*>(s.a + swizzled(my_row, 2 * my_half + 1)) =
            make_uint4(out_words[4], out_words[5], out_words[6], out_words[7]);
        if (my_half == 0) { s.row_scale[my_row] = __half2float(__ushort_as_half(raw_scale)); }
    };

    constexpr int kXChunks = Tokens * 4 / kThreads;
    const std::int8_t* x_src[kXChunks];
    unsigned x_dst[kXChunks];
    int x_bytes[kXChunks];
#pragma unroll
    for (int i = 0; i < kXChunks; ++i) {
        const int item  = tid + i * kThreads;
        const int token = item >> 2;
        const int chunk = item & 3;
        const bool ok   = token < live;
        x_src[i]   = qx + static_cast<std::int64_t>(ok ? token0 + token : token0) * k + chunk * 16;
        x_dst[i]   = static_cast<unsigned>(offsetof(Stage, x)) + swizzled(token, chunk);
        x_bytes[i] = ok ? 16 : 0;
    }
    const bool stages_scale = tid < Tokens;
    const float* scale_src  = x_scale + (tid < live ? token0 + tid : token0);
    const unsigned scale_dst =
        static_cast<unsigned>(offsetof(Stage, token_scale)) + static_cast<unsigned>(tid) * 4u;
    const int scale_bytes = tid < live ? 4 : 0;
    const auto issue_x    = [&](int step, Stage& s) {
        const unsigned base = smem_addr(&s);
#pragma unroll
        for (int i = 0; i < kXChunks; ++i) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                         :
                         : "r"(base + x_dst[i]), "l"(x_src[i] + step * kStepK), "r"(x_bytes[i]));
        }
        if (stages_scale) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                         :
                         : "r"(base + scale_dst),
                           "l"(scale_src + static_cast<std::int64_t>(step) * tokens),
                           "r"(scale_bytes));
        }
    };

    unsigned a_off[2];
    unsigned b_off[2];
    {
        const int a_row = wm * Cfg::kWarpRows + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int b_tok = wn * 32 + (lane >> 4) * 8 + (lane & 7);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            a_off[ks] = static_cast<unsigned>(offsetof(Stage, a)) +
                        swizzled(a_row, 2 * ks + (lane >> 4));
            b_off[ks] = static_cast<unsigned>(offsetof(Stage, x)) +
                        swizzled(b_tok, 2 * ks + ((lane >> 3) & 1));
        }
    }

    float acc[kHalves][MT][4][4];
#pragma unroll
    for (int h = 0; h < kHalves; ++h) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { acc[h][mt][nt][e] = 0.0f; }
            }
        }
    }

    const auto multiply = [&](const Stage& s, int half, int (&g)[MT][4][4]) {
        const unsigned base  = smem_addr(&s);
        const unsigned b_add = static_cast<unsigned>(half * kHalfTokens * kStepK);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned b[4][2];
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                ldmatrix_x4(b[2 * np][0], b[2 * np][1], b[2 * np + 1][0], b[2 * np + 1][1],
                            base + b_off[ks] + b_add + np * 16 * kStepK);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3, base + a_off[ks] + mt * 16 * kStepK);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    int* d = g[mt][nt];
                    if (ks == 0) {
                        mma_s8_from(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b[nt][0], b[nt][1],
                                    kFloatMagic, kFloatMagic, kFloatMagic, kFloatMagic);
                    } else {
                        mma_s8(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b[nt][0], b[nt][1]);
                    }
                }
            }
        }
    };
    float row_scale[MT][2], token_scale[kHalves][4][2];
    const auto load_scales = [&](const Stage& s) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            row_scale[mt][0] = s.row_scale[wm * Cfg::kWarpRows + mt * 16 + gid];
            row_scale[mt][1] = s.row_scale[wm * Cfg::kWarpRows + mt * 16 + gid + 8];
        }
#pragma unroll
        for (int h = 0; h < kHalves; ++h) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                token_scale[h][nt][0] =
                    s.token_scale[h * kHalfTokens + wn * 32 + nt * 8 + 2 * lid];
                token_scale[h][nt][1] =
                    s.token_scale[h * kHalfTokens + wn * 32 + nt * 8 + 2 * lid + 1];
            }
        }
    };
    const auto update = [&](int half, const int (&g)[MT][4][4]) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const float rs0 = row_scale[mt][0];
            const float rs1 = row_scale[mt][1];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const float ts0 = token_scale[half][nt][0];
                const float ts1 = token_scale[half][nt][1];
                acc[half][mt][nt][0] = fmaf(
                    rs0 * ts0, __int_as_float(g[mt][nt][0]) - kFloatMagicValue, acc[half][mt][nt][0]);
                acc[half][mt][nt][1] = fmaf(
                    rs0 * ts1, __int_as_float(g[mt][nt][1]) - kFloatMagicValue, acc[half][mt][nt][1]);
                acc[half][mt][nt][2] = fmaf(
                    rs1 * ts0, __int_as_float(g[mt][nt][2]) - kFloatMagicValue, acc[half][mt][nt][2]);
                acc[half][mt][nt][3] = fmaf(
                    rs1 * ts1, __int_as_float(g[mt][nt][3]) - kFloatMagicValue, acc[half][mt][nt][3]);
            }
        }
    };

    load(0);
    decode(stages[0]);
    if (steps > 1) { load(1); }
    issue_x(0, stages[0]);
    cp_commit();
    int g[MT][4][4];
    for (int step = 0; step < steps; ++step) {
        cp_wait<0>();
        __syncthreads();
        if (step + 1 < steps) { issue_x(step + 1, stages[(step + 1) & 1]); }
        cp_commit();
        const Stage& s  = stages[step & 1];
        const auto next = [&] {
            if (step + 1 < steps) {
                decode(stages[(step + 1) & 1]);
                if (step + 2 < steps) { load(step + 2); }
            }
        };
        if constexpr (kHalves == 1) {
            if (pong) {
                if (step > 0) { update(0, g); }
                next();
                multiply(s, 0, g);
                load_scales(s);
            } else {
                multiply(s, 0, g);
                next();
                load_scales(s);
                update(0, g);
            }
        } else {
            if (pong) {
                if (step > 0) { update(1, g); }
                next();
                load_scales(s);
                multiply(s, 0, g);
                update(0, g);
                multiply(s, 1, g);
            } else {
                load_scales(s);
                multiply(s, 0, g);
                next();
                update(0, g);
                multiply(s, 1, g);
                update(1, g);
            }
        }
    }
    if (pong) { update(kHalves - 1, g); }

    constexpr int kChunks = kRows / 8;
    __nv_bfloat16* staged = reinterpret_cast<__nv_bfloat16*>(q4_tall_a8_smem);
#pragma unroll
    for (int h = 0; h < kHalves; ++h) {
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int row = wm * Cfg::kWarpRows + mt * 16 + gid;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int tok = wn * 32 + nt * 8 + 2 * lid;
                staged[tok * kOutLd + row]           = __float2bfloat16_rn(acc[h][mt][nt][0]);
                staged[(tok + 1) * kOutLd + row]     = __float2bfloat16_rn(acc[h][mt][nt][1]);
                staged[tok * kOutLd + row + 8]       = __float2bfloat16_rn(acc[h][mt][nt][2]);
                staged[(tok + 1) * kOutLd + row + 8] = __float2bfloat16_rn(acc[h][mt][nt][3]);
            }
        }
        __syncthreads();
        const int half_token0 = h * kHalfTokens;
#pragma unroll
        for (int i = 0; i < kHalfTokens * kChunks / kThreads; ++i) {
            const int item  = tid + i * kThreads;
            const int token = item / kChunks;
            const int chunk = item % kChunks;
            if (half_token0 + token < live) {
                *reinterpret_cast<uint4*>(
                    out + static_cast<std::int64_t>(token0 + half_token0 + token) * rows +
                    row_block * kRows + chunk * 8) =
                    *reinterpret_cast<const uint4*>(&staged[token * kOutLd + chunk * 8]);
            }
        }
    }
}

template <int Tokens>
void launch(const std::int8_t* qx, const float* x_scale, const std::uint8_t* codes,
            const std::uint16_t* scales, __nv_bfloat16* out, std::int32_t rows, std::int32_t k,
            std::int32_t tokens, cudaStream_t stream) {
    constexpr std::size_t kSmem = Config<Tokens>::shared_bytes();
    static const bool opted_in  = [] {
        CUDA_CHECK(cudaFuncSetAttribute(q4_rowsplit_tall_a8_kernel<Tokens>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(kSmem)));
        return true;
    }();
    (void)opted_in;
    const std::int32_t row_blocks  = rows / kRows;
    const std::int32_t token_tiles = (tokens + Tokens - 1) / Tokens;
    const unsigned grid = static_cast<unsigned>(static_cast<std::int64_t>(row_blocks) * token_tiles);
    q4_rowsplit_tall_a8_kernel<Tokens><<<grid, kThreads, kSmem, stream>>>(
        qx, x_scale, codes, scales, out, rows, k, tokens, token_tiles);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::q4_tall_a8
