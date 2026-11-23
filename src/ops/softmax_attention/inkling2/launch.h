#pragma once
#include "ninfer/ops/inkling2_attention.h"

namespace ninfer::ops::inkling2_detail {
void rel_launch(const Tensor&, const Tensor&, const Tensor&, Tensor&, cudaStream_t);
void swa_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                const Tensor&, const CyclicKVCacheLayerView&, int, Tensor&, cudaStream_t);
void full_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                 const PagedKVLayerView&, unsigned, float*, Tensor&, cudaStream_t);
void dspark_swa_launch(const Tensor&, const Tensor&, const Tensor&, unsigned,
                       const CyclicKVCacheLayerView&, int, Tensor&, cudaStream_t);
void dspark_full_launch(const Tensor&, const Tensor&, const Tensor&, const PagedKVLayerView&,
                        unsigned, float*, Tensor&, cudaStream_t);

inline unsigned splits(unsigned keys, int tokens) {
    if (tokens >= 16) return 0u;
    if (keys <= 256u) return 1u;
    const unsigned token_tiles = tokens == 1 ? 1u : static_cast<unsigned>((tokens + 1) / 2);
    const unsigned cap         = 128u / token_tiles;
    const unsigned min_part    = keys >= 16384u ? 512u : 256u;
    const unsigned n           = (keys + min_part - 1u) / min_part;
    const unsigned clamped     = n < cap ? n : cap;
    return clamped > 0u ? clamped : 1u;
}
} // namespace ninfer::ops::inkling2_detail
