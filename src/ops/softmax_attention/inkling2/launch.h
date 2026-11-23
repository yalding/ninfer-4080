#pragma once
#include "ninfer/ops/inkling2_attention.h"

namespace ninfer::ops::inkling2_detail {
void rel_launch(const Tensor&, const Tensor&, const Tensor&, Tensor&, cudaStream_t);
void swa_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                const Tensor&, const CyclicKVCacheLayerView&, int, Tensor&, cudaStream_t);
void full_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                 const PagedKVLayerView&, unsigned, float*, Tensor&, cudaStream_t);

inline unsigned splits(unsigned keys, int tokens) {
    if (tokens >= 16) return 0u;
    if (keys <= 256u) return 1u;
    const unsigned n = (keys + 255u) / 256u;
    return n < 128u ? n : 128u;
}
} // namespace ninfer::ops::inkling2_detail
