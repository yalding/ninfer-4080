#pragma once
#include "ninfer/ops/inkling2_attention.h"

namespace ninfer::ops::inkling2_detail {
void rel_launch(const Tensor&, const Tensor&, const Tensor&, Tensor&, cudaStream_t);
void swa_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                const Tensor&, const CyclicKVCacheLayerView&, int, Tensor&, cudaStream_t);
void full_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                 const PagedKVLayerView&, unsigned, float*, Tensor&, cudaStream_t);

inline unsigned splits(unsigned keys, int tokens) {
    const unsigned n = (keys + 511u) / 512u;
    return tokens <= 16 ? (n < 256u ? n : 256u) : 1u;
}
} // namespace ninfer::ops::inkling2_detail
