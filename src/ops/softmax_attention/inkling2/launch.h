#pragma once
#include "ninfer/ops/inkling2_attention.h"

namespace ninfer::ops::inkling2_detail {
void rel_launch(const Tensor&, const Tensor&, const Tensor&, Tensor&, cudaStream_t);
void swa_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                const Tensor&, const CyclicKVCacheLayerView&, int, unsigned, float*, bool, Tensor&,
                cudaStream_t);
void full_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                 const PagedKVLayerView&, unsigned, float*, Tensor&, cudaStream_t);
void swa_batched_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                        const Tensor&, const Tensor&, const CyclicKVCacheLayerView&, unsigned,
                        float*, bool, Tensor&, cudaStream_t);
void kv_append_batched_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                              const PagedKVBatchLayerView&, cudaStream_t);
void full_batched_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                         const PagedKVBatchLayerView&, unsigned, float*, Tensor&, cudaStream_t);
void dspark_swa_launch(const Tensor&, const Tensor&, const Tensor&, unsigned, const int*,
                       const CyclicKVCacheLayerView&, int, Tensor&, cudaStream_t);
void dspark_full_launch(const Tensor&, const Tensor&, const Tensor&, const PagedKVLayerView&,
                        unsigned, unsigned, const int*, float*, Tensor&, cudaStream_t);

inline unsigned splits(unsigned keys, int tokens) {
    if (tokens > 16) return 0u;
    if (keys <= 128u) return 1u;
    if (keys <= 256u) return 2u;
    const unsigned cap      = 128u;
    const unsigned min_part = keys >= 16384u ? 512u : 256u;
    unsigned n              = (keys + min_part - 1u) / min_part;
    if (n > cap) n = cap;
    if (n >= 4u) n = ((n + 3u) / 4u) * 4u;
    if (n > cap) n = (cap / 4u) * 4u;
    return n > 0u ? n : 1u;
}

inline unsigned swa_splits(unsigned keys, int tokens) {
    if (tokens >= 16) return 0u;
    if (keys <= 128u) return 1u;
    if (keys <= 256u) return 2u;
    return 4u;
}
} // namespace ninfer::ops::inkling2_detail
