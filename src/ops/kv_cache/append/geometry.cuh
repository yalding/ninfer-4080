#pragma once

namespace ninfer::ops {

inline constexpr int kKVCacheAppendFullHeadDim = 256;

template <int HeadDimValue, int KVHeadsValue>
struct KVCacheAppendFullGeometry {
    static_assert((HeadDimValue == 256 && (KVHeadsValue == 4 || KVHeadsValue == 2)) ||
                  (HeadDimValue == 128 && KVHeadsValue == 8));
    static constexpr int HeadDim    = HeadDimValue;
    static constexpr int KVHeads    = KVHeadsValue;
    static constexpr int Int8Groups = HeadDim / 64;
};

using KVCacheAppendD256Kv4 = KVCacheAppendFullGeometry<256, 4>;
using KVCacheAppendD256Kv2 = KVCacheAppendFullGeometry<256, 2>;
using KVCacheAppendD128Kv8 = KVCacheAppendFullGeometry<128, 8>;

} // namespace ninfer::ops
