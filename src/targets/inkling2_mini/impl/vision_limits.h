#pragma once

#include <algorithm>
#include <cstdint>

namespace ninfer::targets::inkling2_mini::detail {

inline std::uint32_t vision_native_capacity(std::uint32_t max_tokens) {
    return std::max(4U, max_tokens);
}

inline std::uint32_t vision_merged_capacity(std::uint32_t max_tokens,
                                            std::uint32_t max_merged_tokens) {
    const auto native = vision_native_capacity(max_tokens);
    return std::max(1U, std::min(max_merged_tokens, 1U + (native - 1U) / 4U));
}

} // namespace ninfer::targets::inkling2_mini::detail
