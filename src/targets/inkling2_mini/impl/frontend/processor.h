#pragma once

#include <ninfer/targets/inkling2_mini/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

namespace ninfer::targets::inkling2_mini::frontend_internal {

struct PreprocessedImage {
    std::int32_t grid_h = 0;
    std::int32_t grid_w = 0;
    std::array<std::uint8_t, 32> content_digest{};
    std::shared_ptr<const PreparedMediaPayload> payload;
};

struct MediaRequestMetrics {
    std::size_t hits               = 0;
    std::size_t misses             = 0;
    std::size_t singleflight_waits = 0;
    std::size_t built_patch_bytes  = 0;
    std::size_t reused_patch_bytes = 0;
    double build_seconds           = 0.0;
};

class MediaPreprocessCache {
public:
    MediaPreprocessCache(std::size_t capacity_bytes, std::size_t live_capacity_bytes,
                         std::uint32_t preprocess_threads, std::uint32_t max_patches,
                         std::uint32_t max_merged_tokens);
    ~MediaPreprocessCache();

    MediaPreprocessCache(const MediaPreprocessCache&)            = delete;
    MediaPreprocessCache& operator=(const MediaPreprocessCache&) = delete;

    [[nodiscard]] PreprocessedImage prepare_image(const OwnedMedia& media,
                                                  const PreparationControl& control,
                                                  MediaRequestMetrics& metrics);
    [[nodiscard]] MediaCacheSummary summary() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::array<std::uint8_t, 32>
sha256_bytes(std::span<const std::uint8_t> input, const std::function<void()>& checkpoint = {});

} // namespace ninfer::targets::inkling2_mini::frontend_internal
