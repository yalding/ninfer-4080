#pragma once
#include <ninfer/targets/inkling2_mini/frontend.h>
#include <array>
#include <optional>
#include <string>

namespace ninfer::targets::inkling2_mini {
enum class RewriteCheckpointKind : std::uint8_t { TurnClosure, ResponseReplay };

struct RewriteCheckpointSpec {
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
};

struct PromptIdentity {
    bool reusable = true;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<std::uint32_t> rewrite_execution_frontiers;
};

struct PreparedSessionKey {
    std::uint16_t size = 0;
    std::array<char, kMaximumContextCacheSessionKeyBytes> bytes{};

    std::string_view view() const noexcept { return {bytes.data(), size}; }

    friend bool operator==(const PreparedSessionKey&, const PreparedSessionKey&) noexcept = default;
};

struct PreparedCacheOpportunity {
    PromptCacheMarkerKind kind       = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence = SharedCandidateEvidence::None;
    std::uint32_t frontier           = 0;
    std::uint32_t input_order        = 0;
};

struct PreparedContextCache {
    std::optional<PreparedSessionKey> session_key;
    runtime::RetentionClass retention = runtime::RetentionClass::RecentPrivate;
    std::vector<PreparedCacheOpportunity> opportunities;
    bool update_session_index = true;
};

struct PreparedMediaPayload {
    std::vector<std::uint16_t> patches;
    std::uint32_t native_patches = 0;
    std::uint32_t merged_tokens  = 0;
    std::int32_t padded_h20      = 0;
    std::int32_t padded_w20      = 0;
    std::vector<std::int32_t> coords;
    std::vector<std::int32_t> pos_indices;
    std::vector<float> pos_weights;
    std::vector<std::int32_t> merge_indices;

    [[nodiscard]] std::size_t payload_bytes() const noexcept {
        return patches.size() * sizeof(std::uint16_t) + coords.size() * sizeof(std::int32_t) +
               pos_indices.size() * sizeof(std::int32_t) + pos_weights.size() * sizeof(float) +
               merge_indices.size() * sizeof(std::int32_t);
    }
};

struct PreparedVisionItem {
    std::uint32_t token_begin = 0;
    std::uint32_t token_end   = 0;
    std::int32_t grid_h       = 0;
    std::int32_t grid_w       = 0;
    std::array<std::uint8_t, 32> content_digest{};

    friend bool operator==(const PreparedVisionItem&, const PreparedVisionItem&) noexcept = default;
};

struct PreparedPromptData {
    std::vector<TokenId> token_ids;
    std::vector<std::int32_t> positions;
    PromptIdentity identity;
    PreparedContextCache context_cache;
    PromptPreparationStats prepare;
    std::vector<PreparedVisionItem> vision_items;
    std::vector<std::shared_ptr<const PreparedMediaPayload>> media_payloads;
    bool continuation        = false;
    bool starts_in_reasoning = false;

    bool has_media() const noexcept { return !vision_items.empty(); }

    void release_all_media_payloads() noexcept { media_payloads.clear(); }
};

class PreparedPromptAccess {
public:
    static const PreparedPromptData& view(const PreparedPrompt& prompt);
    static PreparedPromptData take(PreparedPrompt&& prompt);
};
} // namespace ninfer::targets::inkling2_mini
