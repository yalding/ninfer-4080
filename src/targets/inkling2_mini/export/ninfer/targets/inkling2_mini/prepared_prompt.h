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

struct PreparedPromptData {
    std::vector<TokenId> token_ids;
    std::vector<std::int32_t> positions;
    PromptIdentity identity;
    PreparedContextCache context_cache;
    PromptPreparationStats prepare;
    bool continuation        = false;
    bool starts_in_reasoning = false;

    bool has_media() const noexcept { return false; }

    void release_all_media_payloads() noexcept {}
};

class PreparedPromptAccess {
public:
    static const PreparedPromptData& view(const PreparedPrompt& prompt);
    static PreparedPromptData take(PreparedPrompt&& prompt);
};
} // namespace ninfer::targets::inkling2_mini
