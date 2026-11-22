#include <ninfer/targets/inkling2_mini/frontend.h>
#include <ninfer/targets/inkling2_mini/frontend_resources.h>
#include <ninfer/targets/inkling2_mini/prepared_prompt.h>
#include "targets/inkling2_mini/impl/frontend/tokenizer.h"
#include "targets/inkling2_mini/impl/frontend/chat_template.h"
#include "targets/inkling2_mini/impl/frontend/output_session.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace ninfer::targets::inkling2_mini {
namespace fi = frontend_internal;

namespace {
void check_control(const PreparationControl& control) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled, "prompt preparation cancelled");
    }
    if (control.deadline != std::chrono::steady_clock::time_point{} &&
        std::chrono::steady_clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::Cancelled, "prompt preparation deadline exceeded");
    }
}
} // namespace

class Frontend::Impl {
public:
    std::shared_ptr<const fi::Tokenizer> tokenizer;
    std::uint32_t max_context = 131072;
    StopPolicy stop;
};

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

Frontend::~Frontend()                              = default;
Frontend::Frontend(const Frontend&)                = default;
Frontend& Frontend::operator=(const Frontend&)     = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;

Frontend make_frontend(const FrontendResources& resources, FrontendOptions options) {
    if (options.max_context == 0) {
        throw std::invalid_argument("maximum context must be positive");
    }
    if (resources.chat_template_jinja.find("Thinking effort level:") == std::string::npos ||
        resources.chat_template_jinja.find("<|content_model_end_sampling|>") == std::string::npos) {
        throw std::invalid_argument("unsupported Inkling chat template resource");
    }
    auto impl       = std::make_shared<Frontend::Impl>();
    impl->tokenizer = std::make_shared<fi::Tokenizer>(
        fi::TokenizerResources{resources.tokenizer_json, resources.tokenizer_config_json,
                               resources.generation_config_json});
    if (!impl->tokenizer->has_exact_token_domain(kTokenDomain) ||
        impl->tokenizer->encode("<|content_model_end_sampling|>") != std::vector<int>{200006}) {
        throw std::invalid_argument("unsupported Inkling tokenizer token domain");
    }
    impl->max_context    = options.max_context;
    impl->stop.token_ids = {200006};
    return Frontend(std::move(impl));
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    check_control(control);
    const auto started       = std::chrono::steady_clock::now();
    const auto rendered      = fi::render_chat(input);
    const auto token_started = std::chrono::steady_clock::now();
    // Markers bind to byte frontiers in this exact serialization, never to separately tokenized
    // text.
    std::vector<std::size_t> bytes = rendered.message_boundaries;
    if (input.context_cache.markers.size() > kMaximumExplicitPromptCacheMarkers) {
        throw std::invalid_argument("at most four explicit prompt cache markers are supported");
    }
    std::vector<std::optional<std::size_t>> marker_indices;
    for (const auto& marker : input.context_cache.markers) {
        if ((marker.kind != PromptCacheMarkerKind::SharedStablePrefix &&
             marker.kind != PromptCacheMarkerKind::PrivateLongAnchor) ||
            (marker.kind == PromptCacheMarkerKind::SharedStablePrefix &&
             marker.evidence == SharedCandidateEvidence::None)) {
            throw std::invalid_argument("invalid cache marker kind or evidence");
        }
        std::optional<std::size_t> offset;
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count >= rendered.message_boundaries.size() ||
                marker.after_message_part_count || marker.leading_instruction_bytes ||
                marker.after_tool_count) {
                throw std::invalid_argument("invalid cache message boundary");
            }
            offset = rendered.message_boundaries[marker.after_message_count];
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            if (!marker.after_message_count || marker.after_message_count > input.messages.size() ||
                !marker.after_message_part_count || marker.leading_instruction_bytes ||
                marker.after_tool_count) {
                throw std::invalid_argument("invalid cache message-part boundary");
            }
            if (const auto& spans = rendered.message_part_spans[marker.after_message_count - 1];
                marker.after_message_part_count <= spans.size()) {
                offset = spans[marker.after_message_part_count - 1].end;
            }
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (marker.after_message_count || marker.after_message_part_count ||
                !marker.leading_instruction_bytes || marker.after_tool_count) {
                throw std::invalid_argument("invalid cache leading-instruction boundary");
            }
            if (!input.messages.empty() && (input.messages[0].role == ChatRole::System ||
                                            input.messages[0].role == ChatRole::Developer)) {
                std::size_t remaining = marker.leading_instruction_bytes;
                for (const auto span : rendered.message_part_spans[0]) {
                    if (remaining <= span.end - span.begin) {
                        offset = span.begin + remaining;
                        break;
                    }
                    remaining -= span.end - span.begin;
                }
            }
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_message_count || marker.after_message_part_count ||
                marker.leading_instruction_bytes || !marker.after_tool_count) {
                throw std::invalid_argument("invalid cache tool boundary");
            }
            if (marker.after_tool_count <= rendered.tool_boundaries.size()) {
                offset = rendered.tool_boundaries[marker.after_tool_count - 1];
            }
            break;
        default:
            throw std::invalid_argument("invalid cache marker location");
        }
        if (offset) {
            marker_indices.push_back(bytes.size());
            bytes.push_back(*offset);
        } else {
            marker_indices.push_back(std::nullopt);
        }
    }
    std::optional<std::size_t> tool_index;
    if (!rendered.tool_boundaries.empty()) {
        tool_index = bytes.size();
        bytes.push_back(rendered.tool_boundaries.back());
    }
    const std::size_t rewrite_index = bytes.size();
    bytes.push_back(input.options.continuation == PromptContinuationMode::ContinueFinalAssistant
                        ? rendered.message_begins.back()
                        : rendered.text.size());
    const auto encoded = impl_->tokenizer->encode_with_boundaries(
        rendered.text, bytes, {.max_tokens = static_cast<std::size_t>(impl_->max_context) + 1},
        rendered.literal_spans);
    auto result = prepare_tokens(encoded.input_ids, true);
    auto& data  = *result.data_;
    data.continuation =
        input.options.continuation == PromptContinuationMode::ContinueFinalAssistant;
    if (data.continuation) {
        const auto& last = input.messages.back();
        data.starts_in_reasoning =
            !last.reasoning_content.empty() && last.parts.empty() && last.tool_calls.empty();
    }
    const auto frontier = [&](std::size_t message_count) -> std::optional<std::uint32_t> {
        const auto value = encoded.boundaries.at(message_count).exact_frontier;
        return value ? std::optional<std::uint32_t>(static_cast<std::uint32_t>(*value))
                     : std::nullopt;
    };
    // Inkling emits its own assistant opener. The complete request prefix is the replay boundary;
    // continuation instead retains the history before the final assistant serialization.
    if (const auto count = frontier(rewrite_index)) {
        data.identity.rewrite_checkpoint =
            RewriteCheckpointSpec{RewriteCheckpointKind::ResponseReplay, *count};
    }
    for (std::size_t i = 0; i < rendered.message_boundaries.size(); ++i) {
        if (const auto count = frontier(i); count && *count) {
            data.identity.rewrite_execution_frontiers.push_back(*count);
        }
    }
    if (data.identity.rewrite_checkpoint) {
        data.identity.rewrite_execution_frontiers.push_back(
            data.identity.rewrite_checkpoint->frontier);
        auto& frontiers = data.identity.rewrite_execution_frontiers;
        std::sort(frontiers.begin(), frontiers.end());
        frontiers.erase(std::unique(frontiers.begin(), frontiers.end()), frontiers.end());
    }
    if (input.context_cache.session_key) {
        const auto& key = *input.context_cache.session_key;
        if (key.empty() || key.size() > kMaximumContextCacheSessionKeyBytes) {
            throw std::invalid_argument("invalid context cache session key");
        }
        PreparedSessionKey value;
        value.size = static_cast<std::uint16_t>(key.size());
        std::memcpy(value.bytes.data(), key.data(), key.size());
        data.context_cache.session_key = value;
    }
    data.context_cache.update_session_index = input.context_cache.update_session_index;
    switch (input.context_cache.retention) {
    case CacheRetentionHint::Default:
        data.context_cache.retention = data.context_cache.session_key
                                           ? runtime::RetentionClass::LiveSession
                                           : runtime::RetentionClass::RecentPrivate;
        break;
    case CacheRetentionHint::LiveSession:
        if (!data.context_cache.session_key) {
            throw std::invalid_argument("LiveSession retention requires a session key");
        }
        data.context_cache.retention = runtime::RetentionClass::LiveSession;
        break;
    case CacheRetentionHint::Disposable:
        data.context_cache.retention = runtime::RetentionClass::Disposable;
        break;
    default:
        throw std::invalid_argument("invalid context cache retention hint");
    }
    const auto add_opportunity = [&](PromptCacheMarkerKind kind, SharedCandidateEvidence evidence,
                                     std::uint32_t count, std::uint32_t order) {
        if (!count) { return; }
        const auto duplicate =
            std::find_if(data.context_cache.opportunities.begin(),
                         data.context_cache.opportunities.end(), [&](const auto& existing) {
                             return existing.kind == kind && existing.frontier == count;
                         });
        if (duplicate == data.context_cache.opportunities.end()) {
            data.context_cache.opportunities.push_back({kind, evidence, count, order});
        } else if (kind == PromptCacheMarkerKind::SharedStablePrefix) {
            duplicate->evidence |= evidence;
        }
    };
    for (std::size_t i = 0; i < input.context_cache.markers.size(); ++i) {
        const auto& marker = input.context_cache.markers[i];
        if (marker_indices[i]) {
            if (const auto count = frontier(*marker_indices[i])) {
                add_opportunity(marker.kind, marker.evidence, *count,
                                static_cast<std::uint32_t>(i));
            }
        }
    }
    const std::size_t boundaries = input.messages.size();
    const auto private_count = std::min<std::size_t>(input.context_cache.automatic_private_anchors,
                                                     boundaries == 0 ? 0 : boundaries - 1);
    for (std::size_t remaining = private_count, i = boundaries; remaining; --remaining) {
        --i;
        if (const auto count = frontier(i); count && *count < data.token_ids.size()) {
            add_opportunity(PromptCacheMarkerKind::PrivateLongAnchor, SharedCandidateEvidence::None,
                            *count,
                            static_cast<std::uint32_t>(input.context_cache.markers.size() +
                                                       private_count - remaining));
        }
    }
    if (input.context_cache.allow_engine_automatic_shared_prefixes) {
        std::uint32_t order = static_cast<std::uint32_t>(input.context_cache.markers.size()) +
                              static_cast<std::uint32_t>(private_count);
        std::size_t leading = 0;
        while (leading < input.messages.size() &&
               (input.messages[leading].role == ChatRole::System ||
                input.messages[leading].role == ChatRole::Developer)) {
            ++leading;
        }
        if (leading) {
            if (const auto count = frontier(leading)) {
                add_opportunity(PromptCacheMarkerKind::SharedStablePrefix,
                                SharedCandidateEvidence::EngineStructural, *count, order++);
            }
        }
        if (tool_index) {
            if (const auto count = frontier(*tool_index)) {
                add_opportunity(PromptCacheMarkerKind::SharedStablePrefix,
                                SharedCandidateEvidence::EngineStructural, *count, order++);
            }
        }
        if (boundaries > 1) {
            if (const auto count = frontier(boundaries - 1)) {
                add_opportunity(PromptCacheMarkerKind::SharedStablePrefix,
                                SharedCandidateEvidence::EngineStructural, *count, order++);
            }
        }
        add_opportunity(PromptCacheMarkerKind::SharedStablePrefix,
                        SharedCandidateEvidence::EngineObserved,
                        static_cast<std::uint32_t>(data.token_ids.size()), order);
    }
    data.prepare.tokenize_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - token_started).count();
    data.prepare.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    check_control(control);
    return result;
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    return prepare(std::move(input), control).summary().prompt_tokens;
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> ids, bool reusable) const {
    if (ids.empty()) { throw std::invalid_argument("prompt token sequence must not be empty"); }
    if (ids.size() > impl_->max_context) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           "prompt exceeds maximum context");
    }
    for (auto id : ids) {
        if (!impl_->tokenizer->is_valid_token(id)) {
            throw std::invalid_argument("prompt token outside Inkling vocabulary");
        }
    }
    auto data       = std::make_unique<PreparedPromptData>();
    data->token_ids = std::move(ids);
    data->positions.resize(data->token_ids.size());
    std::iota(data->positions.begin(), data->positions.end(), 0);
    data->identity.reusable = reusable;
    return PreparedPrompt(std::move(data));
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    return impl_->tokenizer->encode(text, {.parse_added_tokens = false});
}

PromptCapabilities Frontend::prompt_capabilities() const noexcept {
    return fi::prompt_capabilities();
}

MediaCacheSummary Frontend::media_cache_summary() const { return {}; }

const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->stop; }

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller,
                                            const OutputOptions& options,
                                            const ThinkingControlOptions& thinking) const {
    if (thinking.budget) {
        throw std::invalid_argument("Inkling does not support injected thinking-budget control");
    }
    const auto& data = PreparedPromptAccess::view(prompt);
    auto stop        = caller;
    if (caller.include_model_defaults) {
        for (auto id : impl_->stop.token_ids) {
            if (std::find(stop.token_ids.begin(), stop.token_ids.end(), id) ==
                stop.token_ids.end()) {
                stop.token_ids.push_back(id);
            }
        }
    }
    return inkling2_mini::make_output_session(impl_->tokenizer, stop, options, data.continuation,
                                              data.starts_in_reasoning);
}

PreparedPrompt::PreparedPrompt() noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}

PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::operator bool() const noexcept { return bool(data_); }

PromptSummary PreparedPrompt::summary() const {
    if (!data_) { throw std::logic_error("empty prepared prompt"); }
    return {static_cast<std::uint32_t>(data_->token_ids.size()), false};
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept {
    return data_ ? data_->prepare : PromptPreparationStats{};
}

const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (!prompt.data_) { throw std::logic_error("empty prepared prompt"); }
    return *prompt.data_;
}

PreparedPromptData PreparedPromptAccess::take(PreparedPrompt&& prompt) {
    if (!prompt.data_) { throw std::logic_error("empty prepared prompt"); }
    auto data = std::move(prompt.data_);
    return std::move(*data);
}
} // namespace ninfer::targets::inkling2_mini
