#include "targets/inkling2_mini/impl/frontend/output_session.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::inkling2_mini {
namespace {
enum class Phase { Header, Content, Reasoning, Tool };

struct Decoder {
    Phase phase           = Phase::Header;
    OutputChannel channel = OutputChannel::Content;
    std::string pending, tool, header;
    std::optional<std::string> matched_stop;
    std::uint32_t reasoning_tokens = 0;
    std::vector<GeneratedToolCall> calls;
    ToolCallParseDiagnostics diagnostic;
    bool finished = false;
};

std::size_t complete_utf8_prefix(std::string_view text) {
    std::size_t p = 0;
    while (p < text.size()) {
        const auto lead     = static_cast<unsigned char>(text[p]);
        const std::size_t n = lead < 0x80                    ? 1
                              : lead >= 0xc2 && lead <= 0xdf ? 2
                              : lead >= 0xe0 && lead <= 0xef ? 3
                              : lead >= 0xf0 && lead <= 0xf4 ? 4
                                                             : 1;
        if (p + n > text.size()) { return p; }
        p += n;
    }
    return p;
}

void append(PublishedOutput& output, OutputChannel channel, std::string text) {
    if (text.empty()) { return; }
    // Vocabulary byte pieces need not form valid UTF-8 individually. Complete sequences are
    // held in pending; malformed model bytes become replacement characters at publication.
    std::string valid;
    for (std::size_t p = 0; p < text.size();) {
        const auto first    = static_cast<unsigned char>(text[p]);
        const std::size_t n = first < 0x80                     ? 1
                              : first >= 0xc2 && first <= 0xdf ? 2
                              : first >= 0xe0 && first <= 0xef ? 3
                              : first >= 0xf0 && first <= 0xf4 ? 4
                                                               : 0;
        bool good           = n != 0 && p + n <= text.size();
        for (std::size_t j = 1; good && j < n; ++j) {
            const auto next = static_cast<unsigned char>(text[p + j]);
            good            = next >= 0x80 && next <= 0xbf;
            if (j == 1 && ((first == 0xe0 && next < 0xa0) || (first == 0xed && next >= 0xa0) ||
                           (first == 0xf0 && next < 0x90) || (first == 0xf4 && next >= 0x90))) {
                good = false;
            }
        }
        if (good) {
            valid.append(text, p, n);
            p += n;
        } else {
            valid += "\xef\xbf\xbd";
            ++p;
        }
    }
    if (!output.empty() && output.back().channel == channel) {
        output.back().text += valid;
    } else {
        output.push_back({channel, std::move(valid)});
    }
}
} // namespace

class OutputSession::Impl {
public:
    std::shared_ptr<const frontend_internal::Tokenizer> tokenizer;
    StopPolicy policy;
    OutputOptions options;
    Decoder state, preview;
    PublishedOutput deltas;
    bool ready = false;

    bool flush(bool final) {
        std::size_t match         = std::string::npos;
        const StopString* matched = nullptr;
        for (const auto& stop : policy.strings) {
            if (stop.channel != preview.channel || stop.text.empty()) { continue; }
            const auto p = preview.pending.find(stop.text);
            if (p < match) {
                match   = p;
                matched = &stop;
            }
        }
        if (matched) {
            append(deltas, preview.channel,
                   preview.pending.substr(
                       0, match + (matched->include_in_output ? matched->text.size() : 0)));
            preview.matched_stop = matched->text;
            preview.pending.clear();
            preview.finished = true;
            return true;
        }
        std::size_t keep = 0;
        if (!final) {
            for (const auto& stop : policy.strings) {
                if (stop.channel != preview.channel) { continue; }
                for (std::size_t n = 1; n < stop.text.size() && n <= preview.pending.size(); ++n) {
                    if (preview.pending.compare(preview.pending.size() - n, n, stop.text, 0, n) ==
                        0) {
                        keep = std::max(keep, n);
                    }
                }
            }
        }
        auto count = preview.pending.size() - keep;
        if (!final) {
            count = complete_utf8_prefix(std::string_view(preview.pending).substr(0, count));
        }
        append(deltas, preview.channel, preview.pending.substr(0, count));
        preview.pending.erase(0, count);
        return false;
    }

    void close_tool() {
        if (preview.phase != Phase::Tool) { return; }
        try {
            const auto json = nlohmann::json::parse(preview.tool);
            if (!json.is_object() || !json.contains("name") || !json.at("name").is_string() ||
                !json.contains("args") || !json.at("args").is_object()) {
                throw std::invalid_argument("invalid tool call payload");
            }
            auto name = json.at("name").get<std::string>();
            if (name.empty() || name.size() > options.tool_name_max_length ||
                (!preview.header.empty() && preview.header != name)) {
                preview.diagnostic.fallback_reason = ToolCallParseFallbackReason::InvalidToolName;
                throw std::invalid_argument("tool call header mismatch");
            }
            preview.calls.push_back({std::move(name), json.at("args").dump()});
            ++preview.diagnostic.structured_call_count;
        } catch (const std::exception&) {
            if (preview.diagnostic.fallback_reason == ToolCallParseFallbackReason::None) {
                preview.diagnostic.fallback_reason =
                    ToolCallParseFallbackReason::MalformedStructure;
            }
            preview.channel = OutputChannel::Content;
            preview.pending += preview.tool;
        }
        preview.tool.clear();
    }
};

OutputSession make_output_session(std::shared_ptr<const frontend_internal::Tokenizer> tokenizer,
                                  const StopPolicy& policy, const OutputOptions& options,
                                  bool continuation, bool reasoning) {
    auto impl       = std::make_unique<OutputSession::Impl>();
    impl->tokenizer = std::move(tokenizer);
    impl->policy    = policy;
    impl->options   = options;
    if (continuation) {
        impl->state.phase   = reasoning ? Phase::Reasoning : Phase::Content;
        impl->state.channel = reasoning ? OutputChannel::Reasoning : OutputChannel::Content;
    }
    return OutputSession(std::move(impl));
}

OutputSession::OutputSession() noexcept = default;

OutputSession::OutputSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

runtime::OutputDecision OutputSession::preview_model(std::span<const TokenId> tokens,
                                                     std::uint32_t remaining, FinishReason limit) {
    if (!impl_ || impl_->ready) {
        throw std::logic_error("output preview already pending or empty session");
    }
    impl_->preview = impl_->state;
    impl_->deltas.clear();
    runtime::OutputDecision decision;
    if (impl_->preview.finished) { throw std::logic_error("output session already finished"); }
    for (auto id : tokens.first(std::min<std::size_t>(tokens.size(), remaining))) {
        const auto token = impl_->tokenizer->decoded_token(id);
        ++decision.accepted_tokens;
        const bool stop = std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(),
                                    id) != impl_->policy.token_ids.end();
        if (stop) {
            impl_->close_tool();
            if (impl_->policy.publish_stop_token) { impl_->preview.pending += token.bytes; }
            impl_->flush(true);
            impl_->preview.finished = true;
            decision.finish_reason  = FinishReason::StopToken;
            break;
        }
        if (impl_->options.raw) {
            impl_->preview.pending += token.bytes;
        } else if (id == 200001) {
            impl_->close_tool();
            impl_->flush(true);
            impl_->preview.phase = Phase::Header;
            impl_->preview.header.clear();
        } else if (id == 200008 || id == 200004 || id == 200022 || id == 200057) {
            impl_->close_tool();
            impl_->flush(true);
            impl_->preview.phase = id == 200008 ? Phase::Reasoning : Phase::Content;
            impl_->preview.channel =
                id == 200008 ? OutputChannel::Reasoning : OutputChannel::Content;
        } else if (id == 200049) {
            impl_->flush(true);
            impl_->preview.phase                  = Phase::Tool;
            impl_->preview.diagnostic.marker_seen = true;
        } else if (id == 200010) {
            impl_->close_tool();
            impl_->flush(true);
            impl_->preview.phase = Phase::Header;
        } else if (impl_->preview.phase == Phase::Tool) {
            impl_->preview.tool += token.bytes;
        } else if (impl_->preview.phase == Phase::Header) {
            impl_->preview.header += token.bytes;
        } else if (!token.special || impl_->options.preserve_special_tokens) {
            impl_->preview.pending += token.bytes;
            if (impl_->preview.phase == Phase::Reasoning) { ++impl_->preview.reasoning_tokens; }
        }
        if (impl_->flush(false)) {
            decision.finish_reason = FinishReason::StopString;
            break;
        }
    }
    if (decision.finish_reason == FinishReason::None && decision.accepted_tokens == remaining) {
        decision.finish_reason = limit;
        impl_->close_tool();
        impl_->flush(true);
        impl_->preview.finished = decision.finished();
    }
    impl_->ready = true;
    return decision;
}

std::uint32_t OutputSession::model_token_budget_remaining(std::uint32_t remaining) const noexcept {
    return remaining;
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept { return {}; }

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId>, std::uint32_t) {
    throw std::logic_error("Inkling has no injected control sequence");
}

void OutputSession::validate_generation_capacity(std::uint32_t) const {}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (!impl_ || impl_->ready) {
        throw std::logic_error("output preview already pending or empty session");
    }
    impl_->preview = impl_->state;
    impl_->deltas.clear();
    impl_->close_tool();
    impl_->flush(true);
    impl_->preview.finished = true;
    impl_->ready            = true;
    return {0, reason};
}

PublishedOutput OutputSession::commit_preview() {
    if (!impl_ || !impl_->ready) { throw std::logic_error("no output preview to commit"); }
    impl_->state = std::move(impl_->preview);
    impl_->ready = false;
    return std::move(impl_->deltas);
}

std::vector<GeneratedToolCall> OutputSession::take_tool_calls() noexcept {
    if (!impl_) { return {}; }
    return std::exchange(impl_->state.calls, {});
}

ToolCallParseDiagnostics OutputSession::tool_call_parse_diagnostics() const noexcept {
    return impl_ ? impl_->state.diagnostic : ToolCallParseDiagnostics{};
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ ? impl_->state.reasoning_tokens : 0;
}

ThinkingBudgetStats OutputSession::thinking_stats() const noexcept { return {}; }

std::optional<std::string> OutputSession::matched_stop_string() const {
    return impl_ ? impl_->state.matched_stop : std::nullopt;
}
} // namespace ninfer::targets::inkling2_mini
