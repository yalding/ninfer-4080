#include "targets/inkling2_mini/impl/frontend/chat_template.h"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>

namespace ninfer::targets::inkling2_mini::frontend_internal {
namespace {
using Json = nlohmann::json;

// Jinja's tojson uses sorted keys, compact separators and HTML-safe escaping.
std::string tojson(const Json& value) {
    const auto raw = value.dump(-1, ' ', true);
    std::string out;
    for (char c : raw) {
        switch (c) {
        case '<':
            out += "\\u003c";
            break;
        case '>':
            out += "\\u003e";
            break;
        case '&':
            out += "\\u0026";
            break;
        case '\'':
            out += "\\u0027";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

std::string_view role_token(ChatRole role) {
    switch (role) {
    case ChatRole::System:
    case ChatRole::Developer:
        return "<|message_system|>";
    case ChatRole::User:
        return "<|message_user|>";
    case ChatRole::Assistant:
        return "<|message_model|>";
    case ChatRole::Tool:
        return "<|message_tool|>";
    }
    throw std::invalid_argument("unsupported Inkling message role");
}

std::string_view effort(const PromptOptions& options) {
    if (!options.enable_thinking) { return "0"; }
    if (!options.reasoning_effort) { return "0.9"; }
    switch (*options.reasoning_effort) {
    case ReasoningEffort::Minimal:
        return "0.1";
    case ReasoningEffort::Low:
        return "0.2";
    case ReasoningEffort::Medium:
        return "0.7";
    case ReasoningEffort::High:
        return "0.9";
    case ReasoningEffort::XHigh:
        return "0.99";
    }
    throw std::invalid_argument("unsupported Inkling reasoning effort");
}
} // namespace

PromptCapabilities prompt_capabilities() noexcept {
    PromptCapabilities out;
    out.enable_thinking                 = true;
    out.reasoning_effort.minimal        = true;
    out.reasoning_effort.low            = true;
    out.reasoning_effort.medium         = true;
    out.reasoning_effort.high           = true;
    out.reasoning_effort.xhigh          = true;
    out.reasoning_effort.default_effort = ReasoningEffort::High;
    return out;
}

RenderedChat render_chat(const PromptInput& input) {
    if (input.options.continuation == PromptContinuationMode::ContinueFinalAssistant &&
        (input.messages.empty() || input.messages.back().role != ChatRole::Assistant)) {
        throw std::invalid_argument("continuation requires a final assistant message");
    }
    RenderedChat out;
    out.message_part_spans.resize(input.messages.size());
    const auto literal = [&](std::string_view text) {
        const auto begin = out.text.size();
        out.text += text;
        if (!text.empty()) { out.literal_spans.push_back({begin, out.text.size()}); }
    };
    if (!input.options.tool_jsons.empty()) {
        Json tools = Json::array();
        for (const auto& encoded : input.options.tool_jsons) {
            const Json tool = Json::parse(encoded);
            const auto& fn  = tool.contains("function") ? tool.at("function") : tool;
            if (!fn.is_object() || !fn.contains("name") || !fn.at("name").is_string()) {
                throw std::invalid_argument("tool declaration requires a function name");
            }
            tools.push_back({{"description", fn.value("description", std::string{})},
                             {"name", fn.at("name")},
                             {"parameters", fn.value("parameters", Json::object())},
                             {"type", tool.value("type", std::string("function"))}});
        }
        out.text += "<|message_system|>tool_declare<|content_xml|>[";
        for (std::size_t i = 0; i < tools.size(); ++i) {
            if (i) { out.text += ','; }
            literal(tojson(tools[i]));
            out.tool_boundaries.push_back(out.text.size());
        }
        out.text += ']';
        out.text += "<|end_message|>";
        // Only the complete declaration is an independently reusable serialization boundary.
        out.tool_boundaries.back() = out.text.size();
    }
    bool effort_emitted    = false;
    const auto emit_effort = [&] {
        out.text += "<|message_system|><|content_text|>Thinking effort level: ";
        out.text += effort(input.options);
        out.text += "<|end_message|>";
        effort_emitted = true;
    };
    out.message_boundaries.push_back(out.text.size());
    for (std::size_t index = 0; index < input.messages.size(); ++index) {
        const auto& message  = input.messages[index];
        const bool assistant = message.role == ChatRole::Assistant;
        const bool continued =
            assistant && index + 1 == input.messages.size() &&
            input.options.continuation == PromptContinuationMode::ContinueFinalAssistant;
        if (!effort_emitted && message.role != ChatRole::System &&
            message.role != ChatRole::Developer) {
            emit_effort();
        }
        out.message_begins.push_back(out.text.size());
        std::string header(role_token(message.role));
        if (message.role == ChatRole::Tool && !message.tool_call_id.empty()) {
            for (const auto& prior : input.messages) {
                if (prior.role != ChatRole::Assistant) { continue; }
                for (const auto& call : prior.tool_calls) {
                    if (call.id == message.tool_call_id) { header += call.name; }
                }
            }
        }
        const auto emit = [&](std::string_view marker, std::string_view text, bool open) {
            out.text += header;
            out.text += marker;
            const auto begin = out.text.size();
            literal(text);
            const auto end = out.text.size();
            if (!open) { out.text += "<|end_message|>"; }
            return ByteSpan{begin, end};
        };
        if (assistant && !message.reasoning_content.empty()) {
            emit("<|content_thinking|>", message.reasoning_content,
                 continued && message.parts.empty() && message.tool_calls.empty());
        }
        if (message.parts.empty() &&
            !(assistant && (!message.tool_calls.empty() || !message.reasoning_content.empty()))) {
            emit("<|content_text|>", {}, continued);
        }
        for (std::size_t part_index = 0; part_index < message.parts.size(); ++part_index) {
            const auto& part = message.parts[part_index];
            if (part.kind == MessagePartKind::Text) {
                out.message_part_spans[index].push_back(emit(
                    "<|content_text|>", part.text,
                    continued && part_index + 1 == message.parts.size() &&
                        message.tool_calls.empty()));
            } else if (part.kind == MessagePartKind::Media) {
                if (part.media.kind != MediaKind::Image) {
                    throw std::invalid_argument("Inkling-2-Mini supports image media only");
                }
                if (message.role != ChatRole::User && message.role != ChatRole::Tool) {
                    throw std::invalid_argument(
                        "Inkling-2-Mini image content is only supported for user and tool messages");
                }
                if (continued && part_index + 1 == message.parts.size() &&
                    message.tool_calls.empty()) {
                    throw std::invalid_argument(
                        "Inkling-2-Mini continuation cannot end on an open image part");
                }
                out.text += header;
                out.text += "<|content_image|>";
                const auto ph_begin = out.text.size();
                out.text += "<|unused_200054|>";
                const auto ph_end = out.text.size();
                out.text += "<|end_message|>";
                out.image_placeholders.push_back({index, part_index, ph_begin, ph_end});
                out.message_part_spans[index].push_back(ByteSpan{ph_begin, out.text.size()});
            } else {
                throw std::invalid_argument("unsupported Inkling message part kind");
            }
        }
        for (const auto& call : message.tool_calls) {
            if (!assistant) { throw std::invalid_argument("tool calls require assistant role"); }
            const Json args =
                call.arguments_json.empty() ? Json::object() : Json::parse(call.arguments_json);
            if (!args.is_object()) {
                throw std::invalid_argument("tool call arguments must be an object");
            }
            out.text += "<|message_model|>";
            literal(call.name);
            out.text += "<|content_invoke_tool_json|>";
            literal("{\"name\":" + tojson(call.name) + ",\"args\":" + tojson(args) + "}");
            out.text += "<|end_message|>";
        }
        if (assistant && !continued) { out.text += "<|content_model_end_sampling|>"; }
        out.message_boundaries.push_back(out.text.size());
    }
    if (!effort_emitted) { emit_effort(); }
    return out;
}
} // namespace ninfer::targets::inkling2_mini::frontend_internal
