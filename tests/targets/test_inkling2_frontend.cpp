#include <ninfer/targets/inkling2_mini/frontend.h>
#include <ninfer/targets/inkling2_mini/frontend_resources.h>
#include <ninfer/targets/inkling2_mini/prepared_prompt.h>
#include "targets/inkling2_mini/impl/frontend/chat_template.h"
#include "targets/inkling2_mini/impl/frontend/tokenizer.h"
#include "targets/inkling2_mini/impl/load/bindings.h"
#include "artifact/reader.h"
#include "text/unicode.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ik = ninfer::targets::inkling2_mini;
namespace fi = ik::frontend_internal;
using namespace ninfer;

namespace {
void require(bool value, const char* description) {
    if (!value) { throw std::runtime_error(description); }
}

std::string read(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot open " + path.string()); }
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

ChatMessage message(ChatRole role, std::string text) {
    ChatMessage value;
    value.role = role;
    value.parts.push_back({MessagePartKind::Text, std::move(text), {}});
    return value;
}

void tokenizer_semantic_tests() {
    nlohmann::json vocab = nlohmann::json::object();
    std::array<std::string, 256> encoder;
    int extra = 256;
    for (int byte = 0; byte < 256; ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        encoder[byte] = ninfer::text::unicode_internal::codepoint_to_utf8(visible ? byte : extra++);
        vocab[encoder[byte]] = byte;
    }
    const auto mapped = [&](std::string_view text) {
        std::string result;
        for (unsigned char byte : text) { result += encoder[byte]; }
        return result;
    };
    vocab["ab"]                = 256;
    vocab[mapped("\xc3\xa1")]  = 257;
    vocab[mapped("a\xcc\x81")] = 258;
    nlohmann::json root        = {{"normalizer", nullptr},
                                  {"added_tokens", nlohmann::json::array()},
                                  {"model",
                                   {{"type", "BPE"},
                                    {"ignore_merges", true},
                                    {"vocab", vocab},
                                    {"merges", nlohmann::json::array()}}}};
    const auto encoded         = root.dump();
    const fi::Tokenizer tokenizer(
        {encoded, R"({"added_tokens_decoder":{}})", R"({"eos_token_id":0})"});
    require(tokenizer.encode("ab") == std::vector<int>{256},
            "ignore_merges emits whole piece even without a merge rule");
    require(tokenizer.encode("\xc3\xa1") == std::vector<int>{257} &&
                tokenizer.encode("a\xcc\x81") == std::vector<int>{258},
            "null normalizer preserves canonically distinct UTF-8");
    require(tokenizer.decode(std::vector<int>{258}) == "a\xcc\x81",
            "decomposed text is not normalized during decoding");
}

void template_tests() {
    PromptInput input;
    input.messages.push_back(message(ChatRole::System, "Be concise."));
    input.messages.push_back(message(ChatRole::User, "Hi!"));
    const std::string expected =
        "<|message_system|><|content_text|>Be concise.<|end_message|>"
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
        "<|message_user|><|content_text|>Hi!<|end_message|>";
    require(fi::render_chat(input).text == expected, "chat serialization/default effort");
    const std::pair<ReasoningEffort, std::string_view> efforts[] = {
        {ReasoningEffort::Minimal, "0.1"},
        {ReasoningEffort::Low, "0.2"},
        {ReasoningEffort::Medium, "0.7"},
        {ReasoningEffort::High, "0.9"},
        {ReasoningEffort::XHigh, "0.99"}};
    for (const auto& [effort, value] : efforts) {
        input.options.reasoning_effort = effort;
        require(fi::render_chat(input).text.find("Thinking effort level: " + std::string(value) +
                                                 "<|") != std::string::npos,
                "effort mapping");
    }
    input.options.enable_thinking = false;
    require(fi::render_chat(input).text.find("Thinking effort level: 0<|") != std::string::npos,
            "thinking disabled");
    input = {};
    input.messages.push_back(message(ChatRole::System, "Only system."));
    require(fi::render_chat(input).text.ends_with("Thinking effort level: 0.9<|end_message|>"),
            "all-system effort placement");
    input                    = {};
    input.options.tool_jsons = {
        R"({"type":"function","function":{"name":"echo","parameters":{"type":"object"},"description":"a<&'"}})"};
    auto call = message(ChatRole::Assistant, "");
    call.parts.clear();
    call.tool_calls.push_back({"c1", "echo", R"({"z":2,"a":"<&'"})"});
    input.messages.push_back(std::move(call));
    input.messages.push_back(message(ChatRole::Tool, "done"));
    input.messages.back().tool_call_id = "c1";
    const auto tool_render             = fi::render_chat(input).text;
    require(
        tool_render.starts_with("<|message_system|>tool_declare<|content_xml|>[{\"description\":"
                                "\"a\\u003c\\u0026\\u0027\",\"name\":\"echo\",\"parameters\":{"
                                "\"type\":\"object\"},\"type\":\"function\"}]<|end_message|>"),
        "tool declaration canonical JSON");
    require(
        tool_render.find(
            "<|message_model|>echo<|content_invoke_tool_json|>{\"name\":\"echo\",\"args\":{\"a\":"
            "\"\\u003c\\u0026\\u0027\",\"z\":2}}<|end_message|><|content_model_end_sampling|>") !=
            std::string::npos,
        "historical tool call JSON and assistant EOS");
    require(tool_render.ends_with("<|message_tool|>echo<|content_text|>done<|end_message|>"),
            "tool response function name");
    input = {};
    input.messages.push_back(message(ChatRole::Assistant, "partial"));
    input.options.continuation = PromptContinuationMode::ContinueFinalAssistant;
    require(fi::render_chat(input).text.ends_with("<|message_model|><|content_text|>partial"),
            "assistant continuation remains open");
}

void output_tests(const ik::Frontend& frontend, const fi::Tokenizer& tokenizer) {
    auto prompt = frontend.prepare_tokens({1});
    auto output = frontend.make_output_session(prompt, {});
    const auto ids =
        tokenizer.encode("<|message_model|><|content_thinking|>Let me think.<|end_message|>"
                         "<|message_model|><|content_text|>Hello, world!<|end_message|>"
                         "<|message_model|>echo<|content_invoke_tool_json|>{\"name\":\"echo\","
                         "\"args\":{\"x\":1}}<|end_message|><|content_model_end_sampling|>");
    std::string reasoning, content;
    bool finished = false;
    for (auto id : ids) {
        const auto old_reasoning = output.reasoning_tokens();
        const auto decision =
            output.preview_model(std::span<const TokenId>(&id, 1), 1024, FinishReason::OutputLimit);
        require(output.reasoning_tokens() == old_reasoning,
                "preview must not mutate published state");
        require(decision.accepted_tokens == 1, "output accepts every consumed token");
        for (auto& delta : output.commit_preview()) {
            (delta.channel == OutputChannel::Reasoning ? reasoning : content) += delta.text;
        }
        finished = decision.finished();
    }
    require(finished && reasoning == "Let me think." && content == "Hello, world!",
            "streaming channel parser and EOS");
    require(output.reasoning_tokens() > 0, "reasoning token accounting");
    const auto calls = output.take_tool_calls();
    require(calls.size() == 1 && calls[0].name == "echo" && calls[0].arguments_json == R"({"x":1})",
            "structured tool call parsing");
    require(output.take_tool_calls().empty(), "tool calls consumed exactly once");
    StopPolicy stop;
    stop.strings.push_back({"world", OutputChannel::Content, false});
    auto stopped           = frontend.make_output_session(prompt, stop);
    const auto stopped_ids = tokenizer.encode("<|message_model|><|content_text|>Hello, world!more");
    content.clear();
    for (auto id : stopped_ids) {
        const auto decision = stopped.preview_model(std::span<const TokenId>(&id, 1), 1024,
                                                    FinishReason::OutputLimit);
        for (auto& delta : stopped.commit_preview()) { content += delta.text; }
        if (decision.finished()) {
            require(decision.finish_reason == FinishReason::StopString, "stop-string reason");
            break;
        }
    }
    require(content == "Hello, " && stopped.matched_stop_string() == "world",
            "withheld stop-string suffix");
    auto malformed = frontend.make_output_session(prompt, {});
    const auto bad = tokenizer.encode("<|message_model|>echo<|content_invoke_tool_json|>{bad}<|end_"
                                      "message|><|content_model_end_sampling|>");
    (void)malformed.preview_model(bad, 1024, FinishReason::OutputLimit);
    content.clear();
    for (auto& delta : malformed.commit_preview()) { content += delta.text; }
    require(malformed.take_tool_calls().empty() && content == "{bad}",
            "malformed JSON returns visible content");
    std::array<int, 256> byte_ids;
    byte_ids.fill(-1);
    for (int id = 0; id < 256; ++id) {
        const auto bytes = tokenizer.decoded_token(id).bytes;
        if (bytes.size() == 1) { byte_ids[static_cast<unsigned char>(bytes[0])] = id; }
    }
    require(byte_ids[0xe2] >= 0 && byte_ids[0x82] >= 0 && byte_ids[0xac] >= 0 &&
                byte_ids[0xff] >= 0,
            "reference byte tokens exist");
    auto unicode = frontend.make_output_session(prompt, {});
    const std::vector<TokenId> unicode_ids{
        200001, 200004, byte_ids[0xe2], byte_ids[0x82], byte_ids[0xac], byte_ids[0xff], 200006};
    content.clear();
    for (std::size_t i = 0; i < unicode_ids.size(); ++i) {
        (void)unicode.preview_model(std::span(&unicode_ids[i], 1), 1024, FinishReason::OutputLimit);
        const auto deltas = unicode.commit_preview();
        if (i == 2 || i == 3) { require(deltas.empty(), "incomplete UTF-8 must remain buffered"); }
        for (const auto& delta : deltas) { content += delta.text; }
    }
    require(content == "\xe2\x82\xac\xef\xbf\xbd",
            "streamed UTF-8 pieces and invalid-byte replacement");
    auto preserved = frontend.make_output_session(
        prompt, {}, {.raw = false, .preserve_special_tokens = true});
    const auto structured =
        tokenizer.encode("<|message_model|><|content_text|>ok<|begin_of_text|><|end_message|>"
                         "<|content_model_end_sampling|>");
    (void)preserved.preview_model(structured, 1024, FinishReason::OutputLimit);
    content.clear();
    for (const auto& delta : preserved.commit_preview()) { content += delta.text; }
    require(content == "ok<|begin_of_text|>",
            "preserve_special_tokens retains non-framing special tokens while consuming Inkling "
            "structural tags");
    auto raw_output = frontend.make_output_session(prompt, {}, {.raw = true});
    (void)raw_output.preview_model(structured, 1024, FinishReason::OutputLimit);
    content.clear();
    for (const auto& delta : raw_output.commit_preview()) { content += delta.text; }
    require(content == "<|message_model|><|content_text|>ok<|begin_of_text|><|end_message|>",
            "raw output preserves structural tokens except stop token");
}

void cache_identity_tests(const ik::Frontend& frontend, const fi::Tokenizer& tokenizer) {
    PromptInput input;
    input.messages.push_back(message(ChatRole::System, "Alpha beta"));
    input.messages.back().parts.push_back({MessagePartKind::Text, "Gamma", {}});
    input.messages.push_back(message(ChatRole::User, "First part"));
    input.messages.back().parts.push_back({MessagePartKind::Text, "Second part", {}});
    input.options.tool_jsons = {R"({"name":"echo","parameters":{}})"};
    input.context_cache.allow_engine_automatic_shared_prefixes = false;
    input.context_cache.session_key                            = "inkling-test-session";
    input.context_cache.markers                                = {
        {.after_message_count = 1},
        {.after_message_count      = 2,
         .location                 = PromptCacheMarkerLocation::MessagePartBoundary,
         .after_message_part_count = 1},
        {.location                  = PromptCacheMarkerLocation::LeadingInstructionBoundary,
         .leading_instruction_bytes = 5},
        {.location = PromptCacheMarkerLocation::ToolBoundary, .after_tool_count = 1}};
    const auto rendered = fi::render_chat(input);
    const auto prepared = frontend.prepare(input);
    const auto& data    = ik::PreparedPromptAccess::view(prepared);
    require(data.context_cache.retention == ninfer::runtime::RetentionClass::LiveSession,
            "named session receives live-session retention by default");
    require(data.context_cache.opportunities.size() == 4, "all cache marker locations resolve");
    const std::size_t expected_offsets[] = {
        rendered.message_boundaries[1], rendered.message_part_spans[1][0].end,
        rendered.message_part_spans[0][0].begin + 5, rendered.tool_boundaries[0]};
    for (std::size_t i = 0; i < 4; ++i) {
        const auto prefix =
            tokenizer.encode(std::string_view(rendered.text).substr(0, expected_offsets[i]));
        require(data.context_cache.opportunities[i].frontier == prefix.size(),
                "cache frontier binds exact full-stream token prefix");
        require(std::equal(prefix.begin(), prefix.end(), data.token_ids.begin()),
                "cache marker never retokenizes a mismatching prefix");
    }
    require(data.identity.rewrite_checkpoint &&
                data.identity.rewrite_checkpoint->kind ==
                    ik::RewriteCheckpointKind::ResponseReplay &&
                data.identity.rewrite_checkpoint->frontier == data.token_ids.size(),
            "new-turn replay checkpoint precedes model-generated assistant opener");
    input = {};
    input.messages.push_back(message(ChatRole::Assistant, "partial"));
    input.options.continuation = PromptContinuationMode::ContinueFinalAssistant;
    const auto continuation    = frontend.prepare(input);
    const auto& continued      = ik::PreparedPromptAccess::view(continuation);
    const auto replay_bytes    = fi::render_chat(input).message_begins.back();
    const auto prefix =
        tokenizer.encode(std::string_view(fi::render_chat(input).text).substr(0, replay_bytes));
    require(continued.identity.rewrite_checkpoint &&
                continued.identity.rewrite_checkpoint->frontier == prefix.size() &&
                continued.identity.rewrite_checkpoint->frontier < continued.token_ids.size(),
            "continuation replay checkpoint precedes final assistant serialization");
}
} // namespace

int main(int argc, char** argv) {
    try {
        template_tests();
        tokenizer_semantic_tests();
        const std::filesystem::path source =
            argc > 1 ? argv[1] : "C:/LLM/thinkingmachines/Inkling-2-Mini";
        if (!std::filesystem::exists(source / "tokenizer.json")) {
            std::cout << "SKIP: pass checkpoint directory to run tokenizer/output qualification\n";
            return 77;
        }
        ik::FrontendResources resources{
            read(source / "tokenizer.json"), read(source / "tokenizer_config.json"),
            read(source / "chat_template.jinja"), read(source / "generation_config.json")};
        const fi::Tokenizer tokenizer({resources.tokenizer_json, resources.tokenizer_config_json,
                                       resources.generation_config_json});
        require(tokenizer.has_exact_token_domain(200058), "token domain");
        const auto cases =
            nlohmann::json::parse(read(std::filesystem::path(NINFER_SOURCE_DIR) /
                                       "tests/fixtures/inkling2_frontend_oracle.json"));
        std::size_t n = 0;
        for (const auto& item : cases.at("cases")) {
            const auto text     = item.at("text").get<std::string>();
            const auto expected = item.at("ids").get<std::vector<int>>();
            const auto ids      = tokenizer.encode(text, {.parse_added_tokens = false});
            if (ids != expected) {
                throw std::runtime_error("tokenizer parity case " + std::to_string(n) + ": " +
                                         item.at("text").dump());
            }
            require(tokenizer.decode(ids) == text, "tokenizer exact byte roundtrip (no NFC)");
            ++n;
        }
        require(tokenizer.encode("<|message_model|>") == std::vector<int>{200001}, "added tokens");
        const auto frontend = ik::make_frontend(resources, {.max_context = 4096});
        PromptInput input;
        input.messages.push_back(message(ChatRole::User, "literal <|message_model|> text"));
        const auto prepared = frontend.prepare(input);
        require(prepared.summary().prompt_tokens ==
                    ik::PreparedPromptAccess::view(prepared).token_ids.size(),
                "prepared token accounting");
        require(frontend.count_tokens(input) == prepared.summary().prompt_tokens,
                "count/prepare parity");
        require(frontend.tokenize_text("<|message_model|>") != std::vector<int>{200001},
                "literal user special markup");
        output_tests(frontend, tokenizer);
        cache_identity_tests(frontend, tokenizer);
        const std::filesystem::path artifact =
            argc > 2 ? argv[2] : "C:/LLM/inkling2_mini_q3g128.ninfer";
        if (std::filesystem::exists(artifact)) {
            ninfer::artifact::Reader reader(artifact);
            ninfer::artifact::Binder binder(reader);
            const auto plan = ik::detail::bind_artifact(binder);
            require(plan.materialization.object_count == 540 &&
                        plan.materialization.device_objects.size() == 536 &&
                        plan.materialization.host_objects.size() == 4,
                    "complete artifact bindings");
            require(plan.materialization.device_capacity_bytes == 11158837248ULL,
                    "resident weights exact arena budget");
        }
        std::cout << "PASS: Inkling frontend template, " << n
                  << " tokenizer oracle cases, output transactions and artifact bindings\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
