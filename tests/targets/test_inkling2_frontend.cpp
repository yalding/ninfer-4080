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
        std::error_code ec;
        std::filesystem::path source =
            argc > 1 ? argv[1] : "C:/LLM/thinkingmachines/Inkling-2-Mini";
        if (!std::filesystem::exists(source / "tokenizer.json", ec)) {
            source = std::filesystem::path(NINFER_SOURCE_DIR) / "models/Inkling-2-Mini";
        }
        if (!std::filesystem::exists(source / "tokenizer.json", ec)) {
            std::cout << "SKIP: pass checkpoint directory to run tokenizer/output qualification\n";
            return 77;
        }
        ik::FrontendResources resources{
            read(source / "tokenizer.json"), read(source / "tokenizer_config.json"),
            read(source / "chat_template.jinja"), read(source / "generation_config.json"),
            std::filesystem::exists(source / "processor_config.json", ec)
                ? read(source / "processor_config.json")
                : std::string{}};
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
        {
            auto vres = resources;
            if (vres.processor_config_json.empty()) {
                vres.processor_config_json =
                    R"({"image_processor":{"size":{"height":40,"width":40},"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]},"image_token":"<|unused_200054|>"})";
            }
            auto make_bmp = [](int w, int h, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
                const int row_stride = ((w * 3 + 3) / 4) * 4;
                const int pixel_bytes = row_stride * h;
                const int file_bytes = 54 + pixel_bytes;
                std::vector<std::uint8_t> bmp(file_bytes, 0);
                bmp[0] = 'B';
                bmp[1] = 'M';
                auto put32 = [&](int off, std::uint32_t v) {
                    bmp[off + 0] = static_cast<std::uint8_t>(v & 0xff);
                    bmp[off + 1] = static_cast<std::uint8_t>((v >> 8) & 0xff);
                    bmp[off + 2] = static_cast<std::uint8_t>((v >> 16) & 0xff);
                    bmp[off + 3] = static_cast<std::uint8_t>((v >> 24) & 0xff);
                };
                put32(2, file_bytes);
                put32(10, 54);
                put32(14, 40);
                put32(18, static_cast<std::uint32_t>(w));
                put32(22, static_cast<std::uint32_t>(h));
                bmp[26] = 1;
                bmp[28] = 24;
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        const int p = 54 + y * row_stride + x * 3;
                        bmp[p + 0]  = b;
                        bmp[p + 1]  = g;
                        bmp[p + 2]  = r;
                    }
                }
                return bmp;
            };
            auto img1 = make_bmp(40, 50, 200, 100, 50);
            auto img2 = make_bmp(40, 50, 10, 20, 30);

            PromptInput img_input;
            ChatMessage msg;
            msg.role = ChatRole::User;
            MessagePart mp;
            mp.kind = MessagePartKind::Media;
            mp.media.kind  = MediaKind::Image;
            mp.media.bytes = img1;
            msg.parts.push_back(std::move(mp));
            msg.parts.push_back({MessagePartKind::Text, "describe", {}});
            img_input.messages.push_back(std::move(msg));

            bool rejected_disabled = false;
            try {
                (void)frontend.prepare(img_input);
            } catch (const std::exception&) { rejected_disabled = true; }
            require(rejected_disabled, "vision disabled rejects image input");

            const auto vfront = ik::make_frontend(
                vres, {.max_context = 4096, .vision_enabled = true, .media_cache_bytes = 256 * 1024});
            const auto vp1     = vfront.prepare(img_input);
            const auto& vdata1 = ik::PreparedPromptAccess::view(vp1);
            require(vdata1.has_media() && vdata1.vision_items.size() == 1 &&
                        vdata1.media_payloads.size() == 1 && vdata1.media_payloads[0] != nullptr,
                    "vision prepare produces one vision item and payload");
            const auto& item1    = vdata1.vision_items[0];
            const auto& payload1 = *vdata1.media_payloads[0];
            require(payload1.merged_tokens == 2 && payload1.native_patches == 6 &&
                        payload1.padded_h20 == 4 && payload1.padded_w20 == 2,
                    "40x50 image pads to 40x80 with 6 active 20x20 patches and 2 merged tokens");
            require(std::find(payload1.patches.begin(), payload1.patches.end(),
                              static_cast<std::uint16_t>(0xBF81)) != payload1.patches.end(),
                    "partially padded patch uses BF16 0xBF81 (-1.00784f) pad fill");
            require(item1.token_begin >= 1 &&
                        vdata1.token_ids[item1.token_begin - 1] == 200005 &&
                        vdata1.token_ids[item1.token_begin] == 200054 &&
                        vdata1.token_ids[item1.token_begin + 1] == 200054 &&
                        vdata1.token_ids[item1.token_end] == 200010,
                    "image expands to <|content_image|>, V x <|unused_200054|>, <|end_message|>");
            const auto vp1_cached = vfront.prepare(img_input);
            require(vfront.media_cache_summary().hits >= 1, "identical image hits media cache");

            PromptInput img2_input = img_input;
            img2_input.messages[0].parts[0].media.bytes = img2;
            const auto vp2     = vfront.prepare(img2_input);
            const auto& vdata2 = ik::PreparedPromptAccess::view(vp2);
            require(vdata2.vision_items[0].content_digest != item1.content_digest,
                    "different same-size images produce distinct content_digest");

            const auto bounded = ik::make_frontend(
                vres, {.max_context = 4096, .vision_enabled = true,
                       .media_live_bytes = payload1.payload_bytes(), .media_preprocess_threads = 1});
            auto live_prompt = bounded.prepare(img_input);
            bool exhausted = false;
            try {
                (void)bounded.prepare(img2_input);
            } catch (const RequestError& error) {
                exhausted = error.kind() == RequestErrorKind::MediaBudgetExceeded;
            }
            require(exhausted && bounded.media_cache_summary().live_bytes == payload1.payload_bytes(),
                    "evicting an image does not release its budget while a prompt holds it");
            live_prompt = {};
            require(bounded.media_cache_summary().live_bytes == 0 &&
                        bounded.prepare(img2_input).summary().has_media,
                    "releasing the prompt returns the live bytes and permits a retry");

            const auto limited = ik::make_frontend(
                vres, {.max_context = 4096, .vision_enabled = true, .vision_max_tokens = 5});
            const auto resized = limited.prepare(img_input);
            const auto& resized_data = ik::PreparedPromptAccess::view(resized);
            require(resized_data.media_payloads[0]->native_patches <= 5,
                    "downsize honors native patch capacity even when merged tokens fit");
            img_input.messages[0].parts[0].media.image_resize_policy =
                ImageResizePolicy::RejectOversized;
            bool rejected_oversized = false;
            try {
                (void)limited.prepare(img_input);
            } catch (const RequestError& error) {
                rejected_oversized = error.kind() == RequestErrorKind::MediaBudgetExceeded;
            }
            require(rejected_oversized, "reject policy honors native patch capacity");
        }
        std::filesystem::path artifact =
            argc > 2 ? argv[2] : "C:/LLM/inkling2_mini_q3g128.ninfer";
        if (!std::filesystem::exists(artifact, ec)) {
            artifact = std::filesystem::path(NINFER_SOURCE_DIR) / "models/inkling2_mini_q3g128.ninfer";
        }
        if (std::filesystem::exists(artifact, ec)) {
            ninfer::artifact::Reader reader(artifact);
            ninfer::artifact::Binder binder(reader);
            const auto plan = ik::detail::bind_artifact(binder);
            const bool has_vis = binder.has_object("vision/patch_embedding");
            require(plan.materialization.object_count == (has_vis ? 871 : 540) &&
                        plan.materialization.device_objects.size() == 536 &&
                        plan.materialization.host_objects.size() == (has_vis ? 5 : 4),
                    "complete artifact bindings");
            require(plan.materialization.device_capacity_bytes == 11158837248ULL,
                    "resident weights exact arena budget");
            if (has_vis) {
                ninfer::artifact::Binder b_overlay(reader);
                const auto overlay_plan = ik::detail::bind_artifact(
                    b_overlay, false, false, false, true, VisionResidency::Overlay);
                require(overlay_plan.materialization.device_objects.size() == 536 &&
                            overlay_plan.materialization.device_capacity_bytes ==
                                11158837248ULL &&
                            overlay_plan.materialization.pinned_objects.size() == 330,
                        "overlay vision bindings keep 0 B permanent VRAM and 330 pinned objects");
                ninfer::artifact::Binder b_resident(reader);
                const auto resident_plan = ik::detail::bind_artifact(
                    b_resident, false, false, false, true, VisionResidency::Resident);
                require(resident_plan.materialization.device_objects.size() == 536 + 330,
                        "resident vision bindings place all 330 vision objects on device");
            }
        }
        std::cout << "PASS: Inkling frontend template, " << n
                  << " tokenizer oracle cases, output transactions and artifact bindings\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
