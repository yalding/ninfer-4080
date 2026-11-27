#pragma once
#include "targets/inkling2_mini/impl/frontend/tokenizer.h"
#include "ninfer/types.h"
#include <optional>
#include <string>
#include <vector>

namespace ninfer::targets::inkling2_mini::frontend_internal {
struct ImagePlaceholder {
    std::size_t message_index     = 0;
    std::size_t part_index        = 0;
    std::size_t placeholder_begin = 0;
    std::size_t placeholder_end   = 0;
};

struct RenderedChat {
    std::string text;
    std::vector<ByteSpan> literal_spans;
    std::vector<std::size_t> message_boundaries;
    std::vector<std::size_t> message_begins;
    std::vector<std::vector<ByteSpan>> message_part_spans;
    std::vector<std::size_t> tool_boundaries;
    std::vector<ImagePlaceholder> image_placeholders;
};

RenderedChat render_chat(const PromptInput& input);
PromptCapabilities prompt_capabilities() noexcept;
} // namespace ninfer::targets::inkling2_mini::frontend_internal
