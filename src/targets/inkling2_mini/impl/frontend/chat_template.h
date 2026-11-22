#pragma once
#include "targets/inkling2_mini/impl/frontend/tokenizer.h"
#include "ninfer/types.h"
#include <optional>
#include <string>
#include <vector>

namespace ninfer::targets::inkling2_mini::frontend_internal {
struct RenderedChat {
    std::string text;
    std::vector<ByteSpan> literal_spans;
    std::vector<std::size_t> message_boundaries;
    std::vector<std::size_t> message_begins;
    std::vector<std::vector<ByteSpan>> message_part_spans;
    std::vector<std::size_t> tool_boundaries;
};

// Implements the registered Inkling-2 text template; media is rejected in Phase 1.
RenderedChat render_chat(const PromptInput& input);
PromptCapabilities prompt_capabilities() noexcept;
} // namespace ninfer::targets::inkling2_mini::frontend_internal
