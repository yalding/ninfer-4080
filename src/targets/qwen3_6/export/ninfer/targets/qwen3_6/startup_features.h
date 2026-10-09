#pragma once

#include "ninfer/types.h"

namespace ninfer::targets::qwen3_6 {

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2;
}

struct StartupFeatures {
    bool vision                            = false;
    VisionResidency vision_residency       = VisionResidency::Resident;
    std::uint32_t vision_max_tokens        = 8192;
    std::uint32_t vision_max_merged_tokens = 16384;
    SpeculativeBackend speculative         = SpeculativeBackend::None;
    ProposalHead proposal_head             = ProposalHead::Full;

    bool operator==(const StartupFeatures&) const = default;

    [[nodiscard]] bool vision_resident() const noexcept {
        return vision && vision_residency == VisionResidency::Resident;
    }

    [[nodiscard]] bool vision_overlay() const noexcept {
        return vision && vision_residency == VisionResidency::Overlay;
    }

    [[nodiscard]] bool vision_cpu() const noexcept {
        return vision && vision_residency == VisionResidency::Cpu;
    }

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool masked_draft() const noexcept {
        return is_masked_draft_backend(speculative);
    }

    [[nodiscard]] bool optimized_proposal() const noexcept {
        return speculative_enabled() && proposal_head == ProposalHead::Optimized;
    }
};

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    return StartupFeatures{
        .vision                   = options.enable_vision,
        .vision_residency         = options.vision_residency,
        .vision_max_tokens        = options.vision_max_tokens > 0 ? options.vision_max_tokens : 8192,
        .vision_max_merged_tokens = options.vision_max_merged_tokens > 0 ? options.vision_max_merged_tokens : 16384,
        .speculative              = options.speculative.backend,
        .proposal_head            = options.speculative.proposal_head,
    };
}

} // namespace ninfer::targets::qwen3_6
