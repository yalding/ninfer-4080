#pragma once

#include "ninfer/ops/linear.h"
#include "ops/linear/q4/q4_launch.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kQ4A8MinTokens     = 64;
inline constexpr std::int32_t kQ4A8WideMinTokens = 256;
inline constexpr std::int32_t kQ4A8Rows          = 128;
inline constexpr std::int32_t kQ4A8StepK         = 64;

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

[[nodiscard]] bool q4_uses_a8(std::int32_t n, std::int32_t k, std::int32_t padded_k,
                              LinearPolicy policy, std::int32_t tokens) noexcept;
[[nodiscard]] std::size_t q4_a8_workspace_capacity_bytes(std::int32_t k, std::int32_t tokens);

void q4_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
