#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <memory>
#include <span>
#include <vector>

namespace ninfer::targets::inkling2_mini::detail {
struct RuntimeModelView;
struct HostStateImage;
std::size_t executor_device_bytes(const EngineOptions& options, std::uint32_t kv_tokens);

class Executor {
public:
    Executor(const RuntimeModelView& model, DeviceContext& device, const EngineOptions& options);
    ~Executor();
    Tensor forward(std::span<const TokenId> ids, std::uint32_t start, std::uint32_t state_slot,
                   bool all_logits = false);
    std::vector<TokenId> draft_dspark(std::uint32_t state_slot, TokenId anchor,
                                      std::uint32_t draft_tokens);
    std::vector<TokenId> verify_dspark(std::uint32_t state_slot, std::span<const TokenId> verify_ids,
                                       std::uint32_t start,
                                       const ResolvedSamplingParameters& sampling,
                                       std::uint32_t logical_position,
                                       std::span<const TokenId> generated_history);
    std::vector<TokenId> speculative_step_dspark(std::uint32_t state_slot, TokenId anchor,
                                                 std::uint32_t draft_tokens, std::uint32_t start,
                                                 const ResolvedSamplingParameters& sampling,
                                                 std::uint32_t logical_position,
                                                 std::span<const TokenId> generated_history);
    std::vector<TokenId> speculative_step_dflash2(std::uint32_t state_slot, TokenId anchor,
                                                  std::uint32_t draft_tokens, std::uint32_t start,
                                                  const ResolvedSamplingParameters& sampling,
                                                  std::uint32_t logical_position,
                                                  std::span<const TokenId> generated_history);
    void truncate_speculative_commit(std::uint32_t state_slot, std::uint32_t accepted_tokens);
    TokenId sample(const Tensor& logits, const ResolvedSamplingParameters& sampling,
                   std::uint32_t logical_position, std::span<const TokenId> generated_history);
    void reset(std::uint32_t slot);
    void copy_state(std::uint32_t source, std::uint32_t destination);
    std::vector<float> causal_score(std::span<const TokenId> ids, std::uint32_t first_target);
    MemorySummary memory_summary() const;
    void reset_memory_peaks();
    void sync();
    std::uint32_t slot_depth(std::uint32_t slot) const;
    std::uint32_t free_pages() const noexcept;
    std::uint32_t used_pages() const noexcept;
    std::uint32_t reclaimable_pages(std::span<const std::uint32_t> slots) const;
    bool can_snapshot_to_host(std::uint32_t slot) const;
    std::shared_ptr<HostStateImage> snapshot_to_host(std::uint32_t slot);
    void restore_from_host(std::uint32_t slot, const HostStateImage& image);
    std::uint32_t host_state_count() const noexcept;
    std::size_t host_kv_bytes() const noexcept;
    std::size_t host_kv_capacity() const noexcept;
    TransferWork fixed_state_transfer_work() const noexcept;
    TransferWork host_kv_transfer_work(std::uint32_t page_count) const;
    TransferWork device_kv_cow_transfer_work() const;
    std::vector<runtime::ContextTransferObservation> take_transfer_observations();
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::targets::inkling2_mini::detail
