#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/paged_kv_cache.h"
#include "core/host_kv_arena.h"
#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <memory>
#include <vector>

namespace ninfer::targets::inkling2_mini::detail {

struct StateLayout {
    DeviceKVPagePoolLayout pages;
    KVExecutionTableLayout tables;
    CyclicKVCacheLayout cyclic;
    TensorRegion convolution;
    std::size_t bytes = 0;
};

// The originating StateImage (and its HostKVArena) must outlive these immutable snapshots.
struct HostStateImage {
    HostKVAllocation pages;
    PinnedHostBuffer fixed;
    std::uint32_t depth = 0;

    explicit HostStateImage(std::size_t fixed_bytes) : fixed(fixed_bytes) {}
};

std::uint32_t state_slot_count(const EngineOptions& options);
StateLayout plan_state(const EngineOptions& options, std::uint32_t kv_tokens);

// A slot owns its short-convolution/SWA image and a shared page mapping. Forking an endpoint
// retains complete pages; only a shared partial tail is copied before the next append.
class StateImage {
public:
    StateImage(const EngineOptions& options, std::uint32_t kv_tokens);
    void reset(std::uint32_t slot, cudaStream_t stream);
    void copy(std::uint32_t source, std::uint32_t destination, cudaStream_t stream);
    void prepare_append(std::uint32_t slot, std::uint32_t start, std::uint32_t count,
                        cudaStream_t stream);
    void commit_append(std::uint32_t slot, std::uint32_t depth);
    Tensor convolution(std::uint32_t slot, int visit, int component) const;
    CyclicKVCacheLayerView swa(int layer) const;
    PagedKVLayerView full(std::uint32_t slot, int layer) const;
    std::uint32_t depth(std::uint32_t slot) const;
    std::uint32_t free_pages() const noexcept;
    std::uint32_t used_pages() const noexcept;
    std::uint32_t reclaimable_pages(std::span<const std::uint32_t> slots) const;
    bool can_snapshot_to_host(std::uint32_t slot) const;
    std::shared_ptr<HostStateImage> snapshot_to_host(std::uint32_t slot, cudaStream_t stream);
    void restore_from_host(std::uint32_t slot, const HostStateImage& image, cudaStream_t stream);
    std::uint32_t host_state_count() const noexcept;
    std::size_t host_kv_bytes() const noexcept;
    std::size_t host_kv_capacity() const noexcept;
    TransferWork fixed_state_transfer_work() const noexcept;
    TransferWork host_kv_transfer_work(std::uint32_t page_count) const;
    TransferWork device_kv_cow_transfer_work() const;
    std::vector<runtime::ContextTransferObservation> take_transfer_observations();

    std::size_t bytes() const noexcept { return backing_.bytes; }

    std::size_t kv_bytes() const noexcept { return layout_.pages.payload_bytes(); }

    std::uint32_t slots() const noexcept { return static_cast<std::uint32_t>(slots_.size()); }

private:
    struct Slot {
        std::vector<std::shared_ptr<DeviceKVPageLease>> pages;
        KVExecutionRowLease row;
        std::uint32_t depth = 0;
    };

    void publish(std::uint32_t slot, cudaStream_t stream);
    StateLayout layout_;
    DeviceBuffer backing_;
    DeviceKVPagePool pages_;
    KVExecutionTablePool tables_;
    CyclicKVCache cyclic_;
    Tensor convolution_;
    KvCacheStorage storage_;
    std::uint32_t max_context_;
    std::vector<Slot> slots_;
    HostKVPageLayout host_layout_;
    std::unique_ptr<HostKVArena> host_arena_;
    std::uint32_t host_state_capacity_;
    std::vector<std::weak_ptr<HostStateImage>> host_images_;
    std::vector<runtime::ContextTransferObservation> transfers_;
};
} // namespace ninfer::targets::inkling2_mini::detail
