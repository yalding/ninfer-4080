#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/paged_kv_cache.h"
#include "core/host_kv_arena.h"
#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::targets::inkling2_mini::detail {

struct StateLayout {
    DeviceKVPagePoolLayout pages;
    KVExecutionTableLayout tables;
    CyclicKVCacheLayout cyclic;
    CyclicKVCacheLayout dflash2_cyclic;
    TensorRegion convolution;
    std::size_t bytes = 0;
};

// The originating StateImage (and its HostKVArena) must outlive these immutable snapshots.
struct HostStateImage {
    HostKVAllocation pages;
    PinnedHostBuffer fixed;
    std::uint32_t depth         = 0;
    std::uint32_t dspark_depth  = 0;
    std::uint32_t dflash2_depth = 0;

    explicit HostStateImage(std::size_t fixed_bytes) : fixed(fixed_bytes) {}
};

std::uint32_t state_slot_count(const EngineOptions& options);
StateLayout plan_state(const EngineOptions& options, std::uint32_t kv_tokens);

// A slot owns its short-convolution/SWA image and a shared page mapping. Forking an endpoint
// retains complete pages; only a shared partial tail is copied before the next append.
class StateImage {
public:
    StateImage(const EngineOptions& options, std::uint32_t kv_tokens,
               const StartupObserver* observer = nullptr);
    void reset(std::uint32_t slot, cudaStream_t stream);
    void copy(std::uint32_t source, std::uint32_t destination, cudaStream_t stream);
    void prepare_append(std::uint32_t slot, std::uint32_t start, std::uint32_t count,
                        cudaStream_t stream);
    void commit_append(std::uint32_t slot, std::uint32_t depth);
    Tensor convolution(std::uint32_t slot, int visit, int component) const;
    Tensor slot_convolution(std::uint32_t slot) const;
    Tensor all_convolutions() const noexcept { return convolution_; }
    CyclicKVCacheLayerView swa(int layer) const;
    PagedKVLayerView full(std::uint32_t slot, int layer) const;
    PagedKVBatchLayerView full_batch(int layer) const;
    CyclicKVCacheLayerView dspark_swa(int layer) const;
    PagedKVLayerView dspark_full(std::uint32_t slot) const;
    std::uint32_t dspark_depth(std::uint32_t slot) const;
    void commit_dspark_append(std::uint32_t slot, std::uint32_t depth);
    bool has_dspark() const noexcept { return has_dspark_; }
    CyclicKVCacheLayerView dflash2_swa(int layer) const;
    std::uint32_t dflash2_depth(std::uint32_t slot) const;
    void commit_dflash2_append(std::uint32_t slot, std::uint32_t depth);
    bool has_dflash2() const noexcept { return has_dflash2_; }
    std::size_t dflash_kv_bytes() const noexcept;
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
        std::uint32_t depth         = 0;
        std::uint32_t dspark_depth  = 0;
        std::uint32_t dflash2_depth = 0;
    };

    void publish(std::uint32_t slot, cudaStream_t stream);
    bool has_dspark_  = false;
    bool has_dflash2_ = false;
    int full_layers_  = 8;
    int swa_layers_   = 40;
    StateLayout layout_;
    DeviceBuffer backing_;
    DeviceKVPagePool pages_;
    KVExecutionTablePool tables_;
    CyclicKVCache cyclic_;
    std::optional<CyclicKVCache> dflash2_cyclic_;
    Tensor convolution_;
    KvCacheStorage storage_;
    std::uint32_t max_context_;
    std::vector<Slot> slots_;
    HostKVPageLayout host_layout_;
    std::unique_ptr<HostKVArena> host_arena_;
    std::uint32_t host_state_capacity_;
    struct SlabPool;
    std::shared_ptr<SlabPool> slab_pool_;
    std::vector<runtime::ContextTransferObservation> transfers_;
};
} // namespace ninfer::targets::inkling2_mini::detail
