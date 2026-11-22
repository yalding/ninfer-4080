#include "targets/inkling2_mini/impl/state/state_image.h"
#include "core/device.h"
#include <algorithm>
#include <chrono>
#include <utility>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace ninfer::targets::inkling2_mini::detail {
namespace {
constexpr int kConvElements = 6656 * 3;
constexpr int kChannels[]   = {1024, 1024, 2304, 2304};
constexpr std::size_t fixed_state_bytes =
    std::size_t(kConvElements) * 48 * 2 + std::size_t(40) * 2 * 128 * 1024 * 8 * 2;
using Clock = std::chrono::steady_clock;

std::uint64_t elapsed(Clock::time_point begin) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
}

} // namespace

std::uint32_t state_slot_count(const EngineOptions& options) {
    const auto slots =
        std::uint64_t(options.max_concurrency) +
        options.context_cache.device_state_slots.value_or(
            options.context_cache.enabled ? std::min(options.max_concurrency, 2U) : 0U);
    if (!slots || slots > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Inkling state slot capacity is invalid");
    return static_cast<std::uint32_t>(slots);
}

StateLayout plan_state(const EngineOptions& options, std::uint32_t kv_tokens) {
    if (options.kv_cache != KvCacheStorage::BFloat16 &&
        options.kv_cache != KvCacheStorage::Int8Group64)
        throw std::invalid_argument("Inkling-2 supports --kv-dtype bf16 or int8");
    if (!kv_tokens || !options.max_context || options.max_context > 131072)
        throw std::invalid_argument("Inkling-2 context must be in [1,131072]");
    const auto slots   = state_slot_count(options);
    const auto storage = paged_kv_storage_layout(options.kv_cache, 128);
    DeviceKVPagePoolSpec spec;
    // One private partial page per state slot avoids capacity loss when a shared endpoint forks.
    spec.page_group_count = (kv_tokens + 63) / 64 + slots;
    for (int layer = 0; layer < 8; ++layer) {
        spec.geometry.planes.push_back({storage.key.data_dtype, 128, 8, 256});
        spec.geometry.planes.push_back({storage.value.data_dtype, 128, 8, 256});
        if (storage.key.has_scale()) {
            spec.geometry.planes.push_back({DType::FP16, 2, 8, 256});
            spec.geometry.planes.push_back({DType::FP16, 2, 8, 256});
        }
    }
    LayoutBuilder builder;
    StateLayout out;
    out.pages  = plan_device_kv_page_pool(builder, spec);
    out.tables = plan_kv_execution_tables(
        builder, {(options.max_context + 63) / 64, static_cast<int>(slots)});
    out.cyclic      = plan_cyclic_kv_cache(builder, 40, 1024, 8, 128, static_cast<int>(slots));
    out.convolution = builder.add_tensor(DType::BF16, {kConvElements, 48, static_cast<int>(slots)},
                                         256, "Inkling convolution states");
    out.bytes       = builder.finish(256);
    return out;
}

StateImage::StateImage(const EngineOptions& options, std::uint32_t tokens)
    : layout_(plan_state(options, tokens)), backing_(layout_.bytes),
      pages_({backing_.p, backing_.bytes}, layout_.pages),
      tables_({backing_.p, backing_.bytes}, layout_.tables, pages_),
      cyclic_({backing_.p, backing_.bytes}, layout_.cyclic),
      convolution_(layout_.convolution.bind({backing_.p, backing_.bytes})),
      storage_(options.kv_cache), max_context_(options.max_context),
      slots_(state_slot_count(options)),
      host_layout_(plan_host_kv_page_layout(layout_.pages.spec.geometry)),
      host_state_capacity_(options.context_cache.enabled ? options.context_cache.host_state_slots
                                                         : 0) {
    backing_.fill();
    // The compute stream is nonblocking, so it does not wait for this default-stream fill.
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        slots_[i].row = tables_.acquire(static_cast<int>(i));
        slots_[i].pages.reserve((max_context_ + 63) / 64);
    }
    if (host_state_capacity_ && options.context_cache.host_kv_capacity_bytes)
        host_arena_ =
            std::make_unique<HostKVArena>(options.context_cache.host_kv_capacity_bytes,
                                          std::span<const HostKVPageLayout>(&host_layout_, 1));
}

void StateImage::reset(std::uint32_t slot, cudaStream_t stream) {
    auto& state = slots_.at(slot);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    state.pages.clear();
    state.depth = 0;
    auto* conv  = static_cast<std::byte*>(convolution_.data) + convolution_.nb[2] * slot;
    CUDA_CHECK(cudaMemsetAsync(conv, 0, kConvElements * 48ULL * 2, stream));
    // Old ring cells are not visible at frontier zero; causal reads establish their validity.
}

void StateImage::publish(std::uint32_t slot, cudaStream_t stream) {
    auto& state = slots_.at(slot);
    std::vector<DeviceKVPageHandle> handles;
    handles.reserve(state.pages.size());
    for (const auto& page : state.pages) handles.push_back(page->handle());
    if (!handles.empty()) tables_.publish(state.row.handle(), 0, handles, stream);
}

void StateImage::copy(std::uint32_t source, std::uint32_t destination, cudaStream_t stream) {
    if (source == destination) return;
    auto& src = slots_.at(source);
    auto& dst = slots_.at(destination);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto begin = Clock::now();
    dst.pages        = src.pages;
    dst.depth        = src.depth;
    const auto bytes = std::size_t(kConvElements) * 48 * 2;
    CUDA_CHECK(cudaMemcpyAsync(
        static_cast<std::byte*>(convolution_.data) + convolution_.nb[2] * destination,
        static_cast<std::byte*>(convolution_.data) + convolution_.nb[2] * source, bytes,
        cudaMemcpyDeviceToDevice, stream));
    cyclic_.copy_slot_from(cyclic_, source, destination, stream);
    publish(destination, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    transfers_.push_back({runtime::ContextResourceClass::State,
                          runtime::ContextTransferDirection::DeviceToDevice,
                          1,
                          0,
                          {fixed_state_bytes, 81},
                          elapsed(begin)});
}

void StateImage::prepare_append(std::uint32_t slot, std::uint32_t start, std::uint32_t count,
                                cudaStream_t stream) {
    auto& state = slots_.at(slot);
    if (start != state.depth || !count || start > max_context_ || count > max_context_ - start)
        throw std::logic_error("Inkling append must extend its exact state frontier");
    const auto needed = (start + count + 63) / 64;
    const bool cow    = start % 64 && !state.pages.empty() && state.pages.back().use_count() > 1;
    const auto extra  = needed - static_cast<std::uint32_t>(state.pages.size()) + (cow ? 1U : 0U);
    if (extra == 0) return;
    auto reservation = pages_.reserve(extra);
    if (!reservation) throw std::runtime_error("Inkling KV page capacity exhausted");
    if (cow) {
        auto replacement =
            std::make_shared<DeviceKVPageLease>(pages_.materialize_one(*reservation));
        pages_.copy_page(state.pages.back()->handle(), replacement->handle(), stream);
        const auto work = plan_device_kv_copy_work(host_layout_, 1);
        transfers_.push_back({runtime::ContextResourceClass::MainKV,
                              runtime::ContextTransferDirection::DeviceToDevice, work.payload_bytes,
                              1, work, 0});
        state.pages.back() = std::move(replacement);
    }
    while (state.pages.size() < needed)
        state.pages.push_back(
            std::make_shared<DeviceKVPageLease>(pages_.materialize_one(*reservation)));
    publish(slot, stream);
}

void StateImage::commit_append(std::uint32_t slot, std::uint32_t depth) {
    slots_.at(slot).depth = depth;
}

Tensor StateImage::convolution(std::uint32_t slot, int visit, int component) const {
    if (slot >= slots_.size() || visit < 0 || visit >= 48 || component < 0 || component >= 4)
        throw std::out_of_range("Inkling convolution state index");
    int offset = 0;
    for (int i = 0; i < component; ++i) offset += kChannels[i] * 3;
    auto* ptr = static_cast<std::byte*>(convolution_.data) + slot * convolution_.nb[2] +
                visit * convolution_.nb[1] + offset * 2;
    return Tensor(ptr, DType::BF16, {kChannels[component], 3});
}

CyclicKVCacheLayerView StateImage::swa(int layer) const { return cyclic_.layer_view(layer); }

PagedKVLayerView StateImage::full(std::uint32_t slot, int layer) const {
    if (layer < 0 || layer >= 8) throw std::out_of_range("Inkling full attention layer");
    const auto n    = storage_ == KvCacheStorage::Int8Group64 ? 4 : 2;
    const auto base = layer * n;
    PagedKVLayerView view;
    view.k_pages = pages_.plane(base);
    view.v_pages = pages_.plane(base + 1);
    if (n == 4) {
        view.k_scale_pages = pages_.plane(base + 2);
        view.v_scale_pages = pages_.plane(base + 3);
    }
    view.block_table  = tables_.row(slots_.at(slot).row.handle());
    view.head_dim     = 128;
    view.num_kv_heads = 8;
    view.storage      = storage_;
    return view;
}

std::uint32_t StateImage::depth(std::uint32_t slot) const { return slots_.at(slot).depth; }

std::uint32_t StateImage::free_pages() const noexcept { return pages_.available_pages(); }

std::uint32_t StateImage::used_pages() const noexcept { return pages_.allocated_pages(); }

std::uint32_t StateImage::reclaimable_pages(std::span<const std::uint32_t> slots) const {
    struct References {
        std::uint32_t selected = 0;
        long total             = 0;
    };

    std::unordered_map<const DeviceKVPageLease*, References> refs;
    std::vector<bool> seen(slots_.size());
    for (const auto slot : slots) {
        if (slot >= slots_.size()) throw std::out_of_range("Inkling reclaimable state slot");
        if (seen[slot]) continue;
        seen[slot] = true;
        for (const auto& page : slots_[slot].pages) {
            auto& entry = refs[page.get()];
            ++entry.selected;
            entry.total = page.use_count();
        }
    }
    std::uint32_t count = 0;
    for (const auto& [page, entry] : refs)
        if (entry.selected == entry.total) ++count;
    return count;
}

std::uint32_t StateImage::host_state_count() const noexcept {
    return static_cast<std::uint32_t>(
        std::count_if(host_images_.begin(), host_images_.end(),
                      [](const auto& image) { return !image.expired(); }));
}

std::size_t StateImage::host_kv_bytes() const noexcept {
    return host_arena_ ? host_arena_->occupied_bytes() : 0;
}

std::size_t StateImage::host_kv_capacity() const noexcept {
    return host_arena_ ? host_arena_->capacity_bytes() : 0;
}

bool StateImage::can_snapshot_to_host(std::uint32_t slot) const {
    const auto& state = slots_.at(slot);
    return host_arena_ && state.depth && host_state_count() < host_state_capacity_ &&
           host_arena_->can_allocate(host_layout_, static_cast<unsigned>(state.pages.size()));
}

std::shared_ptr<HostStateImage> StateImage::snapshot_to_host(std::uint32_t slot,
                                                             cudaStream_t stream) {
    if (!can_snapshot_to_host(slot))
        throw std::runtime_error("Inkling host checkpoint capacity exhausted");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto& state = slots_.at(slot);
    auto allocation =
        host_arena_->allocate(host_layout_, static_cast<unsigned>(state.pages.size()));
    if (!allocation) throw std::runtime_error("Inkling host KV capacity exhausted");
    const auto conv_bytes = std::size_t(kConvElements) * 48 * 2;
    const auto ring_bytes = std::size_t(128) * 1024 * 8 * 2;
    auto image            = std::make_shared<HostStateImage>(conv_bytes + 40 * 2 * ring_bytes);
    image->pages          = std::move(*allocation);
    image->depth          = state.depth;
    auto* fixed           = static_cast<std::byte*>(image->fixed.data());
    const auto begin      = Clock::now();
    CUDA_CHECK(cudaMemcpyAsync(
        fixed, static_cast<std::byte*>(convolution_.data) + slot * convolution_.nb[2], conv_bytes,
        cudaMemcpyDeviceToHost, stream));
    std::size_t offset = conv_bytes;
    for (int layer = 0; layer < 40; ++layer) {
        const auto ring = cyclic_.layer_view(layer);
        for (const auto* plane : {&ring.k, &ring.v}) {
            CUDA_CHECK(cudaMemcpyAsync(fixed + offset,
                                       static_cast<std::byte*>(plane->data) + slot * plane->nb[3],
                                       ring_bytes, cudaMemcpyDeviceToHost, stream));
            offset += ring_bytes;
        }
    }
    std::vector<DeviceKVPageHandle> handles;
    handles.reserve(state.pages.size());
    for (const auto& page : state.pages) handles.push_back(page->handle());
    pages_.copy_to_host(handles, host_arena_->writable_view(image->pages), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto work = plan_host_kv_transfer_work(
        host_layout_, static_cast<unsigned>(handles.size()), pages_.contiguous_run_count(handles));
    const auto ns = elapsed(begin);
    transfers_.push_back({runtime::ContextResourceClass::MainKV,
                          runtime::ContextTransferDirection::DeviceToHost, work.payload_bytes,
                          static_cast<unsigned>(handles.size()), work, ns});
    transfers_.push_back({runtime::ContextResourceClass::State,
                          runtime::ContextTransferDirection::DeviceToHost,
                          1,
                          0,
                          {fixed_state_bytes, 81},
                          0});
    std::erase_if(host_images_, [](const auto& item) { return item.expired(); });
    host_images_.push_back(image);
    return image;
}

void StateImage::restore_from_host(std::uint32_t slot, const HostStateImage& image,
                                   cudaStream_t stream) {
    auto& state = slots_.at(slot);
    if (state.depth || !state.pages.empty() || !host_arena_ || image.depth == 0 ||
        image.depth > max_context_ || image.pages.page_count() != (image.depth + 63) / 64)
        throw std::invalid_argument(
            "Inkling host restore requires an empty slot and matching image");
    auto reservation = pages_.reserve(image.pages.page_count());
    if (!reservation)
        throw std::runtime_error("Inkling device KV capacity exhausted during restore");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::shared_ptr<DeviceKVPageLease>> restored;
    restored.reserve(image.pages.page_count());
    for (unsigned i = 0; i < image.pages.page_count(); ++i)
        restored.push_back(
            std::make_shared<DeviceKVPageLease>(pages_.materialize_one(*reservation)));
    std::vector<DeviceKVPageHandle> handles;
    handles.reserve(restored.size());
    for (const auto& page : restored) handles.push_back(page->handle());
    const auto begin = Clock::now();
    pages_.copy_from_host(host_arena_->view(image.pages), handles, stream);
    const auto conv_bytes = std::size_t(kConvElements) * 48 * 2;
    const auto ring_bytes = std::size_t(128) * 1024 * 8 * 2;
    const auto* fixed     = static_cast<const std::byte*>(image.fixed.data());
    CUDA_CHECK(
        cudaMemcpyAsync(static_cast<std::byte*>(convolution_.data) + slot * convolution_.nb[2],
                        fixed, conv_bytes, cudaMemcpyHostToDevice, stream));
    std::size_t offset = conv_bytes;
    for (int layer = 0; layer < 40; ++layer) {
        const auto ring = cyclic_.layer_view(layer);
        for (const auto* plane : {&ring.k, &ring.v}) {
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(plane->data) + slot * plane->nb[3],
                                       fixed + offset, ring_bytes, cudaMemcpyHostToDevice, stream));
            offset += ring_bytes;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto work = plan_host_kv_transfer_work(
        host_layout_, static_cast<unsigned>(handles.size()), pages_.contiguous_run_count(handles));
    const auto ns = elapsed(begin);
    transfers_.push_back({runtime::ContextResourceClass::MainKV,
                          runtime::ContextTransferDirection::HostToDevice, work.payload_bytes,
                          static_cast<unsigned>(handles.size()), work, ns});
    transfers_.push_back({runtime::ContextResourceClass::State,
                          runtime::ContextTransferDirection::HostToDevice,
                          1,
                          0,
                          {fixed_state_bytes, 81},
                          0});
    state.pages = std::move(restored);
    state.depth = image.depth;
    publish(slot, stream);
}

TransferWork StateImage::fixed_state_transfer_work() const noexcept {
    return {fixed_state_bytes, 81};
}

TransferWork StateImage::host_kv_transfer_work(std::uint32_t page_count) const {
    return page_count ? plan_host_kv_transfer_work(host_layout_, page_count, 1) : TransferWork{};
}

TransferWork StateImage::device_kv_cow_transfer_work() const {
    return plan_device_kv_copy_work(host_layout_, 1);
}

std::vector<runtime::ContextTransferObservation> StateImage::take_transfer_observations() {
    return std::exchange(transfers_, {});
}
} // namespace ninfer::targets::inkling2_mini::detail
