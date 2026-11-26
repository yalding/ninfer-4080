#include "targets/inkling2_mini/impl/state/state_image.h"
#include "core/device.h"
#include "core/startup.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer;
namespace ik = ninfer::targets::inkling2_mini::detail;

namespace {
void require(bool condition, const char* label) {
    if (!condition) { throw std::runtime_error(label); }
}

template <class F>
void rejects(F operation, const char* label) {
    try {
        operation();
    } catch (const std::exception&) { return; }
    throw std::runtime_error(label);
}

std::vector<std::int32_t> table(ik::StateImage& state, std::uint32_t slot, std::size_t count) {
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::int32_t> out(count);
    CUDA_CHECK(cudaMemcpy(out.data(), state.full(slot, 0).block_table.data,
                          out.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    return out;
}

void* page_address(const Tensor& plane, int page) {
    return static_cast<std::byte*>(plane.data) + plane.nb[3] * page;
}

std::vector<unsigned char> copy_bytes(const void* address, std::size_t count) {
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> out(count);
    CUDA_CHECK(cudaMemcpy(out.data(), address, count, cudaMemcpyDeviceToHost));
    return out;
}

void require_bytes(const void* address, std::size_t count, unsigned char value, const char* label) {
    const auto bytes = copy_bytes(address, count);
    require(std::all_of(bytes.begin(), bytes.end(), [=](auto byte) { return byte == value; }),
            label);
}

void qualify(KvCacheStorage storage) {
    DeviceContext device;
    EngineOptions options;
    options.max_context                          = 256;
    options.max_concurrency                      = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 2;
    options.context_cache.host_kv_capacity_bytes =
        (storage == KvCacheStorage::RK4V4E8 ? 2ULL : 4ULL) * 1024 * 1024;
    options.kv_cache                             = storage;
    ik::StateImage state(options, 256);
    state.prepare_append(0, 0, 65, device.stream);
    state.commit_append(0, 65);
    auto original = table(state, 0, 2);
    require(state.used_pages() == 2, "initial two KV page groups");
    for (int layer = 0; layer < 8; ++layer) {
        const auto view = state.full(0, layer);
        for (const Tensor* plane :
             {&view.k_pages, &view.v_pages, &view.k_scale_pages, &view.v_scale_pages}) {
            if (!plane->data) { continue; }
            for (int page : original) {
                CUDA_CHECK(cudaMemsetAsync(page_address(*plane, page), 0x30 + layer,
                                           static_cast<std::size_t>(plane->nb[3]), device.stream));
            }
        }
    }
    // A distinct pattern at every visit/component catches offset and copy-extent mistakes.
    for (int visit = 0; visit < 48; ++visit) {
        for (int component = 0; component < 4; ++component) {
            const auto view = state.convolution(0, visit, component);
            CUDA_CHECK(
                cudaMemsetAsync(view.data, 1 + visit * 4 + component, view.bytes(), device.stream));
        }
    }
    for (int layer = 0; layer < 40; ++layer) {
        const auto view = state.swa(layer);
        CUDA_CHECK(cudaMemsetAsync(view.k.data, 0x20 + layer,
                                   static_cast<std::size_t>(view.k.nb[3]), device.stream));
        CUDA_CHECK(cudaMemsetAsync(view.v.data, 0x70 + layer,
                                   static_cast<std::size_t>(view.v.nb[3]), device.stream));
    }
    require(state.can_snapshot_to_host(0), "host arena can retain complete source image");
    auto host = state.snapshot_to_host(0, device.stream);
    {
        const auto obs = state.take_transfer_observations();
        require(obs.size() == 2, "host snapshot records MainKV and State D2H observations");
        const auto kv_w = state.host_kv_transfer_work(2);
        require(obs[0].resource == runtime::ContextResourceClass::MainKV &&
                    obs[0].direction == runtime::ContextTransferDirection::DeviceToHost &&
                    obs[0].units == kv_w.payload_bytes && obs[0].page_count == 2 &&
                    obs[0].work == kv_w,
                "host snapshot MainKV D2H observation matches transfer work");
        require(obs[1].resource == runtime::ContextResourceClass::State &&
                    obs[1].direction == runtime::ContextTransferDirection::DeviceToHost &&
                    obs[1].units == 1 && obs[1].page_count == 0 &&
                    obs[1].work == state.fixed_state_transfer_work() &&
                    obs[1].work.copy_operations == 81,
                "host snapshot State D2H observation records count and payload bytes");
    }
    require(host->depth == 65 && host->pages.page_count() == 2 && state.host_state_count() == 1 &&
                state.host_kv_bytes() > 0 && state.depth(0) == 65 && state.used_pages() == 2,
            "host snapshot retains exact frontier without changing source ownership");
    require(!state.can_snapshot_to_host(0),
            "bounded host KV arena refuses a second complete image");
    rejects([&] { (void)state.snapshot_to_host(0, device.stream); },
            "host KV capacity overflow must reject");
    rejects([&] { state.restore_from_host(0, *host, device.stream); },
            "host restore must reject occupied destination");
    state.copy(0, 1, device.stream);
    {
        const auto obs = state.take_transfer_observations();
        require(obs.size() == 1 && obs[0].resource == runtime::ContextResourceClass::State &&
                    obs[0].direction == runtime::ContextTransferDirection::DeviceToDevice &&
                    obs[0].units == 1 && obs[0].work == state.fixed_state_transfer_work(),
                "device snapshot copy records State D2D observation");
    }
    require(state.depth(1) == 65 && state.used_pages() == 2 && table(state, 1, 2) == original,
            "snapshot shares paged prefix without allocating KV copies");
    const std::uint32_t only_source[]      = {0};
    const std::uint32_t both_slots[]       = {0, 1};
    const std::uint32_t duplicate_source[] = {0, 0};
    require(
        state.reclaimable_pages(only_source) == 0 && state.reclaimable_pages(both_slots) == 2 &&
            state.reclaimable_pages(duplicate_source) == 0,
        "pressure projection counts shared page reclamation only after every owner is selected");
    for (int visit = 0; visit < 48; ++visit) {
        for (int component = 0; component < 4; ++component) {
            const auto view = state.convolution(1, visit, component);
            require_bytes(view.data, view.bytes(),
                          static_cast<unsigned char>(1 + visit * 4 + component),
                          "snapshot convolution state is complete and independently addressed");
        }
    }
    for (int layer = 0; layer < 40; ++layer) {
        const auto view = state.swa(layer);
        require_bytes(static_cast<std::byte*>(view.k.data) + view.k.nb[3],
                      static_cast<std::size_t>(view.k.nb[3]),
                      static_cast<unsigned char>(0x20 + layer),
                      "snapshot preserves complete SWA K ring");
        require_bytes(static_cast<std::byte*>(view.v.data) + view.v.nb[3],
                      static_cast<std::size_t>(view.v.nb[3]),
                      static_cast<unsigned char>(0x70 + layer),
                      "snapshot preserves complete SWA V ring");
    }
    state.prepare_append(0, 65, 1, device.stream);
    {
        const auto obs   = state.take_transfer_observations();
        const auto cow_w = state.device_kv_cow_transfer_work();
        require(obs.size() == 1 && obs[0].resource == runtime::ContextResourceClass::MainKV &&
                    obs[0].direction == runtime::ContextTransferDirection::DeviceToDevice &&
                    obs[0].units == cow_w.payload_bytes && obs[0].page_count == 1 &&
                    obs[0].work == cow_w,
                "partial-tail COW records MainKV D2D observation");
    }
    state.commit_append(0, 66);
    const auto forked = table(state, 0, 2);
    require(forked[0] == original[0] && forked[1] != original[1] && state.used_pages() == 3,
            "fork retains complete shared page and copies only shared partial tail");
    require(state.reclaimable_pages(only_source) == 1 && state.reclaimable_pages(both_slots) == 3,
            "pressure projection distinguishes private tails from complete shared pages");
    for (int layer = 0; layer < 8; ++layer) {
        const auto view = state.full(0, layer);
        for (const Tensor* plane :
             {&view.k_pages, &view.v_pages, &view.k_scale_pages, &view.v_scale_pages}) {
            if (!plane->data) { continue; }
            require_bytes(page_address(*plane, forked[1]), static_cast<std::size_t>(plane->nb[3]),
                          static_cast<unsigned char>(0x30 + layer),
                          "COW copies every layer and codec plane");
            CUDA_CHECK(cudaMemsetAsync(page_address(*plane, forked[1]), 0x55,
                                       static_cast<std::size_t>(plane->nb[3]), device.stream));
            require_bytes(page_address(*plane, original[1]), static_cast<std::size_t>(plane->nb[3]),
                          static_cast<unsigned char>(0x30 + layer),
                          "fork writes preserve retained page bytes");
        }
    }
    // Once the tail is private, extending within it must allocate zero additional pages.
    state.prepare_append(0, 66, 1, device.stream);
    state.commit_append(0, 67);
    require(state.used_pages() == 3 && table(state, 0, 2) == forked,
            "private partial-page decode needs no reservation");
    const auto conv = state.convolution(0, 47, 3);
    CUDA_CHECK(cudaMemsetAsync(conv.data, 0x55, conv.bytes(), device.stream));
    const auto retained_conv = state.convolution(1, 47, 3);
    require_bytes(retained_conv.data, retained_conv.bytes(), 192,
                  "fork convolution writes preserve retained image");
    const auto ring = state.swa(39);
    CUDA_CHECK(
        cudaMemsetAsync(ring.k.data, 0x55, static_cast<std::size_t>(ring.k.nb[3]), device.stream));
    require_bytes(static_cast<std::byte*>(ring.k.data) + ring.k.nb[3],
                  static_cast<std::size_t>(ring.k.nb[3]), 0x20 + 39,
                  "fork SWA writes preserve retained image");
    state.reset(1, device.stream);
    require(state.depth(1) == 0 && state.used_pages() == 2,
            "reset releases only snapshot-owned partial page");
    const auto reset_conv = state.convolution(1, 47, 3);
    require_bytes(reset_conv.data, reset_conv.bytes(), 0, "reset clears convolution history");
    require(state.depth(0) == 67 && table(state, 0, 2) == forked, "reset preserves active state");
    state.reset(0, device.stream);
    require(state.used_pages() == 0, "all KV leases released after both states reset");
    state.restore_from_host(1, *host, device.stream);
    {
        const auto obs = state.take_transfer_observations();
        require(obs.size() == 2, "host restore records MainKV and State H2D observations");
        const auto kv_w = state.host_kv_transfer_work(2);
        require(obs[0].resource == runtime::ContextResourceClass::MainKV &&
                    obs[0].direction == runtime::ContextTransferDirection::HostToDevice &&
                    obs[0].units == kv_w.payload_bytes && obs[0].page_count == 2 &&
                    obs[0].work == kv_w,
                "host restore MainKV H2D observation matches transfer work");
        require(obs[1].resource == runtime::ContextResourceClass::State &&
                    obs[1].direction == runtime::ContextTransferDirection::HostToDevice &&
                    obs[1].units == 1 && obs[1].page_count == 0 &&
                    obs[1].work == state.fixed_state_transfer_work(),
                "host restore State H2D observation records count and payload bytes");
    }
    require(state.depth(1) == 65 && state.used_pages() == 2 && state.host_state_count() == 1,
            "host restore reconstitutes an empty destination and retains host source");
    const auto restored = table(state, 1, 2);
    for (int layer = 0; layer < 8; ++layer) {
        const auto view = state.full(1, layer);
        for (const Tensor* plane :
             {&view.k_pages, &view.v_pages, &view.k_scale_pages, &view.v_scale_pages}) {
            if (!plane->data) { continue; }
            for (int page : restored) {
                require_bytes(page_address(*plane, page), static_cast<std::size_t>(plane->nb[3]),
                              static_cast<unsigned char>(0x30 + layer),
                              "host roundtrip preserves all paged codec planes");
            }
        }
    }
    for (int visit = 0; visit < 48; ++visit) {
        for (int component = 0; component < 4; ++component) {
            const auto view = state.convolution(1, visit, component);
            require_bytes(view.data, view.bytes(),
                          static_cast<unsigned char>(1 + visit * 4 + component),
                          "host roundtrip preserves every convolution block");
        }
    }
    for (int layer = 0; layer < 40; ++layer) {
        const auto view = state.swa(layer);
        require_bytes(static_cast<std::byte*>(view.k.data) + view.k.nb[3],
                      static_cast<std::size_t>(view.k.nb[3]),
                      static_cast<unsigned char>(0x20 + layer),
                      "host roundtrip preserves complete SWA K rings");
        require_bytes(static_cast<std::byte*>(view.v.data) + view.v.nb[3],
                      static_cast<std::size_t>(view.v.nb[3]),
                      static_cast<unsigned char>(0x70 + layer),
                      "host roundtrip preserves complete SWA V rings");
    }
    host.reset();
    require(state.host_state_count() == 0 && state.host_kv_bytes() == 0 &&
                state.can_snapshot_to_host(1),
            "expired host image releases state slot and KV arena allocation");
    state.reset(1, device.stream);
}

void qualify_pinned_slab_pool() {
    DeviceContext device;
    EngineOptions options;
    options.max_context                          = 256;
    options.max_concurrency                      = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 1;
    options.context_cache.host_kv_capacity_bytes = 2ULL * 1024 * 1024;
    options.kv_cache                             = KvCacheStorage::Int8Group64;
    bool saw_host_state_pin                      = false;
    StartupObserver observer;
    observer.callback = [&](const StartupEvent& ev) {
        if (ev.phase == StartupPhase::HostStatePin && ev.status == StartupStatus::Complete) {
            saw_host_state_pin = ev.current > 0 && ev.current == ev.total &&
                                 ev.progress_unit == StartupProgressUnit::Bytes;
        }
    };
    ik::StateImage state(options, 256, &observer);
    require(saw_host_state_pin, "StateImage emits StartupPhase::HostStatePin");
    state.prepare_append(0, 0, 16, device.stream);
    state.commit_append(0, 16);
    auto h1        = state.snapshot_to_host(0, device.stream);
    const void* p1 = h1->fixed.data();
    h1.reset();
    auto h2        = state.snapshot_to_host(0, device.stream);
    const void* p2 = h2->fixed.data();
    require(p1 != nullptr && p1 == p2, "pre-pinned HostStateImage slab is reused without cudaMallocHost");
}

void qualify_dflash2() {
    DeviceContext device;
    EngineOptions options;
    options.max_context                          = 256;
    options.max_concurrency                      = 1;
    options.kv_cache                             = KvCacheStorage::RK4V4E8;
    options.speculative.backend                  = SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens             = 7;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 1;
    options.context_cache.host_kv_capacity_bytes = 2ULL * 1024 * 1024;
    ik::StateImage state(options, 256);
    require(state.has_dflash2() && !state.has_dspark(), "DFlash2 state mode");
    require(state.dflash_kv_bytes() == 5ULL * 2ULL * 128ULL * 2048ULL * 8ULL * 2ULL * 2ULL,
            "DFlash2 2048-ring KV byte accounting");
    state.prepare_append(0, 0, 16, device.stream);
    state.commit_append(0, 16);
    state.commit_dflash2_append(0, 16);
    for (int l = 0; l < 5; ++l) {
        const auto view = state.dflash2_swa(l);
        CUDA_CHECK(cudaMemsetAsync(view.k.data, 0x40 + l,
                                   static_cast<std::size_t>(view.k.nb[3]), device.stream));
        CUDA_CHECK(cudaMemsetAsync(view.v.data, 0x50 + l,
                                   static_cast<std::size_t>(view.v.nb[3]), device.stream));
    }
    auto host = state.snapshot_to_host(0, device.stream);
    require(host->dflash2_depth == 16, "DFlash2 host snapshot depth");
    state.copy(0, 1, device.stream);
    require(state.dflash2_depth(1) == 16, "DFlash2 device copy depth");
    for (int l = 0; l < 5; ++l) {
        const auto view = state.dflash2_swa(l);
        require_bytes(static_cast<std::byte*>(view.k.data) + view.k.nb[3],
                      static_cast<std::size_t>(view.k.nb[3]), static_cast<unsigned char>(0x40 + l),
                      "DFlash2 copy preserves K ring");
        require_bytes(static_cast<std::byte*>(view.v.data) + view.v.nb[3],
                      static_cast<std::size_t>(view.v.nb[3]), static_cast<unsigned char>(0x50 + l),
                      "DFlash2 copy preserves V ring");
    }
    state.reset(1, device.stream);
    require(state.dflash2_depth(1) == 0, "DFlash2 reset clears depth");
    state.restore_from_host(1, *host, device.stream);
    require(state.dflash2_depth(1) == 16, "DFlash2 host restore restores depth");
    for (int l = 0; l < 5; ++l) {
        const auto view = state.dflash2_swa(l);
        require_bytes(static_cast<std::byte*>(view.k.data) + view.k.nb[3],
                      static_cast<std::size_t>(view.k.nb[3]), static_cast<unsigned char>(0x40 + l),
                      "DFlash2 host restore preserves K ring");
        require_bytes(static_cast<std::byte*>(view.v.data) + view.v.nb[3],
                      static_cast<std::size_t>(view.v.nb[3]), static_cast<unsigned char>(0x50 + l),
                      "DFlash2 host restore preserves V ring");
    }
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    try {
        qualify(KvCacheStorage::BFloat16);
        qualify(KvCacheStorage::Int8Group64);
        qualify(KvCacheStorage::RK4V4E8);
        qualify_dflash2();
        qualify_pinned_slab_pool();
        EngineOptions options;
        options.max_context                      = 131072;
        options.kv_cache                         = KvCacheStorage::Int8Group64;
        options.context_cache.device_state_slots = 1;
        const auto layout                        = ik::plan_state(options, 131072);
        require(layout.bytes < 2500ULL * 1024 * 1024,
                "128K state/KV allocation fits planned VRAM budget");
        options.kv_cache = KvCacheStorage::RK4V4E8;
        const auto rk4_layout = ik::plan_state(options, 131072);
        require(rk4_layout.bytes < 1450ULL * 1024 * 1024,
                "128K RK4V4E8 state/KV allocation fits planned 128K VRAM budget");
        std::cout << "PASS: Inkling BF16/int8/rk4v4-e8 KV page sharing/COW, device/host convolution and SWA "
                     "snapshots, DFlash2 ring, reset and 128K budget\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
