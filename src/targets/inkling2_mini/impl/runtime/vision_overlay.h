#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "targets/inkling2_mini/impl/load/bindings.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::targets::inkling2_mini::detail {

class GroupRebase {
public:
    GroupRebase(std::span<const std::byte> block, PinnedRange range, std::byte* target)
        : begin_(block.data() + range.offset), bytes_(range.bytes), target_(target) {}

    [[nodiscard]] const void* pointer(const void* source) const {
        if (source == nullptr) { return nullptr; }
        const auto* byte = static_cast<const std::byte*>(source);
        if (byte < begin_ || byte > begin_ + bytes_) {
            throw std::logic_error(
                "Inkling2 vision overlay weight does not address its pinned group");
        }
        return target_ + (byte - begin_);
    }

    void operator()(Weight& weight) const {
        weight.payload = pointer(weight.payload);
        weight.qdata   = pointer(weight.qdata);
        weight.qhigh   = pointer(weight.qhigh);
        weight.scales  = pointer(weight.scales);
    }

    void operator()(Tensor& tensor) const { tensor.data = const_cast<void*>(pointer(tensor.data)); }

private:
    const std::byte* begin_ = nullptr;
    std::size_t bytes_      = 0;
    std::byte* target_      = nullptr;
};

class VisionWeightStream {
public:
    VisionWeightStream(DeviceContext& device, const VisionOverlayLayout& layout,
                       std::span<const std::byte> pinned_block, std::byte* staging)
        : device_(device), layout_(layout), block_(pinned_block) {
        prelude_ = staging;
        merger_  = prelude_ + vision_staging_align(layout.prelude.bytes);
        slot_[0] = merger_ + vision_staging_align(layout.merger.bytes);
        slot_[1] = slot_[0] + vision_staging_align(layout.slot_bytes);

        try {
            for (cudaEvent_t& event : uploaded_) {
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
            for (cudaEvent_t& event : compute_fence_) {
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventCreateWithFlags(&prelude_event_, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&merger_event_, cudaEventDisableTiming));
        } catch (...) {
            destroy_events();
            throw;
        }
    }

    ~VisionWeightStream() { destroy_events(); }

    VisionWeightStream(const VisionWeightStream&)            = delete;
    VisionWeightStream& operator=(const VisionWeightStream&) = delete;

    [[nodiscard]] VisionWeights window_weights(const VisionWeights& host) const {
        VisionWeights out = host;
        const GroupRebase prelude(block_, layout_.prelude, prelude_);
        prelude(out.patch_embedding);
        prelude(out.class_embedding);
        prelude(out.cls_position_embedding);
        prelude(out.position_embedding);
        prelude(out.ln_pre_weight);
        prelude(out.ln_pre_bias);
        for (std::size_t index = 0; index < out.layers.size(); ++index) {
            const GroupRebase rebase(block_, layout_.layers[index], slot_[index % 2]);
            auto& layer = out.layers[index];
            rebase(layer.ln_1_weight);
            rebase(layer.ln_1_bias);
            rebase(layer.qkv);
            rebase(layer.qkv_bias);
            rebase(layer.output);
            rebase(layer.output_bias);
            rebase(layer.ls_1_gamma);
            rebase(layer.ln_2_weight);
            rebase(layer.ln_2_bias);
            rebase(layer.fc1);
            rebase(layer.fc1_bias);
            rebase(layer.fc2);
            rebase(layer.fc2_bias);
            rebase(layer.ls_2_gamma);
        }
        const GroupRebase merger(block_, layout_.merger, merger_);
        merger(out.proj_in);
        merger(out.proj_out);
        return out;
    }

    void reset(cudaStream_t compute) {
        const cudaStream_t copy = device_.transfer_stream;
        CUDA_CHECK(cudaEventRecord(compute_fence_[0], compute));
        CUDA_CHECK(cudaStreamWaitEvent(copy, compute_fence_[0], 0));
        CUDA_CHECK(cudaMemcpyAsync(prelude_, block_.data() + layout_.prelude.offset,
                                   layout_.prelude.bytes, cudaMemcpyHostToDevice, copy));
        CUDA_CHECK(cudaEventRecord(prelude_event_, copy));
        CUDA_CHECK(cudaMemcpyAsync(merger_, block_.data() + layout_.merger.offset,
                                   layout_.merger.bytes, cudaMemcpyHostToDevice, copy));
        CUDA_CHECK(cudaEventRecord(merger_event_, copy));
        next_upload_ = 0;
        upload_next_layer();
        upload_next_layer();
    }

    void prelude_ready(cudaStream_t compute) const {
        CUDA_CHECK(cudaStreamWaitEvent(compute, prelude_event_, 0));
    }

    void merger_ready(cudaStream_t compute) const {
        CUDA_CHECK(cudaStreamWaitEvent(compute, merger_event_, 0));
    }

    void begin_layer(std::uint32_t layer, cudaStream_t compute) const {
        CUDA_CHECK(cudaStreamWaitEvent(compute, uploaded_[layer % 2], 0));
    }

    void end_layer(std::uint32_t layer, cudaStream_t compute) {
        if (layer + 2 < layout_.layers.size()) {
            CUDA_CHECK(cudaEventRecord(compute_fence_[layer % 2], compute));
            CUDA_CHECK(cudaStreamWaitEvent(device_.transfer_stream, compute_fence_[layer % 2], 0));
            upload_next_layer();
        }
    }

private:
    void upload_next_layer() {
        if (next_upload_ >= layout_.layers.size()) { return; }
        const std::uint32_t layer = next_upload_++;
        const std::uint32_t slot  = layer % 2;
        const PinnedRange range   = layout_.layers[layer];
        const cudaStream_t copy   = device_.transfer_stream;
        CUDA_CHECK(cudaMemcpyAsync(slot_[slot], block_.data() + range.offset, range.bytes,
                                   cudaMemcpyHostToDevice, copy));
        CUDA_CHECK(cudaEventRecord(uploaded_[slot], copy));
    }

    void destroy_events() noexcept {
        for (cudaEvent_t& event : uploaded_) {
            if (event != nullptr) {
                (void)cudaEventDestroy(event);
                event = nullptr;
            }
        }
        for (cudaEvent_t& event : compute_fence_) {
            if (event != nullptr) {
                (void)cudaEventDestroy(event);
                event = nullptr;
            }
        }
        if (prelude_event_ != nullptr) {
            (void)cudaEventDestroy(prelude_event_);
            prelude_event_ = nullptr;
        }
        if (merger_event_ != nullptr) {
            (void)cudaEventDestroy(merger_event_);
            merger_event_ = nullptr;
        }
    }

    DeviceContext& device_;
    const VisionOverlayLayout& layout_;
    std::span<const std::byte> block_;
    std::byte* prelude_           = nullptr;
    std::byte* merger_            = nullptr;
    std::byte* slot_[2]           = {nullptr, nullptr};
    cudaEvent_t uploaded_[2]      = {nullptr, nullptr};
    cudaEvent_t compute_fence_[2] = {nullptr, nullptr};
    cudaEvent_t prelude_event_    = nullptr;
    cudaEvent_t merger_event_     = nullptr;
    std::uint32_t next_upload_    = 0;
};

} // namespace ninfer::targets::inkling2_mini::detail
