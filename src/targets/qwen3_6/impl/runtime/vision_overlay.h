#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_6/vision.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class GroupRebase {
public:
    GroupRebase(std::span<const std::byte> block, PinnedRange range, std::byte* target)
        : begin_(block.data() + range.offset), bytes_(range.bytes), target_(target) {}

    [[nodiscard]] const void* pointer(const void* source) const {
        if (source == nullptr) { return nullptr; }
        const auto* byte = static_cast<const std::byte*>(source);
        if (byte < begin_ || byte > begin_ + bytes_) {
            throw std::logic_error("Vision overlay weight does not address its pinned group");
        }
        return target_ + (byte - begin_);
    }

    void operator()(Weight& weight) const {
        weight.payload = pointer(weight.payload);
        weight.qdata   = pointer(weight.qdata);
        weight.qhigh   = pointer(weight.qhigh);
        weight.scales  = pointer(weight.scales);
    }

    void operator()(Tensor& tensor) const {
        tensor.data = const_cast<void*>(pointer(tensor.data));
    }

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
        merger_  = prelude_ + staging_align(layout.prelude.bytes);
        slot_[0] = merger_ + staging_align(layout.merger.bytes);
        slot_[1] = slot_[0] + staging_align(layout.slot_bytes);

        try {
            for (cudaEvent_t& event : uploaded_) {
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventCreateWithFlags(&prelude_event_, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&merger_event_, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&compute_fence_, cudaEventDisableTiming));

            const cudaStream_t copy = device_.transfer_stream;
            CUDA_CHECK(cudaMemcpyAsync(prelude_, block_.data() + layout.prelude.offset,
                                       layout.prelude.bytes, cudaMemcpyHostToDevice, copy));
            CUDA_CHECK(cudaEventRecord(prelude_event_, copy));
            CUDA_CHECK(cudaMemcpyAsync(merger_, block_.data() + layout.merger.offset,
                                       layout.merger.bytes, cudaMemcpyHostToDevice, copy));
            CUDA_CHECK(cudaEventRecord(merger_event_, copy));
        } catch (...) {
            destroy_events();
            throw;
        }
        upload_bytes_ = layout.prelude.bytes + layout.merger.bytes;
    }

    ~VisionWeightStream() {
        destroy_events();
    }

    VisionWeightStream(const VisionWeightStream&)            = delete;
    VisionWeightStream& operator=(const VisionWeightStream&) = delete;

    [[nodiscard]] VisionWeights window_weights(const VisionWeights& host) const {
        VisionWeights out = host;
        const GroupRebase prelude(block_, layout_.prelude, prelude_);
        prelude(out.common.patch_embedding);
        prelude(out.common.patch_embedding_bias);
        prelude(out.common.position_embedding);
        for (std::size_t index = 0; index < out.common.layers.size(); ++index) {
            const GroupRebase rebase(block_, layout_.layers[index], slot_[index % 2]);
            auto& layer = out.common.layers[index];
            rebase(layer.norm1_weight);
            rebase(layer.norm1_bias);
            rebase(layer.norm2_weight);
            rebase(layer.norm2_bias);
            rebase(layer.qkv);
            rebase(layer.qkv_bias);
            rebase(layer.output);
            rebase(layer.output_bias);
            rebase(layer.fc1);
            rebase(layer.fc1_bias);
            rebase(layer.fc2);
            rebase(layer.fc2_bias);
        }
        const GroupRebase merger(block_, layout_.merger, merger_);
        merger(out.common.merger_norm_weight);
        merger(out.common.merger_norm_bias);
        merger(out.common.merger_fc1);
        merger(out.common.merger_fc1_bias);
        merger(out.merger_fc2);
        merger(out.merger_fc2_bias);
        return out;
    }

    void reset(cudaStream_t compute) {
        CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
        CUDA_CHECK(cudaStreamWaitEvent(device_.transfer_stream, compute_fence_, 0));
        next_upload_ = 0;
        upload_next_layer();
        upload_next_layer();
    }

    void prelude_ready(cudaStream_t compute) {
        CUDA_CHECK(cudaStreamWaitEvent(compute, prelude_event_, 0));
    }

    void merger_ready(cudaStream_t compute) {
        CUDA_CHECK(cudaStreamWaitEvent(compute, merger_event_, 0));
    }

    void arrive(std::uint32_t layer, cudaStream_t compute) {
        CUDA_CHECK(cudaStreamWaitEvent(compute, uploaded_[layer % 2], 0));
        if (layer + 2 < layout_.layers.size()) {
            CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
            CUDA_CHECK(cudaStreamWaitEvent(device_.transfer_stream, compute_fence_, 0));
            upload_next_layer();
        }
    }

    [[nodiscard]] std::size_t uploaded_bytes() const noexcept { return upload_bytes_; }

private:
    void upload_next_layer() {
        if (next_upload_ >= layout_.layers.size()) { return; }
        const std::uint32_t layer = next_upload_++;
        const PinnedRange range   = layout_.layers[layer];
        CUDA_CHECK(cudaMemcpyAsync(slot_[layer % 2], block_.data() + range.offset, range.bytes,
                                   cudaMemcpyHostToDevice, device_.transfer_stream));
        CUDA_CHECK(cudaEventRecord(uploaded_[layer % 2], device_.transfer_stream));
        upload_bytes_ += range.bytes;
    }

    void destroy_events() noexcept {
        for (cudaEvent_t& event : uploaded_) {
            if (event != nullptr) { (void)cudaEventDestroy(event); event = nullptr; }
        }
        if (prelude_event_ != nullptr) { (void)cudaEventDestroy(prelude_event_); prelude_event_ = nullptr; }
        if (merger_event_ != nullptr) { (void)cudaEventDestroy(merger_event_); merger_event_ = nullptr; }
        if (compute_fence_ != nullptr) { (void)cudaEventDestroy(compute_fence_); compute_fence_ = nullptr; }
    }

    DeviceContext& device_;
    const VisionOverlayLayout& layout_;
    std::span<const std::byte> block_;
    std::byte* prelude_        = nullptr;
    std::byte* merger_         = nullptr;
    std::byte* slot_[2]        = {nullptr, nullptr};
    cudaEvent_t uploaded_[2]   = {nullptr, nullptr};
    cudaEvent_t prelude_event_ = nullptr;
    cudaEvent_t merger_event_  = nullptr;
    cudaEvent_t compute_fence_ = nullptr;
    std::uint32_t next_upload_ = 0;
    std::size_t upload_bytes_  = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
