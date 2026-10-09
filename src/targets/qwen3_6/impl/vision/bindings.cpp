#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6 {

VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder,
                                        artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };

    VisionBackbonePlan out;
    out.patch_embedding = bind("vision/patch_embedding", NumericFormat::Q6G64_F16S,
                               {VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim});
    out.patch_embedding_bias =
        bind("vision/patch_embedding_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding =
        bind("vision/position_embedding", NumericFormat::BF16,
             {VisionBackboneConfig::position_embeddings, VisionBackboneConfig::hidden});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        VisionLayerPlan& target  = out.layers[layer];
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv               = bind(prefix + "attention/qkv", NumericFormat::Q4G64_F16S,
                                        {3 * VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.qkv_bias          = bind(prefix + "attention/qkv_bias", NumericFormat::BF16,
                                        {3 * VisionBackboneConfig::hidden});
        target.output            = bind(prefix + "attention/output", NumericFormat::Q5G64_F16S,
                                        {VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.output_bias       = bind(prefix + "attention/output_bias", NumericFormat::BF16,
                                        {VisionBackboneConfig::hidden});
        target.fc1               = bind(prefix + "mlp/fc1", NumericFormat::Q4G64_F16S,
                                        {VisionBackboneConfig::intermediate, VisionBackboneConfig::hidden});
        target.fc1_bias          = bind(prefix + "mlp/fc1_bias", NumericFormat::BF16,
                                        {VisionBackboneConfig::intermediate});
        target.fc2               = bind(prefix + "mlp/fc2", NumericFormat::Q5G64_F16S,
                                        {VisionBackboneConfig::hidden, VisionBackboneConfig::intermediate});
        target.fc2_bias =
            bind(prefix + "mlp/fc2_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_weight =
            bind(prefix + "norm1/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_bias =
            bind(prefix + "norm1/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_weight =
            bind(prefix + "norm2/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_bias =
            bind(prefix + "norm2/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    }
    return out;
}

VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder,
                                               artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerInputPlan{
        .fc1      = bind("vision/merger/fc1", NumericFormat::W8G32_F16S,
                         {VisionBackboneConfig::merger_hidden, VisionBackboneConfig::merger_hidden}),
        .fc1_bias = bind("vision/merger/fc1_bias", NumericFormat::BF16,
                         {VisionBackboneConfig::merger_hidden}),
    };
}

VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder,
                                             artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerNormPlan{
        .weight =
            bind("vision/merger/norm/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden}),
        .bias =
            bind("vision/merger/norm/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden}),
    };
}

VisionCommonWeights materialize_vision_common(const artifact::MaterializedArtifact& materialized,
                                              const VisionBackbonePlan& backbone,
                                              const VisionMergerInputPlan& merger_input,
                                              const VisionMergerNormPlan& merger_norm) {
    using artifact::NumericFormat;

    VisionCommonWeights out;
    out.patch_embedding = artifact::materialized_weight(
        materialized, backbone.patch_embedding, NumericFormat::Q6G64_F16S,
        VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim);
    out.patch_embedding_bias =
        artifact::materialized_tensor(materialized, backbone.patch_embedding_bias,
                                      NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding = artifact::materialized_tensor(
        materialized, backbone.position_embedding, NumericFormat::BF16,
        {VisionBackboneConfig::hidden, VisionBackboneConfig::position_embeddings});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const VisionLayerPlan& source = backbone.layers[layer];
        VisionLayerWeights& target    = out.layers[layer];
        target.qkv                    = artifact::materialized_weight(
            materialized, source.qkv, NumericFormat::Q4G64_F16S, 3 * VisionBackboneConfig::hidden,
            VisionBackboneConfig::hidden);
        target.qkv_bias = artifact::materialized_tensor(
            materialized, source.qkv_bias, NumericFormat::BF16, {3 * VisionBackboneConfig::hidden});
        target.output = artifact::materialized_weight(
            materialized, source.output, NumericFormat::Q5G64_F16S, VisionBackboneConfig::hidden,
            VisionBackboneConfig::hidden);
        target.output_bias = artifact::materialized_tensor(
            materialized, source.output_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.fc1 = artifact::materialized_weight(
            materialized, source.fc1, NumericFormat::Q4G64_F16S, VisionBackboneConfig::intermediate,
            VisionBackboneConfig::hidden);
        target.fc1_bias =
            artifact::materialized_tensor(materialized, source.fc1_bias, NumericFormat::BF16,
                                          {VisionBackboneConfig::intermediate});
        target.fc2 = artifact::materialized_weight(
            materialized, source.fc2, NumericFormat::Q5G64_F16S, VisionBackboneConfig::hidden,
            VisionBackboneConfig::intermediate);
        target.fc2_bias = artifact::materialized_tensor(
            materialized, source.fc2_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_weight = artifact::materialized_tensor(
            materialized, source.norm1_weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_bias = artifact::materialized_tensor(
            materialized, source.norm1_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_weight = artifact::materialized_tensor(
            materialized, source.norm2_weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_bias = artifact::materialized_tensor(
            materialized, source.norm2_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    }

    out.merger_fc1 = artifact::materialized_weight(
        materialized, merger_input.fc1, NumericFormat::W8G32_F16S,
        VisionBackboneConfig::merger_hidden, VisionBackboneConfig::merger_hidden);
    out.merger_fc1_bias =
        artifact::materialized_tensor(materialized, merger_input.fc1_bias, NumericFormat::BF16,
                                      {VisionBackboneConfig::merger_hidden});
    out.merger_norm_weight = artifact::materialized_tensor(
        materialized, merger_norm.weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.merger_norm_bias = artifact::materialized_tensor(
        materialized, merger_norm.bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    return out;
}

namespace {

void update_range(const void* ptr, std::size_t bytes, const std::byte* base, std::size_t block_size,
                  std::size_t& min_offset, std::size_t& max_end) {
    if (ptr == nullptr || bytes == 0) { return; }
    const auto* byte_ptr = static_cast<const std::byte*>(ptr);
    if (byte_ptr < base || byte_ptr + bytes > base + block_size) {
        throw std::logic_error("Vision overlay weight is outside the pinned block");
    }
    const auto offset = static_cast<std::size_t>(byte_ptr - base);
    min_offset = std::min(min_offset, offset);
    max_end    = std::max(max_end, offset + bytes);
}

void update_weight_range(const Weight& w, const std::byte* base, std::size_t block_size,
                         std::size_t& min_offset, std::size_t& max_end) {
    update_range(w.payload, static_cast<std::size_t>(w.payload_bytes), base, block_size, min_offset, max_end);
}

void update_tensor_range(const Tensor& t, const std::byte* base, std::size_t block_size,
                         std::size_t& min_offset, std::size_t& max_end) {
    update_range(t.data, t.bytes(), base, block_size, min_offset, max_end);
}

PinnedRange make_range(std::size_t min_offset, std::size_t max_end) {
    if (max_end <= min_offset) {
        throw std::logic_error("Vision overlay group has no valid weights");
    }
    return PinnedRange{min_offset, max_end - min_offset};
}

} // namespace

VisionOverlayLayout vision_overlay_layout(const VisionWeights& weights,
                                          std::span<const std::byte> pinned_block) {
    const auto* base = pinned_block.data();
    const auto size  = pinned_block.size();
    if (base == nullptr || size == 0) {
        throw std::invalid_argument("vision_overlay_layout requires a non-empty pinned block");
    }

    VisionOverlayLayout out;
    std::size_t min_off = std::numeric_limits<std::size_t>::max();
    std::size_t max_end = 0;
    update_weight_range(weights.common.patch_embedding, base, size, min_off, max_end);
    update_tensor_range(weights.common.patch_embedding_bias, base, size, min_off, max_end);
    update_tensor_range(weights.common.position_embedding, base, size, min_off, max_end);
    out.prelude = make_range(min_off, max_end);

    out.layers.reserve(weights.common.layers.size());
    for (const auto& layer : weights.common.layers) {
        min_off = std::numeric_limits<std::size_t>::max();
        max_end = 0;
        update_tensor_range(layer.norm1_weight, base, size, min_off, max_end);
        update_tensor_range(layer.norm1_bias, base, size, min_off, max_end);
        update_tensor_range(layer.norm2_weight, base, size, min_off, max_end);
        update_tensor_range(layer.norm2_bias, base, size, min_off, max_end);
        update_weight_range(layer.qkv, base, size, min_off, max_end);
        update_tensor_range(layer.qkv_bias, base, size, min_off, max_end);
        update_weight_range(layer.output, base, size, min_off, max_end);
        update_tensor_range(layer.output_bias, base, size, min_off, max_end);
        update_weight_range(layer.fc1, base, size, min_off, max_end);
        update_tensor_range(layer.fc1_bias, base, size, min_off, max_end);
        update_weight_range(layer.fc2, base, size, min_off, max_end);
        update_tensor_range(layer.fc2_bias, base, size, min_off, max_end);
        out.layers.push_back(make_range(min_off, max_end));
        out.slot_bytes = std::max(out.slot_bytes, out.layers.back().bytes);
    }

    min_off = std::numeric_limits<std::size_t>::max();
    max_end = 0;
    update_tensor_range(weights.common.merger_norm_weight, base, size, min_off, max_end);
    update_tensor_range(weights.common.merger_norm_bias, base, size, min_off, max_end);
    update_weight_range(weights.common.merger_fc1, base, size, min_off, max_end);
    update_tensor_range(weights.common.merger_fc1_bias, base, size, min_off, max_end);
    update_weight_range(weights.merger_fc2, base, size, min_off, max_end);
    update_tensor_range(weights.merger_fc2_bias, base, size, min_off, max_end);
    out.merger = make_range(min_off, max_end);

    out.staging_bytes = staging_align(out.prelude.bytes) + staging_align(out.merger.bytes) +
                        2 * staging_align(out.slot_bytes);
    return out;
}

} // namespace ninfer::targets::qwen3_6
