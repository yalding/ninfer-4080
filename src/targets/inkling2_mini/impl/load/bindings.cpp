#include "targets/inkling2_mini/impl/load/bindings.h"
#include "artifact/typed_binding.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::targets::inkling2_mini::detail {
using artifact::NumericFormat;

namespace {
void update_range(const void* ptr, std::size_t bytes, const std::byte* base, std::size_t block_size,
                  std::size_t& min_offset, std::size_t& max_end) {
    if (ptr == nullptr || bytes == 0) { return; }
    const auto* byte_ptr = static_cast<const std::byte*>(ptr);
    if (byte_ptr < base || byte_ptr + bytes > base + block_size) {
        throw std::logic_error("Inkling2 vision overlay weight is outside the pinned block");
    }
    const auto offset = static_cast<std::size_t>(byte_ptr - base);
    min_offset        = std::min(min_offset, offset);
    max_end           = std::max(max_end, offset + bytes);
}

void update_weight_range(const Weight& w, const std::byte* base, std::size_t block_size,
                         std::size_t& min_offset, std::size_t& max_end) {
    update_range(w.payload, static_cast<std::size_t>(w.payload_bytes), base, block_size, min_offset,
                 max_end);
}

void update_tensor_range(const Tensor& t, const std::byte* base, std::size_t block_size,
                         std::size_t& min_offset, std::size_t& max_end) {
    update_range(t.data, t.bytes(), base, block_size, min_offset, max_end);
}

PinnedRange make_range(std::size_t min_offset, std::size_t max_end) {
    if (max_end <= min_offset) {
        throw std::logic_error("Inkling2 vision overlay group has no valid weights");
    }
    return PinnedRange{min_offset, max_end - min_offset};
}

VisionOverlayLayout compute_vision_overlay_layout(const VisionWeights& weights,
                                                  std::span<const std::byte> pinned_block) {
    const auto* base = pinned_block.data();
    const auto size  = pinned_block.size();
    if (base == nullptr || size == 0) {
        throw std::invalid_argument("vision_overlay_layout requires a non-empty pinned block");
    }
    VisionOverlayLayout out;
    std::size_t min_off = std::numeric_limits<std::size_t>::max();
    std::size_t max_end = 0;
    update_tensor_range(weights.patch_embedding, base, size, min_off, max_end);
    update_tensor_range(weights.class_embedding, base, size, min_off, max_end);
    update_tensor_range(weights.cls_position_embedding, base, size, min_off, max_end);
    update_tensor_range(weights.position_embedding, base, size, min_off, max_end);
    update_tensor_range(weights.ln_pre_weight, base, size, min_off, max_end);
    update_tensor_range(weights.ln_pre_bias, base, size, min_off, max_end);
    out.prelude = make_range(min_off, max_end);

    out.layers.reserve(weights.layers.size());
    for (const auto& layer : weights.layers) {
        min_off = std::numeric_limits<std::size_t>::max();
        max_end = 0;
        update_tensor_range(layer.ln_1_weight, base, size, min_off, max_end);
        update_tensor_range(layer.ln_1_bias, base, size, min_off, max_end);
        update_weight_range(layer.qkv, base, size, min_off, max_end);
        update_tensor_range(layer.qkv_bias, base, size, min_off, max_end);
        update_weight_range(layer.output, base, size, min_off, max_end);
        update_tensor_range(layer.output_bias, base, size, min_off, max_end);
        update_tensor_range(layer.ls_1_gamma, base, size, min_off, max_end);
        update_tensor_range(layer.ln_2_weight, base, size, min_off, max_end);
        update_tensor_range(layer.ln_2_bias, base, size, min_off, max_end);
        update_weight_range(layer.fc1, base, size, min_off, max_end);
        update_tensor_range(layer.fc1_bias, base, size, min_off, max_end);
        update_weight_range(layer.fc2, base, size, min_off, max_end);
        update_tensor_range(layer.fc2_bias, base, size, min_off, max_end);
        update_tensor_range(layer.ls_2_gamma, base, size, min_off, max_end);
        out.layers.push_back(make_range(min_off, max_end));
        out.slot_bytes = std::max(out.slot_bytes, out.layers.back().bytes);
    }

    min_off = std::numeric_limits<std::size_t>::max();
    max_end = 0;
    update_weight_range(weights.proj_in, base, size, min_off, max_end);
    update_weight_range(weights.proj_out, base, size, min_off, max_end);
    out.merger = make_range(min_off, max_end);

    out.staging_bytes = vision_staging_align(out.prelude.bytes) +
                        vision_staging_align(out.merger.bytes) +
                        2 * vision_staging_align(out.slot_bytes);
    return out;
}
} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, bool load_mtp, bool load_draft_head,
                               bool load_dflash2, bool enable_vision,
                               VisionResidency vision_residency) {
    ArtifactLoadPlan out;
    auto& plan           = out.bindings;
    plan.has_mtp         = load_mtp;
    plan.has_dflash2     = load_dflash2;
    plan.has_vision      = enable_vision;
    plan.vision_overlay  = enable_vision && vision_residency == VisionResidency::Overlay;
    plan.frontend        = bind_frontend_resources(binder);
    plan.token_embedding = artifact::bind_device_tensor(binder, "text/token_embedding",
                                                        NumericFormat::Q4G64_F16S, {201024, 2304});
    plan.token_norm =
        artifact::bind_device_tensor(binder, "text/token_norm", NumericFormat::BF16, {2304});
    for (std::size_t i = 0; i < kPhysicalLayers; ++i) {
        auto& layer           = plan.layers[i];
        const auto prefix     = "text/layers/" + std::to_string(i) + "/";
        const auto moe_format = i < 20 ? NumericFormat::Q3G128_F16S : NumericFormat::Q4G64_F16S;
        if (!is_full_attention_layer(i)) {
            layer.attention_input_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/input_norm", NumericFormat::BF16, {2304});
        }
        layer.attention_qgr = artifact::bind_device_tensor(binder, prefix + "attention/qgr",
                                                           NumericFormat::Q4G64_F16S, {6528, 2304});
        layer.attention_k   = artifact::bind_device_tensor(binder, prefix + "attention/k",
                                                           NumericFormat::Q4G64_F16S, {1024, 2304});
        layer.attention_v   = artifact::bind_device_tensor(binder, prefix + "attention/v",
                                                           NumericFormat::Q4G64_F16S, {1024, 2304});
        layer.attention_output = artifact::bind_device_tensor(
            binder, prefix + "attention/output", NumericFormat::Q4G64_F16S, {2304, 3072});
        layer.attention_sink_proj = artifact::bind_device_tensor(
            binder, prefix + "attention/sink_proj", NumericFormat::BF16, {24, 2304});
        layer.attention_q_clip   = artifact::bind_device_tensor(binder, prefix + "attention/q_clip",
                                                                NumericFormat::FP32, {24, 128});
        layer.attention_rel_clip = artifact::bind_device_tensor(
            binder, prefix + "attention/rel_clip", NumericFormat::FP32, {24, 16});
        layer.attention_rel_bias = artifact::bind_device_tensor(
            binder, prefix + "attention/rel_bias", NumericFormat::FP32, {1024, 16});
        layer.attention_k_conv = artifact::bind_device_tensor(binder, prefix + "attention/k_conv",
                                                              NumericFormat::FP32, {1024, 4});
        layer.attention_v_conv = artifact::bind_device_tensor(binder, prefix + "attention/v_conv",
                                                              NumericFormat::FP32, {1024, 4});
        layer.attention_post_conv = artifact::bind_device_tensor(
            binder, prefix + "attention/post_conv", NumericFormat::FP32, {2304, 4});
        layer.ffn_res_query      = artifact::bind_device_tensor(binder, prefix + "ffn/res_query",
                                                                NumericFormat::FP32, {2304});
        layer.ffn_input_norm     = artifact::bind_device_tensor(binder, prefix + "ffn/input_norm",
                                                                NumericFormat::BF16, {2304});
        layer.ffn_router         = artifact::bind_device_tensor(binder, prefix + "ffn/router",
                                                                NumericFormat::BF16, {128, 2304});
        layer.ffn_exp_probs_bias = artifact::bind_device_tensor(
            binder, prefix + "ffn/exp_probs_bias", NumericFormat::FP32, {128});
        layer.ffn_routed_gate_up = artifact::bind_device_tensor(
            binder, prefix + "ffn/routed_gate_up", moe_format, {294912, 2304});
        layer.ffn_routed_down = artifact::bind_device_tensor(binder, prefix + "ffn/routed_down",
                                                             moe_format, {294912, 1152});
        layer.ffn_post_conv   = artifact::bind_device_tensor(binder, prefix + "ffn/post_conv",
                                                             NumericFormat::FP32, {2304, 4});
        if (i > 0) {
            layer.attention_res_query = artifact::bind_device_tensor(
                binder, prefix + "attention/res_query", NumericFormat::FP32,
                {is_full_attention_layer(i) ? 3ULL : 1ULL, 2304});
        }
        if (is_full_attention_layer(i)) {
            layer.attention_q_input_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/q_input_norm", NumericFormat::BF16, {2304});
        }
        if (is_full_attention_layer(i)) {
            layer.attention_k_input_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/k_input_norm", NumericFormat::BF16, {2304});
        }
        if (is_full_attention_layer(i)) {
            layer.attention_v_input_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/v_input_norm", NumericFormat::BF16, {2304});
        }
        if (is_full_attention_layer(i)) {
            layer.attention_k_conv_proj = artifact::bind_device_tensor(
                binder, prefix + "attention/k_conv_proj", NumericFormat::BF16, {32, 2304});
        }
        if (is_full_attention_layer(i)) {
            layer.attention_v_conv_proj = artifact::bind_device_tensor(
                binder, prefix + "attention/v_conv_proj", NumericFormat::BF16, {32, 2304});
        }
        if (is_loop_layer(i)) {
            layer.loop_msg_query = artifact::bind_device_tensor(binder, prefix + "loop/msg_query",
                                                                NumericFormat::FP32, {4, 2304});
        }
        if (is_loop_layer(i)) {
            layer.loop_ctrl_down = artifact::bind_device_tensor(binder, prefix + "loop/ctrl_down",
                                                                NumericFormat::BF16, {4, 64, 4608});
        }
        if (is_loop_layer(i)) {
            layer.loop_ctrl_gain_forget = artifact::bind_device_tensor(
                binder, prefix + "loop/ctrl_gain_forget", NumericFormat::BF16, {4, 2, 2304, 64});
        }
    }
    plan.output_res_query =
        artifact::bind_device_tensor(binder, "text/output_res_query", NumericFormat::FP32, {2304});
    plan.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {2304});
    plan.output_head    = artifact::bind_device_tensor(binder, "text/output_head",
                                                       NumericFormat::Q6G64_F16S, {201024, 2304});
    if (load_mtp || binder.has_object("mtp/fuse_weight")) {
        const auto mtp_placement = load_mtp ? artifact::TensorPlacement::Device
                                            : artifact::TensorPlacement::ValidateOnly;
        auto& mtp                = plan.mtp;
        mtp.fuse_weight          = artifact::bind_tensor(binder, "mtp/fuse_weight",
                                                         NumericFormat::FP32, {4, 4}, mtp_placement);
        mtp.mask_embed           = artifact::bind_tensor(binder, "mtp/mask_embed",
                                                         NumericFormat::BF16, {2304}, mtp_placement);
        mtp.final_norm           = artifact::bind_tensor(binder, "mtp/final_norm",
                                                         NumericFormat::BF16, {2304}, mtp_placement);
        for (std::size_t i = 0; i < kMtpLayers; ++i) {
            auto& l        = mtp.layers[i];
            const auto pfx = "mtp/layers/" + std::to_string(i) + "/";
            l.dflare_norm  = artifact::bind_tensor(binder, pfx + "dflare_norm", NumericFormat::BF16,
                                                   {2304}, mtp_placement);
            l.input_norm   = artifact::bind_tensor(binder, pfx + "input_norm", NumericFormat::BF16,
                                                   {2304}, mtp_placement);
            l.post_attention_norm = artifact::bind_tensor(
                binder, pfx + "post_attention_norm", NumericFormat::BF16, {2304}, mtp_placement);
            l.q_norm     = artifact::bind_tensor(binder, pfx + "attention/q_norm",
                                                 NumericFormat::BF16, {128}, mtp_placement);
            l.k_norm     = artifact::bind_tensor(binder, pfx + "attention/k_norm",
                                                 NumericFormat::BF16, {128}, mtp_placement);
            l.k_norm_ctx = artifact::bind_tensor(binder, pfx + "attention/k_norm_ctx",
                                                 NumericFormat::BF16, {128}, mtp_placement);
            l.q          = artifact::bind_tensor(binder, pfx + "attention/q",
                                                 NumericFormat::Q4G64_F16S, {3072, 2304}, mtp_placement);
            l.k          = artifact::bind_tensor(binder, pfx + "attention/k",
                                                 NumericFormat::Q4G64_F16S, {1024, 2304}, mtp_placement);
            l.v          = artifact::bind_tensor(binder, pfx + "attention/v",
                                                 NumericFormat::Q4G64_F16S, {1024, 2304}, mtp_placement);
            l.k_ctx      = artifact::bind_tensor(binder, pfx + "attention/k_ctx",
                                                 NumericFormat::Q4G64_F16S, {1024, 2304}, mtp_placement);
            l.v_ctx      = artifact::bind_tensor(binder, pfx + "attention/v_ctx",
                                                 NumericFormat::Q4G64_F16S, {1024, 2304}, mtp_placement);
            l.output     = artifact::bind_tensor(binder, pfx + "attention/output",
                                                 NumericFormat::Q4G64_F16S, {2304, 3072}, mtp_placement);
            l.gate_up    = artifact::bind_tensor(binder, pfx + "mlp/gate_up",
                                                 NumericFormat::Q4G64_F16S, {18432, 2304}, mtp_placement);
            l.down       = artifact::bind_tensor(binder, pfx + "mlp/down",
                                                 NumericFormat::Q4G64_F16S, {2304, 9216}, mtp_placement);
        }
        mtp.markov_dflare_norm       = artifact::bind_tensor(binder, "mtp/markov/dflare_norm",
                                                             NumericFormat::BF16, {2304}, mtp_placement);
        mtp.markov_w1                = artifact::bind_tensor(binder, "mtp/markov/w1",
                                                             NumericFormat::BF16, {201024, 256}, mtp_placement);
        mtp.markov_rnn_proj_weight   = artifact::bind_tensor(binder, "mtp/markov/rnn_proj_weight",
                                                             NumericFormat::BF16, {768, 5120}, mtp_placement);
        mtp.markov_rnn_proj_bias     = artifact::bind_tensor(binder, "mtp/markov/rnn_proj_bias",
                                                             NumericFormat::FP32, {768}, mtp_placement);
        mtp.markov_hidden_out_weight = artifact::bind_tensor(binder, "mtp/markov/hidden_out_weight",
                                                             NumericFormat::BF16, {2304, 256}, mtp_placement);
        mtp.markov_hidden_out_bias   = artifact::bind_tensor(binder, "mtp/markov/hidden_out_bias",
                                                             NumericFormat::FP32, {2304}, mtp_placement);
        if (binder.has_object("mtp/draft_head")) {
            const auto draft_placement = (load_mtp && load_draft_head)
                                             ? artifact::TensorPlacement::Device
                                             : artifact::TensorPlacement::ValidateOnly;
            mtp.has_draft_head = load_mtp && load_draft_head;
            mtp.draft_head = artifact::bind_tensor(
                binder, "mtp/draft_head", NumericFormat::Q4G64_F16S, {32768, 2304}, draft_placement);
            mtp.draft_head_token_ids = artifact::bind_tensor(
                binder, "mtp/draft_head_token_ids", NumericFormat::I32, {32768}, draft_placement);
        }
    }
    if (load_dflash2 || binder.has_object("dflash2/fc")) {
        const auto d2_placement = load_dflash2 ? artifact::TensorPlacement::Device
                                               : artifact::TensorPlacement::ValidateOnly;
        auto& d2                = plan.dflash2;
        d2.fc = artifact::bind_tensor(binder, "dflash2/fc", NumericFormat::Q4G64_F16S,
                                      {2304, 9216}, d2_placement);
        d2.hidden_norm = artifact::bind_tensor(binder, "dflash2/hidden_norm", NumericFormat::BF16,
                                               {2304}, d2_placement);
        for (std::size_t i = 0; i < kDFlash2Layers; ++i) {
            auto& l        = d2.layers[i];
            const auto pfx = "dflash2/layers/" + std::to_string(i) + "/";
            l.input_norm   = artifact::bind_tensor(binder, pfx + "input_norm", NumericFormat::BF16,
                                                   {2304}, d2_placement);
            l.post_attention_norm = artifact::bind_tensor(
                binder, pfx + "post_attention_norm", NumericFormat::BF16, {2304}, d2_placement);
            l.attn_conv_base = artifact::bind_tensor(
                binder, pfx + "attention_conv/base_kernel", NumericFormat::BF16, {2, 2, 2304},
                d2_placement);
            l.attn_conv_proj = artifact::bind_tensor(
                binder, pfx + "attention_conv/kernel_projection", NumericFormat::BF16, {576, 2304},
                d2_placement);
            l.q_norm = artifact::bind_tensor(binder, pfx + "attention/q_norm", NumericFormat::BF16,
                                             {128}, d2_placement);
            l.k_norm = artifact::bind_tensor(binder, pfx + "attention/k_norm", NumericFormat::BF16,
                                             {128}, d2_placement);
            l.q      = artifact::bind_tensor(binder, pfx + "attention/q", NumericFormat::Q4G64_F16S,
                                             {4096, 2304}, d2_placement);
            l.k      = artifact::bind_tensor(binder, pfx + "attention/k", NumericFormat::Q4G64_F16S,
                                             {1024, 2304}, d2_placement);
            l.v      = artifact::bind_tensor(binder, pfx + "attention/v", NumericFormat::Q4G64_F16S,
                                             {1024, 2304}, d2_placement);
            l.output = artifact::bind_tensor(binder, pfx + "attention/output",
                                             NumericFormat::Q4G64_F16S, {2304, 4096}, d2_placement);
            l.mlp_conv_base = artifact::bind_tensor(
                binder, pfx + "mlp_conv/base_kernel", NumericFormat::BF16, {2, 2, 2304},
                d2_placement);
            l.mlp_conv_proj = artifact::bind_tensor(
                binder, pfx + "mlp_conv/kernel_projection", NumericFormat::BF16, {576, 2304},
                d2_placement);
            l.gate_up = artifact::bind_tensor(binder, pfx + "mlp/gate_up",
                                              NumericFormat::Q4G64_F16S, {13824, 2304}, d2_placement);
            l.down    = artifact::bind_tensor(binder, pfx + "mlp/down", NumericFormat::Q4G64_F16S,
                                              {2304, 6912}, d2_placement);
        }
        d2.final_norm = artifact::bind_tensor(binder, "dflash2/final_norm", NumericFormat::BF16,
                                              {2304}, d2_placement);
        d2.selector_hidden_proj = artifact::bind_tensor(
            binder, "dflash2/selector/hidden_projection", NumericFormat::BF16, {256, 2304},
            d2_placement);
        d2.predecessor_codebook = artifact::bind_tensor(
            binder, "dflash2/selector/predecessor_codebook", NumericFormat::BF16, {201024, 256},
            d2_placement);
        d2.successor_codebook = artifact::bind_tensor(
            binder, "dflash2/selector/successor_codebook", NumericFormat::BF16, {201024, 256},
            d2_placement);
    }
    if (enable_vision || binder.has_object("vision/patch_embedding")) {
        const auto vis_placement =
            !enable_vision ? artifact::TensorPlacement::ValidateOnly
                           : (vision_residency == VisionResidency::Overlay
                                  ? artifact::TensorPlacement::PinnedHost
                                  : artifact::TensorPlacement::Device);
        auto& v                  = plan.vision;
        v.patch_embedding        = artifact::bind_tensor(binder, "vision/patch_embedding",
                                                         NumericFormat::BF16, {1024, 1200}, vis_placement);
        v.class_embedding        = artifact::bind_tensor(binder, "vision/class_embedding",
                                                         NumericFormat::BF16, {1024}, vis_placement);
        v.cls_position_embedding = artifact::bind_tensor(binder, "vision/cls_position_embedding",
                                                         NumericFormat::BF16, {1024}, vis_placement);
        v.position_embedding     = artifact::bind_tensor(binder, "vision/position_embedding",
                                                         NumericFormat::BF16, {1024, 1024}, vis_placement);
        v.ln_pre_weight          = artifact::bind_tensor(binder, "vision/ln_pre/weight",
                                                         NumericFormat::BF16, {1024}, vis_placement);
        v.ln_pre_bias            = artifact::bind_tensor(binder, "vision/ln_pre/bias",
                                                         NumericFormat::BF16, {1024}, vis_placement);
        for (std::size_t i = 0; i < kVisionLayers; ++i) {
            auto& l        = v.layers[i];
            const auto pfx = "vision/layers/" + std::to_string(i) + "/";
            l.ln_1_weight  = artifact::bind_tensor(binder, pfx + "ln_1/weight", NumericFormat::BF16,
                                                   {1024}, vis_placement);
            l.ln_1_bias    = artifact::bind_tensor(binder, pfx + "ln_1/bias", NumericFormat::BF16,
                                                   {1024}, vis_placement);
            l.qkv          = artifact::bind_tensor(binder, pfx + "attention/qkv",
                                                   NumericFormat::Q4G64_F16S, {3072, 1024}, vis_placement);
            l.qkv_bias     = artifact::bind_tensor(binder, pfx + "attention/qkv_bias",
                                                   NumericFormat::BF16, {3072}, vis_placement);
            l.output       = artifact::bind_tensor(binder, pfx + "attention/output",
                                                   NumericFormat::Q5G64_F16S, {1024, 1024}, vis_placement);
            l.output_bias  = artifact::bind_tensor(binder, pfx + "attention/output_bias",
                                                   NumericFormat::BF16, {1024}, vis_placement);
            l.ls_1_gamma   = artifact::bind_tensor(binder, pfx + "ls_1/gamma", NumericFormat::BF16,
                                                   {1024}, vis_placement);
            l.ln_2_weight  = artifact::bind_tensor(binder, pfx + "ln_2/weight", NumericFormat::BF16,
                                                   {1024}, vis_placement);
            l.ln_2_bias    = artifact::bind_tensor(binder, pfx + "ln_2/bias", NumericFormat::BF16,
                                                   {1024}, vis_placement);
            l.fc1          = artifact::bind_tensor(binder, pfx + "mlp/fc1",
                                                   NumericFormat::Q4G64_F16S, {4096, 1024}, vis_placement);
            l.fc1_bias     = artifact::bind_tensor(binder, pfx + "mlp/fc1_bias",
                                                   NumericFormat::BF16, {4096}, vis_placement);
            l.fc2          = artifact::bind_tensor(binder, pfx + "mlp/fc2",
                                                   NumericFormat::Q5G64_F16S, {1024, 4096}, vis_placement);
            l.fc2_bias     = artifact::bind_tensor(binder, pfx + "mlp/fc2_bias",
                                                   NumericFormat::BF16, {1024}, vis_placement);
            l.ls_2_gamma   = artifact::bind_tensor(binder, pfx + "ls_2/gamma", NumericFormat::BF16,
                                                   {1024}, vis_placement);
        }
        v.proj_in  = artifact::bind_tensor(binder, "vision/merger/proj_in",
                                           NumericFormat::W8G32_F16S, {4096, 4096}, vis_placement);
        v.proj_out = artifact::bind_tensor(binder, "vision/merger/proj_out",
                                           NumericFormat::W8G32_F16S, {2304, 4096}, vis_placement);
    }
    out.materialization = binder.finish();
    return out;
}

RuntimeModelView materialize_model(artifact::MaterializedArtifact& materialized,
                                   const BindingPlan& plan) {
    RuntimeModelView out;
    out.weights_arena = &materialized.device_arena();
    out.token_embedding = artifact::materialized_weight(materialized, plan.token_embedding,
                                                        NumericFormat::Q4G64_F16S, 201024, 2304);
    out.token_norm =
        artifact::materialized_tensor(materialized, plan.token_norm, NumericFormat::BF16, {2304});
    for (std::size_t i = 0; i < kPhysicalLayers; ++i) {
        auto& layer           = out.layers[i];
        const auto& source    = plan.layers[i];
        layer.full_attention  = is_full_attention_layer(i);
        const auto moe_format = i < 20 ? NumericFormat::Q3G128_F16S : NumericFormat::Q4G64_F16S;
        if (!is_full_attention_layer(i)) {
            layer.attention_input_norm = artifact::materialized_tensor(
                materialized, source.attention_input_norm, NumericFormat::BF16, {2304});
        }
        layer.attention_qgr = artifact::materialized_weight(materialized, source.attention_qgr,
                                                            NumericFormat::Q4G64_F16S, 6528, 2304);
        layer.attention_k   = artifact::materialized_weight(materialized, source.attention_k,
                                                            NumericFormat::Q4G64_F16S, 1024, 2304);
        layer.attention_v   = artifact::materialized_weight(materialized, source.attention_v,
                                                            NumericFormat::Q4G64_F16S, 1024, 2304);
        layer.attention_output = artifact::materialized_weight(
            materialized, source.attention_output, NumericFormat::Q4G64_F16S, 2304, 3072);
        layer.attention_sink_proj = artifact::materialized_tensor(
            materialized, source.attention_sink_proj, NumericFormat::BF16, {2304, 24});
        layer.attention_q_clip = artifact::materialized_tensor(
            materialized, source.attention_q_clip, NumericFormat::FP32, {128, 24});
        layer.attention_rel_clip = artifact::materialized_tensor(
            materialized, source.attention_rel_clip, NumericFormat::FP32, {16, 24});
        layer.attention_rel_bias = artifact::materialized_tensor(
            materialized, source.attention_rel_bias, NumericFormat::FP32, {16, 1024});
        layer.attention_k_conv = artifact::materialized_tensor(
            materialized, source.attention_k_conv, NumericFormat::FP32, {4, 1024});
        layer.attention_v_conv = artifact::materialized_tensor(
            materialized, source.attention_v_conv, NumericFormat::FP32, {4, 1024});
        layer.attention_post_conv = artifact::materialized_tensor(
            materialized, source.attention_post_conv, NumericFormat::FP32, {4, 2304});
        layer.ffn_res_query  = artifact::materialized_tensor(materialized, source.ffn_res_query,
                                                             NumericFormat::FP32, {2304});
        layer.ffn_input_norm = artifact::materialized_tensor(materialized, source.ffn_input_norm,
                                                             NumericFormat::BF16, {2304});
        layer.ffn_router     = artifact::materialized_weight(materialized, source.ffn_router,
                                                             NumericFormat::BF16, 128, 2304);
        layer.ffn_exp_probs_bias = artifact::materialized_weight(
            materialized, source.ffn_exp_probs_bias, NumericFormat::FP32, 128, 1);
        layer.ffn_routed_gate_up = artifact::materialized_weight(
            materialized, source.ffn_routed_gate_up, moe_format, 294912, 2304);
        layer.ffn_routed_down = artifact::materialized_weight(materialized, source.ffn_routed_down,
                                                              moe_format, 294912, 1152);
        layer.ffn_post_conv   = artifact::materialized_tensor(materialized, source.ffn_post_conv,
                                                              NumericFormat::FP32, {4, 2304});
        if (i > 0) {
            layer.attention_res_query = artifact::materialized_tensor(
                materialized, source.attention_res_query, NumericFormat::FP32,
                {2304, is_full_attention_layer(i) ? 3 : 1});
        }
        if (is_full_attention_layer(i)) {
            layer.attention_q_input_norm = artifact::materialized_tensor(
                materialized, source.attention_q_input_norm, NumericFormat::BF16, {2304});
        }
        if (is_full_attention_layer(i)) {
            layer.attention_k_input_norm = artifact::materialized_tensor(
                materialized, source.attention_k_input_norm, NumericFormat::BF16, {2304});
        }
        if (is_full_attention_layer(i)) {
            layer.attention_v_input_norm = artifact::materialized_tensor(
                materialized, source.attention_v_input_norm, NumericFormat::BF16, {2304});
        }
        if (is_full_attention_layer(i)) {
            layer.attention_k_conv_proj = artifact::materialized_tensor(
                materialized, source.attention_k_conv_proj, NumericFormat::BF16, {2304, 32});
        }
        if (is_full_attention_layer(i)) {
            layer.attention_v_conv_proj = artifact::materialized_tensor(
                materialized, source.attention_v_conv_proj, NumericFormat::BF16, {2304, 32});
        }
        if (is_loop_layer(i)) {
            layer.loop_msg_query = artifact::materialized_tensor(
                materialized, source.loop_msg_query, NumericFormat::FP32, {2304, 4});
        }
        if (is_loop_layer(i)) {
            layer.loop_ctrl_down = artifact::materialized_tensor(
                materialized, source.loop_ctrl_down, NumericFormat::BF16, {4608, 64, 4});
        }
        if (is_loop_layer(i)) {
            layer.loop_ctrl_gain_forget = artifact::materialized_tensor(
                materialized, source.loop_ctrl_gain_forget, NumericFormat::BF16, {64, 2304, 2, 4});
        }
    }
    out.output_res_query = artifact::materialized_tensor(materialized, plan.output_res_query,
                                                         NumericFormat::FP32, {2304});
    out.final_norm =
        artifact::materialized_tensor(materialized, plan.final_norm, NumericFormat::BF16, {2304});
    out.output_head = artifact::materialized_weight(materialized, plan.output_head,
                                                    NumericFormat::Q6G64_F16S, 201024, 2304);
    out.has_mtp     = plan.has_mtp;
    if (plan.has_mtp) {
        const auto& src = plan.mtp;
        auto& dst       = out.mtp;
        dst.fuse_weight = artifact::materialized_tensor(materialized, src.fuse_weight, NumericFormat::FP32, {4, 4});
        dst.mask_embed  = artifact::materialized_tensor(materialized, src.mask_embed, NumericFormat::BF16, {2304});
        dst.final_norm  = artifact::materialized_tensor(materialized, src.final_norm, NumericFormat::BF16, {2304});
        for (std::size_t i = 0; i < kMtpLayers; ++i) {
            const auto& sl       = src.layers[i];
            auto& dl             = dst.layers[i];
            dl.dflare_norm       = artifact::materialized_tensor(materialized, sl.dflare_norm, NumericFormat::BF16, {2304});
            dl.input_norm        = artifact::materialized_tensor(materialized, sl.input_norm, NumericFormat::BF16, {2304});
            dl.post_attention_norm = artifact::materialized_tensor(materialized, sl.post_attention_norm, NumericFormat::BF16, {2304});
            dl.q_norm            = artifact::materialized_tensor(materialized, sl.q_norm, NumericFormat::BF16, {128});
            dl.k_norm            = artifact::materialized_tensor(materialized, sl.k_norm, NumericFormat::BF16, {128});
            dl.k_norm_ctx        = artifact::materialized_tensor(materialized, sl.k_norm_ctx, NumericFormat::BF16, {128});
            dl.q                 = artifact::materialized_weight(materialized, sl.q, NumericFormat::Q4G64_F16S, 3072, 2304);
            dl.k                 = artifact::materialized_weight(materialized, sl.k, NumericFormat::Q4G64_F16S, 1024, 2304);
            dl.v                 = artifact::materialized_weight(materialized, sl.v, NumericFormat::Q4G64_F16S, 1024, 2304);
            dl.k_ctx             = artifact::materialized_weight(materialized, sl.k_ctx, NumericFormat::Q4G64_F16S, 1024, 2304);
            dl.v_ctx             = artifact::materialized_weight(materialized, sl.v_ctx, NumericFormat::Q4G64_F16S, 1024, 2304);
            dl.output            = artifact::materialized_weight(materialized, sl.output, NumericFormat::Q4G64_F16S, 2304, 3072);
            dl.gate_up           = artifact::materialized_weight(materialized, sl.gate_up, NumericFormat::Q4G64_F16S, 18432, 2304);
            dl.down              = artifact::materialized_weight(materialized, sl.down, NumericFormat::Q4G64_F16S, 2304, 9216);
        }
        dst.markov_dflare_norm       = artifact::materialized_tensor(materialized, src.markov_dflare_norm, NumericFormat::BF16, {2304});
        dst.markov_w1                = artifact::materialized_tensor(materialized, src.markov_w1, NumericFormat::BF16, {256, 201024});
        dst.markov_rnn_proj_weight   = artifact::materialized_tensor(materialized, src.markov_rnn_proj_weight, NumericFormat::BF16, {5120, 768});
        dst.markov_rnn_proj_bias     = artifact::materialized_tensor(materialized, src.markov_rnn_proj_bias, NumericFormat::FP32, {768});
        dst.markov_hidden_out_weight = artifact::materialized_tensor(materialized, src.markov_hidden_out_weight, NumericFormat::BF16, {256, 2304});
        dst.markov_hidden_out_bias   = artifact::materialized_tensor(materialized, src.markov_hidden_out_bias, NumericFormat::FP32, {2304});
        dst.has_draft_head           = src.has_draft_head;
        out.has_draft_head           = src.has_draft_head;
        if (src.has_draft_head) {
            dst.draft_head = artifact::materialized_weight(
                materialized, src.draft_head, NumericFormat::Q4G64_F16S, 32768, 2304);
            dst.draft_head_token_ids = artifact::materialized_tensor(
                materialized, src.draft_head_token_ids, NumericFormat::I32, {32768});
        }
    }
    out.has_dflash2 = plan.has_dflash2;
    if (plan.has_dflash2) {
        const auto& src = plan.dflash2;
        auto& dst       = out.dflash2;
        dst.fc          = artifact::materialized_weight(materialized, src.fc,
                                                        NumericFormat::Q4G64_F16S, 2304, 9216);
        dst.hidden_norm = artifact::materialized_tensor(materialized, src.hidden_norm,
                                                        NumericFormat::BF16, {2304});
        for (std::size_t i = 0; i < kDFlash2Layers; ++i) {
            const auto& sl       = src.layers[i];
            auto& dl             = dst.layers[i];
            dl.input_norm        = artifact::materialized_tensor(materialized, sl.input_norm, NumericFormat::BF16, {2304});
            dl.post_attention_norm = artifact::materialized_tensor(materialized, sl.post_attention_norm, NumericFormat::BF16, {2304});
            dl.attn_conv_base    = artifact::materialized_tensor(materialized, sl.attn_conv_base, NumericFormat::BF16, {2304, 2, 2});
            dl.attn_conv_proj    = artifact::materialized_tensor(materialized, sl.attn_conv_proj, NumericFormat::BF16, {2304, 576});
            dl.q_norm            = artifact::materialized_tensor(materialized, sl.q_norm, NumericFormat::BF16, {128});
            dl.k_norm            = artifact::materialized_tensor(materialized, sl.k_norm, NumericFormat::BF16, {128});
            dl.q                 = artifact::materialized_weight(materialized, sl.q, NumericFormat::Q4G64_F16S, 4096, 2304);
            dl.k                 = artifact::materialized_weight(materialized, sl.k, NumericFormat::Q4G64_F16S, 1024, 2304);
            dl.v                 = artifact::materialized_weight(materialized, sl.v, NumericFormat::Q4G64_F16S, 1024, 2304);
            dl.output            = artifact::materialized_weight(materialized, sl.output, NumericFormat::Q4G64_F16S, 2304, 4096);
            dl.mlp_conv_base     = artifact::materialized_tensor(materialized, sl.mlp_conv_base, NumericFormat::BF16, {2304, 2, 2});
            dl.mlp_conv_proj     = artifact::materialized_tensor(materialized, sl.mlp_conv_proj, NumericFormat::BF16, {2304, 576});
            dl.gate_up           = artifact::materialized_weight(materialized, sl.gate_up, NumericFormat::Q4G64_F16S, 13824, 2304);
            dl.down              = artifact::materialized_weight(materialized, sl.down, NumericFormat::Q4G64_F16S, 2304, 6912);
        }
        dst.final_norm = artifact::materialized_tensor(materialized, src.final_norm, NumericFormat::BF16, {2304});
        dst.selector_hidden_proj = artifact::materialized_tensor(
            materialized, src.selector_hidden_proj, NumericFormat::BF16, {2304, 256});
        dst.predecessor_codebook = artifact::materialized_tensor(
            materialized, src.predecessor_codebook, NumericFormat::BF16, {256, 201024});
        dst.successor_codebook = artifact::materialized_tensor(
            materialized, src.successor_codebook, NumericFormat::BF16, {256, 201024});
    }
    out.has_vision             = plan.has_vision;
    out.vision_overlay_enabled = plan.vision_overlay;
    if (plan.has_vision) {
        const auto& src             = plan.vision;
        auto& dst                   = out.vision;
        dst.patch_embedding         = artifact::materialized_tensor(materialized, src.patch_embedding,
                                                                    NumericFormat::BF16, {1200, 1024});
        dst.class_embedding         = artifact::materialized_tensor(materialized, src.class_embedding,
                                                                    NumericFormat::BF16, {1024});
        dst.cls_position_embedding  = artifact::materialized_tensor(
            materialized, src.cls_position_embedding, NumericFormat::BF16, {1024});
        dst.position_embedding      = artifact::materialized_tensor(materialized, src.position_embedding,
                                                                    NumericFormat::BF16, {1024, 1024});
        dst.ln_pre_weight           = artifact::materialized_tensor(materialized, src.ln_pre_weight,
                                                                    NumericFormat::BF16, {1024});
        dst.ln_pre_bias             = artifact::materialized_tensor(materialized, src.ln_pre_bias,
                                                                    NumericFormat::BF16, {1024});
        for (std::size_t i = 0; i < kVisionLayers; ++i) {
            const auto& sl = src.layers[i];
            auto& dl       = dst.layers[i];
            dl.ln_1_weight = artifact::materialized_tensor(materialized, sl.ln_1_weight,
                                                           NumericFormat::BF16, {1024});
            dl.ln_1_bias   = artifact::materialized_tensor(materialized, sl.ln_1_bias,
                                                           NumericFormat::BF16, {1024});
            dl.qkv         = artifact::materialized_weight(materialized, sl.qkv,
                                                           NumericFormat::Q4G64_F16S, 3072, 1024);
            dl.qkv_bias    = artifact::materialized_tensor(materialized, sl.qkv_bias,
                                                           NumericFormat::BF16, {3072});
            dl.output      = artifact::materialized_weight(materialized, sl.output,
                                                           NumericFormat::Q5G64_F16S, 1024, 1024);
            dl.output_bias = artifact::materialized_tensor(materialized, sl.output_bias,
                                                           NumericFormat::BF16, {1024});
            dl.ls_1_gamma  = artifact::materialized_tensor(materialized, sl.ls_1_gamma,
                                                           NumericFormat::BF16, {1024});
            dl.ln_2_weight = artifact::materialized_tensor(materialized, sl.ln_2_weight,
                                                           NumericFormat::BF16, {1024});
            dl.ln_2_bias   = artifact::materialized_tensor(materialized, sl.ln_2_bias,
                                                           NumericFormat::BF16, {1024});
            dl.fc1         = artifact::materialized_weight(materialized, sl.fc1,
                                                           NumericFormat::Q4G64_F16S, 4096, 1024);
            dl.fc1_bias    = artifact::materialized_tensor(materialized, sl.fc1_bias,
                                                           NumericFormat::BF16, {4096});
            dl.fc2         = artifact::materialized_weight(materialized, sl.fc2,
                                                           NumericFormat::Q5G64_F16S, 1024, 4096);
            dl.fc2_bias    = artifact::materialized_tensor(materialized, sl.fc2_bias,
                                                           NumericFormat::BF16, {1024});
            dl.ls_2_gamma  = artifact::materialized_tensor(materialized, sl.ls_2_gamma,
                                                           NumericFormat::BF16, {1024});
        }
        dst.proj_in  = artifact::materialized_weight(materialized, src.proj_in,
                                                   NumericFormat::W8G32_F16S, 4096, 4096);
        dst.proj_out = artifact::materialized_weight(materialized, src.proj_out,
                                                   NumericFormat::W8G32_F16S, 2304, 4096);
        if (plan.vision_overlay) {
            out.pinned_weights = materialized.pinned_block();
            out.vision_overlay = compute_vision_overlay_layout(dst, out.pinned_weights);
        }
    }
    return out;
}
} // namespace ninfer::targets::inkling2_mini::detail
