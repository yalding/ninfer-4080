#pragma once
#include <ninfer/targets/inkling2_mini/frontend_resources.h>
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include "ninfer/types.h"
#include <array>
#include <span>
#include <vector>

namespace ninfer::targets::inkling2_mini::detail {
inline constexpr std::size_t kPhysicalLayers = 24;
inline constexpr std::size_t kMtpLayers      = 3;
inline constexpr std::size_t kDFlash2Layers  = 5;
inline constexpr std::size_t kVisionLayers   = 23;

inline constexpr bool is_full_attention_layer(std::size_t i) noexcept { return i % 6 == 5; }

inline constexpr bool is_loop_layer(std::size_t i) noexcept { return i >= 6 && i < 18; }

struct LayerPlan {
    artifact::ObjectHandle attention_input_norm;
    artifact::ObjectHandle attention_qgr;
    artifact::ObjectHandle attention_k;
    artifact::ObjectHandle attention_v;
    artifact::ObjectHandle attention_output;
    artifact::ObjectHandle attention_sink_proj;
    artifact::ObjectHandle attention_q_clip;
    artifact::ObjectHandle attention_rel_clip;
    artifact::ObjectHandle attention_rel_bias;
    artifact::ObjectHandle attention_k_conv;
    artifact::ObjectHandle attention_v_conv;
    artifact::ObjectHandle attention_post_conv;
    artifact::ObjectHandle ffn_res_query;
    artifact::ObjectHandle ffn_input_norm;
    artifact::ObjectHandle ffn_router;
    artifact::ObjectHandle ffn_exp_probs_bias;
    artifact::ObjectHandle ffn_routed_gate_up;
    artifact::ObjectHandle ffn_routed_down;
    artifact::ObjectHandle ffn_post_conv;
    artifact::ObjectHandle attention_res_query;
    artifact::ObjectHandle attention_q_input_norm;
    artifact::ObjectHandle attention_k_input_norm;
    artifact::ObjectHandle attention_v_input_norm;
    artifact::ObjectHandle attention_k_conv_proj;
    artifact::ObjectHandle attention_v_conv_proj;
    artifact::ObjectHandle loop_msg_query;
    artifact::ObjectHandle loop_ctrl_down;
    artifact::ObjectHandle loop_ctrl_gain_forget;
};

struct MtpLayerPlan {
    artifact::ObjectHandle dflare_norm;
    artifact::ObjectHandle input_norm;
    artifact::ObjectHandle post_attention_norm;
    artifact::ObjectHandle q_norm;
    artifact::ObjectHandle k_norm;
    artifact::ObjectHandle k_norm_ctx;
    artifact::ObjectHandle q;
    artifact::ObjectHandle k;
    artifact::ObjectHandle v;
    artifact::ObjectHandle k_ctx;
    artifact::ObjectHandle v_ctx;
    artifact::ObjectHandle output;
    artifact::ObjectHandle gate_up;
    artifact::ObjectHandle down;
};

struct MtpPlan {
    artifact::ObjectHandle fuse_weight;
    artifact::ObjectHandle mask_embed;
    artifact::ObjectHandle final_norm;
    std::array<MtpLayerPlan, kMtpLayers> layers;
    artifact::ObjectHandle markov_dflare_norm;
    artifact::ObjectHandle markov_w1;
    artifact::ObjectHandle markov_rnn_proj_weight;
    artifact::ObjectHandle markov_rnn_proj_bias;
    artifact::ObjectHandle markov_hidden_out_weight;
    artifact::ObjectHandle markov_hidden_out_bias;
    bool has_draft_head = false;
    artifact::ObjectHandle draft_head;
    artifact::ObjectHandle draft_head_token_ids;
};

struct DFlash2LayerPlan {
    artifact::ObjectHandle input_norm;
    artifact::ObjectHandle post_attention_norm;
    artifact::ObjectHandle attn_conv_base;
    artifact::ObjectHandle attn_conv_proj;
    artifact::ObjectHandle q_norm;
    artifact::ObjectHandle k_norm;
    artifact::ObjectHandle q;
    artifact::ObjectHandle k;
    artifact::ObjectHandle v;
    artifact::ObjectHandle output;
    artifact::ObjectHandle mlp_conv_base;
    artifact::ObjectHandle mlp_conv_proj;
    artifact::ObjectHandle gate_up;
    artifact::ObjectHandle down;
};

struct DFlash2Plan {
    artifact::ObjectHandle fc;
    artifact::ObjectHandle hidden_norm;
    std::array<DFlash2LayerPlan, kDFlash2Layers> layers;
    artifact::ObjectHandle final_norm;
    artifact::ObjectHandle selector_hidden_proj;
    artifact::ObjectHandle predecessor_codebook;
    artifact::ObjectHandle successor_codebook;
};

struct VisionLayerPlan {
    artifact::ObjectHandle ln_1_weight;
    artifact::ObjectHandle ln_1_bias;
    artifact::ObjectHandle qkv;
    artifact::ObjectHandle qkv_bias;
    artifact::ObjectHandle output;
    artifact::ObjectHandle output_bias;
    artifact::ObjectHandle ls_1_gamma;
    artifact::ObjectHandle ln_2_weight;
    artifact::ObjectHandle ln_2_bias;
    artifact::ObjectHandle fc1;
    artifact::ObjectHandle fc1_bias;
    artifact::ObjectHandle fc2;
    artifact::ObjectHandle fc2_bias;
    artifact::ObjectHandle ls_2_gamma;
};

struct VisionPlan {
    artifact::ObjectHandle patch_embedding;
    artifact::ObjectHandle class_embedding;
    artifact::ObjectHandle cls_position_embedding;
    artifact::ObjectHandle position_embedding;
    artifact::ObjectHandle ln_pre_weight;
    artifact::ObjectHandle ln_pre_bias;
    std::array<VisionLayerPlan, kVisionLayers> layers;
    artifact::ObjectHandle proj_in;
    artifact::ObjectHandle proj_out;
};

struct BindingPlan {
    FrontendResourcePlan frontend;
    artifact::ObjectHandle token_embedding;
    artifact::ObjectHandle token_norm;
    std::array<LayerPlan, kPhysicalLayers> layers;
    artifact::ObjectHandle output_res_query;
    artifact::ObjectHandle final_norm;
    artifact::ObjectHandle output_head;
    bool has_mtp     = false;
    MtpPlan mtp;
    bool has_dflash2 = false;
    DFlash2Plan dflash2;
    bool has_vision     = false;
    bool vision_overlay = false;
    VisionPlan vision;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

struct LayerWeights {
    bool full_attention = false;
    Tensor attention_input_norm;
    Weight attention_qgr;
    Weight attention_k;
    Weight attention_v;
    Weight attention_output;
    Tensor attention_sink_proj;
    Tensor attention_q_clip;
    Tensor attention_rel_clip;
    Tensor attention_rel_bias;
    Tensor attention_k_conv;
    Tensor attention_v_conv;
    Tensor attention_post_conv;
    Tensor ffn_res_query;
    Tensor ffn_input_norm;
    Weight ffn_router;
    Weight ffn_exp_probs_bias;
    Weight ffn_routed_gate_up;
    Weight ffn_routed_down;
    Tensor ffn_post_conv;
    Tensor attention_res_query;
    Tensor attention_q_input_norm;
    Tensor attention_k_input_norm;
    Tensor attention_v_input_norm;
    Tensor attention_k_conv_proj;
    Tensor attention_v_conv_proj;
    Tensor loop_msg_query;
    Tensor loop_ctrl_down;
    Tensor loop_ctrl_gain_forget;
};

struct MtpLayerWeights {
    Tensor dflare_norm;
    Tensor input_norm;
    Tensor post_attention_norm;
    Tensor q_norm;
    Tensor k_norm;
    Tensor k_norm_ctx;
    Weight q;
    Weight k;
    Weight v;
    Weight k_ctx;
    Weight v_ctx;
    Weight output;
    Weight gate_up;
    Weight down;
};

struct MtpWeights {
    Tensor fuse_weight;
    Tensor mask_embed;
    Tensor final_norm;
    std::array<MtpLayerWeights, kMtpLayers> layers;
    Tensor markov_dflare_norm;
    Tensor markov_w1;
    Tensor markov_rnn_proj_weight;
    Tensor markov_rnn_proj_bias;
    Tensor markov_hidden_out_weight;
    Tensor markov_hidden_out_bias;
    bool has_draft_head = false;
    Weight draft_head;
    Tensor draft_head_token_ids;
};

struct DFlash2LayerWeights {
    Tensor input_norm;
    Tensor post_attention_norm;
    Tensor attn_conv_base;
    Tensor attn_conv_proj;
    Tensor q_norm;
    Tensor k_norm;
    Weight q;
    Weight k;
    Weight v;
    Weight output;
    Tensor mlp_conv_base;
    Tensor mlp_conv_proj;
    Weight gate_up;
    Weight down;
};

struct DFlash2Weights {
    Weight fc;
    Tensor hidden_norm;
    std::array<DFlash2LayerWeights, kDFlash2Layers> layers;
    Tensor final_norm;
    Tensor selector_hidden_proj;
    Tensor predecessor_codebook;
    Tensor successor_codebook;
};

struct VisionLayerWeights {
    Tensor ln_1_weight;
    Tensor ln_1_bias;
    Weight qkv;
    Tensor qkv_bias;
    Weight output;
    Tensor output_bias;
    Tensor ls_1_gamma;
    Tensor ln_2_weight;
    Tensor ln_2_bias;
    Weight fc1;
    Tensor fc1_bias;
    Weight fc2;
    Tensor fc2_bias;
    Tensor ls_2_gamma;
};

struct VisionWeights {
    Tensor patch_embedding;
    Tensor class_embedding;
    Tensor cls_position_embedding;
    Tensor position_embedding;
    Tensor ln_pre_weight;
    Tensor ln_pre_bias;
    std::array<VisionLayerWeights, kVisionLayers> layers;
    Weight proj_in;
    Weight proj_out;
};

struct PinnedRange {
    std::size_t offset = 0;
    std::size_t bytes  = 0;
};

struct VisionOverlayLayout {
    PinnedRange prelude;
    std::vector<PinnedRange> layers;
    PinnedRange merger;
    std::size_t slot_bytes    = 0;
    std::size_t staging_bytes = 0;
};

inline constexpr std::size_t kVisionStagingAlignment = 256;

[[nodiscard]] inline constexpr std::size_t vision_staging_align(std::size_t bytes) noexcept {
    return (bytes + kVisionStagingAlignment - 1) / kVisionStagingAlignment *
           kVisionStagingAlignment;
}

struct RuntimeModelView {
    const DeviceArena* weights_arena = nullptr;
    Weight token_embedding;
    Tensor token_norm;
    std::array<LayerWeights, kPhysicalLayers> layers;
    Tensor output_res_query;
    Tensor final_norm;
    Weight output_head;
    bool has_mtp        = false;
    bool has_draft_head = false;
    MtpWeights mtp;
    bool has_dflash2    = false;
    DFlash2Weights dflash2;
    bool has_vision     = false;
    bool vision_overlay_enabled = false;
    VisionWeights vision;
    std::span<const std::byte> pinned_weights;
    VisionOverlayLayout vision_overlay;
};

[[nodiscard]] ArtifactLoadPlan bind_artifact(artifact::Binder& binder, bool load_mtp = false,
                                             bool load_draft_head = false,
                                             bool load_dflash2 = false,
                                             bool enable_vision = false,
                                             VisionResidency vision_residency = VisionResidency::Resident);
[[nodiscard]] RuntimeModelView materialize_model(artifact::MaterializedArtifact& materialized,
                                                 const BindingPlan& plan);
} // namespace ninfer::targets::inkling2_mini::detail
