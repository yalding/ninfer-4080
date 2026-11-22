#pragma once
#include <ninfer/targets/inkling2_mini/frontend_resources.h>
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include <array>

namespace ninfer::targets::inkling2_mini::detail {
inline constexpr std::size_t kPhysicalLayers = 24;

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

struct BindingPlan {
    FrontendResourcePlan frontend;
    artifact::ObjectHandle token_embedding;
    artifact::ObjectHandle token_norm;
    std::array<LayerPlan, kPhysicalLayers> layers;
    artifact::ObjectHandle output_res_query;
    artifact::ObjectHandle final_norm;
    artifact::ObjectHandle output_head;
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

struct RuntimeModelView {
    const DeviceArena* weights_arena = nullptr;
    Weight token_embedding;
    Tensor token_norm;
    std::array<LayerWeights, kPhysicalLayers> layers;
    Tensor output_res_query;
    Tensor final_norm;
    Weight output_head;
};

[[nodiscard]] ArtifactLoadPlan bind_artifact(artifact::Binder& binder);
[[nodiscard]] RuntimeModelView materialize_model(artifact::MaterializedArtifact& materialized,
                                                 const BindingPlan& plan);
} // namespace ninfer::targets::inkling2_mini::detail
