#include "targets/inkling2_mini/impl/load/bindings.h"
#include "artifact/typed_binding.h"
#include <string>

namespace ninfer::targets::inkling2_mini::detail {
using artifact::NumericFormat;

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, bool load_mtp, bool load_draft_head) {
    ArtifactLoadPlan out;
    auto& plan           = out.bindings;
    plan.has_mtp         = load_mtp;
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
    return out;
}
} // namespace ninfer::targets::inkling2_mini::detail
