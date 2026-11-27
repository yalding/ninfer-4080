#pragma once

#include "ninfer/ops/inkling2_fused.h"

namespace ninfer::ops::detail {
struct Inkling2Branches {
    const void* states[8]{};
    const void* norm[4]{};
    int count = 0;
};
void inkling2_proj_launch(const Tensor&, const Tensor&, Tensor&, cudaStream_t);
void inkling2_sconv_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, Tensor&,
                          const Tensor&, Tensor&, const Tensor&, cudaStream_t);
void inkling2_sconv_kv_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, Tensor&,
                             Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             Tensor&, Tensor&, const Tensor&, const Tensor&, cudaStream_t);
void inkling2_sconv_batched_launch(const Tensor&, const Tensor&, Tensor&, int, int, const Tensor&,
                                   const Tensor&, Tensor&, cudaStream_t);
void inkling2_sconv_kv_batched_launch(const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                      const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                      Tensor&, int, const Tensor&, cudaStream_t);
void inkling2_mix_launch(Inkling2Branches, const Tensor&, Tensor&, cudaStream_t);
void inkling2_control_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            Inkling2Branches, Tensor&, float*, float*, float*, cudaStream_t);
void inkling2_clip_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                         const Tensor&, bool, Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t);
void inkling2_softcap_launch(const Tensor&, float, int, Tensor&, cudaStream_t);
void inkling2_dspark_feature_mix_launch(const Tensor&, const Tensor&, Inkling2Branches, Tensor&,
                                        cudaStream_t);
void inkling2_dspark_rmsnorm_rope_launch(const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                         Tensor&, float, cudaStream_t);
void inkling2_dspark_rnn_step_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                     const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                     Tensor&, Tensor&, void*, float*, cudaStream_t);
void inkling2_dspark_fold_head_launch(const Weight&, const Tensor&, const Tensor&, Tensor&,
                                      Tensor&, Tensor&, cudaStream_t);
void inkling2_dspark_rnn_step_folded_launch(const Tensor&, const Tensor&, const Tensor&,
                                            const Tensor&, const Tensor&, const Tensor&,
                                            const Tensor&, const Tensor&, const Tensor&,
                                            const Tensor&, int, Tensor&, Tensor&, void*, float*,
                                            float*, float*, int*, cudaStream_t);
void inkling2_dspark_commit_conv_launch(const Tensor&, const Tensor&, Tensor&, int, int,
                                        cudaStream_t);
void inkling2_dflash2_kv_materialize_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, const Tensor& scratch, cudaStream_t stream);
void inkling2_dflash2_conv_prepare_launch(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const Tensor& base_kernel,
                                          const Tensor& kernel_proj, Tensor& prepared,
                                          Tensor& finish_delta, void* normed_buf, float* delta_buf,
                                          cudaStream_t stream);
void inkling2_dflash2_conv_finish_add_launch(const Tensor& proj_out, const Tensor& base_kernel,
                                             const Tensor& finish_delta, Tensor& residual,
                                             cudaStream_t stream);
void inkling2_dflash2_topk16_and_select_launch(
    const Tensor& mask_hidden, const Tensor& unary_logits, const Tensor& hidden_projection,
    const Tensor& anchors, const Tensor& predecessor_codebook, const Tensor& successor_codebook,
    const Tensor& base_positions, const SamplingConfig* configs, Tensor& candidate_ids,
    Tensor& drafts, Tensor& proposal_q, void* projected_hidden_buf, std::int32_t* partial_ids,
    float* partial_vals, float* unary_scores, cudaStream_t stream);
void inkling2_vision_patch_embed_launch(
    const Tensor& patches, const Tensor& patch_weight, const Tensor& class_embedding,
    const Tensor& cls_position_embedding, const Tensor& position_table, const Tensor& pos_indices,
    const Tensor& pos_weights, const Tensor& ln_pre_weight, const Tensor& ln_pre_bias, float eps,
    Tensor& out, cudaStream_t stream);
void inkling2_vision_rope_2d_launch(const Tensor& coords, float rope_theta, Tensor& qkv,
                                    cudaStream_t stream);
void inkling2_vision_layerscale_add_launch(const Tensor& proj, const Tensor& bias,
                                           const Tensor& gamma, Tensor& hidden,
                                           cudaStream_t stream);
void inkling2_vision_merge_2x2_launch(const Tensor& src, const Tensor& merge_indices, Tensor& out,
                                      cudaStream_t stream);
} // namespace ninfer::ops::detail
