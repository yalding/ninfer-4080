#pragma once
#include "ninfer/ops/inkling2_moe.h"
#include <cstdint>

namespace ninfer::ops::inkling2_detail {
inline constexpr int kInkling2MoePrefillMinTokens = 4;
inline constexpr int kInkling2MoeSliceMax         = 1024;
inline constexpr int kInkling2MoeVerifyExpertBN    = 8;
inline constexpr int kInkling2MoeSmallExpertBN     = 16;
inline constexpr int kInkling2MoeExpertBN          = 32;

struct Inkling2MoePrefillWorkspace {
    float* router_probs;
    int* token_ids;
    float* token_alpha;
    int* local_rank;
    int* packed_index;
    int* packed_token;
    int* tile_counts;
    int* tile_bases;
    int* expert_offsets;
    int* route_job_experts;
    int* route_job_columns;
    int* route_job_count;
    void* routed_activation;
    void* grouped_output;
    std::int8_t* x_a8_q;
    float* x_a8_scale;
    std::int8_t* act_a8_q;
    float* act_a8_scale;
};

void moe_launch(const Tensor&, const Inkling2MoeWeights&, Tensor&, float*, int*, float*, float*,
                cudaStream_t);
void moe_prefill_launch(const void* x, const Inkling2MoeWeights& w, void* out, int tokens,
                        const Inkling2MoePrefillWorkspace& ws, cudaStream_t s);
void moe_prefill_post_sconv_launch(const void* x, const Inkling2MoeWeights& w,
                                   const Tensor& post_conv_weight, const Tensor& state_in,
                                   Tensor& state_out, const Tensor& residual_in, void* out,
                                   void* moe_trace, int tokens,
                                   const Inkling2MoePrefillWorkspace& ws, cudaStream_t s);
}
