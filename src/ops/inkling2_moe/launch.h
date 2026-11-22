#pragma once
#include "ninfer/ops/inkling2_moe.h"

namespace ninfer::ops::inkling2_detail {
inline constexpr int kInkling2MoePrefillMinTokens = 16;
inline constexpr int kInkling2MoeSliceMax         = 1024;
inline constexpr int kInkling2MoeExpertBN         = 32;

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
};

void moe_launch(const Tensor&, const Inkling2MoeWeights&, Tensor&, float*, int*, float*, float*,
                cudaStream_t);
void moe_prefill_launch(const void* x, const Inkling2MoeWeights& w, void* out, int tokens,
                        const Inkling2MoePrefillWorkspace& ws, cudaStream_t s);
}
