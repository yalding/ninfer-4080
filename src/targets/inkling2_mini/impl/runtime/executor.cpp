#include "targets/inkling2_mini/impl/runtime/executor.h"
#include "targets/inkling2_mini/impl/load/bindings.h"
#include "targets/inkling2_mini/impl/runtime/vision_overlay.h"
#include "targets/inkling2_mini/impl/vision_limits.h"
#include "targets/inkling2_mini/impl/state/state_image.h"
#include <ninfer/targets/inkling2_mini/prepared_prompt.h>
#include "core/arena.h"
#include "core/decode_graph.h"
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/inkling2_attention.h"
#include "ninfer/ops/inkling2_fused.h"
#include "ninfer/ops/inkling2_moe.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/target_logprobs.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace ninfer::targets::inkling2_mini::detail {
namespace {
constexpr int C = 2304, V = 200058, P = 201024;
constexpr std::uint32_t kScoreChunk = 128;
constexpr std::size_t kVisionOverlayStagingBytes = 47'448'064ULL;

std::uint32_t v_native_max(const EngineOptions& options) {
    return vision_native_capacity(options.vision_max_tokens);
}

std::uint32_t v_item_max(const EngineOptions& options) {
    return vision_merged_capacity(options.vision_max_tokens, options.vision_max_merged_tokens);
}

std::uint32_t chunk_capacity(const EngineOptions& options) {
    const std::uint32_t batch =
        std::min<std::uint32_t>(kMaximumConcurrency, std::max(1U, options.max_concurrency));
    const auto prefill = std::min({1024U, options.max_context, std::max(1U, options.prefill_chunk)});
    return std::max(prefill, batch);
}

std::uint32_t kv_tokens(const EngineOptions& options) {
    if (options.kv_capacity.mode != KvCapacityMode::Explicit)
        throw std::logic_error("Inkling Executor requires resolved explicit KV capacity");
    return options.kv_capacity.explicit_tokens;
}

int head_columns(const EngineOptions& options) {
    if (options.purpose == EnginePurpose::CausalScoring)
        return static_cast<int>(std::min(kScoreChunk, chunk_capacity(options)));
    int cols = static_cast<int>(
        std::min<std::uint32_t>(kMaximumConcurrency, std::max(1U, options.max_concurrency)));
    if (options.speculative.backend != SpeculativeBackend::None) {
        std::uint32_t max_drafts = std::max(1U, options.speculative.draft_tokens);
        if (options.speculative.backend == SpeculativeBackend::Mtp &&
            options.speculative.ngram.mode == NgramDraftMode::Chain) {
            max_drafts = std::min(7U, std::max(max_drafts, options.speculative.ngram.max_drafts));
        }
        cols = std::max(cols, static_cast<int>(max_drafts + 1U));
    }
    return cols;
}

std::size_t op_workspace(const EngineOptions& options) {
    const int t       = static_cast<int>(chunk_capacity(options));
    std::size_t bytes = ops::inkling2_moe_workspace_capacity_bytes(QType::Q3G128_F16S, 1, t);
    bytes = std::max(bytes, ops::inkling2_loop_control_workspace_capacity_bytes(3, 1, t));
    bytes = std::max(bytes, ops::inkling2_full_attention_workspace_capacity_bytes(
                                options.kv_cache, options.max_context, 1, t));
    for (const auto [n, k] : {std::pair{6528, 2304}, {1024, 2304}, {2304, 3072}})
        bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                    QType::Q4G64_F16S, n, k, ops::LinearPolicy::AllowA8, 1, t));
    if (options.speculative.backend == SpeculativeBackend::DFlash2) {
        const int w_max = head_columns(options);
        const int k_max = std::max(1, w_max - 1);
        bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                    QType::Q4G64_F16S, C, 9216, ops::LinearPolicy::AllowA8, 1, t));
        for (const auto [n, k] : {std::pair{4096, C}, {1024, C}, {C, 4096}, {13824, C}, {C, 6912}})
            bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                        QType::Q4G64_F16S, n, k, ops::LinearPolicy::A16Only, 1,
                                        w_max));
        bytes = std::max(bytes, ops::sliding_window_attention_workspace_capacity_bytes(
                                    {128, 32, 8}, 2048U, {0U, options.max_context}, 1, w_max, 1));
        bytes = std::max(bytes, ops::inkling2_dflash2_kv_materialize_workspace_bytes(t));
        bytes = std::max(bytes, ops::inkling2_dflash2_conv_prepare_workspace_bytes(w_max));
        bytes = std::max(bytes, ops::inkling2_dflash2_topk16_and_select_workspace_bytes(k_max));
        bytes = std::max(bytes, ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                                    V, {false}, 1, k_max, 1, 1));
    } else if (options.speculative.backend != SpeculativeBackend::None) {
        bytes = std::max(bytes, ops::inkling2_dspark_rnn_step_workspace_bytes());
        bytes = std::max(bytes, ops::inkling2_dspark_rnn_step_folded_workspace_bytes());
        bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                    QType::Q4G64_F16S, 18432, 2304, ops::LinearPolicy::A16Only, 1, 8));
        bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                    QType::Q4G64_F16S, 2304, 9216, ops::LinearPolicy::A16Only, 1, 8));
        bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                    QType::Q4G64_F16S, 32768, 2304, ops::LinearPolicy::A16Only, 1, 1));
        bytes = std::max(bytes, ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                    V, 1, 7, 1, 1));
    }
    bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(QType::Q6G64_F16S, P, C,
                                                                 ops::LinearPolicy::A16Only, 1,
                                                                 head_columns(options)));
    if (options.enable_vision) {
        const int tv = static_cast<int>(v_native_max(options)) + 1;
        const int vm = static_cast<int>(v_item_max(options));
        for (const auto [qt, n, k, cols] :
             {std::tuple{QType::Q4G64_F16S, 3072, 1024, tv},
              std::tuple{QType::Q5G64_F16S, 1024, 1024, tv},
              std::tuple{QType::Q4G64_F16S, 4096, 1024, tv},
              std::tuple{QType::Q5G64_F16S, 1024, 4096, tv},
              std::tuple{QType::W8G32_F16S, 4096, 4096, vm},
              std::tuple{QType::W8G32_F16S, 2304, 4096, vm}}) {
            bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                        qt, n, k, ops::LinearPolicy::A16Only, 1, cols));
        }
    }
    return std::max(bytes, ops::sampling_workspace_capacity_bytes(V, 1, 1));
}

struct ScratchLayout {
    TensorRegion cur, residual, blocks[4], prelude, bank, previous, mix, messages, inputs;
    TensorRegion qgr, kraw, vraw, kconv, vconv, q, k, v, r, sink, dynamic_k, dynamic_v, rel;
    TensorRegion attention, projection, moe, ffn_mix, ids, positions, count, lane, batch_slots;
    TensorRegion logits, sampled, sample_position, counts, score_targets, score_result;
    TensorRegion dspark_features, dspark_mixed, dspark_c_head, dspark_block, dspark_res, dspark_norm;
    TensorRegion dspark_q, dspark_k, dspark_v, dspark_attn, dspark_proj, dspark_mlp_gu, dspark_mlp_act;
    TensorRegion dspark_h_prime, dspark_rnn_state, dspark_pos, dspark_draft_tok, dspark_draft_logits;
    TensorRegion dspark_w_fold_q, dspark_w_fold_scale, dspark_b_fold;
    TensorRegion dspark_pre_conv, dspark_conv_trace, dspark_target_toks, dspark_extents;
    TensorRegion dspark_lengths, dspark_anchors, dspark_licensed, dspark_licensed_cnt, dspark_accepted;
    TensorRegion dflash2_features, dflash2_ctx_proj, dflash2_ctx, dflash2_ids, dflash2_pos;
    TensorRegion dflash2_valid_cols, dflash2_res, dflash2_prep, dflash2_delta;
    TensorRegion dflash2_q, dflash2_k, dflash2_v, dflash2_attn, dflash2_proj, dflash2_gu, dflash2_act;
    TensorRegion dflash2_mask_h, dflash2_unary_logits, dflash2_cand_ids, dflash2_proposal_q;
    TensorRegion dflash2_drafts;
    TensorRegion vision_handoff, vision_scatter_idx;
    LayoutRegion vision_overlay_staging;
    TensorRegion vit_patches, vit_coords, vit_pos_idx, vit_pos_w, vit_merge_idx;
    TensorRegion vit_x, vit_ln, vit_qkv, vit_attn, vit_proj, vit_fc1, vit_merged, vit_proj_in;
    LayoutRegion config;
    std::size_t vision_general_bytes = 0;
    std::size_t bytes;
};

ScratchLayout scratch_layout(const EngineOptions& options) {
    LayoutBuilder b;
    ScratchLayout s;
    const int t = static_cast<int>(chunk_capacity(options));
    auto bf     = [&](std::initializer_list<int> shape) {
        return b.add_tensor(DType::BF16, shape, 256, "Inkling activations");
    };
    auto fp = [&](std::initializer_list<int> shape) {
        return b.add_tensor(DType::FP32, shape, 256, "Inkling controls");
    };
    auto i32 = [&](std::initializer_list<int> shape) {
        return b.add_tensor(DType::I32, shape, 256, "Inkling indices");
    };
    s.cur             = bf({C, t});
    const int max_b   = static_cast<int>(
        std::min<std::uint32_t>(kMaximumConcurrency, std::max(1U, options.max_concurrency)));
    s.ids             = i32({t});
    s.positions       = i32({t});
    s.count           = i32({1});
    s.lane            = i32({1});
    s.batch_slots     = i32({max_b});
    s.logits          = bf({P, head_columns(options)});
    s.sampled         = i32({max_b});
    s.sample_position = i32({max_b});
    s.counts          = i32({V, max_b});
    s.score_targets   = i32({head_columns(options)});
    s.score_result    = fp({head_columns(options)});
    if (options.speculative.backend != SpeculativeBackend::None) {
        const int max_w = std::max(8, head_columns(options));
        if (options.speculative.backend == SpeculativeBackend::DFlash2) {
            s.dflash2_features     = bf({9216, t});
            s.dflash2_ctx_proj     = bf({C, t});
            s.dflash2_ctx          = bf({C, t});
            s.dflash2_ids          = i32({16, 1});
            s.dflash2_pos          = i32({16, 1});
            s.dflash2_valid_cols   = i32({1});
            s.dflash2_res          = bf({C, 16});
            s.dflash2_prep         = bf({C, 16});
            s.dflash2_delta        = bf({144, 2, 16});
            s.dflash2_q            = bf({128, 32, 16, 1});
            s.dflash2_k            = bf({128, 8, 16, 1});
            s.dflash2_v            = bf({128, 8, 16, 1});
            s.dflash2_attn         = bf({128, 32, 16, 1});
            s.dflash2_proj         = bf({C, 16});
            s.dflash2_gu           = bf({13824, 16});
            s.dflash2_act          = bf({6912, 16});
            s.dflash2_mask_h       = bf({C, 15});
            s.dflash2_unary_logits = bf({P, 15});
            s.dflash2_cand_ids     = i32({16, 15, 1});
            s.dflash2_proposal_q   = fp({16, 15, 1});
            s.dflash2_drafts       = i32({15, 1});
        } else {
            const int slots     = static_cast<int>(state_slot_count(options));
            s.dspark_features   = bf({C, t, 4});
            s.dspark_mixed      = bf({C, t, 4});
            s.dspark_c_head     = bf({C, slots});
            s.dspark_block      = bf({C, 8});
            s.dspark_res        = bf({C, 8});
            s.dspark_norm       = bf({C, 8});
            s.dspark_q          = bf({128, 24, 8});
            s.dspark_k          = bf({128, 8, t});
            s.dspark_v          = bf({128, 8, t});
            s.dspark_attn       = bf({128, 24, 8});
            s.dspark_proj       = bf({C, 8});
            s.dspark_mlp_gu     = bf({18432, 8});
            s.dspark_mlp_act    = bf({9216, 8});
            s.dspark_h_prime    = bf({C});
            s.dspark_rnn_state  = fp({256});
            s.dspark_pos        = i32({8});
            s.dspark_draft_tok  = i32({8});
            s.dspark_draft_logits = bf({P, 7});
            if (options.speculative.proposal_head != ProposalHead::Optimized) {
                s.dspark_w_fold_q =
                    b.add_tensor(DType::I8, {256, P}, 256, "Inkling DSpark folded W");
                s.dspark_w_fold_scale =
                    b.add_tensor(DType::FP16, {4, P}, 256, "Inkling DSpark folded scale");
                s.dspark_b_fold = fp({P});
            }
        }
        s.dspark_pre_conv     = bf({6656 * 3, 48});
        s.dspark_conv_trace   = bf({6656 * max_w, 48});
        s.dspark_target_toks  = i32({max_w});
        s.dspark_extents      = i32({1});
        s.dspark_lengths      = i32({1});
        s.dspark_anchors      = i32({1});
        s.dspark_licensed     = i32({max_w});
        s.dspark_licensed_cnt = i32({1});
        s.dspark_accepted     = i32({1});
    }
    if (options.enable_vision) {
        s.vision_handoff     = bf({C, static_cast<int>(v_item_max(options))});
        s.vision_scatter_idx = i32({t});
    }
    s.config = b.add(sizeof(ops::SamplingConfig) * max_b, 256, "Inkling sampling config");
    {
        auto text_scope = b.scope();
        s.residual      = bf({C, t});
        for (auto& block : s.blocks) block = bf({C, t});
        s.prelude   = bf({C, t});
        s.bank      = bf({C, t});
        s.previous  = bf({C, t});
        s.mix       = bf({C, t, 3});
        s.messages  = bf({C, t, 4});
        s.inputs    = bf({C, t, 3});
        s.qgr       = bf({6528, t});
        s.kraw      = bf({1024, t});
        s.vraw      = bf({1024, t});
        s.kconv     = bf({1024, t});
        s.vconv     = bf({1024, t});
        s.q         = bf({128, 24, t});
        s.k         = bf({128, 8, t});
        s.v         = bf({128, 8, t});
        s.r         = fp({16, 24, t});
        s.sink      = fp({24, t});
        s.dynamic_k = fp({32, t});
        s.dynamic_v = fp({32, t});
        s.rel       = bf({1025, 24, std::max(t, 32)});
        s.attention = bf({128, 24, t});
        s.projection = bf({C, t});
        s.moe       = bf({C, t});
        s.ffn_mix   = bf({C, t});
    }
    if (options.enable_vision) {
        auto vision_scope = b.scope();
        const int vn      = static_cast<int>(v_native_max(options));
        const int tv      = vn + 1;
        const int vm      = static_cast<int>(v_item_max(options));
        std::size_t start_offset = 0;
        if (options.vision_residency == VisionResidency::Overlay) {
            s.vision_overlay_staging =
                b.add(kVisionOverlayStagingBytes, 256, "Inkling vision overlay staging");
            start_offset = s.vision_overlay_staging.offset;
        }
        s.vit_patches   = bf({1200, vn});
        if (options.vision_residency != VisionResidency::Overlay) {
            start_offset = s.vit_patches.region.offset;
        }
        s.vit_coords    = i32({4, tv});
        s.vit_pos_idx   = i32({4, vn});
        s.vit_pos_w     = fp({4, vn});
        s.vit_merge_idx = i32({4, vm});
        s.vit_x         = bf({1024, tv});
        s.vit_ln        = bf({1024, tv});
        s.vit_qkv       = bf({3072, tv});
        s.vit_attn      = bf({1024, tv});
        s.vit_proj      = bf({1024, tv});
        s.vit_fc1       = bf({4096, tv});
        s.vit_merged    = bf({4096, vm});
        s.vit_proj_in   = bf({4096, vm});
        s.vision_general_bytes =
            (s.vit_proj_in.region.offset + s.vit_proj_in.region.bytes) - start_offset;
    }
    s.bytes = b.finish(256);
    return s;
}

Tensor region(const TensorRegion& r, DeviceBuffer& backing, std::initializer_list<int> shape) {
    return Tensor(r.region.bind({backing.p, backing.bytes}).data, r.dtype, shape);
}

void copy(const Tensor& src, Tensor& dst, cudaStream_t stream) {
    if (src.data != dst.data)
        CUDA_CHECK(
            cudaMemcpyAsync(dst.data, src.data, src.bytes(), cudaMemcpyDeviceToDevice, stream));
}

Tensor branch(const Tensor& tensor, int index) {
    return Tensor(static_cast<std::byte*>(tensor.data) + index * tensor.nb[2], tensor.dtype,
                  {tensor.ne[0], tensor.ne[1]});
}

Tensor controls(const Tensor& tensor, int begin, int count) {
    if (tensor.ne[0] == 4608)
        return Tensor(static_cast<std::byte*>(tensor.data) + begin * tensor.nb[2], tensor.dtype,
                      {4608, 64, count});
    return Tensor(static_cast<std::byte*>(tensor.data) + begin * tensor.nb[3], tensor.dtype,
                  {64, C, 2, count});
}
} // namespace

std::size_t executor_device_bytes(const EngineOptions& options, std::uint32_t tokens) {
    return plan_state(options, tokens).bytes + scratch_layout(options).bytes +
           op_workspace(options);
}

class Executor::Impl {
public:
    const RuntimeModelView& model;
    DeviceContext& device;
    EngineOptions options;
    StateImage states;
    ScratchLayout layout;
    DeviceBuffer scratch;
    WorkspaceArena workspace;

    struct Graph {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
    };

    struct HostStepStaging {
        std::int32_t count;
        std::int32_t lane;
        std::int32_t sample_position;
        TokenId sampled;
        ops::SamplingConfig config;
        std::int32_t dspark_pos[16];
        TokenId dspark_drafts[16];
        TokenId dspark_licensed[16];
        std::int32_t dspark_licensed_count;
        std::int32_t dflash2_frontier;
        std::int32_t dflash2_valid_cols;
        std::int32_t dflash2_base_pos;
        std::int32_t batch_slots[kMaximumConcurrency];
        TokenId batch_ids[kMaximumConcurrency];
        std::int32_t batch_positions[kMaximumConcurrency];
        std::int32_t batch_sample_positions[kMaximumConcurrency];
        TokenId batch_sampled[kMaximumConcurrency];
        ops::SamplingConfig batch_configs[kMaximumConcurrency];
    };

    std::map<std::pair<std::uint32_t, std::uint32_t>, Graph> graphs;
    std::map<std::pair<int, std::uint32_t>, Graph> batch_graphs;
    std::map<std::tuple<std::uint32_t, std::uint32_t, int>, Graph> draft_graphs;
    std::map<std::tuple<std::uint32_t, std::uint32_t, int>, Graph> dflash2_draft_graphs;
    std::map<std::tuple<std::uint32_t, std::uint32_t, int>, Graph> verify_graphs;
    PinnedHostBuffer pinned_step;
    PinnedHostBuffer pinned_ids;
    PinnedHostBuffer pinned_positions;
    PinnedHostBuffer pinned_vision_idx;
    std::vector<std::int32_t> host_counts;
    std::optional<VisionWeightStream> vision_stream;
    VisionWeights active_vision_weights;
    std::size_t vision_encode_peak_bytes    = 0;
    std::size_t vision_handoff_active_bytes = 0;
    std::size_t vision_handoff_peak_bytes   = 0;
    bool last_spec_active                   = false;
    std::uint32_t last_spec_slot            = 0;
    std::uint32_t last_spec_start           = 0;
    int last_spec_w                         = 0;
    int last_spec_accepted                  = 0;

    Impl(const RuntimeModelView& m, DeviceContext& d, const EngineOptions& o,
         const StartupObserver* observer = nullptr)
        : model(m), device(d), options(o), states(o, kv_tokens(o), observer),
          layout(scratch_layout(o)), scratch(layout.bytes), workspace(op_workspace(o)),
          pinned_step(sizeof(HostStepStaging)),
          pinned_ids(chunk_capacity(o) * sizeof(TokenId)),
          pinned_positions(chunk_capacity(o) * sizeof(std::int32_t)),
          pinned_vision_idx(o.enable_vision ? chunk_capacity(o) * sizeof(std::int32_t)
                                            : sizeof(std::int32_t)),
          host_counts(static_cast<std::size_t>(V) * std::max(1U, o.max_concurrency)) {
        if (model.has_mtp && states.has_dspark() &&
            options.speculative.proposal_head != ProposalHead::Optimized) {
            Tensor w_fold_q = bf(layout.dspark_w_fold_q, 256, P);
            Tensor w_fold_s = bf(layout.dspark_w_fold_scale, 4, P);
            Tensor b_fold   = bf(layout.dspark_b_fold, P);
            ops::inkling2_dspark_fold_head(model.output_head, model.mtp.markov_hidden_out_weight,
                                           model.mtp.markov_hidden_out_bias, w_fold_q, w_fold_s,
                                           b_fold, device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
        }
        if (model.has_vision && options.enable_vision) {
            if (model.vision_overlay_enabled) {
                if (model.vision_overlay.staging_bytes > layout.vision_overlay_staging.bytes) {
                    throw std::logic_error(
                        "Inkling vision overlay staging exceeds planned scratch capacity");
                }
                auto* staging = static_cast<std::byte*>(
                    layout.vision_overlay_staging.bind({scratch.p, scratch.bytes}).data);
                vision_stream.emplace(device, model.vision_overlay, model.pinned_weights, staging);
                active_vision_weights = vision_stream->window_weights(model.vision);
            } else {
                active_vision_weights = model.vision;
            }
        }
    }

    Tensor bf(const TensorRegion& r, int a, int b = 1, int c = 1) {
        return region(r, scratch, {a, b, c});
    }

    void schedule(int t, std::uint32_t slot, std::uint32_t envelope, bool all,
                  bool capture_dspark = false, bool record_conv = false,
                  bool capture_dflash2 = false,
                  std::span<const VisionChunkSlice> vision_slices = {}) {
        auto stream = device.stream;
        Tensor cur = bf(layout.cur, C, t), residual = bf(layout.residual, C, t);
        Tensor ids = bf(layout.ids, t), positions = bf(layout.positions, t);
        Tensor count = bf(layout.count, 1), lane = bf(layout.lane, 1);
        ops::embedding(ids, model.token_embedding, cur, stream);
        ops::rmsnorm(cur, model.token_norm, 1e-6f, false, cur, stream);
        if (!vision_slices.empty()) {
            auto* handoff_base = static_cast<std::byte*>(
                layout.vision_handoff.region.bind({scratch.p, scratch.bytes}).data);
            auto* idx_base = static_cast<std::byte*>(
                layout.vision_scatter_idx.region.bind({scratch.p, scratch.bytes}).data);
            std::size_t idx_offset = 0;
            for (const auto& slice : vision_slices) {
                Tensor src(handoff_base + static_cast<std::size_t>(slice.handoff_col_offset) * C * 2,
                           DType::BF16, {C, static_cast<int>(slice.token_count)});
                Tensor idx(idx_base + idx_offset * sizeof(std::int32_t), DType::I32,
                           {static_cast<int>(slice.token_count)});
                ops::scatter(src, idx, cur, stream);
                idx_offset += slice.token_count;
            }
        }
        std::array<Tensor, 4> blocks;
        for (int i = 0; i < 4; ++i) blocks[i] = bf(layout.blocks[i], C, t);
        Tensor prelude = bf(layout.prelude, C, t), bank = bf(layout.bank, C, t),
               previous = bf(layout.previous, C, t);
        int block_count = 0, full_layer = 0, swa_layer = 0;
        Tensor mix = bf(layout.mix, C, t, 3), messages = bf(layout.messages, C, t, 4),
               inputs = bf(layout.inputs, C, t, 3);
        Tensor qgr = bf(layout.qgr, 6528, t), kraw = bf(layout.kraw, 1024, t),
               vraw = bf(layout.vraw, 1024, t);
        Tensor kc = bf(layout.kconv, 1024, t), vc = bf(layout.vconv, 1024, t);
        Tensor q = bf(layout.q, 128, 24, t), k = bf(layout.k, 128, 8, t),
               v = bf(layout.v, 128, 8, t);
        Tensor r = bf(layout.r, 16, 24, t), sink = bf(layout.sink, 24, t),
               dk = bf(layout.dynamic_k, 32, t), dv = bf(layout.dynamic_v, 32, t);
        Tensor table =
            t <= 16 ? Tensor(layout.rel.region.bind({scratch.p, scratch.bytes}).data, DType::FP32,
                             {1025, 24, t})
                    : bf(layout.rel, 1025, 24, t);
        Tensor attention = bf(layout.attention, 128, 24, t);
        Tensor projection = bf(layout.projection, C, t), moe = bf(layout.moe, C, t),
               ffn          = bf(layout.ffn_mix, C, t);
        auto capture_feat = [&](int slice, const Tensor& src) {
            if (capture_dflash2) {
                auto* dst = static_cast<std::byte*>(bf(layout.dflash2_features, 9216, t).data) +
                            static_cast<std::size_t>(slice) * C * 2;
                CUDA_CHECK(cudaMemcpy2DAsync(dst, 9216 * 2, src.data, C * 2, C * 2, t,
                                             cudaMemcpyDeviceToDevice, stream));
            } else {
                auto* dst = static_cast<std::byte*>(bf(layout.dspark_features, C, t, 4).data) +
                            static_cast<std::size_t>(slice) * C * t * 2;
                CUDA_CHECK(cudaMemcpyAsync(dst, src.data, static_cast<std::size_t>(C) * t * 2,
                                           cudaMemcpyDeviceToDevice, stream));
            }
        };
        auto conv_trace_slice = [&](int visit_idx, int comp_off, int channels) -> Tensor {
            if (!record_conv) return {};
            auto* dst = static_cast<std::byte*>(bf(layout.dspark_conv_trace, 6656 * t, 48).data) +
                        (static_cast<std::size_t>(visit_idx) * 6656 + comp_off) * t * 2;
            return Tensor(dst, DType::BF16, {channels, t});
        };
        auto states_and_cur = [&] {
            std::array<Tensor, 5> values;
            for (int i = 0; i < block_count; ++i) values[i] = blocks[i];
            values[block_count] = cur;
            return values;
        };
        auto layer = [&](int physical, int visit, bool loop) {
            if (capture_dspark || capture_dflash2) {
                if (visit == 5) capture_feat(0, cur);
                else if (visit == 23) capture_feat(1, cur);
                else if (visit == 41) capture_feat(2, cur);
            }
            const auto& w      = model.layers[physical];
            const int branches = w.full_attention ? 3 : 1;
            Tensor mixed = Tensor(mix.data, DType::BF16, {C, t, branches});
            std::array<Tensor, 3> norms;
            for (int j = 0; j < 3; ++j)
                norms[j] =
                    w.full_attention
                        ? (j == 0 ? w.attention_q_input_norm
                                  : (j == 1 ? w.attention_k_input_norm : w.attention_v_input_norm))
                        : w.attention_input_norm;
            Tensor attn_in;
            if (loop) {
                auto values = states_and_cur();
                ops::inkling2_attn_res_mix(std::span(values.data(), block_count + 1),
                                           w.attention_res_query, {}, mixed, stream);
                Tensor msg3(messages.data, DType::BF16, {C, t, 3});
                ops::inkling2_loop_control(mixed, msg3, controls(w.loop_ctrl_down, 0, 3),
                                           controls(w.loop_ctrl_gain_forget, 0, 3), norms, inputs,
                                           workspace, stream);
                attn_in = inputs;
            } else if (physical) {
                auto values = states_and_cur();
                ops::inkling2_attn_res_mix(std::span(values.data(), block_count + 1),
                                           w.attention_res_query, std::span(norms.data(), branches),
                                           mixed, stream);
                attn_in = mixed;
            } else {
                for (int j = 0; j < branches; ++j) {
                    auto dst = branch(inputs, j);
                    ops::rmsnorm(cur, norms[j], 1e-6f, false, dst, stream);
                }
                attn_in = inputs;
            }
            const int kv_idx = (loop || branches == 3) ? 1 : 0;
            const int v_idx  = (loop || branches == 3) ? 2 : 0;
            auto qi = branch(attn_in, 0), ki = branch(attn_in, kv_idx), vi = branch(attn_in, v_idx);
            ops::linear(qi, w.attention_qgr, qgr, ops::LinearPolicy::AllowA8, workspace, stream);
            ops::linear(ki, w.attention_k, kraw, ops::LinearPolicy::AllowA8, workspace, stream);
            ops::linear(vi, w.attention_v, vraw, ops::LinearPolicy::AllowA8, workspace, stream);
            if (w.full_attention) {
                ops::inkling2_small_bf16_proj(ki, w.attention_k_conv_proj, dk, stream);
                ops::inkling2_small_bf16_proj(vi, w.attention_v_conv_proj, dv, stream);
            }
            auto ks = states.convolution(slot, visit, 0), vs = states.convolution(slot, visit, 1);
            ops::inkling2_fused_sconv_kv(kraw, w.attention_k_conv, w.full_attention ? dk : Tensor{},
                                         ks, ks, kc, vraw, w.attention_v_conv,
                                         w.full_attention ? dv : Tensor{}, vs, vs, vc, stream,
                                         conv_trace_slice(visit, 0, 1024),
                                         conv_trace_slice(visit, 1024, 1024));
            ops::inkling2_qkr_clip_v_norm(qgr, kc, vc, w.attention_q_clip, w.attention_rel_clip,
                                          positions, w.full_attention, q, k, v, r, stream);
            ops::inkling2_small_bf16_proj(qi, w.attention_sink_proj, sink, stream);
            ops::inkling2_rel_bias_prepare(r, w.attention_rel_bias, sink, table, stream);
            if (w.full_attention) {
                auto cache = states.full(slot, full_layer++);
                ops::kv_cache_append(k, v, positions, cache, stream);
                ops::inkling2_full_attention(q, qgr, table, positions, cache, envelope, workspace,
                                             attention, stream);
            } else {
                auto cache = states.swa(swa_layer++);
                ops::inkling2_swa_attention(
                    q, k, v, qgr, table, positions, cache, slot,
                    std::min<std::uint32_t>(envelope, 1024U + static_cast<std::uint32_t>(t)),
                    workspace, attention, stream, t == 1);
                if (t != 1) {
                    Tensor kb(k.data, DType::BF16, {128, 8, t, 1}),
                        vb(v.data, DType::BF16, {128, 8, t, 1});
                    Tensor pos(positions.data, DType::I32, {t, 1});
                    ops::kv_cache_append_prefix(
                        kb, vb, pos, count, lane,
                        {static_cast<unsigned>(t), static_cast<unsigned>(t)}, cache, stream);
                }
            }
            Tensor flat(attention.data, DType::BF16, {3072, t});
            ops::linear(flat, w.attention_output, projection, ops::LinearPolicy::AllowA8, workspace,
                        stream);
            auto as = states.convolution(slot, visit, 2);
            auto attn_trace = conv_trace_slice(visit, 2048, C);
            if (physical % 6 == 0) {
                if (block_count >= 4)
                    throw std::logic_error("Inkling residual block schedule overflow");
                copy(cur, blocks[block_count++], stream);
                ops::inkling2_fused_sconv(projection, w.attention_post_conv, {}, as, as, {},
                                          residual, stream, attn_trace);
            } else {
                ops::inkling2_fused_sconv(projection, w.attention_post_conv, {}, as, as, cur,
                                          residual, stream, attn_trace);
            }
            std::array<Tensor, 5> values;
            for (int i = 0; i < block_count; ++i) values[i] = blocks[i];
            values[block_count] = residual;
            const std::array<Tensor, 1> fnorm{w.ffn_input_norm};
            ops::inkling2_attn_res_mix(
                std::span(values.data(), block_count + 1), w.ffn_res_query,
                loop ? std::span<const Tensor>{} : std::span<const Tensor>(fnorm), ffn, stream);
            if (loop) {
                auto msg = branch(messages, 3);
                ops::inkling2_loop_control(ffn, msg, controls(w.loop_ctrl_down, 3, 1),
                                           controls(w.loop_ctrl_gain_forget, 3, 1), fnorm, ffn,
                                           workspace, stream);
            }
            ops::Inkling2MoeWeights mw{w.ffn_router, w.ffn_exp_probs_bias, w.ffn_routed_gate_up,
                                       w.ffn_routed_down};
            auto fs = states.convolution(slot, visit, 3);
            if (t <= 16) {
                ops::inkling2_moe_post_sconv(ffn, mw, w.ffn_post_conv, fs, fs, residual, cur,
                                             conv_trace_slice(visit, 4352, C), workspace, stream);
            } else {
                ops::inkling2_moe(ffn, mw, moe, workspace, stream);
                ops::inkling2_fused_sconv(moe, w.ffn_post_conv, {}, fs, fs, residual, cur, stream,
                                          conv_trace_slice(visit, 4352, C));
            }
        };
        int visit = 0;
        for (int p = 0; p < 6; ++p) layer(p, visit++, false);
        copy(cur, prelude, stream);
        for (int pass = 0; pass < 3; ++pass) {
            if (pass) {
                copy(blocks[2], bank, stream);
                copy(cur, previous, stream);
                copy(prelude, cur, stream);
                block_count = 1;
            }
            for (int p = 6; p < 18; ++p) {
                if (pass) {
                    const std::array<Tensor, 2> bank_states{bank, previous};
                    ops::inkling2_attn_res_mix(bank_states, model.layers[p].loop_msg_query, {},
                                               messages, stream);
                }
                layer(p, visit++, pass != 0);
            }
        }
        for (int p = 18; p < 24; ++p) layer(p, visit++, false);
        auto values = states_and_cur();
        const std::array<Tensor, 1> final_norm{model.final_norm};
        ops::inkling2_attn_res_mix(std::span(values.data(), block_count + 1),
                                   model.output_res_query, final_norm, ffn, stream);
        if (capture_dspark || capture_dflash2) capture_feat(3, ffn);
        const int cols    = all ? t : 1;
        Tensor head_input = all ? ffn
                                : Tensor(static_cast<std::byte*>(ffn.data) + (t - 1) * ffn.nb[1],
                                         DType::BF16, {C, 1});
        auto logits       = bf(layout.logits, P, cols);
        ops::linear(head_input, model.output_head, logits, ops::LinearPolicy::A16Only, workspace,
                    stream);
        ops::inkling2_softcap_slice(logits, 30.0f, V, logits, stream);
    }

    void schedule_batched(int b, std::uint32_t envelope) {
        auto stream    = device.stream;
        Tensor cur     = bf(layout.cur, C, b), residual = bf(layout.residual, C, b);
        Tensor ids     = bf(layout.ids, b), positions = bf(layout.positions, b);
        Tensor b_slots = bf(layout.batch_slots, b);
        Tensor all_conv = states.all_convolutions();
        ops::embedding(ids, model.token_embedding, cur, stream);
        ops::rmsnorm(cur, model.token_norm, 1e-6f, false, cur, stream);
        std::array<Tensor, 4> blocks;
        for (int i = 0; i < 4; ++i) blocks[i] = bf(layout.blocks[i], C, b);
        Tensor prelude = bf(layout.prelude, C, b), bank = bf(layout.bank, C, b),
               previous = bf(layout.previous, C, b);
        int block_count = 0, full_layer = 0, swa_layer = 0;
        Tensor mix = bf(layout.mix, C, b, 3), messages = bf(layout.messages, C, b, 4),
               inputs = bf(layout.inputs, C, b, 3);
        Tensor qgr = bf(layout.qgr, 6528, b), kraw = bf(layout.kraw, 1024, b),
               vraw = bf(layout.vraw, 1024, b);
        Tensor kc = bf(layout.kconv, 1024, b), vc = bf(layout.vconv, 1024, b);
        Tensor q = bf(layout.q, 128, 24, b), k = bf(layout.k, 128, 8, b),
               v = bf(layout.v, 128, 8, b);
        Tensor r = bf(layout.r, 16, 24, b), sink = bf(layout.sink, 24, b),
               dk = bf(layout.dynamic_k, 32, b), dv = bf(layout.dynamic_v, 32, b);
        Tensor table(layout.rel.region.bind({scratch.p, scratch.bytes}).data, DType::FP32,
                     {1025, 24, b});
        Tensor attention = bf(layout.attention, 128, 24, b);
        Tensor projection = bf(layout.projection, C, b), moe = bf(layout.moe, C, b),
               ffn        = bf(layout.ffn_mix, C, b);
        auto states_and_cur = [&] {
            std::array<Tensor, 5> values;
            for (int i = 0; i < block_count; ++i) values[i] = blocks[i];
            values[block_count] = cur;
            return values;
        };
        auto layer = [&](int physical, int visit, bool loop) {
            const auto& w      = model.layers[physical];
            const int branches = w.full_attention ? 3 : 1;
            Tensor mixed(mix.data, DType::BF16, {C, b, branches});
            std::array<Tensor, 3> norms;
            for (int j = 0; j < 3; ++j)
                norms[j] =
                    w.full_attention
                        ? (j == 0 ? w.attention_q_input_norm
                                  : (j == 1 ? w.attention_k_input_norm : w.attention_v_input_norm))
                        : w.attention_input_norm;
            Tensor attn_in;
            if (loop) {
                auto values = states_and_cur();
                ops::inkling2_attn_res_mix(std::span(values.data(), block_count + 1),
                                           w.attention_res_query, {}, mixed, stream);
                Tensor msg3(messages.data, DType::BF16, {C, b, 3});
                ops::inkling2_loop_control(mixed, msg3, controls(w.loop_ctrl_down, 0, 3),
                                           controls(w.loop_ctrl_gain_forget, 0, 3), norms, inputs,
                                           workspace, stream);
                attn_in = inputs;
            } else if (physical) {
                auto values = states_and_cur();
                ops::inkling2_attn_res_mix(std::span(values.data(), block_count + 1),
                                           w.attention_res_query, std::span(norms.data(), branches),
                                           mixed, stream);
                attn_in = mixed;
            } else {
                for (int j = 0; j < branches; ++j) {
                    auto dst = branch(inputs, j);
                    ops::rmsnorm(cur, norms[j], 1e-6f, false, dst, stream);
                }
                attn_in = inputs;
            }
            const int kv_idx = (loop || branches == 3) ? 1 : 0;
            const int v_idx  = (loop || branches == 3) ? 2 : 0;
            auto qi = branch(attn_in, 0), ki = branch(attn_in, kv_idx), vi = branch(attn_in, v_idx);
            ops::linear(qi, w.attention_qgr, qgr, ops::LinearPolicy::A16Only, workspace, stream);
            ops::linear(ki, w.attention_k, kraw, ops::LinearPolicy::A16Only, workspace, stream);
            ops::linear(vi, w.attention_v, vraw, ops::LinearPolicy::A16Only, workspace, stream);
            if (w.full_attention) {
                ops::inkling2_small_bf16_proj(ki, w.attention_k_conv_proj, dk, stream);
                ops::inkling2_small_bf16_proj(vi, w.attention_v_conv_proj, dv, stream);
            }
            ops::inkling2_fused_sconv_kv_batched(
                kraw, w.attention_k_conv, w.full_attention ? dk : Tensor{}, kc, vraw,
                w.attention_v_conv, w.full_attention ? dv : Tensor{}, vc, all_conv, visit, b_slots,
                stream);
            ops::inkling2_qkr_clip_v_norm(qgr, kc, vc, w.attention_q_clip, w.attention_rel_clip,
                                          positions, w.full_attention, q, k, v, r, stream);
            ops::inkling2_small_bf16_proj(qi, w.attention_sink_proj, sink, stream);
            ops::inkling2_rel_bias_prepare(r, w.attention_rel_bias, sink, table, stream);
            if (w.full_attention) {
                auto cache = states.full_batch(full_layer++);
                ops::inkling2_kv_append_batched(k, v, positions, b_slots, cache, stream);
                ops::inkling2_full_attention_batched(q, qgr, table, positions, b_slots, cache,
                                                     envelope, workspace, attention, stream);
            } else {
                auto cache = states.swa(swa_layer++);
                ops::inkling2_swa_attention_batched(
                    q, k, v, qgr, table, positions, b_slots, cache,
                    std::min<std::uint32_t>(envelope, 1025U), workspace, attention, stream, true);
            }
            Tensor flat(attention.data, DType::BF16, {3072, b});
            ops::linear(flat, w.attention_output, projection, ops::LinearPolicy::A16Only, workspace,
                        stream);
            if (physical % 6 == 0) {
                if (block_count >= 4)
                    throw std::logic_error("Inkling residual block schedule overflow");
                copy(cur, blocks[block_count++], stream);
                ops::inkling2_fused_sconv_batched(projection, w.attention_post_conv, all_conv,
                                                  visit, 2, b_slots, {}, residual, stream);
            } else {
                ops::inkling2_fused_sconv_batched(projection, w.attention_post_conv, all_conv,
                                                  visit, 2, b_slots, cur, residual, stream);
            }
            std::array<Tensor, 5> values;
            for (int i = 0; i < block_count; ++i) values[i] = blocks[i];
            values[block_count] = residual;
            const std::array<Tensor, 1> fnorm{w.ffn_input_norm};
            ops::inkling2_attn_res_mix(
                std::span(values.data(), block_count + 1), w.ffn_res_query,
                loop ? std::span<const Tensor>{} : std::span<const Tensor>(fnorm), ffn, stream);
            if (loop) {
                auto msg = branch(messages, 3);
                ops::inkling2_loop_control(ffn, msg, controls(w.loop_ctrl_down, 3, 1),
                                           controls(w.loop_ctrl_gain_forget, 3, 1), fnorm, ffn,
                                           workspace, stream);
            }
            ops::Inkling2MoeWeights mw{w.ffn_router, w.ffn_exp_probs_bias, w.ffn_routed_gate_up,
                                       w.ffn_routed_down};
            ops::inkling2_moe(ffn, mw, moe, workspace, stream);
            ops::inkling2_fused_sconv_batched(moe, w.ffn_post_conv, all_conv, visit, 3, b_slots,
                                              residual, cur, stream);
        };
        int visit = 0;
        for (int p = 0; p < 6; ++p) layer(p, visit++, false);
        copy(cur, prelude, stream);
        for (int pass = 0; pass < 3; ++pass) {
            if (pass) {
                copy(blocks[2], bank, stream);
                copy(cur, previous, stream);
                copy(prelude, cur, stream);
                block_count = 1;
            }
            for (int p = 6; p < 18; ++p) {
                if (pass) {
                    const std::array<Tensor, 2> bank_states{bank, previous};
                    ops::inkling2_attn_res_mix(bank_states, model.layers[p].loop_msg_query, {},
                                               messages, stream);
                }
                layer(p, visit++, pass != 0);
            }
        }
        for (int p = 18; p < 24; ++p) layer(p, visit++, false);
        auto values = states_and_cur();
        const std::array<Tensor, 1> final_norm{model.final_norm};
        ops::inkling2_attn_res_mix(std::span(values.data(), block_count + 1),
                                   model.output_res_query, final_norm, ffn, stream);
        auto logits = bf(layout.logits, P, b);
        ops::linear(ffn, model.output_head, logits, ops::LinearPolicy::A16Only, workspace, stream);
        ops::inkling2_softcap_slice(logits, 30.0f, V, logits, stream);
    }

    void mix_and_append_dspark(int feat_stride, int append_count, std::uint32_t start,
                               std::uint32_t slot, bool already_mixed = false) {
        auto stream = device.stream;
        Tensor mixed = bf(layout.dspark_mixed, C, feat_stride, 4);
        if (!already_mixed) {
            Tensor feats = bf(layout.dspark_features, C, feat_stride, 4);
            std::array<Tensor, 4> norms{model.mtp.layers[0].dflare_norm,
                                        model.mtp.layers[1].dflare_norm,
                                        model.mtp.layers[2].dflare_norm,
                                        model.mtp.markov_dflare_norm};
            ops::inkling2_dspark_feature_mix(feats, model.mtp.fuse_weight, norms, mixed, stream);
        }
        if (append_count <= 0) return;
        const std::size_t slice_bytes = static_cast<std::size_t>(C) * feat_stride * 2;
        auto* mixed_bytes             = static_cast<std::byte*>(mixed.data);
        auto* c_head_dst              = static_cast<std::byte*>(
                               bf(layout.dspark_c_head, C, static_cast<int>(states.slots())).data) +
                           static_cast<std::size_t>(slot) * C * 2;
        CUDA_CHECK(cudaMemcpyAsync(
            c_head_dst, mixed_bytes + 3 * slice_bytes + static_cast<std::size_t>(append_count - 1) * C * 2,
            C * 2, cudaMemcpyDeviceToDevice, stream));

        auto* step  = static_cast<HostStepStaging*>(pinned_step.data());
        step->count = append_count;
        step->lane  = static_cast<int>(slot);
        auto count_t = bf(layout.count, 1), lane_t = bf(layout.lane, 1);
        CUDA_CHECK(cudaMemcpyAsync(count_t.data, &step->count, sizeof(int), cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(lane_t.data, &step->lane, sizeof(int), cudaMemcpyHostToDevice,
                                   stream));
        auto* host_pos = static_cast<std::int32_t*>(pinned_positions.data());
        for (int i = 0; i < append_count; ++i) host_pos[i] = static_cast<int>(start) + i;
        auto pos = bf(layout.positions, append_count);
        CUDA_CHECK(cudaMemcpyAsync(pos.data, host_pos, append_count * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        auto k_ctx = bf(layout.dspark_k, 1024, append_count);
        auto v_ctx = bf(layout.dspark_v, 1024, append_count);
        Tensor k3d(k_ctx.data, DType::BF16, {128, 8, append_count});
        Tensor v3d(v_ctx.data, DType::BF16, {128, 8, append_count});
        Tensor kb(k_ctx.data, DType::BF16, {128, 8, append_count, 1});
        Tensor vb(v_ctx.data, DType::BF16, {128, 8, append_count, 1});
        Tensor pos_2d(pos.data, DType::I32, {append_count, 1});
        int swa_idx = 0;
        for (int layer = 0; layer < 3; ++layer) {
            const auto& ml = model.mtp.layers[layer];
            Tensor c_r(mixed_bytes + static_cast<std::size_t>(layer) * slice_bytes, DType::BF16,
                       {C, append_count});
            ops::linear(c_r, ml.k_ctx, k_ctx, ops::LinearPolicy::A16Only, workspace, stream);
            ops::linear(c_r, ml.v_ctx, v_ctx, ops::LinearPolicy::A16Only, workspace, stream);
            Tensor empty_k;
            ops::inkling2_dspark_rmsnorm_rope(pos, ml.k_norm_ctx, empty_k, k3d, empty_k, 1e6f,
                                              stream);
            if (layer == 1) {
                auto full_cache = states.dspark_full(slot);
                ops::kv_cache_append(k3d, v3d, pos, full_cache, stream);
            } else {
                auto swa_cache = states.dspark_swa(swa_idx++);
                ops::kv_cache_append_prefix(
                    kb, vb, pos_2d, count_t, lane_t,
                    {static_cast<unsigned>(append_count), static_cast<unsigned>(append_count)},
                    swa_cache, stream);
            }
        }
        states.commit_dspark_append(slot, start + static_cast<std::uint32_t>(append_count));
    }

    void draft_dspark_body(std::uint32_t state_slot, int K, std::uint32_t dspark_envelope) {
        const int W  = K + 1;
        auto stream  = device.stream;
        auto pos     = bf(layout.dspark_pos, W);
        Tensor pos0(pos.data, DType::I32, {1});
        auto draft_toks = bf(layout.dspark_draft_tok, W);
        Tensor prev_tok0(draft_toks.data, DType::I32, {1});

        auto block = bf(layout.dspark_block, C, W);
        auto norm  = bf(layout.dspark_norm, C, W);
        Tensor norm_col0(norm.data, DType::BF16, {C, 1});
        Tensor block_col0(block.data, DType::BF16, {C, 1});
        ops::embedding(prev_tok0, model.token_embedding, norm_col0, stream);
        ops::rmsnorm(norm_col0, model.token_norm, 1e-6f, false, block_col0, stream);
        for (int i = 1; i < W; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::byte*>(block.data) + static_cast<std::size_t>(i) * C * 2,
                model.mtp.mask_embed.data, C * 2, cudaMemcpyDeviceToDevice, stream));
        }

        auto q    = bf(layout.dspark_q, 128, 24, W);
        auto k    = bf(layout.dspark_k, 128, 8, W);
        auto v    = bf(layout.dspark_v, 128, 8, W);
        Tensor q_flat(q.data, DType::BF16, {3072, W});
        Tensor k_flat(k.data, DType::BF16, {1024, W});
        Tensor v_flat(v.data, DType::BF16, {1024, W});
        auto attn = bf(layout.dspark_attn, 128, 24, W);
        Tensor attn_flat(attn.data, DType::BF16, {3072, W});
        auto proj = bf(layout.dspark_proj, C, W);
        auto gu   = bf(layout.dspark_mlp_gu, 18432, W);
        auto act  = bf(layout.dspark_mlp_act, 9216, W);
        Tensor gate = gu.slice(0, 0, 9216);
        Tensor up   = gu.slice(0, 9216, 9216);

        int swa_idx = 0;
        for (int layer = 0; layer < 3; ++layer) {
            const auto& ml = model.mtp.layers[layer];
            ops::rmsnorm(block, ml.input_norm, 1e-6f, false, norm, stream);
            ops::linear(norm, ml.q, q_flat, ops::LinearPolicy::A16Only, workspace, stream);
            ops::linear(norm, ml.k, k_flat, ops::LinearPolicy::A16Only, workspace, stream);
            ops::linear(norm, ml.v, v_flat, ops::LinearPolicy::A16Only, workspace, stream);
            ops::inkling2_dspark_rmsnorm_rope(pos, ml.q_norm, ml.k_norm, q, k, 1e6f, stream);
            if (layer == 1) {
                auto full_cache = states.dspark_full(state_slot);
                ops::inkling2_dspark_full_attention(q, k, v, full_cache, pos0, dspark_envelope,
                                                    workspace, attn, stream);
            } else {
                auto swa_cache = states.dspark_swa(swa_idx++);
                ops::inkling2_dspark_swa_attention(q, k, v, pos0, swa_cache,
                                                   static_cast<int>(state_slot), attn, stream);
            }
            ops::linear(attn_flat, ml.output, proj, ops::LinearPolicy::A16Only, workspace, stream);
            ops::residual_add(proj, block, stream);

            ops::rmsnorm(block, ml.post_attention_norm, 1e-6f, false, norm, stream);
            ops::linear(norm, ml.gate_up, gu, ops::LinearPolicy::A16Only, workspace, stream);
            ops::silu_mul(gate, up, act, stream);
            ops::linear(act, ml.down, proj, ops::LinearPolicy::A16Only, workspace, stream);
            ops::residual_add(proj, block, stream);
        }
        ops::rmsnorm(block, model.mtp.final_norm, 1e-6f, false, norm, stream);

        Tensor rnn_state(layout.dspark_rnn_state.region.bind({scratch.p, scratch.bytes}).data,
                         DType::FP32, {256});
        CUDA_CHECK(cudaMemsetAsync(rnn_state.data, 0, 256 * sizeof(float), stream));
        Tensor c_head(
            static_cast<std::byte*>(
                bf(layout.dspark_c_head, C, static_cast<int>(states.slots())).data) +
                static_cast<std::size_t>(state_slot) * C * 2,
            DType::BF16, {C});
        const bool use_shortlist =
            options.speculative.proposal_head == ProposalHead::Optimized && model.has_draft_head;
        if (use_shortlist) {
            auto h_prime = bf(layout.dspark_h_prime, C);
            Tensor h_prime_2d(h_prime.data, DType::BF16, {C, 1});
            auto draft_logits = bf(layout.dspark_draft_logits, P, 1);
            Tensor draft_logits_short(draft_logits.data, DType::BF16, {32768, 1});
            for (int i = 0; i < K; ++i) {
                Tensor h_in(
                    static_cast<std::byte*>(norm.data) + static_cast<std::size_t>(i + 1) * C * 2,
                    DType::BF16, {C});
                Tensor cur_tok(static_cast<std::byte*>(draft_toks.data) +
                                   static_cast<std::size_t>(i) * sizeof(TokenId),
                               DType::I32, {1});
                Tensor next_tok(static_cast<std::byte*>(draft_toks.data) +
                                    static_cast<std::size_t>(i + 1) * sizeof(TokenId),
                                DType::I32, {1});
                ops::inkling2_dspark_rnn_step(
                    cur_tok, model.mtp.markov_w1, model.mtp.markov_rnn_proj_weight,
                    model.mtp.markov_rnn_proj_bias, model.mtp.markov_hidden_out_weight,
                    model.mtp.markov_hidden_out_bias, h_in, c_head, rnn_state, h_prime, workspace,
                    stream);
                ops::linear(h_prime_2d, model.mtp.draft_head, draft_logits_short,
                            ops::LinearPolicy::A16Only, workspace, stream);
                ops::argmax(draft_logits_short, next_tok, 32768, stream);
                ops::proposal_remap_token_ids(
                    next_tok,
                    static_cast<const std::int32_t*>(model.mtp.draft_head_token_ids.data), 32768,
                    stream);
            }
        } else {
            Tensor norm_drafts(static_cast<std::byte*>(norm.data) + static_cast<std::size_t>(C) * 2,
                               DType::BF16, {C, K});
            auto base_logits = bf(layout.dspark_draft_logits, P, K);
            ops::linear(norm_drafts, model.output_head, base_logits, ops::LinearPolicy::A16Only,
                        workspace, stream);
            Tensor w_fold_q = bf(layout.dspark_w_fold_q, 256, P);
            Tensor w_fold_s = bf(layout.dspark_w_fold_scale, 4, P);
            Tensor b_fold   = bf(layout.dspark_b_fold, P);
            for (int i = 0; i < K; ++i) {
                Tensor h_in(
                    static_cast<std::byte*>(norm.data) + static_cast<std::size_t>(i + 1) * C * 2,
                    DType::BF16, {C});
                Tensor base_col(static_cast<std::byte*>(base_logits.data) +
                                    static_cast<std::size_t>(i) * P * 2,
                                DType::BF16, {P});
                Tensor cur_tok(static_cast<std::byte*>(draft_toks.data) +
                                   static_cast<std::size_t>(i) * sizeof(TokenId),
                               DType::I32, {1});
                Tensor next_tok(static_cast<std::byte*>(draft_toks.data) +
                                    static_cast<std::size_t>(i + 1) * sizeof(TokenId),
                                DType::I32, {1});
                ops::inkling2_dspark_rnn_step_folded(
                    cur_tok, model.mtp.markov_w1, model.mtp.markov_rnn_proj_weight,
                    model.mtp.markov_rnn_proj_bias, h_in, c_head, base_col, w_fold_q, w_fold_s,
                    b_fold, V, rnn_state, next_tok, workspace, stream);
            }
        }
    }

    void append_dflash2_context(int append_count, std::uint32_t start, std::uint32_t slot) {
        if (append_count <= 0) return;
        auto stream    = device.stream;
        Tensor feats   = bf(layout.dflash2_features, 9216, append_count);
        Tensor ctx_p   = bf(layout.dflash2_ctx_proj, C, append_count);
        Tensor ctx     = bf(layout.dflash2_ctx, C, append_count);
        ops::linear(feats, model.dflash2.fc, ctx_p, ops::LinearPolicy::AllowA8, workspace, stream);
        ops::rmsnorm(ctx_p, model.dflash2.hidden_norm, 1e-6f, false, ctx, stream);

        auto* step     = static_cast<HostStepStaging*>(pinned_step.data());
        step->count    = append_count;
        step->lane     = static_cast<int>(slot);
        auto count_t   = bf(layout.count, 1);
        auto lane_t    = bf(layout.lane, 1);
        CUDA_CHECK(cudaMemcpyAsync(count_t.data, &step->count, sizeof(int), cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(lane_t.data, &step->lane, sizeof(int), cudaMemcpyHostToDevice,
                                   stream));
        auto* host_pos = static_cast<std::int32_t*>(pinned_positions.data());
        for (int i = 0; i < append_count; ++i) host_pos[i] = static_cast<int>(start) + i;
        auto pos_2d = bf(layout.positions, append_count, 1);
        CUDA_CHECK(cudaMemcpyAsync(pos_2d.data, host_pos, append_count * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));

        std::array<ops::ContextKVMaterializeLayerView, 5> layers{};
        for (std::size_t l = 0; l < 5; ++l) {
            const auto& dl            = model.dflash2.layers[l];
            layers[l].key_weight      = dl.k;
            layers[l].value_weight    = dl.v;
            layers[l].key_norm_weight = dl.k_norm;
            layers[l].cache           = states.dflash2_swa(static_cast<int>(l));
        }
        Tensor ctx_3d(ctx.data, DType::BF16, {C, append_count, 1});
        const ops::ContextKVMaterializeExecutionEnvelope env{
            static_cast<std::uint32_t>(append_count), static_cast<std::uint32_t>(append_count)};
        ops::inkling2_dflash2_kv_materialize(ctx_3d, pos_2d, count_t, lane_t, layers, env,
                                             workspace, stream);
        states.commit_dflash2_append(slot, start + static_cast<std::uint32_t>(append_count));
    }

    void draft_dflash2_body(std::uint32_t, int K, std::uint32_t dflash_envelope,
                            const ops::SamplingConfig* device_config) {
        const int W        = K + 1;
        auto stream        = device.stream;
        auto anchors_t     = bf(layout.dspark_anchors, 1);
        auto lengths_t     = bf(layout.dspark_lengths, 1);
        auto valid_cols_t  = bf(layout.dflash2_valid_cols, 1);
        auto d2_ids_2d     = bf(layout.dflash2_ids, W, 1);
        auto d2_pos_2d     = bf(layout.dflash2_pos, W, 1);
        auto lane_t        = bf(layout.lane, 1);
        ops::prepare_masked_block(anchors_t, lengths_t, valid_cols_t, 200056, d2_ids_2d, d2_pos_2d,
                                  stream);

        Tensor ids_1d(d2_ids_2d.data, DType::I32, {W});
        Tensor pos_1d(d2_pos_2d.data, DType::I32, {W});
        auto res = bf(layout.dflash2_res, C, W);
        ops::embedding(ids_1d, model.token_embedding, res, stream);
        ops::rmsnorm(res, model.token_norm, 1e-6f, false, res, stream);

        auto prep  = bf(layout.dflash2_prep, C, W);
        auto delta = bf(layout.dflash2_delta, 144, 2, W);
        auto q_4d  = region(layout.dflash2_q, scratch, {128, 32, W, 1});
        auto k_4d  = region(layout.dflash2_k, scratch, {128, 8, W, 1});
        auto v_4d  = region(layout.dflash2_v, scratch, {128, 8, W, 1});
        auto a_4d  = region(layout.dflash2_attn, scratch, {128, 32, W, 1});
        Tensor q_3d(q_4d.data, DType::BF16, {128, 32, W});
        Tensor k_3d(k_4d.data, DType::BF16, {128, 8, W});
        Tensor q_flat(q_4d.data, DType::BF16, {4096, W});
        Tensor k_flat(k_4d.data, DType::BF16, {1024, W});
        Tensor v_flat(v_4d.data, DType::BF16, {1024, W});
        Tensor a_flat(a_4d.data, DType::BF16, {4096, W});
        auto proj = bf(layout.dflash2_proj, C, W);
        auto gu   = bf(layout.dflash2_gu, 13824, W);
        auto act  = bf(layout.dflash2_act, 6912, W);
        Tensor gate = gu.slice(0, 0, 6912);
        Tensor up   = gu.slice(0, 6912, 6912);
        const float attn_scale = 1.0f / std::sqrt(128.0f);

        for (int layer = 0; layer < 5; ++layer) {
            const auto& dl = model.dflash2.layers[layer];
            ops::inkling2_dflash2_conv_prepare(res, dl.input_norm, 1e-6f, dl.attn_conv_base,
                                               dl.attn_conv_proj, prep, delta, workspace, stream);
            ops::linear(prep, dl.q, q_flat, ops::LinearPolicy::A16Only, workspace, stream);
            ops::linear(prep, dl.k, k_flat, ops::LinearPolicy::A16Only, workspace, stream);
            ops::linear(prep, dl.v, v_flat, ops::LinearPolicy::A16Only, workspace, stream);
            ops::inkling2_dspark_rmsnorm_rope(pos_1d, dl.q_norm, dl.k_norm, q_3d, k_3d, 10000.0f,
                                              stream);
            ops::sliding_window_attention(q_4d, k_4d, v_4d, d2_pos_2d, valid_cols_t, lane_t,
                                          {128, 32, 8}, 2048U, attn_scale,
                                          states.dflash2_swa(layer), {0U, dflash_envelope},
                                          workspace, a_4d, stream);
            ops::linear(a_flat, dl.output, proj, ops::LinearPolicy::A16Only, workspace, stream);
            ops::inkling2_dflash2_conv_finish_add(proj, dl.attn_conv_base, delta, res, stream);

            ops::inkling2_dflash2_conv_prepare(res, dl.post_attention_norm, 1e-6f,
                                               dl.mlp_conv_base, dl.mlp_conv_proj, prep, delta,
                                               workspace, stream);
            ops::linear(prep, dl.gate_up, gu, ops::LinearPolicy::A16Only, workspace, stream);
            ops::silu_mul(gate, up, act, stream);
            ops::linear(act, dl.down, proj, ops::LinearPolicy::A16Only, workspace, stream);
            ops::inkling2_dflash2_conv_finish_add(proj, dl.mlp_conv_base, delta, res, stream);
        }

        Tensor res_tail(static_cast<std::byte*>(res.data) + static_cast<std::size_t>(C) * 2U,
                        DType::BF16, {C, K});
        auto mask_h       = bf(layout.dflash2_mask_h, C, K);
        auto unary_logits = bf(layout.dflash2_unary_logits, P, K);
        ops::rmsnorm(res_tail, model.dflash2.final_norm, 1e-6f, false, mask_h, stream);
        ops::linear(mask_h, model.output_head, unary_logits, ops::LinearPolicy::A16Only, workspace,
                    stream);

        auto cand_ids   = bf(layout.dflash2_cand_ids, 16, K, 1);
        auto drafts_2d  = bf(layout.dflash2_drafts, K, 1);
        auto proposal_q = bf(layout.dflash2_proposal_q, 16, K, 1);
        Tensor base_pos_t(bf(layout.sample_position, 1).data, DType::I32, {1});
        ops::inkling2_dflash2_topk16_and_select(
            mask_h, unary_logits, model.dflash2.selector_hidden_proj, anchors_t,
            model.dflash2.predecessor_codebook, model.dflash2.successor_codebook, base_pos_t,
            device_config, cand_ids, drafts_2d, proposal_q, workspace, stream);

        auto extents_t   = bf(layout.dspark_extents, 1);
        auto verify_pos  = bf(layout.positions, W, 1);
        auto verify_ids  = bf(layout.ids, W, 1);
        Tensor base_pos_verify(bf(layout.dspark_pos, 1).data, DType::I32, {1});
        ops::speculative_prepare_verify_inputs(anchors_t, drafts_2d, base_pos_verify, extents_t,
                                               verify_ids, verify_pos, stream);
    }
};

Executor::Executor(const RuntimeModelView& model, DeviceContext& device,
                   const EngineOptions& options, const StartupObserver* observer)
    : impl_(std::make_unique<Impl>(model, device, options, observer)) {}

Executor::~Executor() {
    if (impl_) {
        try {
            sync();
        } catch (...) {}
    }
}

Tensor Executor::forward(std::span<const TokenId> ids, std::uint32_t start, std::uint32_t slot,
                         bool all, std::span<const VisionChunkSlice> vision_slices) {
    auto& p     = *impl_;
    const int t = static_cast<int>(ids.size());
    if (ids.empty() || ids.size() > chunk_capacity(p.options) ||
        (all && t > head_columns(p.options)))
        throw std::invalid_argument("Inkling forward chunk exceeds planned token capacity");
    for (auto id : ids)
        if (id < 0 || id >= V) throw std::invalid_argument("Inkling token ID outside vocabulary");
    // Finish prior use of pinned table staging and transient storage before mutating the maps.
    sync();
    if (!vision_slices.empty()) {
        if (!p.options.enable_vision || !p.model.has_vision)
            throw std::invalid_argument("Inkling vision slices require enabled vision model");
        const std::uint32_t cap = v_item_max(p.options);
        auto* host_idx          = static_cast<std::int32_t*>(p.pinned_vision_idx.data());
        std::size_t total_idx   = 0;
        for (const auto& s : vision_slices) {
            if (s.token_count == 0 || s.handoff_col_offset + s.token_count > cap ||
                s.chunk_col_begin + s.token_count > static_cast<std::uint32_t>(t) ||
                total_idx + s.token_count > static_cast<std::size_t>(t))
                throw std::invalid_argument("Inkling vision slice exceeds capacity");
            for (std::uint32_t j = 0; j < s.token_count; ++j)
                host_idx[total_idx++] = static_cast<std::int32_t>(s.chunk_col_begin + j);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            p.layout.vision_scatter_idx.region.bind({p.scratch.p, p.scratch.bytes}).data, host_idx,
            total_idx * sizeof(std::int32_t), cudaMemcpyHostToDevice, p.device.stream));
    }
    p.states.prepare_append(slot, start, t, p.device.stream);
    auto* host_ids = static_cast<TokenId*>(p.pinned_ids.data());
    std::memcpy(host_ids, ids.data(), ids.size_bytes());
    auto* host_pos = static_cast<std::int32_t*>(p.pinned_positions.data());
    for (int i = 0; i < t; ++i) host_pos[i] = static_cast<int>(start) + i;
    auto positions = p.bf(p.layout.positions, t), tokens = p.bf(p.layout.ids, t);
    CUDA_CHECK(cudaMemcpyAsync(tokens.data, host_ids, ids.size_bytes(), cudaMemcpyHostToDevice,
                               p.device.stream));
    CUDA_CHECK(cudaMemcpyAsync(positions.data, host_pos, t * sizeof(int),
                               cudaMemcpyHostToDevice, p.device.stream));
    auto* step  = static_cast<Impl::HostStepStaging*>(p.pinned_step.data());
    step->count = t;
    step->lane  = static_cast<int>(slot);
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.count, 1).data, &step->count, sizeof(int),
                               cudaMemcpyHostToDevice, p.device.stream));
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.lane, 1).data, &step->lane, sizeof(int),
                               cudaMemcpyHostToDevice, p.device.stream));
    const bool capture_dspark = p.model.has_mtp && p.states.has_dspark() && !all &&
                                p.options.purpose != EnginePurpose::CausalScoring;
    const bool capture_dflash2 = p.model.has_dflash2 && p.states.has_dflash2() && !all &&
                                 p.options.purpose != EnginePurpose::CausalScoring;
    if (p.options.use_cuda_graph && t == 1 && !all && vision_slices.empty()) {
        std::uint32_t envelope = std::min(256U, p.options.max_context);
        while (envelope < start + 1) envelope = std::min(envelope * 2, p.options.max_context);
        auto& graph = p.graphs[{slot, envelope}];
        if (!graph.executable.ready()) {
            sync();
            graph.definition.capture(p.device.stream, [&] {
                p.schedule(1, slot, envelope, false, capture_dspark, false, capture_dflash2);
            });
            graph.executable.instantiate(graph.definition);
        }
        graph.executable.launch(p.device.stream);
    } else
        p.schedule(t, slot, start + t, all, capture_dspark, false, capture_dflash2, vision_slices);
    p.states.commit_append(slot, start + t);
    if (capture_dspark) p.mix_and_append_dspark(t, t, start, slot);
    if (capture_dflash2) p.append_dflash2_context(t, start, slot);
    return p.bf(p.layout.logits, P, all ? t : 1);
}

std::uint32_t Executor::vision_handoff_capacity_tokens() const noexcept {
    return impl_->options.enable_vision ? v_item_max(impl_->options) : 0;
}

double Executor::encode_vision_item(const PreparedMediaPayload& payload,
                                    std::uint32_t handoff_col_offset) {
    auto& p = *impl_;
    if (!p.options.enable_vision || !p.model.has_vision)
        throw std::logic_error("Inkling encode_vision_item called without enabled vision");
    const std::uint32_t p_nat = payload.native_patches;
    const std::uint32_t v_tok = payload.merged_tokens;
    if (p_nat == 0 || p_nat > v_native_max(p.options) || v_tok == 0 ||
        handoff_col_offset + v_tok > v_item_max(p.options))
        throw std::invalid_argument("Inkling vision item exceeds configured vision capacity");

    sync();
    const auto t0      = std::chrono::steady_clock::now();
    auto stream        = p.device.stream;
    const int P_vis    = static_cast<int>(p_nat);
    const int T_vis    = P_vis + 1;
    const int V_vis    = static_cast<int>(v_tok);
    const DeviceSpan backing{p.scratch.p, p.scratch.bytes};

    Tensor tp(p.layout.vit_patches.region.bind(backing).data, DType::BF16, {1200, P_vis});
    Tensor tcoords(p.layout.vit_coords.region.bind(backing).data, DType::I32, {4, T_vis});
    Tensor tpi(p.layout.vit_pos_idx.region.bind(backing).data, DType::I32, {4, P_vis});
    Tensor tpw(p.layout.vit_pos_w.region.bind(backing).data, DType::FP32, {4, P_vis});
    Tensor tmerge_idx(p.layout.vit_merge_idx.region.bind(backing).data, DType::I32, {4, V_vis});

    CUDA_CHECK(cudaMemcpyAsync(tp.data, payload.patches.data(),
                               payload.patches.size() * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(tcoords.data, payload.coords.data(),
                               payload.coords.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(tpi.data, payload.pos_indices.data(),
                               payload.pos_indices.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(tpw.data, payload.pos_weights.data(),
                               payload.pos_weights.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(tmerge_idx.data, payload.merge_indices.data(),
                               payload.merge_indices.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));

    Tensor tx(p.layout.vit_x.region.bind(backing).data, DType::BF16, {1024, T_vis});
    Tensor tln(p.layout.vit_ln.region.bind(backing).data, DType::BF16, {1024, T_vis});
    Tensor tqkv(p.layout.vit_qkv.region.bind(backing).data, DType::BF16, {3072, T_vis});
    Tensor tattn(p.layout.vit_attn.region.bind(backing).data, DType::BF16, {1024, T_vis});
    Tensor tproj(p.layout.vit_proj.region.bind(backing).data, DType::BF16, {1024, T_vis});
    Tensor tfc1(p.layout.vit_fc1.region.bind(backing).data, DType::BF16, {4096, T_vis});
    Tensor tmerged(p.layout.vit_merged.region.bind(backing).data, DType::BF16, {4096, V_vis});
    Tensor tproj_in(p.layout.vit_proj_in.region.bind(backing).data, DType::BF16, {4096, V_vis});

    const auto& w = p.active_vision_weights;
    if (p.vision_stream) {
        p.vision_stream->reset(stream);
        p.vision_stream->prelude_ready(stream);
    }

    ops::inkling2_vision_patch_embed(tp, w.patch_embedding, w.class_embedding,
                                     w.cls_position_embedding, w.position_embedding, tpi, tpw,
                                     w.ln_pre_weight, w.ln_pre_bias, 1e-5f, tx, stream);

    for (std::uint32_t l = 0; l < kVisionLayers; ++l) {
        if (p.vision_stream) p.vision_stream->begin_layer(l, stream);
        const auto& wl = w.layers[l];
        ops::layer_norm(tx, wl.ln_1_weight, wl.ln_1_bias, 1e-5f, tln, stream);
        ops::linear(tln, wl.qkv, tqkv, ops::LinearPolicy::A16Only, p.workspace, stream);
        ops::add_bias(wl.qkv_bias, tqkv, stream);
        ops::inkling2_vision_rope_2d(tcoords, 10000.0f, tqkv, stream);
        Tensor qkv3   = tqkv.reshape({64, 48, T_vis});
        Tensor q      = qkv3.slice(1, 0, 16);
        Tensor k      = qkv3.slice(1, 16, 16);
        Tensor v      = qkv3.slice(1, 32, 16);
        Tensor tattn3 = tattn.reshape({64, 16, T_vis});
        ops::packed_softmax_attention(q, k, v, {64, 16, 16}, 0.125f, T_vis, tattn3, stream);
        ops::linear(tattn, wl.output, tproj, ops::LinearPolicy::A16Only, p.workspace, stream);
        ops::inkling2_vision_layerscale_add(tproj, wl.output_bias, wl.ls_1_gamma, tx, stream);

        ops::layer_norm(tx, wl.ln_2_weight, wl.ln_2_bias, 1e-5f, tln, stream);
        ops::linear(tln, wl.fc1, tfc1, ops::LinearPolicy::A16Only, p.workspace, stream);
        ops::add_bias(wl.fc1_bias, tfc1, stream);
        ops::gelu(tfc1, ops::GeluMode::Exact, stream);
        ops::linear(tfc1, wl.fc2, tproj, ops::LinearPolicy::A16Only, p.workspace, stream);
        ops::inkling2_vision_layerscale_add(tproj, wl.fc2_bias, wl.ls_2_gamma, tx, stream);
        if (p.vision_stream) p.vision_stream->end_layer(l, stream);
    }

    ops::inkling2_vision_merge_2x2(tx, tmerge_idx, tmerged, stream);
    if (p.vision_stream) p.vision_stream->merger_ready(stream);
    ops::linear(tmerged, w.proj_in, tproj_in, ops::LinearPolicy::A16Only, p.workspace, stream);
    ops::gelu(tproj_in, ops::GeluMode::Exact, stream);
    auto* handoff_ptr = static_cast<std::byte*>(p.layout.vision_handoff.region.bind(backing).data) +
                        static_cast<std::size_t>(handoff_col_offset) * C * 2;
    Tensor handoff_dst(handoff_ptr, DType::BF16, {C, V_vis});
    ops::linear(tproj_in, w.proj_out, handoff_dst, ops::LinearPolicy::A16Only, p.workspace, stream);

    sync();
    if (p.vision_stream) CUDA_CHECK(cudaStreamSynchronize(p.device.transfer_stream));
    p.vision_encode_peak_bytes    = std::max(p.vision_encode_peak_bytes, p.layout.vision_general_bytes);
    p.vision_handoff_active_bytes = static_cast<std::size_t>(handoff_col_offset + v_tok) * C * 2;
    p.vision_handoff_peak_bytes =
        std::max(p.vision_handoff_peak_bytes, p.vision_handoff_active_bytes);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

TokenId Executor::sample(const Tensor& logits, const ResolvedSamplingParameters& sampling,
                         std::uint32_t logical_position, std::span<const TokenId> history) {
    auto& p                = *impl_;
    auto stream            = p.device.stream;
    const bool use_penalties =
        sampling.presence_penalty != 0.0f || sampling.frequency_penalty != 0.0f;
    int* counts_ptr = nullptr;
    if (use_penalties) {
        std::fill_n(p.host_counts.data(), V, 0);
        for (auto id : history)
            if (id >= 0 && id < V) ++p.host_counts[id];
        auto counts = p.bf(p.layout.counts, V);
        CUDA_CHECK(cudaMemcpyAsync(counts.data, p.host_counts.data(), V * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        counts_ptr = static_cast<int*>(counts.data);
    }
    auto* step           = static_cast<Impl::HostStepStaging*>(p.pinned_step.data());
    step->config         = ops::SamplingConfig{sampling.temperature,
                                               sampling.top_k,
                                               sampling.top_p,
                                               sampling.min_p,
                                               sampling.presence_penalty,
                                               sampling.frequency_penalty,
                                               sampling.seed,
                                               counts_ptr};
    step->sample_position = static_cast<int>(logical_position);
    auto* device_config   = static_cast<ops::SamplingConfig*>(
        p.layout.config.bind({p.scratch.p, p.scratch.bytes}).data);
    auto position = p.bf(p.layout.sample_position, 1), out = p.bf(p.layout.sampled, 1);
    CUDA_CHECK(cudaMemcpyAsync(device_config, &step->config, sizeof(step->config),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(position.data, &step->sample_position, sizeof(step->sample_position),
                               cudaMemcpyHostToDevice, stream));
    ops::sample(logits, out, V, device_config, position, ops::kSamplePurposeDecode, p.workspace,
                stream);
    CUDA_CHECK(cudaMemcpyAsync(&step->sampled, out.data, sizeof(step->sampled),
                               cudaMemcpyDeviceToHost, stream));
    sync();
    return step->sampled;
}

std::vector<TokenId> Executor::decode_batch(std::span<const DecodeRow> rows) {
    auto& p     = *impl_;
    if (rows.empty() || rows.size() > p.options.max_concurrency)
        throw std::invalid_argument("Inkling decode_batch: invalid batch sizes");
    const int b = static_cast<int>(rows.size());
    for (const auto& row : rows)
        if (row.id < 0 || row.id >= V)
            throw std::invalid_argument("Inkling token ID outside vocabulary");

    auto stream            = p.device.stream;
    std::uint32_t max_next = 1;
    auto* step             = static_cast<Impl::HostStepStaging*>(p.pinned_step.data());
    for (int i = 0; i < b; ++i) {
        const auto& row = rows[i];
        p.states.prepare_append(row.slot, row.start, 1, stream);
        step->batch_ids[i]       = row.id;
        step->batch_positions[i] = static_cast<std::int32_t>(row.start);
        step->batch_slots[i]     = static_cast<std::int32_t>(row.slot);
        max_next                 = std::max(max_next, row.start + 1U);
    }
    auto d_ids = p.bf(p.layout.ids, b), d_pos = p.bf(p.layout.positions, b),
         d_slots = p.bf(p.layout.batch_slots, b);
    CUDA_CHECK(cudaMemcpyAsync(d_ids.data, step->batch_ids, b * sizeof(TokenId),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pos.data, step->batch_positions, b * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_slots.data, step->batch_slots, b * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));

    if (p.options.use_cuda_graph) {
        std::uint32_t envelope = std::min(256U, p.options.max_context);
        while (envelope < max_next) envelope = std::min(envelope * 2U, p.options.max_context);
        auto& graph = p.batch_graphs[{b, envelope}];
        if (!graph.executable.ready()) {
            sync();
            graph.definition.capture(stream, [&] { p.schedule_batched(b, envelope); });
            graph.executable.instantiate(graph.definition);
        }
        graph.executable.launch(stream);
    } else {
        p.schedule_batched(b, max_next);
    }

    auto logits_all    = p.bf(p.layout.logits, P, b);
    auto* dev_configs  = static_cast<ops::SamplingConfig*>(
        p.layout.config.bind({p.scratch.p, p.scratch.bytes}).data);
    auto dev_pos_all   = p.bf(p.layout.sample_position, b);
    auto dev_out_all   = p.bf(p.layout.sampled, b);
    auto dev_cnt_all   = p.bf(p.layout.counts, V, b);

    for (int i = 0; i < b; ++i) {
        const auto& row = rows[i];
        const auto& sampling = row.sampling;
        const bool use_penalties =
            sampling.presence_penalty != 0.0f || sampling.frequency_penalty != 0.0f;
        int* counts_ptr = nullptr;
        if (use_penalties) {
            auto* host_row = p.host_counts.data() + static_cast<std::size_t>(i) * V;
            std::fill_n(host_row, V, 0);
            for (auto id : row.history)
                if (id >= 0 && id < V) ++host_row[id];
            counts_ptr = static_cast<int*>(dev_cnt_all.data) + static_cast<std::size_t>(i) * V;
            CUDA_CHECK(cudaMemcpyAsync(counts_ptr, host_row, V * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
        }
        step->batch_configs[i] = ops::SamplingConfig{sampling.temperature,
                                                     sampling.top_k,
                                                     sampling.top_p,
                                                     sampling.min_p,
                                                     sampling.presence_penalty,
                                                     sampling.frequency_penalty,
                                                     sampling.seed,
                                                     counts_ptr};
        step->batch_sample_positions[i] = static_cast<std::int32_t>(row.logical_position);
    }
    CUDA_CHECK(cudaMemcpyAsync(dev_configs, step->batch_configs, b * sizeof(ops::SamplingConfig),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_pos_all.data, step->batch_sample_positions,
                               b * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    for (int i = 0; i < b; ++i) {
        Tensor col_logits(static_cast<std::byte*>(logits_all.data) +
                              static_cast<std::size_t>(i) * P * 2,
                          DType::BF16, {P, 1});
        Tensor col_pos(static_cast<std::byte*>(dev_pos_all.data) +
                           static_cast<std::size_t>(i) * sizeof(std::int32_t),
                       DType::I32, {1});
        Tensor col_out(static_cast<std::byte*>(dev_out_all.data) +
                           static_cast<std::size_t>(i) * sizeof(TokenId),
                       DType::I32, {1});
        ops::sample(col_logits, col_out, V, &dev_configs[i], col_pos, ops::kSamplePurposeDecode,
                    p.workspace, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(step->batch_sampled, dev_out_all.data, b * sizeof(TokenId),
                               cudaMemcpyDeviceToHost, stream));
    sync();
    for (const auto& row : rows) p.states.commit_append(row.slot, row.start + 1U);
    return std::vector<TokenId>(step->batch_sampled, step->batch_sampled + b);
}

void Executor::reset(std::uint32_t slot) { impl_->states.reset(slot, impl_->device.stream); }

void Executor::copy_state(std::uint32_t source, std::uint32_t destination) {
    impl_->states.copy(source, destination, impl_->device.stream);
    if (impl_->model.has_mtp && impl_->states.has_dspark() && source != destination) {
        auto c_heads = impl_->bf(impl_->layout.dspark_c_head, C,
                                 static_cast<int>(impl_->states.slots()));
        auto* base   = static_cast<std::byte*>(c_heads.data);
        CUDA_CHECK(cudaMemcpyAsync(base + static_cast<std::size_t>(destination) * C * 2,
                                   base + static_cast<std::size_t>(source) * C * 2, C * 2,
                                   cudaMemcpyDeviceToDevice, impl_->device.stream));
    }
}

std::vector<TokenId> Executor::draft_dspark(std::uint32_t state_slot, TokenId anchor,
                                            std::uint32_t draft_tokens) {
    auto& p = *impl_;
    if (!p.model.has_mtp || !p.states.has_dspark())
        throw std::logic_error("DSpark draft requested without loaded MTP weights");
    if (draft_tokens == 0 || draft_tokens > 7)
        throw std::invalid_argument("DSpark draft_tokens must be in [1, 7]");
    if (anchor < 0 || anchor >= V)
        throw std::invalid_argument("DSpark anchor token ID outside vocabulary");

    const int K             = static_cast<int>(draft_tokens);
    const int W             = K + 1;
    const std::uint32_t F   = p.states.dspark_depth(state_slot);
    auto stream             = p.device.stream;
    sync();

    auto* step = static_cast<Impl::HostStepStaging*>(p.pinned_step.data());
    for (int i = 0; i < W; ++i) step->dspark_pos[i] = static_cast<std::int32_t>(F) + i;
    step->dspark_drafts[0] = anchor;
    auto pos               = p.bf(p.layout.dspark_pos, W);
    auto draft_toks        = p.bf(p.layout.dspark_draft_tok, W);
    Tensor prev_tok0(draft_toks.data, DType::I32, {1});
    CUDA_CHECK(cudaMemcpyAsync(pos.data, step->dspark_pos, W * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(prev_tok0.data, &step->dspark_drafts[0], sizeof(TokenId),
                               cudaMemcpyHostToDevice, stream));

    if (p.options.use_cuda_graph) {
        std::uint32_t dspark_env = std::min(256U, p.options.max_context);
        while (dspark_env < F + static_cast<std::uint32_t>(W))
            dspark_env = std::min(dspark_env * 2U, p.options.max_context);
        auto& graph = p.draft_graphs[{state_slot, dspark_env, K}];
        if (!graph.executable.ready()) {
            sync();
            graph.definition.capture(stream, [&] { p.draft_dspark_body(state_slot, K, dspark_env); });
            graph.executable.instantiate(graph.definition);
        }
        graph.executable.launch(stream);
    } else {
        p.draft_dspark_body(state_slot, K, F + static_cast<std::uint32_t>(W));
    }

    CUDA_CHECK(cudaMemcpyAsync(
        step->dspark_drafts, static_cast<const std::byte*>(draft_toks.data) + sizeof(TokenId),
        static_cast<std::size_t>(K) * sizeof(TokenId), cudaMemcpyDeviceToHost, stream));
    sync();
    return std::vector<TokenId>(step->dspark_drafts, step->dspark_drafts + K);
}

std::vector<TokenId> Executor::speculative_step_dspark(std::uint32_t state_slot, TokenId anchor,
                                                       std::uint32_t draft_tokens,
                                                       std::uint32_t start,
                                                       const ResolvedSamplingParameters& sampling,
                                                       std::uint32_t logical_position,
                                                       std::span<const TokenId> generated_history) {
    auto& p = *impl_;
    if (!p.model.has_mtp || !p.states.has_dspark())
        throw std::logic_error("DSpark step requested without loaded MTP weights");
    if (draft_tokens == 0 || draft_tokens > 7)
        throw std::invalid_argument("DSpark draft_tokens must be in [1, 7]");
    if (anchor < 0 || anchor >= V)
        throw std::invalid_argument("DSpark anchor token ID outside vocabulary");

    const int K           = static_cast<int>(draft_tokens);
    const int W           = K + 1;
    const std::uint32_t F = p.states.dspark_depth(state_slot);
    if (W > head_columns(p.options))
        throw std::invalid_argument("DSpark verify window out of planned range");

    sync();
    auto stream = p.device.stream;
    auto* step  = static_cast<Impl::HostStepStaging*>(p.pinned_step.data());
    for (int i = 0; i < W; ++i) step->dspark_pos[i] = static_cast<std::int32_t>(F) + i;
    step->dspark_drafts[0] = anchor;
    auto pos_dspark        = p.bf(p.layout.dspark_pos, W);
    auto draft_toks        = p.bf(p.layout.dspark_draft_tok, W);
    Tensor prev_tok0(draft_toks.data, DType::I32, {1});
    CUDA_CHECK(cudaMemcpyAsync(pos_dspark.data, step->dspark_pos, W * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(prev_tok0.data, &step->dspark_drafts[0], sizeof(TokenId),
                               cudaMemcpyHostToDevice, stream));

    auto slot_conv = p.states.slot_convolution(state_slot);
    auto pre_conv  = p.bf(p.layout.dspark_pre_conv, 6656 * 3, 48);
    CUDA_CHECK(cudaMemcpyAsync(pre_conv.data, slot_conv.data, slot_conv.bytes(),
                               cudaMemcpyDeviceToDevice, stream));

    p.states.prepare_append(state_slot, start, static_cast<std::uint32_t>(W), stream);
    auto* host_pos = static_cast<std::int32_t*>(p.pinned_positions.data());
    for (int i = 0; i < W; ++i) host_pos[i] = static_cast<int>(start) + i;
    auto positions = p.bf(p.layout.positions, W), tokens = p.bf(p.layout.ids, W);
    CUDA_CHECK(
        cudaMemcpyAsync(positions.data, host_pos, W * sizeof(int), cudaMemcpyHostToDevice, stream));
    step->count = W;
    step->lane  = static_cast<int>(state_slot);
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.count, 1).data, &step->count, sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.lane, 1).data, &step->lane, sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    if (p.options.use_cuda_graph) {
        std::uint32_t dspark_env = std::min(256U, p.options.max_context);
        while (dspark_env < F + static_cast<std::uint32_t>(W))
            dspark_env = std::min(dspark_env * 2U, p.options.max_context);
        auto& dgraph = p.draft_graphs[{state_slot, dspark_env, K}];
        if (!dgraph.executable.ready()) {
            sync();
            dgraph.definition.capture(stream, [&] { p.draft_dspark_body(state_slot, K, dspark_env); });
            dgraph.executable.instantiate(dgraph.definition);
        }
        dgraph.executable.launch(stream);
    } else {
        p.draft_dspark_body(state_slot, K, F + static_cast<std::uint32_t>(W));
    }

    CUDA_CHECK(cudaMemcpyAsync(tokens.data, draft_toks.data, static_cast<std::size_t>(W) * sizeof(TokenId),
                               cudaMemcpyDeviceToDevice, stream));

    if (p.options.use_cuda_graph) {
        std::uint32_t envelope = std::min(256U, p.options.max_context);
        while (envelope < start + static_cast<std::uint32_t>(W))
            envelope = std::min(envelope * 2U, p.options.max_context);
        auto& vgraph = p.verify_graphs[{state_slot, envelope, W}];
        if (!vgraph.executable.ready()) {
            sync();
            vgraph.definition.capture(stream, [&] {
                p.schedule(W, state_slot, envelope, true, true, true);
            });
            vgraph.executable.instantiate(vgraph.definition);
        }
        vgraph.executable.launch(stream);
    } else {
        p.schedule(W, state_slot, start + static_cast<std::uint32_t>(W), true, true, true);
    }

    auto logits           = p.bf(p.layout.logits, P, W);
    int accepted_total    = 1;
    const bool raw_greedy = sampling.temperature <= 0.0f && sampling.presence_penalty == 0.0f &&
                            sampling.frequency_penalty == 0.0f;
    if (raw_greedy) {
        auto target_toks = p.bf(p.layout.dspark_target_toks, W);
        ops::argmax(logits, target_toks, V, stream);
        CUDA_CHECK(cudaMemcpyAsync(step->dspark_drafts,
                                   static_cast<const std::byte*>(draft_toks.data) + sizeof(TokenId),
                                   static_cast<std::size_t>(K) * sizeof(TokenId),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(step->dspark_licensed, target_toks.data, W * sizeof(TokenId),
                                   cudaMemcpyDeviceToHost, stream));
        sync();
        for (int i = 0; i < K; ++i) {
            if (step->dspark_licensed[i] == step->dspark_drafts[i]) {
                accepted_total = i + 2;
            } else {
                break;
            }
        }
    } else {
        const bool use_penalties =
            sampling.presence_penalty != 0.0f || sampling.frequency_penalty != 0.0f;
        int* counts_ptr = nullptr;
        if (use_penalties) {
            std::fill_n(p.host_counts.data(), V, 0);
            for (auto id : generated_history)
                if (id >= 0 && id < V) ++p.host_counts[id];
            auto counts = p.bf(p.layout.counts, V);
            CUDA_CHECK(cudaMemcpyAsync(counts.data, p.host_counts.data(), V * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
            counts_ptr = static_cast<int*>(counts.data);
        }
        step->config = ops::SamplingConfig{sampling.temperature,
                                           sampling.top_k,
                                           sampling.top_p,
                                           sampling.min_p,
                                           sampling.presence_penalty,
                                           sampling.frequency_penalty,
                                           sampling.seed,
                                           counts_ptr};
        auto* device_config = static_cast<ops::SamplingConfig*>(
            p.layout.config.bind({p.scratch.p, p.scratch.bytes}).data);
        CUDA_CHECK(cudaMemcpyAsync(device_config, &step->config, sizeof(step->config),
                                   cudaMemcpyHostToDevice, stream));

        step->dspark_pos[0] = K;
        step->dspark_pos[1] = static_cast<std::int32_t>(logical_position);
        step->dspark_pos[2] = anchor;
        auto extents_t      = p.bf(p.layout.dspark_extents, 1);
        auto lengths_t      = p.bf(p.layout.dspark_lengths, 1);
        auto anchors_t      = p.bf(p.layout.dspark_anchors, 1);
        CUDA_CHECK(cudaMemcpyAsync(extents_t.data, &step->dspark_pos[0], sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(lengths_t.data, &step->dspark_pos[1], sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(anchors_t.data, &step->dspark_pos[2], sizeof(TokenId),
                                   cudaMemcpyHostToDevice, stream));

        auto target_toks  = p.bf(p.layout.dspark_target_toks, W);
        auto licensed_t   = p.bf(p.layout.dspark_licensed, W);
        auto licensed_cnt = p.bf(p.layout.dspark_licensed_cnt, 1);
        auto accepted_t   = p.bf(p.layout.dspark_accepted, 1);
        Tensor logits_3d(logits.data, DType::BF16, {P, W, 1});
        Tensor drafts_2d(static_cast<std::byte*>(draft_toks.data) + sizeof(TokenId), DType::I32,
                         {K, 1});
        Tensor target_2d(target_toks.data, DType::I32, {W, 1});
        Tensor licensed_2d(licensed_t.data, DType::I32, {W, 1});
        ops::speculative_accept_greedy_drafts(target_2d, logits_3d, drafts_2d, extents_t, lengths_t,
                                              anchors_t, licensed_2d, licensed_cnt, accepted_t, V,
                                              device_config, p.workspace, stream);
        CUDA_CHECK(cudaMemcpyAsync(&step->dspark_licensed_count, licensed_cnt.data, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(step->dspark_licensed, licensed_t.data, W * sizeof(TokenId),
                                   cudaMemcpyDeviceToHost, stream));
        sync();
        accepted_total = step->dspark_licensed_count;
    }

    p.states.commit_append(state_slot, start + static_cast<std::uint32_t>(accepted_total));
    if (accepted_total < W) {
        auto conv_trace = p.bf(p.layout.dspark_conv_trace, 6656 * W, 48);
        ops::inkling2_dspark_commit_conv_state(pre_conv, conv_trace, slot_conv, W, accepted_total,
                                               stream);
    }
    p.mix_and_append_dspark(W, accepted_total, start, state_slot);
    return std::vector<TokenId>(step->dspark_licensed, step->dspark_licensed + accepted_total);
}

std::vector<TokenId> Executor::verify_dspark(std::uint32_t state_slot,
                                             std::span<const TokenId> verify_ids,
                                             std::uint32_t start,
                                             const ResolvedSamplingParameters& sampling,
                                             std::uint32_t logical_position,
                                             std::span<const TokenId> generated_history) {
    auto& p     = *impl_;
    const int W = static_cast<int>(verify_ids.size());
    const int K = W - 1;
    if (W < 2 || W > 8 || W > head_columns(p.options))
        throw std::invalid_argument("DSpark verify window out of planned range");
    for (auto id : verify_ids)
        if (id < 0 || id >= V) throw std::invalid_argument("Inkling token ID outside vocabulary");

    sync();
    auto stream    = p.device.stream;
    auto slot_conv = p.states.slot_convolution(state_slot);
    auto pre_conv  = p.bf(p.layout.dspark_pre_conv, 6656 * 3, 48);
    CUDA_CHECK(cudaMemcpyAsync(pre_conv.data, slot_conv.data, slot_conv.bytes(),
                               cudaMemcpyDeviceToDevice, stream));

    p.states.prepare_append(state_slot, start, static_cast<std::uint32_t>(W), stream);
    auto* host_ids = static_cast<TokenId*>(p.pinned_ids.data());
    std::memcpy(host_ids, verify_ids.data(), verify_ids.size_bytes());
    auto* host_pos = static_cast<std::int32_t*>(p.pinned_positions.data());
    for (int i = 0; i < W; ++i) host_pos[i] = static_cast<int>(start) + i;
    auto positions = p.bf(p.layout.positions, W), tokens = p.bf(p.layout.ids, W);
    CUDA_CHECK(
        cudaMemcpyAsync(tokens.data, host_ids, verify_ids.size_bytes(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(positions.data, host_pos, W * sizeof(int), cudaMemcpyHostToDevice, stream));
    auto* step  = static_cast<Impl::HostStepStaging*>(p.pinned_step.data());
    step->count = W;
    step->lane  = static_cast<int>(state_slot);
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.count, 1).data, &step->count, sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.lane, 1).data, &step->lane, sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    if (p.options.use_cuda_graph) {
        std::uint32_t envelope = std::min(256U, p.options.max_context);
        while (envelope < start + static_cast<std::uint32_t>(W))
            envelope = std::min(envelope * 2, p.options.max_context);
        auto& graph = p.verify_graphs[{state_slot, envelope, W}];
        if (!graph.executable.ready()) {
            sync();
            graph.definition.capture(stream, [&] {
                p.schedule(W, state_slot, envelope, true, true, true);
            });
            graph.executable.instantiate(graph.definition);
        }
        graph.executable.launch(stream);
    } else {
        p.schedule(W, state_slot, start + static_cast<std::uint32_t>(W), true, true, true);
    }
    auto logits = p.bf(p.layout.logits, P, W);

    int accepted_total = 1;
    const bool raw_greedy = sampling.temperature <= 0.0f && sampling.presence_penalty == 0.0f &&
                            sampling.frequency_penalty == 0.0f;
    if (raw_greedy) {
        auto target_toks = p.bf(p.layout.dspark_target_toks, W);
        ops::argmax(logits, target_toks, V, stream);
        CUDA_CHECK(cudaMemcpyAsync(step->dspark_licensed, target_toks.data, W * sizeof(TokenId),
                                   cudaMemcpyDeviceToHost, stream));
        sync();
        for (int i = 0; i < K; ++i) {
            if (step->dspark_licensed[i] == verify_ids[i + 1]) {
                accepted_total = i + 2;
            } else {
                break;
            }
        }
    } else {
        const bool use_penalties =
            sampling.presence_penalty != 0.0f || sampling.frequency_penalty != 0.0f;
        int* counts_ptr = nullptr;
        if (use_penalties) {
            std::fill_n(p.host_counts.data(), V, 0);
            for (auto id : generated_history)
                if (id >= 0 && id < V) ++p.host_counts[id];
            auto counts = p.bf(p.layout.counts, V);
            CUDA_CHECK(cudaMemcpyAsync(counts.data, p.host_counts.data(), V * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
            counts_ptr = static_cast<int*>(counts.data);
        }
        step->config = ops::SamplingConfig{sampling.temperature,
                                           sampling.top_k,
                                           sampling.top_p,
                                           sampling.min_p,
                                           sampling.presence_penalty,
                                           sampling.frequency_penalty,
                                           sampling.seed,
                                           counts_ptr};
        auto* device_config = static_cast<ops::SamplingConfig*>(
            p.layout.config.bind({p.scratch.p, p.scratch.bytes}).data);
        CUDA_CHECK(cudaMemcpyAsync(device_config, &step->config, sizeof(step->config),
                                   cudaMemcpyHostToDevice, stream));

        std::memcpy(step->dspark_drafts, verify_ids.data() + 1, K * sizeof(TokenId));
        step->dspark_pos[0]         = K;
        step->dspark_pos[1]         = static_cast<std::int32_t>(logical_position);
        step->dspark_pos[2]         = verify_ids[0];
        auto drafts_t               = p.bf(p.layout.dspark_draft_tok, K);
        auto extents_t              = p.bf(p.layout.dspark_extents, 1);
        auto lengths_t              = p.bf(p.layout.dspark_lengths, 1);
        auto anchors_t              = p.bf(p.layout.dspark_anchors, 1);
        CUDA_CHECK(cudaMemcpyAsync(drafts_t.data, step->dspark_drafts, K * sizeof(TokenId),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(extents_t.data, &step->dspark_pos[0], sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(lengths_t.data, &step->dspark_pos[1], sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(anchors_t.data, &step->dspark_pos[2], sizeof(TokenId),
                                   cudaMemcpyHostToDevice, stream));

        auto target_toks   = p.bf(p.layout.dspark_target_toks, W);
        auto licensed_t    = p.bf(p.layout.dspark_licensed, W);
        auto licensed_cnt  = p.bf(p.layout.dspark_licensed_cnt, 1);
        auto accepted_t    = p.bf(p.layout.dspark_accepted, 1);
        Tensor logits_3d(logits.data, DType::BF16, {P, W, 1});
        Tensor drafts_2d(drafts_t.data, DType::I32, {K, 1});
        Tensor target_2d(target_toks.data, DType::I32, {W, 1});
        Tensor licensed_2d(licensed_t.data, DType::I32, {W, 1});
        ops::speculative_accept_greedy_drafts(target_2d, logits_3d, drafts_2d, extents_t, lengths_t,
                                              anchors_t, licensed_2d, licensed_cnt, accepted_t, V,
                                              device_config, p.workspace, stream);
        CUDA_CHECK(cudaMemcpyAsync(&step->dspark_licensed_count, licensed_cnt.data, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(step->dspark_licensed, licensed_t.data, W * sizeof(TokenId),
                                   cudaMemcpyDeviceToHost, stream));
        sync();
        accepted_total = step->dspark_licensed_count;
    }

    p.states.commit_append(state_slot, start + static_cast<std::uint32_t>(accepted_total));
    if (accepted_total < W) {
        auto conv_trace = p.bf(p.layout.dspark_conv_trace, 6656 * W, 48);
        ops::inkling2_dspark_commit_conv_state(pre_conv, conv_trace, slot_conv, W, accepted_total,
                                               stream);
    }
    p.mix_and_append_dspark(W, accepted_total, start, state_slot);
    p.last_spec_active   = true;
    p.last_spec_slot     = state_slot;
    p.last_spec_start    = start;
    p.last_spec_w        = W;
    p.last_spec_accepted = accepted_total;
    return std::vector<TokenId>(step->dspark_licensed, step->dspark_licensed + accepted_total);
}

std::vector<TokenId> Executor::speculative_step_dflash2(std::uint32_t state_slot, TokenId anchor,
                                                        std::uint32_t draft_tokens,
                                                        std::uint32_t start,
                                                        const ResolvedSamplingParameters& sampling,
                                                        std::uint32_t logical_position,
                                                        std::span<const TokenId> generated_history) {
    auto& p = *impl_;
    if (!p.model.has_dflash2 || !p.states.has_dflash2())
        throw std::logic_error("DFlash2 step requested without loaded DFlash2 weights");
    if (draft_tokens == 0 || draft_tokens > 15)
        throw std::invalid_argument("DFlash2 draft_tokens must be in [1, 15]");
    if (anchor < 0 || anchor >= V)
        throw std::invalid_argument("DFlash2 anchor token ID outside vocabulary");

    const int K           = static_cast<int>(draft_tokens);
    const int W           = K + 1;
    const std::uint32_t F = p.states.dflash2_depth(state_slot);
    if (W > head_columns(p.options))
        throw std::invalid_argument("DFlash2 verify window out of planned range");

    sync();
    auto stream = p.device.stream;
    auto* step  = static_cast<Impl::HostStepStaging*>(p.pinned_step.data());

    const bool use_penalties =
        sampling.presence_penalty != 0.0f || sampling.frequency_penalty != 0.0f;
    int* counts_ptr = nullptr;
    if (use_penalties) {
        std::fill_n(p.host_counts.data(), V, 0);
        for (auto id : generated_history)
            if (id >= 0 && id < V) ++p.host_counts[id];
        auto counts = p.bf(p.layout.counts, V);
        CUDA_CHECK(cudaMemcpyAsync(counts.data, p.host_counts.data(), V * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        counts_ptr = static_cast<int*>(counts.data);
    }
    step->config = ops::SamplingConfig{sampling.temperature,
                                       sampling.top_k,
                                       sampling.top_p,
                                       sampling.min_p,
                                       sampling.presence_penalty,
                                       sampling.frequency_penalty,
                                       sampling.seed,
                                       counts_ptr};
    auto* device_config = static_cast<ops::SamplingConfig*>(
        p.layout.config.bind({p.scratch.p, p.scratch.bytes}).data);
    CUDA_CHECK(cudaMemcpyAsync(device_config, &step->config, sizeof(step->config),
                               cudaMemcpyHostToDevice, stream));

    step->count              = W;
    step->lane               = static_cast<int>(state_slot);
    step->dflash2_frontier   = static_cast<std::int32_t>(F);
    step->dflash2_valid_cols = W;
    step->dflash2_base_pos   = static_cast<std::int32_t>(logical_position) + 1;
    step->dspark_pos[0]      = static_cast<std::int32_t>(start);
    step->dspark_pos[1]      = K;
    step->dspark_pos[2]      = static_cast<std::int32_t>(logical_position);
    step->dspark_pos[3]      = anchor;

    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.count, 1).data, &step->count, sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.lane, 1).data, &step->lane, sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.dflash2_valid_cols, 1).data,
                               &step->dflash2_valid_cols, sizeof(int), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.sample_position, 1).data, &step->dflash2_base_pos,
                               sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.dspark_pos, 1).data, &step->dspark_pos[0], sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.dspark_extents, 1).data, &step->dspark_pos[1],
                               sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.dspark_lengths, 1).data, &step->dflash2_frontier,
                               sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.dspark_anchors, 1).data, &step->dspark_pos[3],
                               sizeof(TokenId), cudaMemcpyHostToDevice, stream));

    auto slot_conv = p.states.slot_convolution(state_slot);
    auto pre_conv  = p.bf(p.layout.dspark_pre_conv, 6656 * 3, 48);
    CUDA_CHECK(cudaMemcpyAsync(pre_conv.data, slot_conv.data, slot_conv.bytes(),
                               cudaMemcpyDeviceToDevice, stream));

    p.states.prepare_append(state_slot, start, static_cast<std::uint32_t>(W), stream);

    if (p.options.use_cuda_graph) {
        std::uint32_t dflash_env = std::min(256U, p.options.max_context);
        while (dflash_env < F + static_cast<std::uint32_t>(W))
            dflash_env = std::min(dflash_env * 2U, p.options.max_context);
        auto& dgraph = p.dflash2_draft_graphs[{state_slot, dflash_env, K}];
        if (!dgraph.executable.ready()) {
            sync();
            dgraph.definition.capture(stream, [&] {
                p.draft_dflash2_body(state_slot, K, dflash_env, device_config);
            });
            dgraph.executable.instantiate(dgraph.definition);
        }
        dgraph.executable.launch(stream);
    } else {
        p.draft_dflash2_body(state_slot, K, F + static_cast<std::uint32_t>(W), device_config);
    }

    // Restore round_lengths to logical_position for speculative_accept_sparse_drafts.
    CUDA_CHECK(cudaMemcpyAsync(p.bf(p.layout.dspark_lengths, 1).data, &step->dspark_pos[2],
                               sizeof(int), cudaMemcpyHostToDevice, stream));

    if (p.options.use_cuda_graph) {
        std::uint32_t envelope = std::min(256U, p.options.max_context);
        while (envelope < start + static_cast<std::uint32_t>(W))
            envelope = std::min(envelope * 2U, p.options.max_context);
        auto& vgraph = p.verify_graphs[{state_slot, envelope, W}];
        if (!vgraph.executable.ready()) {
            sync();
            vgraph.definition.capture(stream, [&] {
                p.schedule(W, state_slot, envelope, true, false, true, true);
            });
            vgraph.executable.instantiate(vgraph.definition);
        }
        vgraph.executable.launch(stream);
    } else {
        p.schedule(W, state_slot, start + static_cast<std::uint32_t>(W), true, false, true, true);
    }

    auto logits_3d    = Tensor(p.bf(p.layout.logits, P, W).data, DType::BF16, {P, W, 1});
    auto target_2d    = p.bf(p.layout.dspark_target_toks, W, 1);
    auto drafts_2d    = p.bf(p.layout.dflash2_drafts, K, 1);
    auto cand_ids     = p.bf(p.layout.dflash2_cand_ids, 16, K, 1);
    auto proposal_q   = p.bf(p.layout.dflash2_proposal_q, 16, K, 1);
    auto extents_t    = p.bf(p.layout.dspark_extents, 1);
    auto lengths_t    = p.bf(p.layout.dspark_lengths, 1);
    auto anchors_t    = p.bf(p.layout.dspark_anchors, 1);
    auto licensed_2d  = p.bf(p.layout.dspark_licensed, W, 1);
    auto licensed_cnt = p.bf(p.layout.dspark_licensed_cnt, 1);
    auto accepted_t   = p.bf(p.layout.dspark_accepted, 1);

    const bool raw_greedy = sampling.temperature <= 0.0f && !use_penalties;
    if (raw_greedy) {
        Tensor target_1d(target_2d.data, DType::I32, {W});
        Tensor logits_2d(logits_3d.data, DType::BF16, {P, W});
        ops::argmax(logits_2d, target_1d, V, stream);
    }
    ops::speculative_accept_sparse_drafts(
        target_2d, logits_3d, drafts_2d, cand_ids, proposal_q, extents_t, lengths_t, anchors_t,
        licensed_2d, licensed_cnt, accepted_t, V, device_config, {raw_greedy}, p.workspace, stream);

    CUDA_CHECK(cudaMemcpyAsync(&step->dspark_licensed_count, licensed_cnt.data, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(step->dspark_licensed, licensed_2d.data, W * sizeof(TokenId),
                               cudaMemcpyDeviceToHost, stream));
    sync();
    const int accepted_total = step->dspark_licensed_count;

    p.states.commit_append(state_slot, start + static_cast<std::uint32_t>(accepted_total));
    if (accepted_total < W) {
        auto conv_trace = p.bf(p.layout.dspark_conv_trace, 6656 * W, 48);
        ops::inkling2_dspark_commit_conv_state(pre_conv, conv_trace, slot_conv, W, accepted_total,
                                               stream);
    }
    p.append_dflash2_context(accepted_total, start, state_slot);
    p.last_spec_active   = true;
    p.last_spec_slot     = state_slot;
    p.last_spec_start    = start;
    p.last_spec_w        = W;
    p.last_spec_accepted = accepted_total;
    return std::vector<TokenId>(step->dspark_licensed, step->dspark_licensed + accepted_total);
}

void Executor::truncate_speculative_commit(std::uint32_t state_slot, std::uint32_t accepted_tokens) {
    auto& p = *impl_;
    if (!p.last_spec_active || p.last_spec_slot != state_slot ||
        accepted_tokens >= static_cast<std::uint32_t>(p.last_spec_accepted) ||
        accepted_tokens == 0) {
        p.last_spec_active = false;
        return;
    }
    auto stream = p.device.stream;
    p.states.commit_append(state_slot, p.last_spec_start + accepted_tokens);
    auto slot_conv  = p.states.slot_convolution(state_slot);
    auto pre_conv   = p.bf(p.layout.dspark_pre_conv, 6656 * 3, 48);
    auto conv_trace = p.bf(p.layout.dspark_conv_trace, 6656 * p.last_spec_w, 48);
    ops::inkling2_dspark_commit_conv_state(pre_conv, conv_trace, slot_conv, p.last_spec_w,
                                           static_cast<int>(accepted_tokens), stream);
    if (p.model.has_dflash2 && p.states.has_dflash2()) {
        p.append_dflash2_context(static_cast<int>(accepted_tokens), p.last_spec_start, state_slot);
    } else if (p.model.has_mtp && p.states.has_dspark()) {
        p.mix_and_append_dspark(p.last_spec_w, static_cast<int>(accepted_tokens),
                                p.last_spec_start, state_slot, true);
    }
    p.last_spec_active = false;
}

std::vector<float> Executor::causal_score(std::span<const TokenId> ids, std::uint32_t first) {
    if (first == 0 || first >= ids.size())
        throw std::invalid_argument("Inkling causal score target interval");
    auto& p = *impl_;
    reset(0);
    std::vector<float> result;
    result.reserve(ids.size() - first);
    const auto chunk = static_cast<std::size_t>(head_columns(p.options));
    for (std::size_t offset = 0; offset + 1 < ids.size();) {
        const auto n = std::min(chunk, ids.size() - 1 - offset);
        auto logits  = forward(ids.subspan(offset, n), static_cast<unsigned>(offset), 0, true);
        auto target  = p.bf(p.layout.score_targets, static_cast<int>(n)),
             scores  = p.bf(p.layout.score_result, static_cast<int>(n));
        CUDA_CHECK(cudaMemcpyAsync(target.data, ids.data() + offset + 1, n * sizeof(TokenId),
                                   cudaMemcpyHostToDevice, p.device.stream));
        ops::target_logprobs(logits, target, V, scores, p.device.stream);
        std::vector<float> values(n);
        CUDA_CHECK(cudaMemcpyAsync(values.data(), scores.data, n * sizeof(float),
                                   cudaMemcpyDeviceToHost, p.device.stream));
        sync();
        for (std::size_t j = 0; j < n; ++j)
            if (offset + j + 1 >= first) result.push_back(values[j]);
        offset += n;
    }
    reset(0);
    return result;
}

MemorySummary Executor::memory_summary() const {
    const auto& p = *impl_;
    MemorySummary out;
    out.device = p.device.device;
    out.max_context = p.options.max_context;
    out.kv_capacity = kv_tokens(p.options);
    if (const auto* weights = p.model.weights_arena) {
        out.weights = {weights->capacity(), weights->used(), weights->peak_used()};
    }
    out.kv_cache                     = p.options.kv_cache;
    out.sequence.capacity_bytes      = p.states.bytes() + p.scratch.bytes;
    out.sequence.used_bytes          = out.sequence.capacity_bytes;
    out.sequence.peak_used_bytes     = out.sequence.capacity_bytes;
    out.workspace.capacity_bytes     = p.workspace.capacity();
    out.workspace.used_bytes         = p.workspace.used();
    out.workspace.peak_used_bytes    = p.workspace.peak_used();
    out.workspace_logical_peak_bytes = p.workspace.peak_used();
    if (p.states.has_dflash2()) {
        out.text_kv_bytes    = p.states.kv_bytes();
        out.dflash_kv_bytes  = p.states.dflash_kv_bytes();
        out.kv_payload_bytes = out.text_kv_bytes + out.dflash_kv_bytes;
    } else {
        out.kv_payload_bytes = p.states.kv_bytes();
        out.dflash_kv_bytes  = p.states.dflash_kv_bytes();
        out.text_kv_bytes    = out.kv_payload_bytes - out.dflash_kv_bytes;
    }
    out.gdn_state_bytes              = std::size_t(6656) * 3 * 48 * 2 * p.states.slots();
    out.host_state_capacity_slots =
        p.options.context_cache.enabled ? p.options.context_cache.host_state_slots : 0;
    out.host_state_occupied_slots = p.states.host_state_count();
    out.host_kv_capacity_bytes    = p.states.host_kv_capacity();
    out.host_kv_occupied_bytes    = p.states.host_kv_bytes();
    if (p.options.enable_vision) {
        VisionWorkspaceMemorySummary vw;
        vw.aggregate_prompt_tokens = v_native_max(p.options);
        vw.max_item_tokens         = v_item_max(p.options);
        vw.general_capacity_bytes  = p.layout.vision_general_bytes;
        vw.encode_peak_bytes       = p.vision_encode_peak_bytes;
        vw.handoff_offset_bytes    = p.layout.vision_handoff.region.offset;
        vw.handoff_capacity_bytes  = p.layout.vision_handoff.region.bytes;
        vw.handoff_active_bytes    = p.vision_handoff_active_bytes;
        vw.handoff_peak_bytes      = p.vision_handoff_peak_bytes;
        out.vision_workspace       = vw;
    }
    return out;
}

void Executor::reset_memory_peaks() {
    impl_->workspace.reset_peak();
    impl_->vision_encode_peak_bytes  = 0;
    impl_->vision_handoff_peak_bytes = 0;
}

void Executor::sync() { CUDA_CHECK(cudaStreamSynchronize(impl_->device.stream)); }

std::uint32_t Executor::slot_depth(std::uint32_t slot) const { return impl_->states.depth(slot); }

std::uint32_t Executor::free_pages() const noexcept { return impl_->states.free_pages(); }

std::uint32_t Executor::used_pages() const noexcept { return impl_->states.used_pages(); }

std::uint32_t Executor::reclaimable_pages(std::span<const std::uint32_t> slots) const {
    return impl_->states.reclaimable_pages(slots);
}

bool Executor::can_snapshot_to_host(std::uint32_t slot) const {
    return impl_->states.can_snapshot_to_host(slot);
}

std::shared_ptr<HostStateImage> Executor::snapshot_to_host(std::uint32_t slot) {
    return impl_->states.snapshot_to_host(slot, impl_->device.stream);
}

void Executor::restore_from_host(std::uint32_t slot, const HostStateImage& image) {
    impl_->states.restore_from_host(slot, image, impl_->device.stream);
}

std::uint32_t Executor::host_state_count() const noexcept {
    return impl_->states.host_state_count();
}

std::size_t Executor::host_kv_bytes() const noexcept { return impl_->states.host_kv_bytes(); }

std::size_t Executor::host_kv_capacity() const noexcept { return impl_->states.host_kv_capacity(); }

TransferWork Executor::fixed_state_transfer_work() const noexcept {
    return impl_->states.fixed_state_transfer_work();
}

TransferWork Executor::host_kv_transfer_work(std::uint32_t page_count) const {
    return impl_->states.host_kv_transfer_work(page_count);
}

TransferWork Executor::device_kv_cow_transfer_work() const {
    return impl_->states.device_kv_cow_transfer_work();
}

std::vector<runtime::ContextTransferObservation> Executor::take_transfer_observations() {
    return impl_->states.take_transfer_observations();
}
} // namespace ninfer::targets::inkling2_mini::detail
