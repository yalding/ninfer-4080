#include "targets/inkling2_mini/impl/runtime/executor.h"
#include "targets/inkling2_mini/impl/load/bindings.h"
#include "targets/inkling2_mini/impl/state/state_image.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/inkling2_attention.h"
#include "ninfer/ops/inkling2_fused.h"
#include "ninfer/ops/inkling2_moe.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/target_logprobs.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace ninfer::targets::inkling2_mini::detail {
namespace {
constexpr int C = 2304, V = 200058, P = 201024;
constexpr std::uint32_t kScoreChunk = 128;

std::uint32_t chunk_capacity(const EngineOptions& options) {
    return std::min({1024U, options.max_context, std::max(1U, options.prefill_chunk)});
}

std::uint32_t kv_tokens(const EngineOptions& options) {
    if (options.kv_capacity.mode != KvCapacityMode::Explicit)
        throw std::logic_error("Inkling Executor requires resolved explicit KV capacity");
    return options.kv_capacity.explicit_tokens;
}

int head_columns(const EngineOptions& options) {
    if (options.purpose == EnginePurpose::CausalScoring)
        return static_cast<int>(std::min(kScoreChunk, chunk_capacity(options)));
    if (options.speculative.backend != SpeculativeBackend::None)
        return static_cast<int>(std::max(1U, options.speculative.draft_tokens + 1U));
    return 1;
}

std::size_t op_workspace(const EngineOptions& options) {
    const int t       = static_cast<int>(chunk_capacity(options));
    std::size_t bytes = ops::inkling2_moe_workspace_capacity_bytes(QType::Q3G128_F16S, 1, t);
    bytes = std::max(bytes, ops::inkling2_loop_control_workspace_capacity_bytes(3, 1, t));
    bytes = std::max(bytes, ops::inkling2_full_attention_workspace_capacity_bytes(
                                options.kv_cache, options.max_context, 1, t));
    for (const auto [n, k] : {std::pair{6528, 2304}, {1024, 2304}, {2304, 3072}})
        bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                    QType::Q4G64_F16S, n, k, ops::LinearPolicy::A16Only, 1, t));
    if (options.speculative.backend != SpeculativeBackend::None) {
        bytes = std::max(bytes, ops::inkling2_dspark_rnn_step_workspace_bytes());
        bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                    QType::Q4G64_F16S, 18432, 2304, ops::LinearPolicy::A16Only, 1, 8));
        bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                    QType::Q4G64_F16S, 2304, 9216, ops::LinearPolicy::A16Only, 1, 8));
        bytes = std::max(bytes, ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                    V, 1, 7, 1, 1));
    }
    bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(QType::Q6G64_F16S, P, C,
                                                                 ops::LinearPolicy::A16Only, 1,
                                                                 head_columns(options)));
    return std::max(bytes, ops::sampling_workspace_capacity_bytes(V, 1, 1));
}

struct ScratchLayout {
    TensorRegion cur, residual, blocks[4], prelude, bank, previous, mix, messages, inputs;
    TensorRegion qgr, kraw, vraw, kconv, vconv, q, k, v, r, sink, dynamic_k, dynamic_v, rel;
    TensorRegion attention, projection, moe, ffn_mix, ids, positions, count, lane;
    TensorRegion logits, sampled, sample_position, counts, score_targets, score_result;
    TensorRegion dspark_features, dspark_mixed, dspark_c_head, dspark_block, dspark_res, dspark_norm;
    TensorRegion dspark_q, dspark_k, dspark_v, dspark_attn, dspark_proj, dspark_mlp_gu, dspark_mlp_act;
    TensorRegion dspark_h_prime, dspark_rnn_state, dspark_pos, dspark_draft_tok, dspark_draft_logits;
    TensorRegion dspark_pre_conv, dspark_conv_trace, dspark_target_toks, dspark_extents;
    TensorRegion dspark_lengths, dspark_anchors, dspark_licensed, dspark_licensed_cnt, dspark_accepted;
    LayoutRegion config;
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
    s.cur      = bf({C, t});
    s.residual = bf({C, t});
    for (auto& block : s.blocks) block = bf({C, t});
    s.prelude         = bf({C, t});
    s.bank            = bf({C, t});
    s.previous        = bf({C, t});
    s.mix             = bf({C, t, 3});
    s.messages        = bf({C, t, 4});
    s.inputs          = bf({C, t, 3});
    s.qgr             = bf({6528, t});
    s.kraw            = bf({1024, t});
    s.vraw            = bf({1024, t});
    s.kconv           = bf({1024, t});
    s.vconv           = bf({1024, t});
    s.q               = bf({128, 24, t});
    s.k               = bf({128, 8, t});
    s.v               = bf({128, 8, t});
    s.r               = fp({16, 24, t});
    s.sink            = fp({24, t});
    s.dynamic_k       = fp({32, t});
    s.dynamic_v       = fp({32, t});
    s.rel             = fp({1025, 24, t});
    s.attention       = bf({128, 24, t});
    s.projection      = bf({C, t});
    s.moe             = bf({C, t});
    s.ffn_mix         = bf({C, t});
    s.ids             = i32({t});
    s.positions       = i32({t});
    s.count           = i32({1});
    s.lane            = i32({1});
    s.logits          = bf({P, head_columns(options)});
    s.sampled         = i32({1});
    s.sample_position = i32({1});
    s.counts          = i32({V});
    s.score_targets   = i32({head_columns(options)});
    s.score_result    = fp({head_columns(options)});
    if (options.speculative.backend != SpeculativeBackend::None) {
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
        s.dspark_draft_logits = bf({P, 1});
        s.dspark_pre_conv   = bf({6656 * 3, 48});
        s.dspark_conv_trace = bf({6656 * 8, 48});
        s.dspark_target_toks = i32({8});
        s.dspark_extents    = i32({1});
        s.dspark_lengths    = i32({1});
        s.dspark_anchors    = i32({1});
        s.dspark_licensed   = i32({8});
        s.dspark_licensed_cnt = i32({1});
        s.dspark_accepted   = i32({1});
    }
    s.config          = b.add(sizeof(ops::SamplingConfig), 256, "Inkling sampling config");
    s.bytes           = b.finish(256);
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
        std::int32_t dspark_pos[8];
        TokenId dspark_drafts[8];
        TokenId dspark_licensed[8];
        std::int32_t dspark_licensed_count;
    };

    std::map<std::pair<std::uint32_t, std::uint32_t>, Graph> graphs;
    std::map<std::tuple<std::uint32_t, std::uint32_t, int>, Graph> verify_graphs;
    PinnedHostBuffer pinned_step;
    PinnedHostBuffer pinned_ids;
    PinnedHostBuffer pinned_positions;
    std::vector<std::int32_t> host_counts;

    Impl(const RuntimeModelView& m, DeviceContext& d, const EngineOptions& o)
        : model(m), device(d), options(o), states(o, kv_tokens(o)), layout(scratch_layout(o)),
          scratch(layout.bytes), workspace(op_workspace(o)),
          pinned_step(sizeof(HostStepStaging)),
          pinned_ids(chunk_capacity(o) * sizeof(TokenId)),
          pinned_positions(chunk_capacity(o) * sizeof(std::int32_t)), host_counts(V) {}

    Tensor bf(const TensorRegion& r, int a, int b = 1, int c = 1) {
        return region(r, scratch, {a, b, c});
    }

    void schedule(int t, std::uint32_t slot, std::uint32_t envelope, bool all,
                  bool capture_dspark = false, bool record_conv = false) {
        auto stream = device.stream;
        Tensor cur = bf(layout.cur, C, t), residual = bf(layout.residual, C, t);
        Tensor ids = bf(layout.ids, t), positions = bf(layout.positions, t);
        Tensor count = bf(layout.count, 1), lane = bf(layout.lane, 1);
        ops::embedding(ids, model.token_embedding, cur, stream);
        ops::rmsnorm(cur, model.token_norm, 1e-6f, false, cur, stream);
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
        Tensor table = bf(layout.rel, 1025, 24, t), attention = bf(layout.attention, 128, 24, t);
        Tensor projection = bf(layout.projection, C, t), moe = bf(layout.moe, C, t),
               ffn          = bf(layout.ffn_mix, C, t);
        auto capture_feat = [&](int slice, const Tensor& src) {
            auto* dst = static_cast<std::byte*>(bf(layout.dspark_features, C, t, 4).data) +
                        static_cast<std::size_t>(slice) * C * t * 2;
            CUDA_CHECK(cudaMemcpyAsync(dst, src.data, static_cast<std::size_t>(C) * t * 2,
                                       cudaMemcpyDeviceToDevice, stream));
        };
        auto record_conv_comp = [&](int visit_idx, int comp_off, const Tensor& src) {
            auto* dst = static_cast<std::byte*>(bf(layout.dspark_conv_trace, 6656 * 8, 48).data) +
                        (static_cast<std::size_t>(visit_idx) * 6656 + comp_off) * t * 2;
            CUDA_CHECK(cudaMemcpyAsync(dst, src.data, src.bytes(), cudaMemcpyDeviceToDevice, stream));
        };
        auto states_and_cur = [&] {
            std::array<Tensor, 5> values;
            for (int i = 0; i < block_count; ++i) values[i] = blocks[i];
            values[block_count] = cur;
            return values;
        };
        auto layer = [&](int physical, int visit, bool loop) {
            if (capture_dspark) {
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
            ops::linear(qi, w.attention_qgr, qgr, ops::LinearPolicy::A16Only, workspace, stream);
            ops::linear(ki, w.attention_k, kraw, ops::LinearPolicy::A16Only, workspace, stream);
            ops::linear(vi, w.attention_v, vraw, ops::LinearPolicy::A16Only, workspace, stream);
            if (w.full_attention) {
                ops::inkling2_small_bf16_proj(ki, w.attention_k_conv_proj, dk, stream);
                ops::inkling2_small_bf16_proj(vi, w.attention_v_conv_proj, dv, stream);
            }
            auto ks = states.convolution(slot, visit, 0), vs = states.convolution(slot, visit, 1);
            if (record_conv) {
                record_conv_comp(visit, 0, kraw);
                record_conv_comp(visit, 1024, vraw);
            }
            ops::inkling2_fused_sconv(kraw, w.attention_k_conv, w.full_attention ? dk : Tensor{},
                                      ks, ks, {}, kc, stream);
            ops::inkling2_fused_sconv(vraw, w.attention_v_conv, w.full_attention ? dv : Tensor{},
                                      vs, vs, {}, vc, stream);
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
                ops::inkling2_swa_attention(q, k, v, qgr, table, positions, cache, slot, attention,
                                            stream);
                Tensor kb(k.data, DType::BF16, {128, 8, t, 1}),
                    vb(v.data, DType::BF16, {128, 8, t, 1});
                Tensor pos(positions.data, DType::I32, {t, 1});
                ops::kv_cache_append_prefix(kb, vb, pos, count, lane,
                                            {static_cast<unsigned>(t), static_cast<unsigned>(t)},
                                            cache, stream);
            }
            Tensor flat(attention.data, DType::BF16, {3072, t});
            ops::linear(flat, w.attention_output, projection, ops::LinearPolicy::A16Only, workspace,
                        stream);
            auto as = states.convolution(slot, visit, 2);
            if (record_conv) record_conv_comp(visit, 2048, projection);
            if (physical % 6 == 0) {
                if (block_count >= 4)
                    throw std::logic_error("Inkling residual block schedule overflow");
                copy(cur, blocks[block_count++], stream);
                ops::inkling2_fused_sconv(projection, w.attention_post_conv, {}, as, as, {},
                                          residual, stream);
            } else {
                ops::inkling2_fused_sconv(projection, w.attention_post_conv, {}, as, as, cur,
                                          residual, stream);
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
            auto fs = states.convolution(slot, visit, 3);
            if (record_conv) record_conv_comp(visit, 4352, moe);
            ops::inkling2_fused_sconv(moe, w.ffn_post_conv, {}, fs, fs, residual, cur, stream);
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
        if (capture_dspark) capture_feat(3, ffn);
        const int cols    = all ? t : 1;
        Tensor head_input = all ? ffn
                                : Tensor(static_cast<std::byte*>(ffn.data) + (t - 1) * ffn.nb[1],
                                         DType::BF16, {C, 1});
        auto logits       = bf(layout.logits, P, cols);
        ops::linear(head_input, model.output_head, logits, ops::LinearPolicy::A16Only, workspace,
                    stream);
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
};

Executor::Executor(const RuntimeModelView& model, DeviceContext& device,
                   const EngineOptions& options)
    : impl_(std::make_unique<Impl>(model, device, options)) {}

Executor::~Executor() {
    if (impl_) {
        try {
            sync();
        } catch (...) {}
    }
}

Tensor Executor::forward(std::span<const TokenId> ids, std::uint32_t start, std::uint32_t slot,
                         bool all) {
    auto& p     = *impl_;
    const int t = static_cast<int>(ids.size());
    if (ids.empty() || ids.size() > chunk_capacity(p.options) ||
        (all && t > head_columns(p.options)))
        throw std::invalid_argument("Inkling forward chunk exceeds planned token capacity");
    for (auto id : ids)
        if (id < 0 || id >= V) throw std::invalid_argument("Inkling token ID outside vocabulary");
    // Finish prior use of pinned table staging and transient storage before mutating the maps.
    sync();
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
    const bool capture_dspark = p.model.has_mtp &&
                                p.options.speculative.backend != SpeculativeBackend::None && !all &&
                                p.options.purpose != EnginePurpose::CausalScoring;
    if (p.options.use_cuda_graph && t == 1 && !all) {
        std::uint32_t envelope = std::min(256U, p.options.max_context);
        while (envelope < start + 1) envelope = std::min(envelope * 2, p.options.max_context);
        auto& graph = p.graphs[{slot, envelope}];
        if (!graph.executable.ready()) {
            sync();
            graph.definition.capture(p.device.stream,
                                     [&] { p.schedule(1, slot, envelope, false, capture_dspark, false); });
            graph.executable.instantiate(graph.definition);
        }
        graph.executable.launch(p.device.stream);
    } else
        p.schedule(t, slot, start + t, all, capture_dspark, false);
    p.states.commit_append(slot, start + t);
    if (capture_dspark) p.mix_and_append_dspark(t, t, start, slot);
    return p.bf(p.layout.logits, P, all ? t : 1);
}

TokenId Executor::sample(const Tensor& logits, const ResolvedSamplingParameters& sampling,
                         std::uint32_t logical_position, std::span<const TokenId> history) {
    auto& p                = *impl_;
    auto stream            = p.device.stream;
    const bool use_penalties =
        sampling.presence_penalty != 0.0f || sampling.frequency_penalty != 0.0f;
    int* counts_ptr = nullptr;
    if (use_penalties) {
        std::fill(p.host_counts.begin(), p.host_counts.end(), 0);
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

    auto block = p.bf(p.layout.dspark_block, C, W);
    auto norm  = p.bf(p.layout.dspark_norm, C, W);
    Tensor norm_col0(norm.data, DType::BF16, {C, 1});
    Tensor block_col0(block.data, DType::BF16, {C, 1});
    ops::embedding(prev_tok0, p.model.token_embedding, norm_col0, stream);
    ops::rmsnorm(norm_col0, p.model.token_norm, 1e-6f, false, block_col0, stream);
    for (int i = 1; i < W; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<std::byte*>(block.data) + static_cast<std::size_t>(i) * C * 2,
            p.model.mtp.mask_embed.data, C * 2, cudaMemcpyDeviceToDevice, stream));
    }

    auto q         = p.bf(p.layout.dspark_q, 128, 24, W);
    auto k         = p.bf(p.layout.dspark_k, 128, 8, W);
    auto v         = p.bf(p.layout.dspark_v, 128, 8, W);
    Tensor q_flat(q.data, DType::BF16, {3072, W});
    Tensor k_flat(k.data, DType::BF16, {1024, W});
    Tensor v_flat(v.data, DType::BF16, {1024, W});
    auto attn      = p.bf(p.layout.dspark_attn, 128, 24, W);
    Tensor attn_flat(attn.data, DType::BF16, {3072, W});
    auto proj      = p.bf(p.layout.dspark_proj, C, W);
    auto gu        = p.bf(p.layout.dspark_mlp_gu, 18432, W);
    auto act       = p.bf(p.layout.dspark_mlp_act, 9216, W);
    Tensor gate = gu.slice(0, 0, 9216);
    Tensor up   = gu.slice(0, 9216, 9216);

    int swa_idx = 0;
    for (int layer = 0; layer < 3; ++layer) {
        const auto& ml = p.model.mtp.layers[layer];
        ops::rmsnorm(block, ml.input_norm, 1e-6f, false, norm, stream);
        ops::linear(norm, ml.q, q_flat, ops::LinearPolicy::A16Only, p.workspace, stream);
        ops::linear(norm, ml.k, k_flat, ops::LinearPolicy::A16Only, p.workspace, stream);
        ops::linear(norm, ml.v, v_flat, ops::LinearPolicy::A16Only, p.workspace, stream);
        ops::inkling2_dspark_rmsnorm_rope(pos, ml.q_norm, ml.k_norm, q, k, 1e6f, stream);
        if (layer == 1) {
            auto full_cache = p.states.dspark_full(state_slot);
            ops::inkling2_dspark_full_attention(q, k, v, full_cache, F, p.workspace, attn, stream);
        } else {
            auto swa_cache = p.states.dspark_swa(swa_idx++);
            ops::inkling2_dspark_swa_attention(q, k, v, F, swa_cache, static_cast<int>(state_slot),
                                               attn, stream);
        }
        ops::linear(attn_flat, ml.output, proj, ops::LinearPolicy::A16Only, p.workspace, stream);
        ops::residual_add(proj, block, stream);

        ops::rmsnorm(block, ml.post_attention_norm, 1e-6f, false, norm, stream);
        ops::linear(norm, ml.gate_up, gu, ops::LinearPolicy::A16Only, p.workspace, stream);
        ops::silu_mul(gate, up, act, stream);
        ops::linear(act, ml.down, proj, ops::LinearPolicy::A16Only, p.workspace, stream);
        ops::residual_add(proj, block, stream);
    }
    ops::rmsnorm(block, p.model.mtp.final_norm, 1e-6f, false, norm, stream);

    Tensor rnn_state(p.layout.dspark_rnn_state.region.bind({p.scratch.p, p.scratch.bytes}).data,
                     DType::FP32, {256});
    CUDA_CHECK(cudaMemsetAsync(rnn_state.data, 0, 256 * sizeof(float), stream));
    Tensor c_head(
        static_cast<std::byte*>(
            p.bf(p.layout.dspark_c_head, C, static_cast<int>(p.states.slots())).data) +
            static_cast<std::size_t>(state_slot) * C * 2,
        DType::BF16, {C});
    auto h_prime      = p.bf(p.layout.dspark_h_prime, C);
    Tensor h_prime_2d(h_prime.data, DType::BF16, {C, 1});
    auto draft_logits = p.bf(p.layout.dspark_draft_logits, P, 1);

    for (int i = 0; i < K; ++i) {
        Tensor h_in(static_cast<std::byte*>(norm.data) + static_cast<std::size_t>(i + 1) * C * 2,
                    DType::BF16, {C});
        Tensor cur_tok(
            static_cast<std::byte*>(draft_toks.data) + static_cast<std::size_t>(i) * sizeof(TokenId),
            DType::I32, {1});
        Tensor next_tok(static_cast<std::byte*>(draft_toks.data) +
                            static_cast<std::size_t>(i + 1) * sizeof(TokenId),
                        DType::I32, {1});
        ops::inkling2_dspark_rnn_step(cur_tok, p.model.mtp.markov_w1,
                                      p.model.mtp.markov_rnn_proj_weight,
                                      p.model.mtp.markov_rnn_proj_bias,
                                      p.model.mtp.markov_hidden_out_weight,
                                      p.model.mtp.markov_hidden_out_bias, h_in, c_head, rnn_state,
                                      h_prime, p.workspace, stream);
        ops::linear(h_prime_2d, p.model.output_head, draft_logits, ops::LinearPolicy::A16Only,
                    p.workspace, stream);
        ops::inkling2_softcap_slice(draft_logits, 30.0f, V, draft_logits, stream);
        ops::argmax(draft_logits, next_tok, V, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(
        step->dspark_drafts, static_cast<const std::byte*>(draft_toks.data) + sizeof(TokenId),
        static_cast<std::size_t>(K) * sizeof(TokenId), cudaMemcpyDeviceToHost, stream));
    sync();
    return std::vector<TokenId>(step->dspark_drafts, step->dspark_drafts + K);
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
            std::fill(p.host_counts.begin(), p.host_counts.end(), 0);
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
    return std::vector<TokenId>(step->dspark_licensed, step->dspark_licensed + accepted_total);
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
    out.kv_payload_bytes             = p.states.kv_bytes();
    out.dflash_kv_bytes              = p.states.dflash_kv_bytes();
    out.text_kv_bytes                = out.kv_payload_bytes - out.dflash_kv_bytes;
    out.gdn_state_bytes              = std::size_t(6656) * 3 * 48 * 2 * p.states.slots();
    out.host_state_capacity_slots =
        p.options.context_cache.enabled ? p.options.context_cache.host_state_slots : 0;
    out.host_state_occupied_slots = p.states.host_state_count();
    out.host_kv_capacity_bytes    = p.states.host_kv_capacity();
    out.host_kv_occupied_bytes    = p.states.host_kv_bytes();
    return out;
}

void Executor::reset_memory_peaks() { impl_->workspace.reset_peak(); }

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
