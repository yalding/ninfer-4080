#include "ninfer/ops/inkling2_attention.h"
#include "ops/softmax_attention/inkling2/launch.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
void shape(const Tensor& x, DType dtype, int a, int b = 1, int c = 1, int d = 1) {
    if (!x.data || x.dtype != dtype || !x.is_contiguous() || x.ne[0] != a || x.ne[1] != b ||
        x.ne[2] != c || x.ne[3] != d || a <= 0 || b <= 0 || c <= 0 || d <= 0)
        throw std::invalid_argument("inkling2_attention: invalid tensor shape, dtype or layout");
}

void common(const Tensor& q, const Tensor& gate, const Tensor& rel, const Tensor& pos,
            Tensor& out) {
    const int t = q.ne[2];
    shape(q, DType::BF16, 128, 24, t);
    shape(out, DType::BF16, 128, 24, t);
    shape(gate, DType::BF16, 6528, t);
    if (rel.dtype != DType::FP32 && rel.dtype != DType::BF16)
        throw std::invalid_argument("inkling2_attention: invalid rel dtype");
    if (rel.dtype == DType::BF16 && t <= 16)
        throw std::invalid_argument("inkling2_attention: BF16 rel requires prompt T > 16");
    shape(rel, rel.dtype, 1025, 24, t);
    shape(pos, DType::I32, t);
    if (out.data == q.data || out.data == gate.data || out.data == rel.data)
        throw std::invalid_argument("inkling2_attention: output aliases input");
}

void plane(const Tensor& x, DType dtype, int leading, int heads, int pages) {
    if (!x.data || x.dtype != dtype || x.ne[0] != leading || x.ne[1] != 64 || x.ne[2] != heads ||
        x.ne[3] != pages || pages <= 0 || x.nb[0] != static_cast<std::int64_t>(dtype_size(dtype)) ||
        x.nb[1] < x.nb[0] * leading || x.nb[2] < x.nb[1] * 64 || x.nb[3] < x.nb[1] * 64)
        throw std::invalid_argument("inkling2_attention: invalid paged plane");
}
} // namespace

void inkling2_rel_bias_prepare(const Tensor& r, const Tensor& bias, const Tensor& sink,
                               Tensor& table, cudaStream_t stream) {
    const int t = r.ne[2];
    shape(r, DType::FP32, 16, 24, t);
    shape(bias, DType::FP32, 16, 1024);
    shape(sink, sink.dtype, 24, t);
    if (sink.dtype != DType::FP32 && sink.dtype != DType::BF16)
        throw std::invalid_argument("inkling2_rel_bias_prepare: sink dtype");
    if (table.dtype != DType::FP32 && table.dtype != DType::BF16)
        throw std::invalid_argument("inkling2_rel_bias_prepare: table dtype");
    shape(table, table.dtype, 1025, 24, t);
    if (table.data == r.data || table.data == bias.data || table.data == sink.data)
        throw std::invalid_argument("inkling2_rel_bias_prepare: aliases");
    inkling2_detail::rel_launch(r, bias, sink, table, stream);
}

void inkling2_swa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& gate,
                            const Tensor& rel, const Tensor& pos,
                            const CyclicKVCacheLayerView& cache, int lane, Tensor& out,
                            cudaStream_t stream) {
    common(q, gate, rel, pos, out);
    const int t = q.ne[2];
    shape(k, DType::BF16, 128, 8, t);
    shape(v, DType::BF16, 128, 8, t);
    if (cache.capacity != 1024 || cache.padded_capacity < 1024 || cache.head_dim != 128 ||
        cache.num_kv_heads != 8 || lane < 0 || lane >= cache.lane_capacity)
        throw std::invalid_argument("inkling2_swa_attention: cache geometry");
    shape(cache.k, DType::BF16, 128, cache.padded_capacity, 8, cache.lane_capacity);
    shape(cache.v, DType::FP16, 128, cache.padded_capacity, 8, cache.lane_capacity);
    if (out.data == k.data || out.data == v.data || out.data == cache.k.data ||
        out.data == cache.v.data)
        throw std::invalid_argument("inkling2_swa_attention: aliases");
    inkling2_detail::swa_launch(q, k, v, gate, rel, pos, cache, lane, 1u, nullptr, false, out,
                                stream);
}

void inkling2_swa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& gate,
                            const Tensor& rel, const Tensor& pos,
                            const CyclicKVCacheLayerView& cache, int lane,
                            std::uint32_t max_visible_keys, WorkspaceArena& workspace, Tensor& out,
                            cudaStream_t stream, bool append_to_ring) {
    common(q, gate, rel, pos, out);
    const int t = q.ne[2];
    shape(k, DType::BF16, 128, 8, t);
    shape(v, DType::BF16, 128, 8, t);
    if (cache.capacity != 1024 || cache.padded_capacity < 1024 || cache.head_dim != 128 ||
        cache.num_kv_heads != 8 || lane < 0 || lane >= cache.lane_capacity || max_visible_keys == 0)
        throw std::invalid_argument("inkling2_swa_attention: cache geometry");
    shape(cache.k, DType::BF16, 128, cache.padded_capacity, 8, cache.lane_capacity);
    shape(cache.v, DType::FP16, 128, cache.padded_capacity, 8, cache.lane_capacity);
    if (out.data == k.data || out.data == v.data || out.data == cache.k.data ||
        out.data == cache.v.data)
        throw std::invalid_argument("inkling2_swa_attention: aliases");
    const unsigned swa_keys =
        std::min<unsigned>(max_visible_keys, 1024u + static_cast<unsigned>(t));
    const unsigned ns = inkling2_detail::swa_splits(swa_keys, t);
    auto scope        = workspace.scope();
    float* scratch    = nullptr;
    if (ns > 1u) {
        const std::size_t bytes =
            ((std::size_t(t) * 24 * ns * 130 * sizeof(float)) + 255) & ~std::size_t(255);
        scratch = static_cast<float*>(workspace.alloc_bytes(bytes).data);
    }
    inkling2_detail::swa_launch(q, k, v, gate, rel, pos, cache, lane, swa_keys, scratch,
                                append_to_ring && (t == 1), out, stream);
}

std::size_t inkling2_full_attention_workspace_capacity_bytes(KvCacheStorage storage,
                                                             std::uint32_t keys, std::int32_t lo,
                                                             std::int32_t hi) {
    if ((storage != KvCacheStorage::BFloat16 && storage != KvCacheStorage::Int8Group64 &&
         storage != KvCacheStorage::RK4V4E8 &&
         storage != KvCacheStorage::RotatedInt4KeyInt4ValueGroup64) ||
        keys == 0 || keys > static_cast<unsigned>(std::numeric_limits<int>::max()) || lo <= 0 ||
        hi < lo)
        throw std::invalid_argument("inkling2_full_attention: workspace envelope");
    std::size_t peak = 0;
    for (int t : {lo, std::min(hi, 16), hi}) {
        if (t < lo) continue;
        peak = std::max(peak, std::size_t(t) * 24 * inkling2_detail::splits(keys, t) * 130 *
                                  sizeof(float));
    }
    return std::max<std::size_t>(256, (peak + 255) & ~std::size_t(255));
}

void inkling2_full_attention(const Tensor& q, const Tensor& gate, const Tensor& rel,
                             const Tensor& pos, const PagedKVLayerView& cache, std::uint32_t keys,
                             WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    common(q, gate, rel, pos, out);
    const auto bytes = q.ne[2] > 16
                           ? std::size_t(0)
                           : inkling2_full_attention_workspace_capacity_bytes(
                                 cache.storage, keys, q.ne[2], q.ne[2]);
    if (cache.head_dim != 128 || cache.num_kv_heads != 8)
        throw std::invalid_argument("inkling2_full_attention: cache geometry");
    shape(cache.block_table, DType::I32, cache.block_table.ne[0]);
    if ((keys + 63) / 64 > static_cast<unsigned>(cache.block_table.ne[0]))
        throw std::invalid_argument("inkling2_full_attention: table envelope");
    const int pages = cache.k_pages.ne[3];
    if (cache.storage == KvCacheStorage::BFloat16) {
        plane(cache.k_pages, DType::BF16, 128, 8, pages);
        plane(cache.v_pages, DType::FP16, 128, 8, pages);
    } else if (cache.storage == KvCacheStorage::Int8Group64) {
        plane(cache.k_pages, DType::I8, 128, 8, pages);
        plane(cache.v_pages, DType::I8, 128, 8, pages);
        plane(cache.k_scale_pages, DType::FP16, 2, 8, pages);
        plane(cache.v_scale_pages, DType::FP16, 2, 8, pages);
    } else {
        plane(cache.k_pages, DType::U8, 64, 8, pages);
        plane(cache.v_pages, DType::U8, 64, 8, pages);
        plane(cache.k_scale_pages, DType::FP16, 2, 8, pages);
        plane(cache.v_scale_pages, DType::FP16, 2, 8, pages);
    }
    auto scope = workspace.scope();
    float* scratch_ptr =
        bytes > 0 ? static_cast<float*>(workspace.alloc_bytes(bytes).data) : nullptr;
    inkling2_detail::full_launch(q, gate, rel, pos, cache, keys, scratch_ptr, out, stream);
}

void inkling2_swa_attention_batched(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& gate, const Tensor& rel, const Tensor& pos,
                                    const Tensor& slots, const CyclicKVCacheLayerView& cache,
                                    std::uint32_t max_visible_keys, WorkspaceArena& workspace,
                                    Tensor& out, cudaStream_t stream, bool append_to_ring) {
    common(q, gate, rel, pos, out);
    const int b = q.ne[2];
    if (b <= 0 || b > 8)
        throw std::invalid_argument("inkling2_swa_attention_batched: B must be in [1, 8]");
    shape(k, DType::BF16, 128, 8, b);
    shape(v, DType::BF16, 128, 8, b);
    shape(slots, DType::I32, b);
    if (cache.capacity != 1024 || cache.padded_capacity < 1024 || cache.head_dim != 128 ||
        cache.num_kv_heads != 8 || max_visible_keys == 0)
        throw std::invalid_argument("inkling2_swa_attention_batched: cache geometry");
    shape(cache.k, DType::BF16, 128, cache.padded_capacity, 8, cache.lane_capacity);
    shape(cache.v, DType::FP16, 128, cache.padded_capacity, 8, cache.lane_capacity);
    const unsigned swa_keys =
        std::min<unsigned>(max_visible_keys, 1024u + 1u);
    const unsigned ns = inkling2_detail::swa_splits(swa_keys, 1);
    auto scope        = workspace.scope();
    float* scratch    = nullptr;
    if (ns > 1u) {
        const std::size_t bytes =
            ((std::size_t(b) * 24 * ns * 130 * sizeof(float)) + 255) & ~std::size_t(255);
        scratch = static_cast<float*>(workspace.alloc_bytes(bytes).data);
    }
    inkling2_detail::swa_batched_launch(q, k, v, gate, rel, pos, slots, cache, swa_keys, scratch,
                                        append_to_ring, out, stream);
}

void inkling2_kv_append_batched(const Tensor& k, const Tensor& v, const Tensor& pos,
                                const Tensor& slots, const PagedKVBatchLayerView& cache,
                                cudaStream_t stream) {
    const int b = k.ne[2];
    if (b <= 0 || b > 8)
        throw std::invalid_argument("inkling2_kv_append_batched: B must be in [1, 8]");
    shape(k, DType::BF16, 128, 8, b);
    shape(v, DType::BF16, 128, 8, b);
    shape(pos, DType::I32, b);
    shape(slots, DType::I32, b);
    if (cache.head_dim != 128 || cache.num_kv_heads != 8)
        throw std::invalid_argument("inkling2_kv_append_batched: cache geometry");
    inkling2_detail::kv_append_batched_launch(k, v, pos, slots, cache, stream);
}

void inkling2_full_attention_batched(const Tensor& q, const Tensor& gate, const Tensor& rel,
                                     const Tensor& pos, const Tensor& slots,
                                     const PagedKVBatchLayerView& cache,
                                     std::uint32_t keys, WorkspaceArena& workspace,
                                     Tensor& out, cudaStream_t stream) {
    common(q, gate, rel, pos, out);
    const int b = q.ne[2];
    if (b <= 0 || b > 8)
        throw std::invalid_argument("inkling2_full_attention_batched: B must be in [1, 8]");
    shape(slots, DType::I32, b);
    if (cache.head_dim != 128 || cache.num_kv_heads != 8)
        throw std::invalid_argument("inkling2_full_attention_batched: cache geometry");
    if ((keys + 63) / 64 > static_cast<unsigned>(cache.block_tables.ne[0]))
        throw std::invalid_argument("inkling2_full_attention_batched: table envelope");
    const int pages = cache.k_pages.ne[3];
    if (cache.storage == KvCacheStorage::BFloat16) {
        plane(cache.k_pages, DType::BF16, 128, 8, pages);
        plane(cache.v_pages, DType::FP16, 128, 8, pages);
    } else if (cache.storage == KvCacheStorage::Int8Group64) {
        plane(cache.k_pages, DType::I8, 128, 8, pages);
        plane(cache.v_pages, DType::I8, 128, 8, pages);
        plane(cache.k_scale_pages, DType::FP16, 2, 8, pages);
        plane(cache.v_scale_pages, DType::FP16, 2, 8, pages);
    } else {
        plane(cache.k_pages, DType::U8, 64, 8, pages);
        plane(cache.v_pages, DType::U8, 64, 8, pages);
        plane(cache.k_scale_pages, DType::FP16, 2, 8, pages);
        plane(cache.v_scale_pages, DType::FP16, 2, 8, pages);
    }
    const auto bytes =
        inkling2_full_attention_workspace_capacity_bytes(cache.storage, keys, b, b);
    auto scope         = workspace.scope();
    float* scratch_ptr = static_cast<float*>(workspace.alloc_bytes(bytes).data);
    inkling2_detail::full_batched_launch(q, gate, rel, pos, slots, cache, keys, scratch_ptr, out,
                                         stream);
}

void inkling2_dspark_swa_attention(const Tensor& q, const Tensor& block_k, const Tensor& block_v,
                                   std::uint32_t context_keys,
                                   const CyclicKVCacheLayerView& cache, std::int32_t lane,
                                   Tensor& out, cudaStream_t stream) {
    const int t = q.ne[2];
    if (t <= 0 || t > 8)
        throw std::invalid_argument("inkling2_dspark_swa_attention: T must be in [1, 8]");
    shape(q, DType::BF16, 128, 24, t);
    shape(out, DType::BF16, 128, 24, t);
    shape(block_k, DType::BF16, 128, 8, t);
    shape(block_v, DType::BF16, 128, 8, t);
    if (cache.capacity != 1024 || cache.padded_capacity < 1024 || cache.head_dim != 128 ||
        cache.num_kv_heads != 8 || lane < 0 || lane >= cache.lane_capacity)
        throw std::invalid_argument("inkling2_dspark_swa_attention: cache geometry");
    shape(cache.k, DType::BF16, 128, cache.padded_capacity, 8, cache.lane_capacity);
    shape(cache.v, DType::FP16, 128, cache.padded_capacity, 8, cache.lane_capacity);
    inkling2_detail::dspark_swa_launch(q, block_k, block_v, context_keys, nullptr, cache, lane, out,
                                       stream);
}

void inkling2_dspark_swa_attention(const Tensor& q, const Tensor& block_k, const Tensor& block_v,
                                   const Tensor& context_keys_pos,
                                   const CyclicKVCacheLayerView& cache, std::int32_t lane,
                                   Tensor& out, cudaStream_t stream) {
    const int t = q.ne[2];
    if (t <= 0 || t > 8)
        throw std::invalid_argument("inkling2_dspark_swa_attention: T must be in [1, 8]");
    shape(q, DType::BF16, 128, 24, t);
    shape(out, DType::BF16, 128, 24, t);
    shape(block_k, DType::BF16, 128, 8, t);
    shape(block_v, DType::BF16, 128, 8, t);
    if (!context_keys_pos.data || context_keys_pos.dtype != DType::I32 ||
        !context_keys_pos.is_contiguous() || context_keys_pos.ne[0] < 1)
        throw std::invalid_argument("inkling2_dspark_swa_attention: context_keys_pos");
    if (cache.capacity != 1024 || cache.padded_capacity < 1024 || cache.head_dim != 128 ||
        cache.num_kv_heads != 8 || lane < 0 || lane >= cache.lane_capacity)
        throw std::invalid_argument("inkling2_dspark_swa_attention: cache geometry");
    shape(cache.k, DType::BF16, 128, cache.padded_capacity, 8, cache.lane_capacity);
    shape(cache.v, DType::FP16, 128, cache.padded_capacity, 8, cache.lane_capacity);
    inkling2_detail::dspark_swa_launch(q, block_k, block_v, 0u,
                                       static_cast<const int*>(context_keys_pos.data), cache, lane,
                                       out, stream);
}

void inkling2_dspark_full_attention(const Tensor& q, const Tensor& block_k, const Tensor& block_v,
                                    const PagedKVLayerView& cache, std::uint32_t context_keys,
                                    WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    const int t = q.ne[2];
    if (t <= 0 || t > 8)
        throw std::invalid_argument("inkling2_dspark_full_attention: T must be in [1, 8]");
    shape(q, DType::BF16, 128, 24, t);
    shape(out, DType::BF16, 128, 24, t);
    shape(block_k, DType::BF16, 128, 8, t);
    shape(block_v, DType::BF16, 128, 8, t);
    if (cache.head_dim != 128 || cache.num_kv_heads != 8)
        throw std::invalid_argument("inkling2_dspark_full_attention: cache geometry");
    if (context_keys > 0) {
        shape(cache.block_table, DType::I32, cache.block_table.ne[0]);
        if ((context_keys + 63) / 64 > static_cast<unsigned>(cache.block_table.ne[0]))
            throw std::invalid_argument("inkling2_dspark_full_attention: table envelope");
        const int pages = cache.k_pages.ne[3];
        if (cache.storage == KvCacheStorage::BFloat16) {
            plane(cache.k_pages, DType::BF16, 128, 8, pages);
            plane(cache.v_pages, DType::FP16, 128, 8, pages);
        } else if (cache.storage == KvCacheStorage::Int8Group64) {
            plane(cache.k_pages, DType::I8, 128, 8, pages);
            plane(cache.v_pages, DType::I8, 128, 8, pages);
            plane(cache.k_scale_pages, DType::FP16, 2, 8, pages);
            plane(cache.v_scale_pages, DType::FP16, 2, 8, pages);
        } else {
            plane(cache.k_pages, DType::U8, 64, 8, pages);
            plane(cache.v_pages, DType::U8, 64, 8, pages);
            plane(cache.k_scale_pages, DType::FP16, 2, 8, pages);
            plane(cache.v_scale_pages, DType::FP16, 2, 8, pages);
        }
    }
    const auto keys_env = std::max<std::uint32_t>(1, context_keys);
    const auto bytes = inkling2_full_attention_workspace_capacity_bytes(cache.storage, keys_env, t, t);
    auto scope       = workspace.scope();
    float* scratch_ptr =
        context_keys > 0 ? static_cast<float*>(workspace.alloc_bytes(bytes).data) : nullptr;
    inkling2_detail::dspark_full_launch(q, block_k, block_v, cache, context_keys, context_keys,
                                        nullptr, scratch_ptr, out, stream);
}

void inkling2_dspark_full_attention(const Tensor& q, const Tensor& block_k, const Tensor& block_v,
                                    const PagedKVLayerView& cache, const Tensor& context_keys_pos,
                                    std::uint32_t max_context_keys, WorkspaceArena& workspace,
                                    Tensor& out, cudaStream_t stream) {
    const int t = q.ne[2];
    if (t <= 0 || t > 8)
        throw std::invalid_argument("inkling2_dspark_full_attention: T must be in [1, 8]");
    shape(q, DType::BF16, 128, 24, t);
    shape(out, DType::BF16, 128, 24, t);
    shape(block_k, DType::BF16, 128, 8, t);
    shape(block_v, DType::BF16, 128, 8, t);
    if (!context_keys_pos.data || context_keys_pos.dtype != DType::I32 ||
        !context_keys_pos.is_contiguous() || context_keys_pos.ne[0] < 1)
        throw std::invalid_argument("inkling2_dspark_full_attention: context_keys_pos");
    if (cache.head_dim != 128 || cache.num_kv_heads != 8)
        throw std::invalid_argument("inkling2_dspark_full_attention: cache geometry");
    if (max_context_keys > 0) {
        shape(cache.block_table, DType::I32, cache.block_table.ne[0]);
        const int pages = cache.k_pages.ne[3];
        if (cache.storage == KvCacheStorage::BFloat16) {
            plane(cache.k_pages, DType::BF16, 128, 8, pages);
            plane(cache.v_pages, DType::FP16, 128, 8, pages);
        } else if (cache.storage == KvCacheStorage::Int8Group64) {
            plane(cache.k_pages, DType::I8, 128, 8, pages);
            plane(cache.v_pages, DType::I8, 128, 8, pages);
            plane(cache.k_scale_pages, DType::FP16, 2, 8, pages);
            plane(cache.v_scale_pages, DType::FP16, 2, 8, pages);
        } else {
            plane(cache.k_pages, DType::U8, 64, 8, pages);
            plane(cache.v_pages, DType::U8, 64, 8, pages);
            plane(cache.k_scale_pages, DType::FP16, 2, 8, pages);
            plane(cache.v_scale_pages, DType::FP16, 2, 8, pages);
        }
    }
    const auto keys_env = std::max<std::uint32_t>(1, max_context_keys);
    const auto bytes = inkling2_full_attention_workspace_capacity_bytes(cache.storage, keys_env, t, t);
    auto scope       = workspace.scope();
    float* scratch_ptr =
        max_context_keys > 0 ? static_cast<float*>(workspace.alloc_bytes(bytes).data) : nullptr;
    inkling2_detail::dspark_full_launch(q, block_k, block_v, cache, 0u, max_context_keys,
                                        static_cast<const int*>(context_keys_pos.data), scratch_ptr,
                                        out, stream);
}
} // namespace ninfer::ops
