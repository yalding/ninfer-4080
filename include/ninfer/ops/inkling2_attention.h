#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Precomputes the compact on-the-fly relative-position bias table plus zero-sink offset:
 *   `rel_table[d, h, t] = dot(r_in[:, h, t], rel_bias[:, d]) - sink[h, t]` for `d in [0, 1024)`
 *   `rel_table[1024, h, t] = -sink[h, t]`
 *
 * Shapes (`T > 0`):
 *   - `r_in`: contiguous FP32 `[16, 24, T]`
 *   - `rel_bias`: contiguous FP32 `[16, 1024]`
 *   - `sink`: contiguous FP32 or BF16 `[24, T]`
 *   - `rel_table`: contiguous FP32 `[1025, 24, T]`
 */
void inkling2_rel_bias_prepare(const Tensor& r_in, const Tensor& rel_bias, const Tensor& sink,
                               Tensor& rel_table, cudaStream_t stream);

/**
 * Causal Sliding-Window Grouped-Query Attention with relative position bias, zero-sink softmax,
 * and fused `sigmoid(gate)` output epilogue for Inkling-2-Mini (`D=128, Hq=24, Hkv=8, W=1024`).
 *
 * Must be called BEFORE appending `[positions[0], positions[0] + T)` to `context` when using a
 * cyclic cache so keys in `[max(0, L-1024), L)` are read from `context` safely alongside
 * current-chunk `[L, L+T)` from `query_k`/`query_v`.
 *
 * Shapes (`T > 0`):
 *   - `q`: contiguous BF16 `[128, 24, T]`
 *   - `query_k`, `query_v`: contiguous BF16 `[128, 8, T]`
 *   - `gate_qgr`: contiguous BF16 `[6528, T]` (reads gate rows `[3072, 6144)`)
 *   - `rel_table`: contiguous FP32 `[1025, 24, T]`
 *   - `positions`: contiguous I32 `[T]`, strictly sequential `[L, L+1, ..., L+T-1]`
 *   - `lane`: scalar cyclic lane index in `[0, context.lane_capacity)`
 *   - `out`: contiguous BF16 `[128, 24, T]` (equals `sum * sigmoid(gate)`)
 */
void inkling2_swa_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                            const Tensor& gate_qgr, const Tensor& rel_table,
                            const Tensor& positions, const CyclicKVCacheLayerView& context,
                            std::int32_t lane, Tensor& out, cudaStream_t stream);

/**
 * Split-KV Sliding-Window Attention overload using `workspace` and CUDA-Graph-safe
 * `max_visible_keys` envelope. When `append_to_ring` is true and `T == 1`, also commits
 * `query_k`/`query_v` into the cyclic KV ring cache at `positions[0] & 1023`.
 */
void inkling2_swa_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                            const Tensor& gate_qgr, const Tensor& rel_table,
                            const Tensor& positions, const CyclicKVCacheLayerView& context,
                            std::int32_t lane, std::uint32_t max_visible_keys,
                            WorkspaceArena& workspace, Tensor& out, cudaStream_t stream,
                            bool append_to_ring = false);

/**
 * Returns caller-owned transient workspace capacity in bytes required by
 * `inkling2_full_attention` for `storage in {BFloat16, Int8Group64}`, visible key envelope, and
 * query token interval `[min_tokens, max_tokens]`.
 */
[[nodiscard]] std::size_t inkling2_full_attention_workspace_capacity_bytes(
    KvCacheStorage storage, std::uint32_t max_visible_keys, std::int32_t min_tokens,
    std::int32_t max_tokens);

/**
 * Full-Attention causal GQA (`D=128, Hq=24, Hkv=8`) over paged KV cache (`BFloat16` or
 * `Int8Group64`) with clamped relative-position bias (`rel_table[min(q_pos - k_pos, 1024)]`),
 * zero-sink softmax, and fused `sigmoid(gate)` output epilogue.
 *
 * Expects current tokens `[positions[0]..positions[T-1]]` to ALREADY be appended in `cache`.
 * `max_visible_keys` is a launch/workspace resource envelope (`>= last_pos + 1`).
 */
void inkling2_full_attention(const Tensor& q, const Tensor& gate_qgr, const Tensor& rel_table,
                             const Tensor& positions, const PagedKVLayerView& cache,
                             std::uint32_t max_visible_keys, WorkspaceArena& workspace, Tensor& out,
                             cudaStream_t stream);

/**
 * DSpark Sliding-Window GQA (`D=128, Hq=24, Hkv=8, W=1024`) over `[max(0, F+t-1023), F)` in
 * `context` + non-causal `[F, F+T)` in `block_k`/`block_v` (`1 <= T <= 8`). Standard softmax
 * without relative bias, zero-sink, or output gating.
 */
void inkling2_dspark_swa_attention(const Tensor& q, const Tensor& block_k, const Tensor& block_v,
                                   std::uint32_t context_keys,
                                   const CyclicKVCacheLayerView& context, std::int32_t lane,
                                   Tensor& out, cudaStream_t stream);

/**
 * CUDA-Graph-safe DSpark Sliding-Window GQA reading `context_keys = context_keys_pos[0]` on
 * device.
 */
void inkling2_dspark_swa_attention(const Tensor& q, const Tensor& block_k, const Tensor& block_v,
                                   const Tensor& context_keys_pos,
                                   const CyclicKVCacheLayerView& context, std::int32_t lane,
                                   Tensor& out, cudaStream_t stream);

/**
 * DSpark Full-Attention GQA (`D=128, Hq=24, Hkv=8`) over `[0, F)` in `cache` + non-causal
 * `[F, F+T)` in `block_k`/`block_v` (`1 <= T <= 8`). Standard softmax without relative bias,
 * zero-sink, or output gating.
 */
void inkling2_dspark_full_attention(const Tensor& q, const Tensor& block_k, const Tensor& block_v,
                                    const PagedKVLayerView& cache, std::uint32_t context_keys,
                                    WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * CUDA-Graph-safe DSpark Full-Attention GQA reading `context_keys = context_keys_pos[0]` on
 * device and sizing split-KV from `max_context_keys`.
 */
void inkling2_dspark_full_attention(const Tensor& q, const Tensor& block_k, const Tensor& block_v,
                                    const PagedKVLayerView& cache, const Tensor& context_keys_pos,
                                    std::uint32_t max_context_keys, WorkspaceArena& workspace,
                                    Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
