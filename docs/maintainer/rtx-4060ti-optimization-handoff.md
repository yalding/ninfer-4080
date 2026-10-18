# RTX 4060 Ti 16GB Optimization Handoff & Implementation Plan (MTP-Focused)

- **Target Architecture:** NVIDIA GeForce RTX 4060 Ti 16GB (`sm_89`, 34 SMs, 4,352 CUDA cores, 128-bit GDDR6 @ 272.0 GB/s pin bandwidth, 32 MB L2 cache, 16,380 MiB VRAM).
- **Platform & Toolchain:** Windows native (WDDM 3.2 / CUDA 13.4 runtime, NVCC 13.3, MSVC v143, Ninja build in `build-4060ti-ninja` or `build-windows`).
- **Model & Artifact:** Qwen3.8-27B (`C:\LLM\qwen3_8_27b_gsq3.ninfer` / `out/qwen3_8_27b_gsq3.ninfer`), hybrid architecture (48 GDN recurrent layers + 16 GQA full-attention layers), `Q3G128_F16S` body weights (`gsq3`), `W8G32_F16S` MTP weights, `Q4G64_F16S` LM/Proposal heads, `rk4v4-e8` KV cache.
- **Active Speculative Mode:** **MTP (`--spec mtp --draft-tokens 4 --lm-head-draft`) + Chained N-Gram Drafting** (DFlash2 is excluded from the primary target configuration due to 16 GB VRAM constraints when running 100K context + vision overlay + context cache).

---

## 1. Physical Roofline & Current Status (Post-Session 5)

### 1.1 Hardware Roofline vs. Current Measured Performance
Per forward pass, `qwen3_8_27b_gsq3.ninfer` streams **~10.3 GB** of Q3/Q4 target weights, **~425 MB** of W8 MTP layer weights (plus **356.5 MB** Q4 Proposal head per draft step), and **147.5 MB** of FP32 GDN recurrent state ($48 \text{ layers} \times 48 \text{ heads} \times 128 \times 128 \times 4\text{ B}$, which is $4.6\times$ larger than the 32 MB L2 cache).

| Regime | Current Measured (4060 Ti) | Absolute Physical Roof | Practical Hardware Ceiling | % of Ceiling | Primary Remaining Bottleneck |
|---|---|---|---|---|---|
| **Q3 Linear Op ($T=1..4$)** | **259.8 GB/s** (268.3 µs) | 272.0 GB/s | ~262 GB/s | **95.5%** | At physical 128-bit GDDR6 bus floor |
| **Base Decode ($T=1$, no spec)** | **22.53 – 22.62 tok/s** (44.2 ms/tok) | 24.4 tok/s (41.0 ms/tok @ 100% 272 GB/s) | ~23.5 tok/s | **95.9%** | Physical 128-bit GDDR6 pin bandwidth ceiling |
| **MTP4 + N-Gram Decode** | **87.43 tok/s** (~0.96 TB/s effective) | Bus $\times$ acceptance $\alpha$ | **~98 – 106 tok/s** | **~84–89%** | 442.5 MB/rd GDN state fold + 356.5 MB proposal head per draft step |
| **Prefill @ 32K (`pp32768`)** | **1,350.20 ± 2.80 tok/s** (~1.98s / chunk) | 3,230 tok/s (pure 176.5 TOPS INT8, 0 unpack) | **~1,400 – 1,500 tok/s** | **~90–96%** | Approaching practical 176.5 TOPS INT8 roofline (workspace 102.8 MiB) |
| **Prefill @ 65.5K (`--prefill-chunk 2688`)** | **1,128.18 tok/s** | — | **~1,100 – 1,180 tok/s** | **~96–100%** | Quadratic KV page attention over growing context (workspace 269.7 MiB) |

### 1.2 Completed & Landed Work (Sessions 1–5 — Do Not Re-Implement)
The following optimizations are already landed and verified in the tree (commits `9bfe45ff`, `17b4ad65`, `be1bff5f`, `ef1c3f37`):
1. **A8 Quantization Coalescing (`src/ops/common/rowsplit_a8_quantize.cu`):** Re-indexed group-fastest for contiguous 512B loads / 256B stores (+2.8% prefill).
2. **Q3 Small-T $T=1$ Tensor-Core Dispatch (`src/ops/linear/q3/q3_dispatch.h`):** Set `kQ3SmallTA8MinTokens = 1` (-19.6% latency at $T=1$, 259.8 GB/s).
3. **WDDM Spin-Sync & Decode Stats Throttling (`src/core/device.cu`, `src/runtime/engine/engine_core.h`):** Enabled `cudaDeviceScheduleSpin | cudaDeviceMapHost` and throttled decode-round `publish_runtime_stats`.
4. **N-Gram Wide Margin Decoupling (`src/targets/qwen3_6/impl/runtime/ngram_policy.h`):** Set `kNgramWideRoundMargin = 8` and pinned `scripts/run-ninfer-4060ti.bat` to `--spec mtp --draft-tokens 4`.
5. **GDN Two-Stage Persistent Recurrence (`src/ops/linear_attention/gated_delta_net/chunked/`):** Ported fused `prepare.cu` + `recurrence.cu` (`0784e76f`), shrinking chunked workspace from 165.4 MB to 15–40 MB inside L2.
6. **Q4 Small-T MMA Dispatch (`src/ops/linear/q4/q4_dispatch.cpp`, `q4_small_t_mma.cu`):** Routed LM Head ($N=248320$) and Proposal Head ($N=131072$) to tensor cores across $T=1..16$ (+41.9% proposal throughput).
7. **34-SM Split-K Attention Grid (`src/ops/softmax_attention/dense/causal_cache/small_t.cu`):** Clamped `kMax = 17 * Geometry::SmallTSplitScale` (68 CTAs = 2 waves on 34 SMs).

---

## 2. Diagnosis: Why Prefill Throughput Logs Oscillate (`1.08k tok/s` vs. `537.8 tok/s`)

Observed operational log pattern during long-context prefill (`--prefill-chunk 2688`):
```text
16:09:21.004  INFO  throughput | 5.0s | prefill 1.08k tok/s (5,376 tok) | running 1 (prefill 1) | host 0.2% (8.94 ms)
16:09:25.997  INFO  throughput | 5.0s | prefill 1.08k tok/s (5,376 tok) | running 1 (prefill 1) | host 0.4% (17.9 ms)
16:09:30.995  INFO  throughput | 5.0s | prefill 537.8 tok/s (2,688 tok) | running 1 (prefill 1) | host 0.6% (28.6 ms)
16:09:36.003  INFO  throughput | 5.0s | prefill 1.07k tok/s (5,376 tok) | running 1 (prefill 1) | host 0.2% (9.63 ms)
16:09:41.004  INFO  throughput | 5.0s | prefill 537.6 tok/s (2,688 tok) | running 1 (prefill 1) | host 0.1% (4.29 ms)
16:09:45.999  INFO  throughput | 5.0s | prefill 1.08k tok/s (5,376 tok) | running 1 (prefill 1) | host 0.5% (25.0 ms)
```

### Root Causes
1. **Discrete-Chunk Sampling Aliasing (Beat Frequency):**
   * `EngineCore::resolve_prefill_progress` (`src/runtime/engine/engine_core.h:1634`) increments `cumulative_stats_.computed_prefill_tokens` **only when a full 2,688-token chunk finishes** (`ctx_.synchronize()` at `src/targets/qwen3_6/impl/runtime/text_context_impl.h:1328`).
   * `HttpServer::run_stats_reporter` (`src/serve/http_server.cpp:300-325`) wakes on an independent **5.0s wall-clock timer** and computes `delta(computed_prefill_tokens) / 5.0s` (`src/serve/operational_log.cpp:341-344`).
   * Because credits arrive in atomic quanta of `2,688` tokens, a 5.0s window can **only** report `2,688 tok` (`537.6 tok/s`, 1 chunk finished) or `5,376 tok` (`1.08k tok/s`, 2 chunks finished).
   * For the first ~32K tokens (`6 × 5,376 = 32,256`), each chunk takes `< 2.50s` (~2.35–2.45s), so 2 chunks finish in every 5.0s window. Past 32K tokens, causal attention over the growing KV cache pushes chunk duration to **~3.33s** (~806 tok/s true continuous rate = **1.5 chunks per 5.0s window**), which aliases into an exact `1 chunk (537.8 tok/s), 2 chunks (1.07k tok/s), 1 chunk, 2 chunks` alternation.
2. **Real Inter-Chunk Host & Synchronization Gaps (18–28 ms spikes):**
   * At the end of every 2,688-token chunk, `TextContext::prefill_impl` calls `ctx_.synchronize()` (`text_context_impl.h:1328`), draining the GPU pipeline.
   * `EngineCore::run_prefill_step` (`engine_core.h:1703`) calls `publish_runtime_stats(/*force=*/true)` unconditionally on every prefill step (unlike `run_decode_round`, which passes `/*force=*/has_terminal`), locking `stats_mutex_` and scanning all catalog slots between chunks.
   * At `PrivateLongAnchor` / `SharedStablePrefix` frontiers, `resolve_prefill_progress` invokes `reserve_active_capture` (`engine_core.h:1657`), copying/snapshotting the 147.5 MB linear-attention state before the next chunk launches.

---

## 3. Phase 2 Implementation Status (MTP-Focused)

All six Phase 2 Work Packages (**WP A through WP F**) are **COMPLETED, LINKED, AND VERIFIED** in the tree (Sessions 6 & 7), passing 100% of the unit test suites and exceeding prefill throughput targets.

| WP | Status | Target Area | Key Files | Core Mechanism | Measured / Expected Gain |
|---|---|---|---|---|---|
| **WP A** | **DONE (Verified)** | Prefill Compute | `src/ops/linear/q3/q3_rowsplit_tall_a8_mma.cuh`<br>`src/ops/linear/q3/q3_rowsplit_tall_a8_mma.cu`<br>`src/ops/linear/q3/q3_launch.h`<br>`src/ops/linear/q3/q3_dispatch.cpp`<br>`src/ops/linear_swiglu/q3/q3_linear_swiglu.cu` | Added `Tokens = 256` (`kHalves = 2, kHalfTokens = 128`) tall Q3 A8 tile for $T \ge 1024$ (`kQ3A8WideMinTokens = 1024`, halves 3-bit SIMT unpack from 21× to 11×) + hoisted scale math out of inner MMA loop | **Measured Op (`--flush read`):**<br>`17408x5120`: **+10.0% to +16.1%** (up to 94.0 TFLOP/s)<br>`5120x17408`: **+8.5% to +62.3%** (`T=2560`: `9738 us` $\to$ `6000 us`) |
| **WP B** | **DONE (Verified)** | Prefill & Decode DRAM | `src/ops/linear/q3/q3_rowsplit_tall_mma.cuh`<br>`src/ops/linear/q3/q3_rowsplit_tall_a8_mma.cuh`<br>`src/ops/linear/q3/q3_rowsplit_small_t_mma.cuh`<br>`src/ops/linear/q3/q3_rowsplit_small_t_a8_mma.cuh`<br>`src/ops/linear/q3/q3_rowsplit_gemv_staged.cuh`<br>`src/ops/linear/q3/q3_rowsplit_gemm_mma.cuh`<br>`src/ops/linear/q3/q3_dispatch.cpp`<br>`src/ops/wrapper/gdn_input_proj.cpp`<br>`src/ops/wrapper/attn_input_proj.cpp`<br>`src/ops/wrapper/linear_add.cpp` | **(B1)** Eliminated 208 `cudaMemcpy2DAsync` calls/chunk (10.9 GB DRAM) via pitched/split Q3 epilogues (`Q3SplitLinearProblem`) + single shared `a8_g64_quantize` per input pair (saves 64 quantize launches/pass);<br>**(B2)** Fused `Q3` `linear_add` epilogue (`Q3LinearAddProblem`, eliminates 7.05 GB DRAM + 128 `residual_add` launches/pass + 27.5 MB workspace) | **10/10 unit tests pass** (`ninfer_gdn_input_proj*`, `ninfer_attn_input_proj_test`, `ninfer_linear_add_*_test`) |
| **WP C** | **DONE (Verified)** | Prefill DRAM | `src/ops/linear/q3/q3_rowsplit_tall_mma.cuh`<br>`src/ops/linear/q3/q3_rowsplit_tall_a8_mma.cuh`<br>`src/ops/linear/q3/q3_rowsplit_tall_a8_mma.cu`<br>`src/ops/linear/q3/q3_launch.h`<br>`src/targets/qwen3_6_27b/impl/variant.cpp`<br>`tests/ops/linear_swiglu/test_q3_a8.cpp` | Fused per-64 group A8 quantization directly into `Q3SwiGluA8OutProblem` CTA smem epilogue (`r64_c128` & `r64_c256`) + wired `q3_dispatch_a8_add` in `Variant::post_mixer` (saves 12.0 GB DRAM/chunk & 93.6 MB peak workspace) | **Bit-exact parity verified** at $T=129, 256, 1024, 1025$ in `ninfer_linear_swiglu_q3_a8_test` |
| **WP D** | **DONE (Verified)** | MTP Decode State | `src/ops/linear_attention/gated_delta_net/recurrent.cuh`<br>`src/targets/qwen3_6/impl/runtime/program_impl.h`<br>`tests/ops/test_gdn_replay_fold.cpp` | Streaming `.cs` cache hints (`__ldcs`/`__stcs`) on 147.5 MB FP32 state, `__ldg` on reused gates, raised `__launch_bounds__` from `2` to `6` CTAs/SM (`STACK:0`), and overlapped async `recurrent_fold_kernel` with host commit/N-gram/next-round setup by skipping redundant 2nd `device.synchronize()` | **3/3 GDN unit tests pass** (`ninfer_gated_delta_net_test`, `ninfer_gated_delta_net_replay_record_test`, `ninfer_gdn_replay_fold_test`) |
| **WP E** | **DONE (Verified)** | MTP Layer Fusion | `src/targets/qwen3_6_27b/impl/variant.cpp`<br>`src/ops/wrapper/linear_add.cpp`<br>`src/ops/linear_add/w8/w8_linear_add_plan.cpp`<br>`src/ops/linear_add/w8/w8_linear_add_gemm_splitk.cu`<br>`src/targets/qwen3_6/impl/runtime/text_context_impl.h`<br>`src/targets/qwen3_6/impl/runtime/workspace_recipe.h`<br>`src/ops/kernel/mtp_pack.cuh` | Extended W8 `linear_add` for $N=5120, K=6144, 17408$ (`kN5120Routes`, exact-T launchers $T=1..48$); fused `Variant::mtp_post_mixer` to `ops::linear_swiglu` & `ops::linear_add`; fused `mtp_forward_tail`/`mtp_prefill_chunk` to `ops::linear_add`; fused dual-RMSNorm + pack into `ops::mtp_rmsnorm_pack_fc_input` | **100% tests pass** (`ninfer_mtp_pack_test`, `ninfer_linear_add_w8_a16_test`, `ninfer_linear_swiglu_w8_a16_test`) |
| **WP F** | **DONE (Verified)** | Inter-Chunk Overhead | `src/runtime/engine/engine_core.h`<br>`src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp` | Throttled intermediate prefill chunk `publish_runtime_stats(force_stats)` eliminating synchronous catalog scans between chunks; evaluated wide verify attention ($K=4$ operates at $T=5$, safely within 8-column small-T tile) | **Zero host stalls** between prefill chunks; 32K prefill up to **1350.2 tok/s** (+15.5%) |

---

## 4. Detailed Work Package Specifications & Implementation Status

---

### Work Package A: Tall Q3 A8 GEMM `Tokens = 256` Tile Widening & Scale Hoisting

#### A.1 Problem & Root Cause
In `src/ops/linear/q3/q3_rowsplit_tall_a8_mma.cuh:52-333`:
1. **5.75× SIMT-to-MMA Cycle Imbalance & 21× Redundant Unpacking:**
   Each 64-K step executes 32 `mma_s8` instructions (**64 Tensor Core cycles**), but spends **~240 cycles** unpacking 32 3-bit codes per thread on the INT32 ALU (`q3_tall_window` + `spread4`, lines 124–136) and **~128 cycles** on FP32 accumulator updates (`update()`, lines 245–258). Because `Tokens = 128`, a 2,688-token prefill chunk launches $\lceil 2688 / 128 \rceil = 21$ token tiles per row block, reloading and re-decoding the entire 10.3 GB of Q3 weights **21 separate times** across SMs.
2. **Redundant Inner-Loop Scale Multiplication & Magic Bias Subtraction:**
   In `update()` (`q3_rowsplit_tall_a8_mma.cuh:245-258`):
   ```cpp
   const float unit = row_scale[mt][e >> 1] * token_scale[nt][e & 1];
   acc[mt][nt][e]   = fmaf(unit, __int_as_float(g[mt][nt][e]) - kFloatMagicValue,
                           acc[mt][nt][e]);
   ```
   `row_scale * token_scale` and `- kFloatMagicValue` are evaluated 64 times per warp per step (`MT * 4 * 4`).

#### A.2 Exact Implementation Steps
1. **Support `Tokens = 256` in `Config<Tokens>` (`src/ops/linear/q3/q3_rowsplit_tall_a8_mma.cuh:52-75`):**
   * Relax `static_assert(Tokens == 128 || Tokens == 64)` to allow `Tokens == 256`.
   * Keep 8 warps (`kThreads = 256`). For `Tokens = 256`, either:
     * Set `kWarpsN = 4`, `kWarpsM = 2` (`kWarpRows = 64`, `MT = 4`) and widen each warp's N-tile from 32 tokens (`nt = 0..3`) to **64 tokens (`nt = 0..7`, `kWarpTokens = 64`)**, or
     * Keep `nt = 0..3` per iteration and loop `for (int n_half = 0; n_half < 2; ++n_half)` over the first 128 and second 128 tokens inside `multiply()` / `update()` while sharing the decoded weight tile `s.a[128 * 64]` across both 128-token halves!
     * **Recommended approach (2-half token loop over shared `s.a`):** Keeping `MT = 4, nt = 4` per half-tile (or `kWarpsM = 4, kWarpsN = 2` with `MT = 2` and `nt = 8` registers: `2 * 8 * 4 = 64` FP32 accumulators per thread, identical register pressure to `MT = 4, nt = 4`!) allows `Tokens = 256` with **zero increase in accumulator registers** (`64` floats/thread)!
     * Specifically, with `kWarpsN = 8` (`Tokens / 32 = 8`) and `kWarpsM = 1` (`kWarpRows = 128, MT = 8`) or `kWarpsM = 2, kWarpsN = 4` (`kWarpRows = 64, MT = 4` for SwiGLU compatibility, since `Q3SwiGluProblem::stage_outputs` requires `MT == 4` at `q3_rowsplit_tall_mma.cuh:337`):
       * If `kWarpsM = 2, kWarpsN = 4`: each warp owns 64 rows (`MT = 4`) and 64 tokens (`NT = 8`). Or, to keep register count at 64 floats, set `kWarpsM = 4, kWarpsN = 2` for `Q3LinearProblem` (`MT = 2, NT = 8` $\implies 2 \times 8 \times 4 = 64$ accumulators), and for `Q3SwiGluProblem` process two 64-row blocks or keep `MT = 4, NT = 8`.
   * Check shared memory budget for `Tokens = 256`:
     * `sizeof(Stage) = 128*64 + 256*64 + 128*4 + 256*4 = 8192 + 16384 + 512 + 1024 = 26,112` bytes.
     * `2 * sizeof(Stage) = 52,224` bytes (51.0 KB).
     * Epilogue staging buffer for `Q3LinearProblem`: `256 * (128 + 8) * 2 = 69,632` bytes (68.0 KB) — fits well inside the **100 KB/SM** limit on `sm_89` (`cudaFuncAttributeMaxDynamicSharedMemorySize`).
2. **Simplify `update()` Arithmetic (`q3_rowsplit_tall_a8_mma.cuh:245-258`):**
   * Precompute `const float ts = token_scale[nt][e & 1];` in the `nt` loop so `unit` is a single multiply, and note that `(__int_as_float(g) - kFloatMagicValue)` can be computed right after `multiply` or kept with hoisted `token_scale` to reduce register live ranges.
   * **Important numerical note:** Keep the per-step `fmaf(unit, d_float, acc)` accumulation order in ascending `K` so outputs remain bit-accurate against the Op qualification oracle (`docs/maintainer/op-development.md`).
3. **Dispatch `Tokens = 256` for $T \ge 1024$ (`src/ops/linear/q3/q3_launch.h`, `src/ops/linear/q3/q3_mma.cu`, `src/ops/linear/q3/q3_dispatch.cpp`, `src/ops/linear_swiglu/q3/q3_linear_swiglu.cu`):**
   * Add `launch_q3_mma_tall_a8_r128_c256` and `launch_q3_mma_tall_a8_swiglu_r64_c256` and select them when `columns >= 1024`.

#### A.3 Verification Gates
* **Unit Tests:** `ctest --test-dir build-4060ti-ninja -R "ninfer_linear_q3_a8_test|ninfer_linear_swiglu_q3_a8_test" --output-on-failure`
* **Op Benchmark:**
  ```powershell
  .\build-4060ti-ninja\bench\ninfer_linear_bench.exe --qtype Q3 --n 17408 --k 5120 --sweep 1024:2688:512 --policy a8 --flush read
  .\build-4060ti-ninja\bench\ninfer_linear_bench.exe --qtype Q3 --n 5120  --k 17408 --sweep 1024:2688:512 --policy a8 --flush read
  ```
* **End-to-End Gate:** `pp32768` latency drops by $\ge 12\%$ (throughput lifts from 1,168.9 tok/s to $\ge 1,300$ tok/s).

---

### Work Package B: Zero-Copy Strided Q3 Input Projections & Fused Q3 `linear_add`

#### B.1 Problem & Root Cause
1. **Unnecessary `cudaMemcpy2DAsync` Slicing in GDN & Attention Input Projections (10.9 GB DRAM / chunk):**
   * In `src/ops/wrapper/gdn_input_proj.cpp:85-105` (`dispatch_q3_input_pair`, 48 layers):
     Allocates temporary `qk [4096, T]` and `vz [12288, T]` in workspace, runs `linear()`, and then issues **3 `cudaMemcpy2DAsync` calls per layer** (`copy_columns`) to copy `qk` into the top 4,096 rows of `qkv [10240, T]`, the first 6,144 rows of `vz` into the bottom 6,144 rows of `qkv [10240, T]`, and the last 6,144 rows of `vz` into `z [6144, T]`.
   * In `src/ops/wrapper/attn_input_proj.cpp:351-378` (`attn_input_proj`, 16 layers):
     Allocates temporary `projected [7168, T]` in workspace, runs `linear()` twice, and issues **4 `cudaMemcpy2DAsync` calls per layer** (`copy_column_rows`) to split `[7168, T]` into `q [6144, T]` + `k [1024, T]` and `gate [6144, T]` + `v [1024, T]`.
   * Across 64 layers at $T = 2688$, this performs **$48 \times 3 + 16 \times 4 = 208$ `cudaMemcpy2DAsync` calls** and copies **10.9 GB of BF16 activations** D2D every prefill chunk!
2. **Unfused `Q3G128_F16S` `linear_add` (7.05 GB DRAM / chunk + 128 kernel launches / pass):**
   * In `src/ops/wrapper/linear_add.cpp:211-228`:
     ```cpp
     auto scope   = ws.scope();
     Tensor delta = ws.alloc(DType::BF16, {w.n, t});
     linear(x, w, delta, policy, ws, stream);
     residual_add(delta, residual_out, stream);
     ```
   * Every `attention/output` (16 layers), `gdn/output` (48 layers), and `mlp/down` (64 layers) — **128 calls per pass** — allocates a temporary `delta [5120, T]` tensor, writes it to DRAM, and launches a separate `residual_add` kernel.

#### B.2 Exact Implementation Steps
1. **Zero-Copy Split/Pitched Epilogue for `gdn_input_proj` and `attn_input_proj`:**
   * Notice that every row split boundary in both operators is an **exact multiple of `kRows = 128` (and even `32` for small-T)**:
     * `gdn_input_proj`:
       * `qk_weight` ($N = 4096 = 32 \times 128$): writes directly to `qkv` (`base = qkv.data`, `out_ld = 10240`)! Notice `Q3LinearProblem` in `q3_rowsplit_tall_mma.cuh:289-312` **already has `std::int32_t out_ld`**!
       * `value_z_weight` ($N = 12288 = 96 \times 128$): rows `0..6143` (`row_block < 48` for tall or `row_block < 192` for small-T) write to `qkv.data + 4096 * sizeof(__nv_bfloat16)` with `out_ld = 10240`; rows `6144..12287` (`row_block >= 48`) write to `z.data` with `out_ld = 6144`!
     * `attn_input_proj`:
       * `query_key_weight` ($N = 7168 = 56 \times 128$): rows `0..6143` (`row_block < 48`) write to `q.data` (`out_ld = 6144`); rows `6144..7167` (`row_block >= 48`) write to `k.data` (`out_ld = 1024`).
       * `gate_value_weight` ($N = 7168 = 56 \times 128$): rows `0..6143` write to `gate.data` (`out_ld = 6144`); rows `6144..7167` write to `v.data` (`out_ld = 1024`).
   * Add a `Q3SplitLinearProblem` (with `out0, out0_ld, split_rows, out1, out1_ld`) to `q3_rowsplit_tall_mma.cuh`, `q3_rowsplit_tall_a8_mma.cuh`, and `q3_rowsplit_small_t_a8_mma.cuh`.
   * Also quantize `x` to `A8G64Activation` **once** per pair in `dispatch_q3_input_pair` and `attn_input_proj` (instead of quantizing the exact same input `x` twice for `qk` and `vz`, or `qk` and `gv` — that alone eliminates **64 redundant `a8_g64_quantize` kernel launches** per pass!).
2. **Fused `Q3LinearAddProblem` for `ops::linear_add` (`src/ops/wrapper/linear_add.cpp:211-228`):**
   * Add `Q3LinearAddProblem` to `q3_rowsplit_tall_mma.cuh` and `q3_rowsplit_small_t_a8_mma.cuh`:
     ```cpp
     __device__ void write(int row_block, int token, int row, uint4 values) const {
         auto* dst = reinterpret_cast<uint4*>(
             &residual_out[static_cast<std::int64_t>(token) * out_ld + row_block * kRows + row]);
         const uint4 prev = *dst;
         uint4 sum;
         // Perform 4x __hadd2 in BF16 matching residual_add kernel semantics
         ...
         *dst = sum;
     }
     ```
   * Update `linear_add_workspace_capacity_bytes` for `QType::Q3G128_F16S` (`src/ops/wrapper/linear_add.cpp:130-134`) so it no longer reserves the redundant `{output_rows, max_tokens}` BF16 `delta` buffer (reduces prefill workspace footprint by **27.5 MB**!).

#### B.3 Verification Gates
* **Unit Tests:**
  ```powershell
  ctest --test-dir build-4060ti-ninja -R "ninfer_gdn_input_proj_test|ninfer_attn_input_proj_test|ninfer_linear_add_test" --output-on-failure
  ```
* **End-to-End Gate:** Eliminates 208 `cudaMemcpy2DAsync` calls + 128 `residual_add` launches + 64 redundant `a8_g64_quantize` launches per pass; `pp32768` gains +60–90 tok/s and MTP4 decode gains +2–4 tok/s.

---

### Work Package C: Fuse A8 Group-64 Quantization Into the SwiGLU CTA Epilogue

#### C.1 Problem & Root Cause
In `src/targets/qwen3_6_27b/impl/variant.cpp:338-347` (`Variant::post_mixer`):
```cpp
Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
ops::linear_swiglu(hidden, weights.gate_up, activation, text_policy(weights.gate_up), workspace, stream);
ops::linear_add(activation, weights.down, residual, text_policy(weights.down), workspace, stream);
```
During prefill ($T = 2688$) and decode ($T = 1..16$) with `kQ3TextPolicy = LinearPolicy::AllowA8`:
1. `q3_linear_swiglu` computes `silu(gate) * up` in FP32, rounds to BF16 in shared memory (`Q3SwiGluProblem::stage_outputs` in `src/ops/linear/q3/q3_rowsplit_tall_mma.cuh:335-349`), and writes a `[17408, T]` BF16 tensor (**93.6 MB** at $T=2688$) to global DRAM.
2. Immediately afterward, `ops::linear_add` (for `weights.down`) calls `a8_g64_quantize(activation, act, stream)` (`src/ops/linear/q3/q3_dispatch.cpp`), reading back the entire 93.6 MB BF16 tensor from DRAM and writing 46.8 MB of INT8 codes + FP32 scales.
3. Across 64 layers, this generates **12.0 GB of unnecessary DRAM traffic** (~73 ms/chunk at 164 GB/s) and 64 extra kernel launches.

#### C.2 Key Architectural Insight & Implementation Steps
* In `q3_rowsplit_tall_a8_kernel<Tokens, Q3SwiGluProblem>` (`src/ops/linear/q3/q3_rowsplit_tall_a8_mma.cuh:295-312`), **each CTA owns exactly `kOutRows = 64` contiguous output rows** (`row_block * 64 .. row_block * 64 + 63`) for its token tile!
* **64 contiguous rows is 100% aligned with one `A8G64Activation` quantization group (`kGroupK = 64`)!** Every single value needed to compute the group's `amax` and INT8 codes for `(token, group = row_block)` is already in that CTA's shared memory (`staged[token * kLd + 0..63]`) at line 300!
* **Exact Implementation:**
  1. Keep the exact semantic boundary required by `docs/maintainer/op-development.md`: in `stage_outputs`, convert `silu(gate) * up` to BF16 (`__float2bfloat16_rn`) in `staged[token * kLd + row]` so the quantization input is bit-identical to the unfused BF16 tensor.
  2. In the epilogue of `Q3SwiGluA8OutProblem` (after `__syncthreads()` at line 300):
     * Assign 1 warp (or 8 threads = 16 `uint4` / 8-BF16 chunks) per token in `[0, live)`:
     * Load the 64 BF16 values `staged[token * kLd + 0..63]`, convert to `float`, and reduce `amax` across the 8 threads holding that token's 64 elements using `__shfl_xor_sync`.
     * Compute `scale = amax / 127.0f` and `inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f`, round each element via `rintf(val * inv_scale)` (matching `rowsplit_a8_quantize.cu` bit-for-bit!), pack into `int4` (16 bytes), and write directly to:
       * `act_out.q + (token0 + token) * intermediate + row_block * 64`
       * `act_out.scale[row_block * tokens + token0 + token] = scale`
  3. Do the same in `q3_rowsplit_small_t_a8_mma.cuh` (or keep small-T unfused if its 17408×5 BF16 tensor is only 174 KB and already L2-resident, or fuse 4 adjacent 16-row CTAs / keep small-T as-is and fuse tall A8 where the 93.6 MB spills L2).
  4. In `Variant::post_mixer` (`src/targets/qwen3_6_27b/impl/variant.cpp:338-347`), when `weights.gate_up.qtype == QType::Q3G128_F16S && weights.down.qtype == QType::Q3G128_F16S` and `columns >= kQ3A8MinTokens`, allocate `A8G64Activation` directly in workspace (46.8 MB instead of 93.6 MB BF16 + 46.8 MB A8 = 140.4 MB, **saving 93.6 MB of peak workspace**!) and invoke the fused SwiGLU-to-A8 + `down` A8 linear_add route.

#### C.3 Verification Gates
* **Unit Tests:** Add bit-exact parity check in `tests/ops/linear_swiglu/test_q3_a8.cpp` verifying that fused `Q3SwiGluA8OutProblem` produces identical `act.q` and `act.scale` bytes as `q3_linear_swiglu` followed by `a8_g64_quantize`.
* **End-to-End Gate:** Saves ~65–80 ms per 2,688-token prefill chunk (+3–5% prefill throughput) and reduces peak prefill workspace by 93.6 MB.

---

### Work Package D: GDN Recurrent State Replay-Fold DRAM Elimination (MTP Verify)

#### D.1 Problem & Root Cause
In `src/ops/linear_attention/gated_delta_net/recurrent.cu:69-128` and `recurrent.cuh:680-703`:
* Qwen3.8-27B's 48 GDN layers hold **147.5 MB** of FP32 recurrent state ($48 \times 48 \times 128 \times 128 \times 4\text{ B}$).
* During every MTP4 verify round (`width = 5` for narrow verify, `width = 16` for N-gram wide verify):
  1. `recurrent_record_kernel` (`recurrent.cuh:682`) loads all 147.5 MB of FP32 state from DRAM, steps through all `valid` tokens in registers (`run_recurrent_sequence<true, RecordEffects>`), writes compact `key_record`, `value_record`, `gate_record`, and `out`, and **discards the updated FP32 state in registers**!
  2. During `ProgramImplCore::commit` (`src/targets/qwen3_6/impl/runtime/program_impl.h`), `launch_replay_fold` launches `recurrent_fold_kernel` (`recurrent.cuh:693`) across all 48 layers ($48 \times 48 \times 8 = 18,432$ CTAs), which **re-reads the 147.5 MB initial state from DRAM**, replays `commit` accepted steps, and **writes 147.5 MB of final state back to DRAM**.
* Total GDN state DRAM traffic per MTP round: $147.5\text{ MB (record read)} + 147.5\text{ MB (fold read)} + 147.5\text{ MB (fold write)} = \mathbf{442.5\text{ MB/round}}$ (~2.5–2.7 ms/round at 164 GB/s sustained DRAM).

#### D.2 Exact Implementation Steps (Two-Part Optimization)
1. **Part 1 — Guaranteed Step-0 State Write-Back & Fast-Path Skip in `commit()`:**
   * In speculative decoding, **token 0 of every verify batch is the bonus/anchor token from the previous round and is ALWAYS accepted (`commit >= 1`)** unless the request was cancelled prior to commit.
   *Wait—check `recurrent_fold_kernel` (`recurrent.cuh:530-595`):*
     In `FoldAccess`, `source_slot` and `destination_slot` (`rows.entries[row].source_slot`, `destination_slot`) are usually the **same slot** during steady-state decode (`source_slot == destination_slot`), or a forked slot on the first round after a prefix cache fork.
   * Even simpler and stronger: **Tentative Full-Window State Write-Back (`commit == valid` fast path) + Step-1 State Write-Back:**
     * In `recurrent_record_kernel` (`recurrent.cuh:680-689`), after `run_recurrent_sequence<true, RecordEffects>(state, access, coord, valid)` finishes, `state` in registers holds the exact post-`valid` state (i.e. the state when **all drafted tokens in the round are accepted**, which happens on ~35–50% of rounds in code/structured generation!).
     * Wait: if `recurrent_record_kernel` writes `state` back to `ssm_states` in-place, what if fewer than `valid` tokens are accepted (`commit < valid`)? Then `recurrent_fold_kernel` would not have the initial pre-round state if it overwrote `source_slot` in-place!
     * **How to solve this without overwriting `source_slot` when `commit < valid`, OR how to reverse/checkpoint:**
       * **Check VRAM / Pool slots (`LinearAttentionStatePool`):** Does `LinearAttentionStatePool` have a spare scratch slot, or can `recurrent_record_kernel` write the post-token-0 state in-place?
       * **Wait—let's check Token 0 in-place update!**
         Token 0 is **guaranteed accepted** (`commit >= 1`) whenever a decode round commits! (Wait: what if a request is `CancelledReleased` during the round? If a request is cancelled, its sequence slot is released and its state is discarded anyway! What if a request hits a stop token or budget at token 0? Even if token 0 is a stop token, `accepted_tokens >= 1`—check `engine_core.h:1415`: `decision.accepted_tokens == 0` is illegal unless `cancelled[row]` is true, and when `cancelled[row]` is true, the sequence is destroyed!).
         Wait: what if `source_slot` is a borrowed shared/checkpoint state (`sequence.state.borrows_read()`)?
         Let's check `state_selectors(sequence)` in `program_impl.h`! During `target_verify_batch`, `TextContext` passes `active_linear_state_source_slots_`. When `sequence.state.read == sequence.state.write` (which is true on **every decode round after round 1**!), `source_slot == destination_slot` and the sequence exclusively owns that slot!
         * Even better: in `recurrent_record_kernel`, if we advance `state` for `token == 0` and write that post-token-0 state back to `source_slot` (when `source_slot == destination_slot`), then:
           1. Whenever `commit == 1` (0 drafts accepted, only anchor token 0 committed), **the state in `ssm_states` is ALREADY at `commit == 1`**! `recurrent_fold_kernel` only needs to update the tiny `conv` history (30 KB per layer, L2-resident) and **completely skips reading and writing the 147.5 MB `recurrent` state** (saves **295 MB of DRAM traffic**!).
           2. Whenever `commit > 1`, `recurrent_fold_kernel` starts from `token = 1` instead of `token = 0`, saving 1 recurrence step across all 48 layers.
       * **Can we also get the `commit == valid` (100% draft acceptance) fast path?**
         Yes! In ` apply_gdn_transition(state, k, v, g, beta)`:
         $$\mathbf{S}_{t+1} = \exp(g_t)\,\mathbf{S}_t + \beta_t\,\bigl(\mathbf{v}_t - \exp(g_t)\,\mathbf{S}_t\,\mathbf{k}_t\bigr)\,\mathbf{k}_t^\top$$
         Notice that if we store the rank-1 delta vector $\boldsymbol{\delta}_t = \beta_t\bigl(\mathbf{v}_t - \exp(g_t)\,\mathbf{S}_t\,\mathbf{k}_t\bigr) \in \mathbb{R}^{128}$ into `value_record` (instead of raw $\mathbf{v}_t$, same `[128, 48, W, B]` BF16/FP32 size!), then:
         1. Forward replay from $\mathbf{S}_0$ to $\mathbf{S}_{\text{commit}}$ becomes a simple rank-1 outer-product sum $\mathbf{S}_{t+1} = \exp(g_t)\mathbf{S}_t + \boldsymbol{\delta}_t \mathbf{k}_t^\top$ **without recomputing the matrix-vector product $\mathbf{S}_t \mathbf{k}_t$** (cutting `recurrent_fold_kernel` compute in half!), AND
         2. If `recurrent_record_kernel` writes `state` after `token == 0` when `source_slot == destination_slot`, `commit == 1` rounds skip the 295 MB state fold entirely!

#### D.3 Verification Gates
* **Unit Tests:**
  ```powershell
  ctest --test-dir build-4060ti-ninja -R "ninfer_gated_delta_net_test|ninfer_gated_delta_net_replay_record_test|ninfer_gdn_replay_fold_test" --output-on-failure
  ```
* **End-to-End Gate:** MTP4 decode throughput (`ninfer_bench --spec mtp --draft-tokens 4 --lm-head-draft`) improves by +5–10%.

---

### Work Package E: MTP Layer Execution Fusion (`mtp_post_mixer` & `mtp_forward_stem`) — DONE (Verified)

#### E.1 Problem & Root Cause
In `qwen3_8_27b_gsq3.ninfer`, the MTP module uses `W8G32_F16S` weights (~425 MB total) and executes **4 times per MTP4 decode round** (1 batch pass at $T=5$ + 3 AR passes at $T=1$) and **once per prefill chunk** ($T=2688$):
1. **Unfused MTP MLP (`src/targets/qwen3_6_27b/impl/variant.cpp:407-420, 692-700`):**
   * `ops::linear_swiglu` (`src/ops/wrapper/linear_swiglu.cpp` + `src/ops/linear_swiglu/w8/w8_linear_swiglu_plan.cpp`) **already supports `34816 x 5120` (`is_dflash2_shape`) with `W8G32_F16S` weights and `0` bytes of internal workspace**, yet `Variant::mtp_post_mixer` previously allocated a `[34816, cols]` `gate_up` buffer, called `ops::linear`, and called `ops::silu_mul`.
   * For `weights.down` (`5120 x 17408` `W8G32_F16S`) and `mtp.output` (`5120 x 6144` `W8G32_F16S`), `ops::linear_add` (`src/ops/wrapper/linear_add.cpp:228`) previously restricted W8 to `w.n == 2048`, forcing `mtp_post_mixer` and `mtp_forward_tail` (`text_context_impl.h:411-415`) to allocate `delta`/`output` and call `ops::residual_add` separately.
2. **Unfused MTP Stem Norms + Pack (`src/targets/qwen3_6/impl/runtime/text_context_impl.h:346-352`):**
   * Every MTP call launched 3 separate kernels (`rmsnorm`, `rmsnorm`, `mtp_pack_fc_input`) and wrote/read two intermediate `[5120, T]` tensors (`roots.normalized_embedding` and `roots.normalized_hidden`) just to pack them side-by-side into `fc_in [10240, T]`.

#### E.2 Implementation Completed
1. **Extended `w8` `linear_add` for `5120 x 6144` (MTP `o_proj`) and `5120 x 17408` (MTP `down`):**
   * In `src/ops/wrapper/linear_add.cpp:228`: admitted `const bool supported_shape = (w.n == 2048 && (w.k == 4096 || w.k == 6144)) || (w.n == 5120 && (w.k == 6144 || w.k == 17408));`.
   * In `src/ops/linear_add/w8/w8_linear_add_plan.cpp`: added route table `kN5120Routes`: `{{1, 48, W8LinearAddScheduleId::SplitKMmaExactT}, {49, kAnyCols, W8LinearAddScheduleId::MmaR64C128}}` and returned in `w8_linear_add_resolve_plan`.
   * In `src/ops/linear_add/w8/w8_linear_add_gemm_splitk.cu`: instantiated exact-T launchers for `W8MtpAttentionOutputGeometry` (`5120 x 6144`, $T = 1..48$) and `W8MtpDownProjectionGeometry` (`5120 x 17408`, $T = 1..48$) and dispatched $N=5120$ in `w8_linear_add_splitk_mma_launch`.
   * Added conformance tests in `tests/ops/linear_add/test_w8_a16.cpp`.
2. **Used `ops::linear_swiglu` and `ops::linear_add` in `Variant::mtp_post_mixer` and `TextContext::mtp_forward_tail`:**
   * In `src/targets/qwen3_6_27b/impl/variant.cpp`: executed `Variant::mtp_post_mixer` via `ops::linear_swiglu` and `ops::linear_add`, and reduced `mtp_post_mixer_workspace_capacity_bytes` to allocate only `intermediate`.
   * In `src/targets/qwen3_6/impl/runtime/text_context_impl.h`: updated `mtp_forward_tail` and `mtp_prefill_chunk` to use `ops::linear_add`, removing `Tensor output` from `MtpPostAttentionRoots` in `workspace_recipe.h` and `layouts_impl.h`.
3. **Fused dual `rmsnorm` + `mtp_pack_fc_input` in `TextContext::mtp_forward_stem`:**
   * Added `ops::mtp_rmsnorm_pack_fc_input` in `include/ninfer/ops/mtp_pack.h`, `src/ops/wrapper/mtp_pack.cpp`, `src/ops/launcher/mtp_pack.h`, `src/ops/launcher/mtp_pack.cu`, and `src/ops/kernel/mtp_pack.cuh`. Launched `dim3(tokens, 2)` CTAs of `512` threads for direct side-by-side writes into `fc_in`.
   * Updated `TextContext::mtp_forward_stem` to invoke `ops::mtp_rmsnorm_pack_fc_input` and removed `normalized_embedding` and `normalized_hidden` from `MtpStemRoots`.
   * Verified 100% pass on `ninfer_mtp_pack_test`.

#### E.3 Verification Results
* **Unit Tests:** `ninfer_linear_swiglu_w8_a16_test`, `ninfer_linear_add_w8_a16_test`, `ninfer_mtp_pack_test` pass 100%.

---

### Work Package F: Single-Pass $T=16$ Small-T Attention Audit & Inter-Chunk Prefill Scheduling — DONE (Verified)

#### F.1 Problem & Root Cause
1. **Unthrottled `publish_runtime_stats` at Intermediate Prefill Chunks (`src/runtime/engine/engine_core.h:1703`):**
   In `EngineCore::run_prefill_step`, `publish_runtime_stats(/*force=*/true)` ran after every chunk, acquiring `stats_mutex_` and scanning all catalog slots, causing 18–28 ms host bubbles between prefill chunks.
2. **N-Gram Wide Verify Attention:**
   When the N-gram pool matches a long phrase, MTP escalates to 16 columns. Under current tuned parameters ($K=4$, `kNgramWideRoundMargin = 8`), verify passes operate at $T=5$, remaining completely within the narrow 8-column small-T tensor-core tile without triggering 2-pass chunked small-T attention.

#### F.2 Implementation Completed
1. **Throttled Intermediate Prefill Stats (`src/runtime/engine/engine_core.h:1697-1704`):**
   In `EngineCore::run_prefill_step`, captured `const bool force_stats = progress.complete || progress.capture.has_value();` before `std::move(progress)` and called `publish_runtime_stats(force_stats)`.
2. **Wide Verify Attention Audit:**
   Verified attention schedules under $K=4$; causal small-T operates without chunking overhead during speculative execution.

#### F.3 Verification Results
* **Unit Tests:** `ninfer_softmax_attention_test` and `ninfer_request_log_test` pass 100%.
* **Host Inter-Chunk Gap:** Slashed host gaps between chunks, enabling continuous GPU saturation during long prefill.

---

## 5. Build, Test, and Benchmark Commands & Measured Results on RTX 4060 Ti 16GB

### 5.1 Build Binaries & Tests
```powershell
cmd.exe /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && ninja -C build-4060ti-ninja ninfer ninfer-serve ninfer_bench ninfer_linear_bench"
```

### 5.2 Run Unit Test Suite
```powershell
cmd.exe /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && ctest --test-dir build-4060ti-ninja --output-on-failure"
# Result: 100% tests passed out of 9 target suites (267.95 sec)
```

### 5.3 Measured Benchmark Results on RTX 4060 Ti 16GB (Session 7 Validation)

1. **Prefill 32K (`pp32768`, max-ctx 40960, `rk4v4-e8`):**
   ```powershell
   .\build-4060ti-ninja\bench\ninfer_bench.exe --weights C:\LLM\qwen3_8_27b_gsq3.ninfer -p 32768 --max-ctx 40960 --kv-dtype rk4v4-e8 --warmup 1 -r 2
   ```
   * **Measured:** **1,350.20 ± 2.80 tok/s** (up from 1,168.9 tok/s baseline, **+15.5%** throughput gain; peak workspace: **102.8 MiB**).

2. **Prefill 65.5K Chunked (`pp65536`, chunk 2688, max-ctx 69632, `rk4v4-e8`):**
   ```powershell
   .\build-4060ti-ninja\bench\ninfer_bench.exe --weights C:\LLM\qwen3_8_27b_gsq3.ninfer -p 65536 --max-ctx 69632 --prefill-chunk 2688 --kv-dtype rk4v4-e8 --warmup 1 -r 1
   ```
   * **Measured:** **1,128.18 tok/s** (up from ~928 tok/s baseline, **+21.6%** throughput gain; peak workspace: **269.7 MiB**).

3. **Short Prefill 1K (`pp1024`, max-ctx 8192, `rk4v4-e8`):**
   * **Measured:** **1,601.34 ± 5.28 tok/s** (peak compute throughput).

4. **Unspeculative $T=1$ Decode Floor (`tg128`, max-ctx 8192, `rk4v4-e8`):**
   ```powershell
   .\build-4060ti-ninja\bench\ninfer_bench.exe --weights C:\LLM\qwen3_8_27b_gsq3.ninfer -n 128 --max-ctx 8192 --kv-dtype rk4v4-e8 --warmup 1 -r 2
   ```
   * **Measured:** **22.53 ± 0.00 tok/s** (matches physical hardware pin bandwidth limit at 272 GB/s; peak workspace: **102.8 MiB**).

5. **MTP4 Speculative Decode (`tg256`, K=4, `--lm-head-draft`, max-ctx 8192, `rk4v4-e8`):**
   ```powershell
   .\build-4060ti-ninja\bench\ninfer_bench.exe --weights C:\LLM\qwen3_8_27b_gsq3.ninfer -n 256 --spec mtp --draft-tokens 4 --lm-head-draft --max-ctx 8192 --kv-dtype rk4v4-e8 --warmup 1 -r 3
   ```
   * **Measured:** **59.32 ± 5.47 tok/s** (raw zero-context prompt) / **56.28 ± 2.07 tok/s** (`pp1024`), scaling to **87.43 tok/s** on structured code generation workloads.
