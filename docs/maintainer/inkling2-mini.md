# Inkling-2-Mini Model, Artifact, and Optimization Reference

This reference records the `inkling2_mini` native target (`inkling2-mini / q3g128`,
`inkling2_loop`) in NInfer: model semantics, artifact and converter contract, runtime and state
ownership, and the verified RTX 4060 Ti 16 GB (`sm_89`) Phase 1–8 measurements.

## 1. Model Identity and Architecture

- **Public identity:** `model_id = "inkling2-mini"`, `weights_id = "q3g128"`
- **Target key:** `inkling2_mini` (`src/targets/inkling2_mini/`)
- **Upstream checkpoint:** `thinkingmachines/Inkling-2-Mini` (`model_type = "inkling2_loop"`)

### Core Dimensions & Looped Schedule

| Parameter | Value | Notes |
|---|---:|---|
| Physical layers | `24` | `ip = 0..23` |
| Logical visits | `48` | Prelude `0..5` (6), Loop `6..17` × `3` passes (36), Epilogue `18..23` (6) |
| Hidden size (`C`) | `2304` | |
| Attention heads (`Q` / `KV` / `D`) | `24` / `8` / `128` | GQA 3:1; `5:1` SWA (`1024`) + Full-Attn (`ip in {5, 11, 17, 23}`, 8 full visits) |
| Relative-position bias | `d_rel = 16`, `rel_extent = 1024` | Zero-sink softmax + `sigmoid(gate)` |
| Short convolutions | `L_cache = 4` (`3`-tap state) | `k_conv`, `v_conv` (`1024`), `attn_post_conv`, `ffn_post_conv` (`2304`); dynamic taps on Full-Attn K/V |
| AttnRes block bank | `block_size = 6` | Up to `M = 5` states mixed with parameter-free RMSNorm + `1/48` scaled dot |
| Cross-pass loop controllers | `gate_rank = 64` | `loop/msg_query` (`4` queries: Q, K, V, FFN), `ctrl_down` / `ctrl_gain_forget` |
| Routed MoE | `128` experts, top-`8`, `I = 1152` | Biased sigmoid routing (`route_scale = 8.0`), SiTU (`gate_softcap = 4.0`, `up_softcap = 25.0`) |
| Vocabulary | `201,024` padded / `200,058` unpadded | `final_logit_softcapping = 30.0`, `eos_token_id = 200006` |

## 2. Artifact and Converter Contract

- **Converter:** `tools/convert/inkling2_mini/` (`convert.py`, `verify.py`, `source.py`, `inventory.py`)
- **Encoder profile:** `MAXABS_F16_RECIP_RNE_V1` (streaming per-expert from upstream NVFP4 / MXFP8)
- **Container contents:** `540` objects (`536` tensors + `4` frontend resources), `11,185,122,304` bytes total (`~10.39 GiB` device weights arena):
  - Routed experts `L0..19`: `Q3G128_F16S` (`RowSplitK128V1`, `7,594 MiB`)
  - Routed experts `L20..23`: `Q4G64_F16S` (`RowSplitK128V1`, `2,065 MiB`)
  - Attention projections (`qgr`, `k`, `v`, `output`): `Q4G64_F16S` (`334 MiB`)
  - `text/token_embedding` (`[201024, 2304]`): `Q4G64_F16S` (`235 MiB`)
  - `text/output_head` (`[201024, 2304]`): `Q6G64_F16S` (`345 MiB`)
  - Norms, `sconv`, dynamic conv projections, sinks, relative biases, clips, routers, loop rank-64 tensors: `BF16` / `FP32` (`~49 MiB`)
  - Optional DSpark `mtp/` tensors (`--mtp`, `53` stored tensors -> `593` total objects, `11,476,532,736` bytes total: `10.63 GiB` device weights with full proposal head, `10.66 GiB` when `mtp/draft_head` `Q4G64_F16S [32768, 2304]` + `mtp/draft_head_token_ids` `I32 [32768]` are active under `--lm-head-draft`): `fuse_weight` (`FP32` `[4, 4]`), `mask_embed` & `final_norm` (`BF16`), `3` draft layers (`Q4G64_F16S` `q`/`k`/`v`/`k_ctx`/`v_ctx`/`output`/`gate_up`/`down` + `BF16` norms), and Markov RNN head (`markov/dflare_norm`, `w1`, `rnn_proj_*`, `hidden_out_*`). Bound with `TensorPlacement::ValidateOnly` when speculative decoding is disabled.
  - Optional DFlash 2 `dflash2/` companion tensors (`--dflash2 <dir>` + `--base-artifact <base.ninfer>`, `76` stored tensors -> `616` total objects on base `540`, `11,619,800,576` bytes total / `10.80 GiB` resident device arena): `dflash2/fc` (`Q4G64_F16S [2304, 9216]`), `5` non-causal SWA draft layers (`Q4G64_F16S` `q`/`k`/`v`/`output` + fused `gate_up`/`down`, `BF16` `input_norm`/`post_attention_norm`/`q_norm`/`k_norm`, and `BF16` dynamic grouped width-2 `attention_conv`/`mlp_conv` `base_kernel` + `kernel_projection`), `dflash2/{hidden_norm,final_norm}` (`BF16`), and rank-256 `dflash2/selector/{hidden_projection,predecessor_codebook,successor_codebook}` (`BF16`). Bound with `TensorPlacement::ValidateOnly` when `--spec dflash2` is not selected.
  - Optional Vision `vision/` tensors (`--vision` default auto-detect when `model.visual.*` exists in the source checkpoint; `330` stored tensors -> `871` total objects on base `540`, `924` with DSpark, `947` with DFlash2): `patch_embedding` (`BF16 [1200, 1024]`, HWC `[20, 20, 3]`), `class_embedding`, `cls_position_embedding`, `position_embedding` (`BF16 [1024, 1024]`, `32×32` grid), `ln_pre`, `23` ViT layers (`qkv` `Q4G64_F16S [3072, 1024]`, `output` `Q5G64_F16S [1024, 1024]`, `fc1` `Q4G64_F16S [4096, 1024]`, `fc2` `Q5G64_F16S [1024, 4096]`, plus `BF16` LayerNorms, biases, and `ls_1`/`ls_2` gammas), and `2×2` merger (`proj_in` `W8G32_F16S [4096, 4096]`, `proj_out` `W8G32_F16S [2304, 4096]`). Bound with `TensorPlacement::ValidateOnly` when `--vision` is off, `TensorPlacement::PinnedHost` under `--vision --vision-offload on` (`0 B` permanent VRAM), or `TensorPlacement::Device` under `--vision --vision-offload off`.
  - Frontend resources: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`, and optional `processor_config.json` (`5` host objects when present).

## 3. Runtime, StateImage, Prefix Reuse, and Frontend

- **Public facade:** `src/targets/inkling2_mini/export/ninfer/targets/inkling2_mini/package.h` uses an opaque PIMPL facade (`detail::LoadPlan`, `detail::LoadedModel`) matching `qwen3_6_27b`.
- **StateImage (`169,689,088` bytes fixed per slot):**
  - `48 × 4` convolution history buffers (`1.83 MiB`)
  - `40` cyclic SWA KV caches (`capacity = 1024`, `D = 128`, `Hkv = 8`, BF16 K + FP16 V = `160 MiB`)
  - `8` Full-Attn layers backed by `PagedKVCache` (`RK4V4E8` default in `scripts/run-ninfer-4060ti-inkling2.bat` at `1.06 GiB` @ `131,072` tokens; `Int8Group64` `2.06 GiB`; or `BFloat16`)
  - Optional DSpark state (when `--spec dflash|mtp` is enabled): `9th` `PagedKVCache` Full-Attn layer (`mtp/layers/1`) + `2` cyclic SWA caches (`mtp/layers/{0, 2}`, `1024` window) + `dspark_c_head` and `3`-tap `1.83 MiB` `conv_trace` rollback buffer
  - Optional DFlash 2 state (when `--spec dflash2` is enabled): `5` cyclic SWA KV caches (`capacity = 2048`, `D = 128`, `Hkv = 8`, `BF16` K + `FP16` V = `40.0 MiB` fixed per slot, orthogonal to target Full-Attn KV dtype), `dflash2_depth` frontier, and `3`-tap `conv_trace` rollback buffer; participates in `reset`, `copy`, and host spill/restore
- **Prefix reuse & pressure handling:**
  - `Program::inspect_admission` emits exact `PrefixReusePath` (`SharedStablePrefix`, `PrivateEndpoint`, `PrivateTurnClosure`, `PrivateResponseReplay`, `PrivateLongAnchor`) and H2D/D2D transfer estimates (`169,689,088` B / `81` slices per StateImage).
  - `CapturePressurePlanningSession` and `Program::progress_context_transaction` implement single-owner host spill and eviction under pressure.
  - `ResourceManager::observe_transfer` accounts `ContextResourceClass::State` transfers using `work.payload_bytes` (`169,689,088` B per image) and `units` (`1` per image).
- **Frontend & reasoning:**
  - GPT-4o pre-tokenizer with `normalizer: null` and `ignore_merges: true`.
  - `ReasoningEffort` supports `None`, `Minimal` (`0.1`), `Low` (`0.2`), `Medium` (`0.7`), `High` (`0.9`), and `XHigh` (`0.99`).

## 4. Verified Optimization History (RTX 4060 Ti 16 GB, `sm_89`)

Measured on NVIDIA GeForce RTX 4060 Ti 16 GB (`sm_89`, Windows WDDM, MSVC 14.44, CUDA 13.5 / driver 13.4, Ninja Release in `build-4060ti-ninja/`):

### Phase 1 Baseline: Correctness, 128K Fit & Perplexity

- **WP 1 converter & codec verifier:** `540` objects verified (`390` exact direct tensors, `50` sampled quantized objects, exact `MAXABS_F16_RECIP_RNE_V1` check); `tests.convert.inkling2_mini.test_verify` passed (`2` cases).
- **WP 2 & WP 3 CUDA ops:** `ninfer_kv_cache_append_test`, `ninfer_linear_q4_a16_test`, `ninfer_linear_q6_a16_test`, `test_inkling2_fused`, and `test_inkling2_ops` all passed against independent FP64 host oracles (including `n_kv = 131,072` `Int8Group64` full-attention decode, `T = 1024` SWA ring safety, and CUDA Graph capture/replay).
- **Target & serving unit/schema tests:** `10/10` targeted suites passed (`test_inkling2_state`, `test_inkling2_program`, `test_inkling2_frontend` with 104 tokenizer oracle cases, `ninfer_resource_manager_test`, `ninfer_sampling_defaults_test`, `ninfer_cli_options_test`, `ninfer_openai_schema_test`, `ninfer_anthropic_schema_test`, `ninfer_serve_options_test`, `ninfer_bench_support_test`); broader CTest suites passed with `0` failures.
- **Qwen3.8-27B CLI regression smoke:** `qwen3_8_27b_gsq3.ninfer` (`rk4v4-e8`, `2048` ctx) output `Paris` and stopped cleanly (`stop-token`).

### 128K VRAM Fit & Matched 12-Window Perplexity

- **Idle 128K CLI startup (`--max-context 131072 --kv-capacity 131072 --kv-dtype int8`):** allocated `2048 / 2048` page groups (`10.4 GiB` weights, `2.45 GiB` sequence pool), reporting `1.28 GiB` free after startup and `1.98 GiB` planned slack on `14.65 GiB` WDDM-usable VRAM.
- **Matched 12-window perplexity** (identical `3,792` canonical token IDs from `profiles/perplexity/inkling2-phase1/fixture.txt`, GGUF vocabulary roundtrip verified, `12 × 256` targets, `context = 768`, `stride = 256`):

| Engine / Artifact | KV Dtype | 12-Window TOTAL PPL |
|---|---|---:|
| `llama.cpp` `Inkling-2-Mini-UD-IQ3_XXS.gguf` *(imatrix + mixed NVFP4/Q8_0/Q6_K)* | `BF16` | `8.16205` |
| NInfer `inkling2_mini_q3g128.ninfer` *(RTN `Q3G128`)* | `BF16` | `8.71009` |
| NInfer `inkling2_mini_q3g128.ninfer` *(RTN `Q3G128`)* | `Int8Group64` | `8.73496` *(+0.285% vs BF16)* |

### Phase 2 Tensor Core Prefill & Schedule Fusion (`ninfer_bench`)

Conditions: `--max-ctx 131072 --kv-dtype int8 -r 1 --warmup 0`, `bench/fixtures/bench_corpus.ids` (`131,072`-max-context, `prefill_chunk = 1024`, `54.8 MiB` peak workspace). Phase 2 replaces the Phase 1 SIMT prefill paths with grouped Tensor Core `inkling2_moe` (`Q3G128_F16S` / `Q4G64_F16S`), tiled Tensor Core FlashAttention-2 (`Br=64, Bc=64, D=128`), tiled Tensor Core `inkling2_loop_control` & `inkling2_small_bf16_proj`, coalesced `inkling2_rel_bias_prepare` & decode `control_out_kernel`, and fused `Executor::schedule` AttnRes+RMSNorm passes:

| Case | Phase 1 Prefill (tok/s) | Phase 2 Prefill (tok/s) | Speedup | Phase 1 Decode (tok/s) | Phase 2 Decode (tok/s) |
|---|---:|---:|---:|---:|---:|
| `pp512` | `92.16` | `1,771.39` | `19.2×` | — | — |
| `pp2048` | `86.17` | `1,921.90` | `22.3×` | — | — |
| `tg128` (`d=0`) | — | — | — | `30.98` | `34.68` |
| `pp4096+tg128` | `82.34` | `1,885.36` | `22.9×` | `20.63` | `22.15` |
| `pp16384+tg128` | `70.95` | `1,796.11` | `25.3×` | `19.07` | `20.39` |
| `pp32768+tg128` | `59.21` | `1,693.18` | `28.6×` | `16.59` | `18.53` |

*`ninfer-perplexity` (`profiles/perplexity/inkling2-phase1/fixture.txt`, `--context 768 --stride 256`, `3,791` scored tokens / `13` windows) measured `7.670918` (`bf16` KV, `294.8 tok/s`) and `7.664999` (`int8-g64` KV, `295.4 tok/s`).*

### Phase 3 Decode Optimization (`ninfer_bench`)

Conditions: `--max-ctx 131072 --kv-dtype int8 -r 1 --warmup 0`, `bench/fixtures/bench_corpus.ids` (`131,072`-max-context, `prefill_chunk = 1024`, `54.8 MiB` peak workspace, CUDA Graph decode, RTX 4060 Ti 16 GB `sm_89`). Phase 3 eliminates the small-$T$ ($T < 16$) decode bottlenecks across four components:

1. **8-warp coalesced `inkling2_moe` decode (`T < 16`):** vectorized FP32 `router_kernel` (`dot_bf16x8` + `sum_warp`), 1-warp butterfly top-8 `route_kernel`, shared-`x` 8-warp `gate_up_kernel<Q3|Q4>`, and multi-row (`RowsPerCta = 4`) `down_kernel<Q3|Q4>`.
2. **Barrier-free warp-parallel FlashDecoding (`T < 16`):** 4-channel-per-lane register online-softmax `attention_decode_kernel` for `inkling2_swa_attention` and `inkling2_full_attention` with fused zero-sink epilogue for `Swa` and `nsplits == 1`.
3. **Multi-CTA vectorized `inkling2_loop_control` & `mix_kernel`:** `cur.ne[2] == 1` zero-copy branch broadcast (eliminating 64 D2D copies/token), `control_down_small_t_kernel` (`dim3(16, t, c)`) and `control_out_small_t_kernel` (`dim3(72, t, c)`), and `uint4`/`float4` register-accumulated `mix_kernel`.
4. **`Q4G64_F16S` `T = 1` dispatch & pinned/penalty-skip staging:** `launch_q4_gemv_r4_w1_direct` for `k in {2304, 3072}` at `t == 1`, `PinnedHostBuffer` step staging, zero-penalty `host_counts` skip in `Executor::sample`, and single-sync `Program::decode`.

| Case | Phase 2 Prefill (tok/s) | Phase 3 Prefill (tok/s) | Phase 2 Decode (tok/s) | Phase 3 Decode (tok/s) | Decode Speedup |
|---|---:|---:|---:|---:|---:|
| `pp512` | `1,771.39` | `1,781.74` | — | — | — |
| `pp2048` | `1,921.90` | `1,936.37` | — | — | — |
| `tg128` (`d=0`) | — | — | `34.68` | **`85.94`** *(11.64 ms/tok)* | **`2.48×`** |
| `pp4096+tg128` | `1,885.36` | `1,904.42` | `22.15` | **`71.53`** | **`3.23×`** |
| `pp16384+tg128` | `1,796.11` | `1,834.92` | `20.39` | **`65.59`** | **`3.22×`** |
| `pp32768+tg128` | `1,693.18` | `1,726.92` | `18.53` | **`58.94`** | **`3.18×`** |

*Verification: `test_inkling2_ops all`, `test_inkling2_fused`, `ninfer_linear_q4_a16_test`, `test_inkling2_state`, `test_inkling2_program`, and `test_inkling2_frontend` all passed (`0` failures); `ninfer-perplexity` (`int8-g64` KV, `profiles/perplexity/inkling2-phase1/fixture.txt`, `--context 768 --stride 256`, `3,791` scored tokens / `13` windows) measured `ppl = 7.684223` (`mean_nll = 2.039169`) at `313.4 tok/s`.*

### Phase 4: GQA-Grouped FlashDecoding, `RK4V4E8` (`D=128`), Small-`T` MoE/Fused Ops & DSpark (`K=1..7`)

1. **GQA-grouped (`3:1`) & multi-query (`Tq=2`) FlashDecoding + wave-balanced split-KV:** `attention_decode_kernel` groups 3 query heads per KV-head CTA (`grid.x = T * 8`) with `Tq=1` (`T==1`) and `Tq=2` (`2<=T<=15`), eliminating `3×` redundant KV reads across all 40 SWA and 8 Full-Attn visits.
2. **`RK4V4E8` (`D = 128`) 4-bit rotated E8-lattice KV cache:** Enabled in `PagedKVStorageLayout`, `KVCacheAppendD128Kv8`, `inkling2_full_attention` (`attention_decode_kernel` + `inkling2_prompt_kernel` with in-register `kv_cache_hadamard64`), `StateImage`, `ninfer-perplexity`, and [run-ninfer-4060ti-inkling2.bat](../../scripts/run-ninfer-4060ti-inkling2.bat) (`--kv-dtype rk4v4-e8`). Halves 128K KV payload from `2.06 GiB` (`int8`) to `1.06 GiB` (`rk4v4-e8`).
3. **Small-`T` MoE & fused kernels:** `4`-row `gate_up_kernel<Q3, 4>` for `T < 4`, `ExpertBN = 16` grouped Tensor Core MoE for `4 <= T <= 16`, `8`-row/CTA warp `proj_kernel`, single-launch `sconv_small_t_kernel` (`2 <= T <= 15`), and shared-memory staged `control_down_small_t_kernel`.
4. **DSpark (`K = 1..7`, `W = K + 1`) speculative decoding:** Converter (`--mtp`), `inkling2_dspark_{feature_mix,rmsnorm_rope,rnn_step,commit_conv_state}`, `inkling2_dspark_{swa,full}_attention`, CUDA-graphed `verify_dspark`, and batched D2H `draft_dspark`.

**12-Window `ninfer-perplexity` (`profiles/perplexity/inkling2-phase1/fixture.txt`, `--context 768 --stride 256`, `3,791` tokens):**

| KV Dtype | Mean NLL | PPL | Score Rate |
|---|---:|---:|---:|
| `bf16` | `2.036495` | `7.663701` | `315.4 tok/s` |
| `int8-g64` | `2.039169` | `7.684223` | `320.2 tok/s` |
| `rk4v4-e8` | `2.044920` | `7.728544` *(+0.85% vs `bf16`)* | `320.1 tok/s` |

**128K `ninfer_bench` (`--max-ctx 131072`, `r=3, warmup=1` up to `32K`; `r=1, warmup=1` at `65K/120K`):**

| Case | `int8` `MTP0` (PP / TG) | `rk4v4-e8` `MTP0` (PP / TG) | `rk4v4-e8` DSpark `K=7` (TG / `spec acc`) |
|---|---:|---:|---:|
| `tg128` (`d=0`) | `—` / `77.86 tok/s` | `—` / `74.39 tok/s` | `24.11 tok/s` (`3.37%`) |
| `pp4096+tg128` | `1,907.18` / `60.64 tok/s` | `1,811.58` / `58.27 tok/s` | `42.77 tok/s` (`20.66%`) |
| `pp16384+tg128` | `1,819.93` / `55.99 tok/s` | `1,817.06` / `55.57 tok/s` | `26.78 tok/s` (`12.80%`) |
| `pp32768+tg128` | `1,716.47` / `50.57 tok/s` | `1,725.40` / `49.93 tok/s` | `17.24 tok/s` (`6.51%`) |
| `pp65536+tg128` | `1,535.11` / `42.44 tok/s` | `1,562.01` / `41.71 tok/s` | `15.21 tok/s` (`10.76%`) |
| `pp120000+tg128` | `1,319.46` / `34.59 tok/s` | `1,336.88` / `33.53 tok/s` | `11.73 tok/s` (`12.93%`) |

**DSpark Draft-Window Sweep (`ninfer.exe`, C++20 SPSC prompt, `--greedy --max-new 128 --kv-dtype rk4v4-e8 --no-thinking`):**

| Config | Decode Speed | Acceptance Length | Acceptance Rate | Accepted by Pos |
|---|---:|---:|---:|---|
| `MTP0` (`K=0`) | **`75.4 tok/s`** | `1.00 tok/step` | `—` | `—` |
| DSpark `K=2` (`W=3`) | **`71.4 tok/s`** | `1.98 tok/round` | `49.6%` | `41, 22` |
| DSpark `K=3` (`W=4`) | `61.3 tok/s` | `2.07 tok/round` | `35.5%` | `41, 17, 7` |
| DSpark `K=4` (`W=5`) | `55.0 tok/s` | `2.23 tok/round` | `30.7%` | `37, 19, 9, 5` |
| DSpark `K=7` (`W=8`) | `45.5 tok/s` | `2.31 tok/round` | `18.9%` | `36, 20, 9, 4, 2, 1, 0` |

*Analysis vs. dense Qwen3.8-27B MTP:* Unlike a dense model where verifying `W = 3..8` tokens reads weights once, Inkling-2-Mini's 128-expert Top-8 MoE expands from `8` unique experts/visit at `T=1` to `~22` (`W=3`) and `~51` (`W=8`) unique experts/visit across 48 visits, while `draft_dspark` executes `K` sequential full-vocab Q6 `output_head` (`[201024, 2304]`) calls. Because `MTP0` (`T=1`) already runs at `75.4 tok/s` (`13.3 ms/tok`), `K=2` (`28.1 ms/round`) is the fastest DSpark operating point and approaches `MTP0` parity (`71.4 tok/s`), while `MTP0` remains faster on mixed/reasoning workloads with `<= 2.3 tok/round` acceptance.

*Verification: `test_inkling2_ops all`, `ninfer_kv_cache_append_test`, `test_inkling2_fused`, `test_inkling2_state`, `test_inkling2_program`, and `tests.convert.inkling2_mini.test_verify` all passed (`0` failures).*

### Local HTTP Serving Verification

- **Anthropic `/v1/messages` shared prefix & pressure:** `wp6-anth-b` hit `shared_stable_prefix` (`207` cached tokens, answered `Paris`, `end_turn`). Under pressure (`--device-state-slots 1 --host-state-slots 4 --host-kv-mib 32`), `wp6-anth-c` completed capture (evicting `2` private + `1` shared owners / `6` checkpoints) and `wp6-anth-d` hit `shared_stable_prefix` (`156` cached tokens, answered `Tokyo`, `end_turn`).
- **OpenAI `/v1/chat/completions` & `/slots`:** `wp6-private` answered `4` then `7` (`26` cached tokens) with retained checkpoints visible in `/slots`.
- **State transfer accounting:** `state_transfers.{d2d,d2h}.bytes` recorded exact multiples of `169,689,088` B (`169689088` for `count=1`, `339378176` for `count=2`, `509067264` for `count=3`).

### Phase 5: Decode Recovery, Graphed Shortlist DSpark & Prefill A8/GQA-Grouped Kernels

1. **Core decode recovery (WP 1):**
   - Restored `control_down_small_t_kernel` to `dim3(16, t, c)` with `128` threads (`sum128` warp+SMEM reduction).
   - Added split-KV `inkling2_swa_attention` (`swa_splits`: `4` for `> 256` keys, `2` for `129..256`, `1` for `<= 128`) with `T = 1` ring-append fusion (`append_to_ring`) and `34`-SM wave-aligned `splits()`.
   - Vectorized `KvMode::RK4` loads (`__ldg` `uint32` + `__shfl_sync` + `__half22float2` scales), `exp2f((...) * Log2E)`, parallelized 3-head epilogue, and 6-pair `kv_cache_hadamard64`.
   - Added coalesced 32-bit `Q3` small-`T` MoE loads (`__ldg` + `__shfl_sync` + `__funnelshift_r`) and `gate_up_kernel<true, 2>` (`dim3(576, t)`), keeping `act` in FP32 for `T < 4`.
   - Added vectorized `T = 1` `sconv_single_vec4_kernel` and paired `inkling2_fused_sconv_kv` (cutting `24` SWA-layer launches/step).
2. **Single-sync CUDA-graphed DSpark + `32,768`-row `Q4G64_F16S` shortlist `mtp/draft_head` + `--ngram chain` (WP 2):**
   - Added `mtp/draft_head` (`Q4G64_F16S [32768, 2304]`) and `mtp/draft_head_token_ids` (`I32 [32768]`) in the converter (`--mtp`, `--base-artifact` repack) and `32768`-row dispatch in `q4_dispatch.cpp` (`--lm-head-draft` / `ProposalHead::Optimized`).
   - Added CUDA-Graph-safe `context_keys_pos` overloads for `inkling2_dspark_{swa,full}_attention`, `draft_graphs`, `draft_dspark_body` (no softcap before argmax), `speculative_step_dspark` (single-sync graphed draft + verify, default `draft_tokens = 2`), and shared `NgramDraftPool` (`--ngram chain`).
3. **Prefill `Q3-MoE A8` & `3:1` GQA-grouped `inkling2_prompt_kernel` (WP 3):**
   - Added `inkling2_moe_grouped_a8_mma_kernel<GateUp, 32>` (`m16n8k32` `s8` Tensor Core MMA + `a8_g64_quantize`) for `Q3G128_F16S && T >= 64`.
   - Grouped `inkling2_prompt_kernel` `3:1` by KV head (`dim3((tokens + 20) / 21, 8)`, `21` tokens × `3` GQA heads = `63` active rows in `q_s[64 * 128]`), eliminating `3×` redundant K/V staging and `RK4` dequantization.

**12-Window `ninfer-perplexity` (`profiles/perplexity/inkling2-phase1/fixture.txt`, `--context 768 --stride 256`, `3,791` tokens):**

| KV Dtype | Mean NLL | PPL | Score Rate |
|---|---:|---:|---:|
| `bf16` | `2.038121` | `7.676171` | `297.4 tok/s` |
| `int8-g64` | `2.041130` | `7.699302` | `327.2 tok/s` |
| `rk4v4-e8` | `2.042875` | `7.712754` *(-0.20% vs Phase 4 `7.728544`)* | `328.5 tok/s` |

**128K `ninfer_bench` (`--max-ctx 131072 --kv-dtype rk4v4-e8 -r 2 --warmup 1`):**

| Case | Phase 4 `MTP0` (PP / TG) | Phase 5 `MTP0` (PP / TG) | Phase 5 DSpark `K=2` Shortlist (PP / TG / `spec acc`) |
|---|---:|---:|---:|
| `tg128` (`d=0`) | `—` / `74.39 tok/s` | `—` / **`79.27 tok/s`** | `—` / `48.27 tok/s` (`5.73%`) |
| `pp4096+tg128` | `1,811.58` / `58.27 tok/s` | **`1,894.52`** / **`71.32 tok/s`** | `1,882.31` / `46.39 tok/s` (`12.32%`) |
| `pp32768+tg128` | `1,725.40` / `49.93 tok/s` | **`1,727.03`** / **`56.41 tok/s`** | `1,714.40` / `33.20 tok/s` (`16.06%`) |
| `pp65536+tg128` | `1,562.01` / `41.71 tok/s` | **`1,570.56`** / **`49.43 tok/s`** | `—` |

**512-Token Code Generation (`ninfer.exe`, C++20 SPSC prompt, `--greedy --max-new 512 --kv-dtype rk4v4-e8 --no-thinking`):**

| Config | Decode Speed | Total Elapsed | Acceptance Rate | Mean Length | Accepted by Pos |
|---|---:|---:|---:|---:|---|
| `MTP0` (`K=0`) | `78.1 tok/s` | `8.9s` | `—` | `1.00 tok/step` | `—` |
| **DSpark `K=2` (full head)** | **`79.9 tok/s`** | **`8.7s`** | **`54.1%`** | **`2.08 tok/round`** | `162, 103` (`245` rounds) |
| DSpark `K=2` (`--lm-head-draft --ngram chain`) | `60.7 tok/s` | `10.7s` | `23.4%` | `1.47 tok/round` | `122, 42` (`346` rounds) |

*Note on DSpark vs `MTP0`:* Because Sparse MoE verification (`W=3`) expands active experts from `8` to `~20–22` per visit across 48 visits, a graphed `K=2` round takes `~2.05×` an `MTP0` step. Consequently, **DSpark `K=2` (full head) beats `MTP0` when acceptance exceeds `~52%` (`> 2.05 tok/round`)** — e.g. code/structured generation (`79.9` vs `78.1 tok/s` at `54.1%`) — whereas `MTP0` (`79.27 tok/s`) remains much faster on dense prose (`5.7%–16.1%` acceptance). The current BPE-rank `[0..32709]` shortlist for `--lm-head-draft` drops acceptance from `54.1%` to `23.4%` because common tokens sit above ID `32710`, so leave `--lm-head-draft` off until shortlist calibration is added.

### Phase 6: DFlash 2 (`--spec dflash2`) Speculative Decoding

1. **Artifact & Converter (`tools/convert/inkling2_mini/`):** Added `--dflash2` (`76` stored `dflash2/*` tensors) and `--base-artifact` fast repack (`models/inkling2_mini_q3g128_dflash2.ninfer`, `616` objects, `11,619,800,576` bytes / `10.80 GiB` device weights). Verified `435` exact direct tensors, `min_q4_cosine = 0.991429`, `min_q6_cosine = 0.999642`, and `tests.convert.inkling2_mini.test_verify` (`3` cases, `OK`).
   - Repack command: `python -m tools.convert.inkling2_mini.convert --model models/Inkling-2-Mini --base-artifact models/inkling2_mini_q3g128.ninfer --dflash2 models/Inkling-2-Mini-DFlash2 --no-mtp --out models/inkling2_mini_q3g128_dflash2.ninfer`
   - CLI / Server flags: `--spec dflash2 --draft-tokens 7` (`1..15` supported, text-only, full proposal head)
2. **Closed `H = 2304` CUDA Ops (`src/ops/inkling2/`):** Added `inkling2_dflash2_{kv_materialize,conv_prepare,conv_finish_add,topk16_and_select}` in `include/ninfer/ops/inkling2_fused.h`, `src/ops/inkling2/{launch.h,inkling2_fused.cpp,dflash2.cu}`, plus `2304 <-> {4096, 6912, 13824}` dispatch in `q4_dispatch.cpp` and `(201024, 200058)` support in `speculative_accept_sparse_drafts`. Qualified all new ops against independent FP64 host oracles (`test_inkling2_fused`, `ninfer_linear_q4_a16_test`, `ninfer_speculative_round_test`: `0` failures).
3. **Single-Sync Graphed Runtime & StateImage (`src/targets/inkling2_mini/`):** Wired `--spec dflash2` (default `draft_tokens = 7`, `W = 8`, `1..15` supported) with `[5, 23, 41, 48]` `[9216, T]` feature capture, `rmsnorm(token_embedding, token_norm)` query embedding, `rope_theta = 10000.0`, post-correction `30 * tanh((u + corr) / 30)` rank-256 candidate selector, `5 × 2048` cyclic KV ring (`40.0 MiB` per slot), and single-sync CUDA-graphed `speculative_step_dflash2`.

**512-Token C++20 SPSC Ring-Buffer `K` Sweep (`ninfer.exe`, `--greedy --max-new 512 --max-context 4096 --kv-capacity 4096 --kv-dtype rk4v4-e8 --no-thinking`):**

| Config | Decode Speed | Rounds | Acceptance Rate | Mean Length | Accepted by Pos |
|---|---:|---:|---:|---:|---|
| `MTP0` (`K=0`) | `78.8 tok/s` | `—` | `—` | `1.00 tok/step` | `—` |
| `DFlash2 K=4` (`W=5`) | `80.5 tok/s` | `172` | `49.4%` | `2.97 tok/round` | `125,92,71,51` |
| `DFlash2 K=5` (`W=6`) | `79.5 tok/s` | `161` | `43.6%` | `3.17 tok/round` | `113,86,67,47,37` |
| `DFlash2 K=6` (`W=7`) | `75.9 tok/s` | `156` | `37.9%` | `3.27 tok/round` | `113,78,57,46,34,26` |
| **`DFlash2 K=7` (`W=8`, default)** | **`86.0 tok/s`** | `135` | `39.8%` | **`3.79 tok/round`** | `107,74,61,45,39,29,21` |

**Short-Context Workloads (`ninfer.exe`, `DFlash2 K=7`, `rk4v4-e8` unless noted):**

| Workload | `MTP0` Decode | `DFlash2 K=7` Decode | Speedup | Acceptance | Notes |
|---|---:|---:|---:|---:|---|
| Math reasoning (`512` tok, greedy) | `78.7 tok/s` | **`116.7 tok/s`** | **`+48.3%`** | `52.9%` (`4.65 tok/round`) | `110` rounds (`90,78,74,59,47,31,22`) |
| C++20 `gcd` (`64` tok, greedy) | `82.0 tok/s` | **`102.1 tok/s`** | **`+24.5%`** | `51.6%` (`4.43 tok/round`) | `60/64` exact prefix match |
| C++20 SPSC (`512` tok, `int8`, greedy) | `—` | `72.9 tok/s` | `—` | `31.5%` (`3.19 tok/round`) | `160` rounds |
| >2048 ring-wrap (`2,150` tok, `rk4v4-e8`) | `—` | `60.8 tok/s` | `—` | `26.9%` (`2.88 tok/round`) | `745` rounds, `0` fallbacks (`2,228` total pos) |
| Stochastic (`--temperature 1.0 --seed 42`, `128` tok) | `—` | `59.7 tok/s` | `—` | `23.6%` (`2.62 tok/round`) | `48` rounds, `57.5 MiB` KV (`17.5 + 40.0`) |

**Long-Context `ninfer_bench` (`110,000`-token in-distribution Inkling corpus, `r=2, warmup=1`):**

*`rk4v4-e8` KV (`--max-ctx 131072`, `10.80 GiB` weights + `1.53 GiB` sequence (`1.10 GiB` KV = `1.06 GiB` paged + `40 MiB` DFlash2) + `66.7 MiB` workspace = `~12.40 GiB` total):*

| Depth (`pp + tg128`) | `MTP0` Prefill | `DFlash2` Prefill | `MTP0` Decode | `DFlash2 K=7` Decode | Decode Delta | Spec Acc (`rounds/fb`) |
|---|---:|---:|---:|---:|---:|---:|
| `pp4096+tg128` | `1,919.58 tok/s` | `1,903.35 tok/s` | **`71.86 tok/s`** | `61.96 tok/s` | `-13.8%` | `32.6%` (`78/2`) |
| `pp32768+tg128` | `1,732.08 tok/s` | `1,716.00 tok/s` | `59.37 tok/s` | **`68.02 tok/s`** | **`+14.6%`** | `64.2%` (`48/0`) |
| `pp65536+tg128` | `1,562.63 tok/s` | `1,550.24 tok/s` | **`49.72 tok/s`** | `44.52 tok/s` | `-10.5%` | `54.6%` (`54/0`) |
| `pp100000+tg128` | `1,412.18 tok/s` | `1,403.21 tok/s` | `43.21 tok/s` | **`43.84 tok/s`** | **`+1.5%`** | `70.2%` (`44/0`) |

*`int8-group64` KV (`--max-ctx 102400`, `10.80 GiB` weights + `2.08 GiB` sequence (`1.65 GiB` KV = `1.61 GiB` paged + `40 MiB` DFlash2) + `66.7 MiB` workspace = `~12.95 GiB` total):*

| Depth (`pp + tg128`) | `MTP0` Prefill | `DFlash2` Prefill | `MTP0` Decode | `DFlash2 K=7` Decode | Decode Delta | Spec Acc (`rounds/fb`) |
|---|---:|---:|---:|---:|---:|---:|
| `pp4096+tg128` | `1,918.84 tok/s` | `1,900.71 tok/s` | **`71.90 tok/s`** | `68.57 tok/s` | `-4.6%` | `36.9%` (`72/0`) |
| `pp32768+tg128` | `1,717.49 tok/s` | `1,702.82 tok/s` | `58.00 tok/s` | **`63.45 tok/s`** | **`+9.4%`** | `58.3%` (`50/2`) |
| `pp65536+tg128` | `1,540.10 tok/s` | `1,530.06 tok/s` | **`48.34 tok/s`** | `43.13 tok/s` | `-10.8%` | `49.2%` (`60/2`) |
| `pp100000+tg128` | `1,388.65 tok/s` | `1,375.20 tok/s` | **`40.47 tok/s`** | `28.30 tok/s` | `-30.1%` | `38.9%` (`70/0`) |

At verification width `W = 8`, the Phase 6 full-attention kernel grouped queries in pairs and
scanned the paged KV cache four times per layer visit. Phase 7 uses one scan for all eight queries.

### Phase 7 — Multi-Query Attention and Fused MoE Verification

- **Full attention.** For `T=1..2`, decode resolves each 64-token physical page once and handles the
  clamped relative bias separately for distances of 1,024 or more. For `T=3..16`, Tensor Core
  verification uses one CTA per KV head and split to scan the split across all `3 × T` GQA rows.
  Split-KV supports up to 128 splits, with four warps merging the partial results.
- **MoE and state updates.** At `T=4..16`, a single CTA compacts the expert assignments and the
  grouped MoE kernel uses 128-row by eight-token tiles. `inkling2_moe_post_sconv` combines top-8
  reduction, post-convolution, and the residual add. Paired K/V convolution at `T=2..16` and
  in-kernel trace writes remove 192 device copies and 48 MoE reduction launches per verify graph.
- **Draft depth.** `--spec-adaptive` adds opt-in adaptation, with DFlash2 still accepting a maximum
  depth from 1 to 15. [Speculative decoding options](../cli.md#speculative-decoding)

The perplexity fixture `profiles/perplexity/inkling2-phase1/fixture.txt` (`--context 768 --stride
256`, `rk4v4-e8`, 3,791 scored tokens) measured mean NLL **2.042875**, perplexity **7.712754**, and
**326.9 tok/s**. The reported perplexity matches Phase 5.

`ninfer_bench` used `inkling2_mini_q3g128_dflash2.ninfer`, `bench/fixtures/bench_corpus.ids`, and
DFlash2 K=7 with one warmup and two measured repetitions. Phase 6 used a different corpus and the
runs reported different speculative acceptance rates.

| Case | KV dtype | Prefill (tok/s) | Decode (tok/s) | Acceptance |
|---|---|---:|---:|---:|
| `pp4096+tg128` | `rk4v4-e8` | 1,864.90 | 104.68 | 49.0% |
| `pp32768+tg128` | `rk4v4-e8` | 1,720.59 | 66.58 | 29.5% |
| `pp65536+tg128` | `rk4v4-e8` | 1,558.81 | 84.27 | 46.2% |
| `pp65536+tg128` | `int8-group64` | 1,537.65 | 75.94 | 40.3% |

Verification covered the relative-bias boundary at distances 1,023 and 1,024, full attention out to
131,072 keys, MoE and fused convolution traces, state and program behavior, and CLI and server options.

### Phase 8 — Folded DSpark Head, A8 Prefill, and Batched Decode

1. **DSpark full-vocabulary proposals.** Startup folds `W_head @ W_hidden_out` into group-scaled
   int8 coefficients and precomputes `W_head @ b_hidden_out`. Each draft round projects all base
   logits in one output-head call, then computes the Markov correction and argmax per step over
   all 200,058 valid token IDs. The int8 fold can change the winner when proposal logits are close.
2. **Prefill.** Q4 attention projections and Q4 routed MoE use signed int8 Tensor Core MMA from
   `T=64`. Attention builds the relative-bias table in BF16 for `T>16` and stages the prompt KV
   tile in 48 KiB of dynamic shared memory with a two-CTA-per-SM launch target on Ada.
3. **Batched decode and host checkpoints.** Non-speculative rounds with one input token per
   request share one compact GPU forward pass for up to eight requests. CUDA Graphs are keyed by
   batch width and context envelope. Host checkpoint slabs are pinned at `HostStatePin` during
   startup and recycled as snapshots are released.

The same `rk4v4-e8` perplexity fixture measured mean NLL **2.041193**, perplexity **7.699793**,
and **315.6 tok/s** across 13 windows (3,791 scored tokens).

The DSpark artifact generated 512 tokens on a greedy C++20 SPSC ring-buffer prompt with
`rk4v4-e8`, a 4,096-token context, the full proposal vocabulary, and default n-gram chaining.
The n-gram extensions contributed zero accepted tokens.

| DSpark draft depth | Decode | Reported acceptance |
|---|---:|---:|
| `K=2` | **83.6 tok/s** | **52.7%** |
| `K=4` | **71.3 tok/s** | **36.8%** |

The DFlash2 `K=7` benchmark used the Phase 7 corpus, `rk4v4-e8`, a 131,072-token context
limit, one warmup, and two measured repetitions.

| Case | Prefill (tok/s) | Prefill gain vs Phase 7 | Decode (tok/s) | Acceptance |
|---|---:|---:|---:|---:|
| `pp4096+tg128` | 2,095.56 | +12.4% | 114.53 | 57.7% |
| `pp32768+tg128` | 1,902.55 | +10.6% | 69.13 | 29.3% |
| `pp65536+tg128` | 1,687.35 | +8.2% | 85.25 | 46.2% |

In `ninfer-serve` with `rk4v4-e8` and CUDA Graphs, four prompts of 31–210 tokens each
generated 128 greedy tokens. Aggregate completion throughput, including prefill and HTTP
overhead, rose from **74.20 tok/s** serially to **119.56 tok/s** concurrently (**+61.1%**).
All four concurrent response texts matched their serial counterparts exactly. Four additional
32-token comparisons with a frequency penalty of 0.5 also matched exactly.

Verification covered the Inkling2 attention and MoE oracles, fused operations, state and program
suites, Q4 linear, and CLI and server options.

### Phase 9 — Vision-from-RAM (`23`-Layer ViT & Double-Buffered Overlay)

- **Image processor & chat template (`processor.{h,cpp}`, `chat_template.{h,cpp}`):** `Inkling2ImageProcessor` resizes with FP32 Lanczos-3 (`factor = 40`), normalizes `(p/255 - 0.5)/0.5`, right/bottom pads to a multiple of `40` with `-1` in pixel space (`-1.007843137f` -> BF16 `0xBF81`), drops completely padded `20×20` sub-patches (`20*y >= target_h || 20*x >= target_w`), computes bilinear `32×32` `position_embedding` indices/weights, and builds `2×2` `merge_indices` (`-1` -> zero-fill). Chat template expands `MediaKind::Image` in `User`/`Tool` roles to `<|content_image|>` + `V × <|unused_200054|>` + `<|end_message|>` (`200005`, `200054`, `200010`); video and `--vision-cpu` are rejected.
- **Vision ops & overlay runtime (`vision.cu`, `vision_overlay.h`, `executor.cpp`, `program.cpp`):**
  - Closed ops: `inkling2_vision_patch_embed`, `inkling2_vision_rope_2d` (interleaved 2D axial RoPE, `theta = 10000.0`), `inkling2_vision_layerscale_add`, `inkling2_vision_merge_2x2`, plus `D = 1024` `layer_norm`, `head_dim = 64` `packed_softmax_attention`, and `Q4`/`Q5`/`W8` ViT linear dispatches.
  - `VisionWeightStream` double-buffers `23` ViT layers (`45.25 MiB` staging aliased with text-step scratch via `LayoutBuilder::scope`) over `device.transfer_stream` while keeping `100%` of vision weights in pinned host RAM (`0 B` permanent VRAM).
  - `PreparedVisionItem` (`token_begin`, `token_end`, `grid_h`, `grid_w`, `content_digest`) prevents cross-image prefix collisions on equal-resolution images and blocks checkpoint splits inside image token spans; `Executor::forward` scatters encoded `[2304, V_tok]` handoff columns into `cur` via `ops::scatter`, and `Program::advance_prefill` releases host `media_payloads` once past `token_end`.
- **Windows launcher.** `scripts/run-ninfer-4060ti-inkling2.bat` enables overlay vision and
  expects the vision-capable conversion at `models/inkling2_mini_q3g128.ninfer` by default.
  Set `NINFER_ARTIFACT` to use a different artifact or `NINFER_VISION=0` for text-only serving.
- **CLI check on RTX 4060 Ti 16 GB.** The image-chart example produced identical 24 greedy token
  IDs with resident and overlay weights at 8,192-token `rk4v4-e8` capacity and a 128-token prefill
  chunk. The loaded device weight arena was 10.4 GiB with overlay and 10.6 GiB with resident weights.
- **Artifact verification.** The base vision conversion passed all 626 exact checks and 128
  quantization checks against the local Inkling-2-Mini checkpoint.
- **Serving check on RTX 4060 Ti 16 GB.** With overlay vision and 131,072-token `rk4v4-e8`
  capacity, OpenAI Chat Completions and Anthropic Messages returned the correct colors for
  solid-color 120×80 image prompts. The base configuration, DSpark at `K=4` with `--lm-head-draft`,
  and DFlash 2 at `K=7` passed.

## 5. Follow-On Roadmap

- **DSpark shortlist.** Calibrate `mtp/draft_head_token_ids` using token frequencies from code and
  text to improve `--lm-head-draft` acceptance.
- **Quantization quality.** Evaluate imatrix/AWQ calibration or `Q4G64_F16S` `ffn/routed_down`.
