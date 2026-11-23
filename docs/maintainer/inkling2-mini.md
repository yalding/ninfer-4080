# Inkling-2-Mini Model, Artifact, and Optimization Reference

This reference records the `inkling2_mini` native target (`inkling2-mini / q3g128`,
`inkling2_loop`) in NInfer: model semantics, artifact and converter contract, runtime and state
ownership, and the verified RTX 4060 Ti 16 GB (`sm_89`) Phase 1–4 measurements.

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
  - Optional DSpark `mtp/` tensors (`--mtp`, `51` stored tensors -> `591` total objects, `11,436,293,632` bytes total / `10.63 GiB` device weights): `fuse_weight` (`FP32` `[4, 4]`), `mask_embed` & `final_norm` (`BF16`), `3` draft layers (`Q4G64_F16S` `q`/`k`/`v`/`k_ctx`/`v_ctx`/`output`/`gate_up`/`down` + `BF16` norms), and Markov RNN head (`markov/dflare_norm`, `w1`, `rnn_proj_*`, `hidden_out_*`). Bound with `TensorPlacement::ValidateOnly` when speculative decoding is disabled.
  - Frontend resources: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`

## 3. Runtime, StateImage, Prefix Reuse, and Frontend

- **Public facade:** `src/targets/inkling2_mini/export/ninfer/targets/inkling2_mini/package.h` uses an opaque PIMPL facade (`detail::LoadPlan`, `detail::LoadedModel`) matching `qwen3_6_27b`.
- **StateImage (`169,689,088` bytes fixed per slot):**
  - `48 × 4` convolution history buffers (`1.83 MiB`)
  - `40` cyclic SWA KV caches (`capacity = 1024`, `D = 128`, `Hkv = 8`, BF16 K + FP16 V = `160 MiB`)
  - `8` Full-Attn layers backed by `PagedKVCache` (`RK4V4E8` default in `scripts/run-ninfer-4060ti-inkling2.bat` at `1.06 GiB` @ `131,072` tokens; `Int8Group64` `2.06 GiB`; or `BFloat16`)
  - Optional DSpark state (when `--spec dflash|mtp` is enabled): `9th` `PagedKVCache` Full-Attn layer (`mtp/layers/1`) + `2` cyclic SWA caches (`mtp/layers/{0, 2}`, `1024` window) + `dspark_c_head` and `3`-tap `1.83 MiB` `conv_trace` rollback buffer
- **Prefix reuse & pressure handling:**
  - `Program::inspect_admission` emits exact `PrefixReusePath` (`SharedStablePrefix`, `PrivateEndpoint`, `PrivateTurnClosure`, `PrivateResponseReplay`, `PrivateLongAnchor`) and H2D/D2D transfer estimates (`169,689,088` B / `81` slices per StateImage).
  - `CapturePressurePlanningSession` and `Program::progress_context_transaction` implement single-owner host spill and eviction under pressure.
  - `ResourceManager::observe_transfer` accounts `ContextResourceClass::State` transfers using `work.payload_bytes` (`169,689,088` B per image) and `units` (`1` per image).
- **Frontend & reasoning:**
  - GPT-4o pre-tokenizer with `normalizer: null` and `ignore_merges: true`.
  - `ReasoningEffort` supports `None`, `Minimal` (`0.1`), `Low` (`0.2`), `Medium` (`0.7`), `High` (`0.9`), and `XHigh` (`0.99`).

## 4. Verified Phase 1 Baseline (RTX 4060 Ti 16 GB, `sm_89`)

Measured on NVIDIA GeForce RTX 4060 Ti 16 GB (`sm_89`, Windows WDDM, MSVC 14.44, CUDA 13.5 / driver 13.4, Ninja Release in `build-4060ti-ninja/`):

### Correctness & Regression Gates

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

## 5. Follow-On Roadmap

1. **Quality upgrade option:** Add imatrix/AWQ calibration or `Q4G64_F16S` `ffn/routed_down` if closing the `8.71` vs `8.16` PPL gap is prioritized over `128K` VRAM headroom.
2. **Lightweight DSpark draft LM-head:** Add a pruned/shortlist `--lm-head-draft` for DSpark to eliminate the `K` full-vocab Q6 `[201024, 2304]` `output_head` calls per draft round.
3. **Vision:** Port Inkling-2's 23-layer ViT (`dim=1024`, `patch=20`, `merge=2`, axial RoPE `theta=10000`).
