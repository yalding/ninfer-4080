# Inkling-2-Mini Model, Artifact, and Phase 1 Reference

This reference records the `inkling2_mini` native target (`inkling2-mini / q3g128`,
`inkling2_loop`) in NInfer: model semantics, artifact and converter contract, runtime and state
ownership, and the verified Phase 1 RTX 4060 Ti 16 GB (`sm_89`) baseline.

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
  - Frontend resources: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`

## 3. Runtime, StateImage, Prefix Reuse, and Frontend

- **Public facade:** `src/targets/inkling2_mini/export/ninfer/targets/inkling2_mini/package.h` uses an opaque PIMPL facade (`detail::LoadPlan`, `detail::LoadedModel`) matching `qwen3_6_27b`.
- **StateImage (`169,689,088` bytes fixed per slot):**
  - `48 × 4` convolution history buffers (`1.83 MiB`)
  - `40` cyclic SWA KV caches (`capacity = 1024`, `D = 128`, `Hkv = 8`, BF16 K + FP16 V = `160 MiB`)
  - `8` Full-Attn layers backed by `PagedKVCache` (`Int8Group64` or `BFloat16`, `2,112 MiB` @ `131,072` tokens in `Int8Group64`)
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

### Phase 1 SIMT Baseline Throughput (`ninfer_bench`)

Conditions: `--max-ctx 131072 --kv-dtype int8 -r 1 --warmup 0`, tiled `131,072`-token corpus. Current Phase 1 kernels use SIMT attention and warp-dot MoE (bulk Tensor Core prefill and MoE are not yet implemented):

| Case | Prefill (tok/s) | Decode (tok/s) | Notes |
|---|---:|---:|---|
| `pp512` | `92.16` | — | |
| `pp2048` | `86.17` | — | |
| `tg128` (`d=0`) | — | `30.98` | CUDA Graph replay |
| `pp4096+tg128` | `82.34` | `20.63` | |
| `pp16384+tg128` | `70.95` | `19.07` | |
| `pp32768+tg128` | `59.21` | `16.59` | `1.33 GiB` free after startup |

*`pp120384+tg128` was intentionally deferred until Tensor Core prefill/MoE optimization replaces the Phase 1 SIMT kernels.*

### Local HTTP Serving Verification

- **Anthropic `/v1/messages` shared prefix & pressure:** `wp6-anth-b` hit `shared_stable_prefix` (`207` cached tokens, answered `Paris`, `end_turn`). Under pressure (`--device-state-slots 1 --host-state-slots 4 --host-kv-mib 32`), `wp6-anth-c` completed capture (evicting `2` private + `1` shared owners / `6` checkpoints) and `wp6-anth-d` hit `shared_stable_prefix` (`156` cached tokens, answered `Tokyo`, `end_turn`).
- **OpenAI `/v1/chat/completions` & `/slots`:** `wp6-private` answered `4` then `7` (`26` cached tokens) with retained checkpoints visible in `/slots`.
- **State transfer accounting:** `state_transfers.{d2d,d2h}.bytes` recorded exact multiples of `169,689,088` B (`169689088` for `count=1`, `339378176` for `count=2`, `509067264` for `count=3`).

## 5. Follow-On Roadmap (Phase 2 / 3)

1. **Tensor Core prefill & MoE:** Replace Phase 1 SIMT prefill attention and warp-dot MoE with tiled Tensor Core FlashAttention prefill and bulk Tensor Core `Q3`/`Q4` MoE kernels.
2. **Quality upgrade option:** Add imatrix/AWQ calibration or `Q4G64_F16S` `ffn/routed_down` if closing the `8.71` vs `8.16` PPL gap is prioritized over `1.28 GiB` 128K slack.
3. **Phase 2a (`RK4V4E8` `D=128` KV):** Halve full-attention KV footprint (`2.06 GiB -> ~1.05 GiB` @ 128K) to unlock 256K+ context on 16 GB.
4. **Phase 2b (DSpark speculator):** Port `mtp.safetensors` (`~450 MiB` quantized) for speculative decode.
5. **Phase 3 (Vision):** Port Inkling-2's 23-layer ViT (`dim=1024`, `patch=20`, `merge=2`, axial RoPE `theta=10000`).
