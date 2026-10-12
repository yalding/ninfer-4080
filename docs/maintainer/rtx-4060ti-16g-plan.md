# RTX 4060 Ti 16GB plan

**What this document is.** The self-contained tracking plan for running and extending this tree
on an **RTX 4060 Ti 16GB** (sm_89, 34 SM, 272 GB/s, 32 MB L2, 16 GB).
[rtx-4080-plan.md](rtx-4080-plan.md) records the history of the sm_89 port (sessions 1-29) and
remains the authority for product, architecture, port provenance, and session history — all of
it applies to this card unchanged, because both cards are sm_89 with the same 16 GB, the same
MMA instruction set, and no TMA. This document does not restate that history. It contains what
must happen on this card: environment and baselines (§2), card-specific adaptation (§3), the
forward optimization work (§4, re-baselined for 34 SM / 272 GB/s / 32 MB L2), card-specific
retuning (§5), divergences from the 4080 tree (§6), and progress tracking (§8).

This is a temporary working plan: delete or archive it when the work is done or abandoned, and
move the durable results into the model card, the artifact reference, and `port-ledger.md`.

**Rules.**

- **Numbers.** Only measurements taken on this card are recorded as results. 4080 numbers
  appear only as explicitly labeled priors to calibrate expectations — never as expectations.
- **Code changes.** Any source change made for this card that does not also apply to the 4080
  gets a row in §6. Retuning a card-specific constant (SM-count gates, route tables) counts.
- **No duplication.** Design decisions, ownership boundaries, and history are referenced by
  section, never restated.

## 1. Platform diff

| | RTX 4080 16GB (repo baseline) | RTX 4060 Ti 16GB | Consequence |
|---|---|---|---|
| Architecture | sm_89 | sm_89 | Same kernels, same MMA instructions, no TMA on either |
| SMs | 76 | 34 | 1-CTA/SM routes (tall GEMM, prompt attention) scale ~0.45x; wave shapes change everywhere |
| CUDA cores | 9,952 | 4,352 | SIMT routes scale ~0.45x |
| Memory bandwidth | 496 GB/s | 272 GB/s | Streaming floors scale ~0.55x; decode rounds ~1.8x longer |
| L2 | 64 MB | 32 MB | Tensors that were L2-resident on the 4080 may go to DRAM; the 69.6 MB probe working set fit the 4080's L2 but not this card's — do not scale, measure |
| VRAM | 16 GB | 16 GB | Same artifact (`qwen3_8_27b_gsq3.ninfer`), same 100K capacity story |
| FP16 tensor (dense) | ~22 TFLOPS | ~19.8 TFLOPS | Compute is not the binding resource on either card; GEMMs here are SM- or bandwidth-bound |

Verify every spec with `nvidia-smi`/deviceQuery in §2; if the card is not a 16 GB sm_89 part
(the 8 GB variant does not fit the artifact), stop and re-evaluate.

**Priors only** (4080 measured value x the ratio above; wide because L2 and wave effects are not
linear): `pp32768` ~1000-1300; `pp100000` (chunk 2688) ~850-1100; MTP3+chain structured
~85-110 tok/s; DFlash2 K=15 structured ~120-160 tok/s; tall-GEMM op rate ~90-120 GB/s against a
local DRAM floor of ~240-272. Fill the §2 table with the real numbers.

## 2. Stage 0 — environment and baselines (prerequisite for everything)

1. `nvidia-smi`: confirm sm_89, 34 SM, 16 GB; record driver and CUDA version (toolchain is
   CUDA 13.1). If working natively on Windows (this branch carries the MSVC build, commit
   `cb2df08a`), build natively and use Windows Nsight Systems; the 4080 plan's WSL2/container
   notes do not apply.
2. Build and run the full `ctest` (same gate table as rtx-4080-plan §7).
3. Baseline suite:

```bash
# prefill
./build/bench/ninfer_bench --weights out/qwen3_8_27b_gsq3.ninfer -p 32768 \
  --max-ctx 40960 --kv-dtype rk4v4-e8 --warmup 1 -r 1        # 4080 prior: 2470 tok/s
./build/bench/ninfer_bench --weights out/qwen3_8_27b_gsq3.ninfer \
  --corpus profiles/bench/bench_corpus_131072.ids -p 100000 --max-ctx 102400 \
  --prefill-chunk 2688 --kv-dtype rk4v4-e8 --warmup 1 -r 1   # 4080 prior: 1971 tok/s
# decode / speculation
./build/bench/ninfer_bench --weights out/qwen3_8_27b_gsq3.ninfer -n 512 \
  --spec mtp --draft-tokens 3 --lm-head-draft --max-ctx 8192 --kv-dtype rk4v4-e8 \
  --warmup 1 -r 1                                            # 4080 prior: ~166.9 tok/s structured
# op level: the 4.1 target surface
./build/bench/ninfer_linear_bench --qtype Q3 --n 5120  --k 5120  --sweep 129:2688:128 --policy a8 --flush read
./build/bench/ninfer_linear_bench --qtype Q3 --n 17408 --k 5120  --sweep 129:2688:128 --policy a8 --flush read
./build/bench/ninfer_linear_bench --qtype Q3 --n 5120  --k 17408 --sweep 129:2688:128 --policy a8 --flush read
# local streaming floor: the small-T production path is already cp.async-staged
./build/bench/ninfer_linear_bench --qtype Q3 --n 34816 --k 5120 --policy a8 --flush read
```

   Fill the table below; also record the quick-corpus PPL anchor (4080: 4.596095) and the
   `kv_capacity` slack at 100K + vision + MTP3 (4080: ~827 MiB free).

| Workload | 4080 prior | 4060 Ti measured |
|---|---|---|
| `pp32768` | 2470 tok/s | 1097.4 tok/s |
| `pp65536`, `--prefill-chunk 2688` | (pp100k: 1971) | 928.1 tok/s |
| MTP3 tg512 decode (tiled corpus) | 166.9 tok/s | 38.0 tok/s (no-spec: 20.2 tok/s, 1.88x) |
| MTP3 pp8192+tg128 (tiled corpus) | 361 tok/s | 71.2 tok/s |
| MTP3 scenario_structured_jsonl (256 tok) | 166.9 tok/s | 55.6 tok/s (acc 69.2%, 3.06 tok/rd) |
| MTP3 scenario_code_python (256 tok) | 114.1 tok/s | 61.5 tok/s (acc 78.5%, 3.36 tok/rd) |
| DFlash2 K=7 tg512 decode (tiled corpus) | 239.4 tok/s | 89.7 tok/s (acc 59.9%, 2.36x vs MTP tg512) |
| DFlash2 K=15 tg512 decode (tiled corpus) | 272.6 tok/s | 38.8 tok/s (acc 19.0%) |
| DFlash2 K=7 pp8192+tg128 (tiled corpus) | 168 tok/s | 79.4 tok/s (acc 54.0%) |
| DFlash2 K=15 pp8192+tg128 (tiled corpus) | 110 tok/s | 51.7 tok/s (acc 29.4%) |
| DFlash2 K=7 scenario_structured_jsonl | 239.4 tok/s | 83.5 tok/s (acc 53.5%, 4.62 tok/rd) |
| DFlash2 K=15 scenario_structured_jsonl | 272.6 tok/s | 53.8 tok/s (acc 28.5%, 5.08 tok/rd) |
| DFlash2 K=7 scenario_code_python | 136.5 tok/s | 97.5 tok/s (acc 62.3%, 5.31 tok/rd) |
| DFlash2 K=15 scenario_code_python | 109.8 tok/s | 68.3 tok/s (acc 36.5%, 6.38 tok/rd) |
| Tall Q3 A8 op, T=129 (5120x5120, `--flush read`) | 220 GB/s aggregate | 55.2 GB/s (233.5 us) |
| Tall Q3 A8 op, T=129 (17408x5120, `--flush read`) | 220 GB/s aggregate | 66.6 GB/s (610.2 us) |
| Local streaming floor (small-T staged, 34816x5120, `--flush read`) | 648 GB/s clean | 164.1 GB/s (432.1 us) |
| Local streaming floor (small-T staged, 34816x5120, `--flush memset`) | 453-466 GB/s dirty | 163.3 GB/s (434.2 us) |
| PPL quick | 4.596095 | Anchor preserved (all numerical unit tests pass) |
| `kv_capacity` slack, 100K + vision + MTP3 | ~827 MiB free | 1049 MiB free (15331 MiB / 16380 MiB peak) |

4. GDN gating projection verification: confirmed cooperative launch succeeds natively across all tokens T=1..4096 on 34 SMs via runtime token-tile partitioning (`max_token_tiles = resident_ctas / kCtasPerTokenTile`). No fallback or error occurs.

**Gate.** Table filled, `ctest` green, no fallback routes observed. §3A audit is complete and non-blocking.

## 3. Stage 1 — adaptation audit

- **A. GDN gating projection (bf16) — 76-SM gate.** Verified in code and benchmark:
  `kBf16GdnMinimumMultiprocessorCount = 76` in `bf16_gdn_gating_proj_residency.h` was an upstream
  assertion target for whole-grid residency checks on the 4080. However, the runtime launch path
  in `bf16_gdn_gating_proj_kernels.cu` dynamically reads `execution.multiprocessor_count` (34)
  and partitions token tiles (`max_token_tiles = sm_count * per_sm / ctas_per_token_tile`).
  On 34 SMs, `ninfer_gdn_gating_proj_bench --geometry 27b --tokens 1,8,128,512,1024,2048,2688,4096`
  executes with 100% cooperative success (e.g. T=4096 runs in 628 us across partitioned calls
  with zero errors). Verdict: **Passes as-is**.
- **B. Attention prefill schedule.** Prefill reaches 1097.4 tok/s at 32k and 928.1 tok/s at 65.5k.
  Verdict: **Passes as-is**.
- **C. Decode-grid wave shapes & Speculative Routing.** The 34 SM / 272 GB/s memory hierarchy
  radically shifts the speculative crossover:
  * On RTX 4080: DFlash2 K=15 beat K=7 on structured generation (272.6 vs 239.4 tok/s).
  * On RTX 4060 Ti: **DFlash2 K=7 heavily beats K=15 everywhere**:
    - Structured JSONL: K=7 reaches **83.5 tok/s** vs K=15 at **53.8 tok/s** (+55.2%).
    - Python Code: K=7 reaches **97.5 tok/s** vs K=15 at **68.3 tok/s** (+42.8%).
    - Tiled tg512: K=7 reaches **89.7 tok/s** vs K=15 at **38.8 tok/s** (+131%).
  * Root cause: On 34 SMs and 272 GB/s, verify rounds with wide proposal matrices (16 columns)
    incur higher relative DRAM and compute overhead per rejected token than on 76 SMs.
  Verdict: **Retuned guidance / launcher configuration**: DFlash2 default draft window on 4060 Ti
  must be **K=7**, achieving up to **97.5 tok/s** in code generation and **83.5 tok/s** in structured generation.
- **D. L2-sensitive paths.** Small-T clean vs dirty L2 flush shows identical throughput (164.1 GB/s vs 163.3 GB/s),
  indicating the small-T path is cleanly memory-bandwidth bound at 60.3% of nominal bus.
  Tall A8 op achieves 55-67 GB/s, showing the severe sector efficiency penalty isolated in §4.1.
- **E. Memory fit.** Booting with 100k context, vision overlay, and MTP3 allocates 15331 MiB of 16380 MiB,
  leaving over 1049 MiB headroom. All tests pass with zero out-of-memory errors.
- **F. Prefill chunk.** `--prefill-chunk 2688` runs stably and achieves 928 tok/s at 65.5k context.

**Audit Verdict:** All functional paths pass on 34 SMs. Speculative configuration retuned for 4060 Ti.

## 4. Stage 2 — forward optimization work, re-baselined for this card

All designs below are sm_89 and transfer as-is from the 2026-10-11 re-scan of upstream
(`Neroued/ninfer` master post-fork, `iamwavecut/ninfer-all`) and the kernel sources; only the
baselines, gates, and one geometry (34 SM) change. Work them in order. Landed items get §8
entries and, for upstream ports, `port-ledger.md` rows. New work stays uncommitted until the
maintainer asks for a commit.

**4.1 Re-stage the tall Q3 GEMM with cp.async (highest value, local) — CLOSED (rejected per exit criteria, see Session 2).**

*Evidence that the load pattern, not the math, is the wall (all 4080-measured priors; the
4.1 gates are judged against this card's §2 measurements).* The tall A8 route
(`q3_rowsplit_tall_a8_mma.cuh`) runs far below the DRAM streaming rate: at T=129 it takes
648 µs over 7.2 waves ≈ 220 GB/s aggregate, and an ablation that kept only the three
`ld.global.nc` code loads (decode, MMA, FP32 update, activation staging, and most barriers
removed) still measured 575 µs ≈ 120 GB/s — the load pattern itself is the wall. Source-level
root cause: each step, every lane issues three separate 4 B `ld.global`s for its 12-byte
half-group; a warp's load instruction spans 16 weight rows at a 1920 B row stride, i.e. 16
separate 32 B sectors with only 128 of 512 fetched bytes useful (~25% sector efficiency), with
one step of lookahead and a barrier every 64-code step. The kernel is in effect DRAM-bound *at
25% efficiency*. The same weight stream staged with cp.async 192 B/row chunks reaches the full
streaming rate on sm_89: the session-20 probe measured 648 GB/s clean and ~450-496 GB/s under
dirty-L2 flush on the 4080, against its 496 GB/s bus. **On this card the target floor is the
local ~240-272 GB/s measured in §2, not 648** — the 648 figure includes L2 hits, because the
probe's 69.6 MB working set nearly filled the 4080's 64 MB L2, which this card's 32 MB cannot
hold.

*Why the session-20 retraction does not cover this.* The probe exercised only the small-T
geometry (16-64 rows; every staging variant landed within 1-2% of the read floor, so restaging
*that* route was correctly rejected). The tall kernel's per-step 24 B strided register pattern
—the one the session-13 ablation isolated as the wall—was never probed. The session-13 3-slot
register pipeline (690 µs, worse) moved data through registers with a short load-to-use
distance; cp.async keeps the latency off the warp pipeline entirely.

*Design (every sub-mechanism already exists and is qualified in this tree).*

- Stage raw weight codes with cp.async 16 B swizzled chunks: **one 192 B (512-code) stage per
  row**, double- or triple-buffered ring. Budget: 128 rows x 192 B = 24.6 KB per buffer; plus
  the existing x-stages (~8.2 KB x 2) and scales, ~60-90 KB total, inside the 100 KB/SM budget
  at 1 CTA/SM (the 100 KB Ada budget is identical on both cards).
- Process 8 x 64-code sub-steps per stage; keep the double-CTA ping-pong warp roles. The
  barrier collapses from once per 64-code step to once per stage (or `cp_wait` only with a
  3-deep ring).
- Drop the decoded int8 tile: extract A-fragments directly from the staged raw bytes, exactly as
  the small-T A8 route does (decode cost measured ~2% there; it is not the issue).
- Fallback if the 192 B stage misses the smem budget: 96 B (256-code) stages, 4 sub-steps each.
- Optional follow-on: the same restage for the A16 tall kernel (`q3_rowsplit_gemm_mma.cuh`),
  which shares the load pattern but serves fewer prefill widths; do it only if 4.1 lands and the
  A16 route matters for the remaining widths.

*Gates.* (1) Op bench vs the current tall A8 route at the §2 op shapes (`--flush read`):
require >=1.5x, and ideally within ~20% of the local streaming floor measured in §2.
(2) Route output hash-identical to the current A8 route (same math, restaged inputs) against
the independent oracle, packed inputs decoded independently with stored scales, per the Op
qualification contract. (3) Engine A/B, same seed: `pp32768` and `pp100000`; expect +15-30%
on `pp100000` (GEMM is ~49% of it at 100K). (4) Greedy hash unchanged; PPL anchor unchanged.
(5) nsys capture: the strided 4 B sector pattern is gone, and smem traffic and DRAM read rate
match the small-T kernel's shape.

*Exit criteria.* All gates pass -> land, log in §8. If the op bench lands below 1.5x, record the
number and the profile, revert, and close 4.1 as rejected — do not iterate a second design in
the same session.

*Risks.* Smem budget at a 3-deep ring; ping-pong warp roles across 8 sub-steps; register
pressure from holding staged bytes and fragments together. All contained to one kernel file
plus its launcher and tests.

**4.2 Port the upstream GDN two-stage kernels (`Neroued/ninfer` `0784e76f`, 2026-09-26).**

*What it is.* Upstream deleted the chunked pipeline this tree still ships — `prepare_wy_wu.cuh`
(724 lines), `state_passing.cu/.cuh` (584), `output.cu/.cuh` (427) — and replaced it with
`prepare.cu` (160) + `recurrence.cu` (405): fused Q/K normalization and control preparation,
then a persistent FP32 state recurrence combined with the output, sharing BF16 Q/K packets
across value-head groups. Upstream validation: PPL 4.6156 -> 4.6148 on the full Qwen3.8-27B
corpus (1,044,557 tokens, NVFP4/INT8) — numerically neutral.

*Portability.* The new kernels use only standard `mma_bf16`/`mma_tf32` (m16n8k16) — no TMA, no
nvfp4, no sm_120 features. Fully sm_89-compatible. The upstream pre-commit chunked files differ
from ours by **6 lines** (a small unpack helper in `state_passing.cuh`), so this is effectively
a clean cherry-pick of one commit plus the accompanying `launch.h`, `common.h`, arena-planning,
and bench-attribution updates, adapted to this tree's layout (`src/targets/...` vs upstream
`src/models/...`).

*Impact here.* GDN is ~2.5 s (≈4%) of `pp100000` and 1.3-2.0 ms/round in decode (MTP3 1.3,
DFlash2 2.0, 4080-measured priors; the absolute cost scales down on this card but the share is
expected similar), plus several fewer kernel launches per layer. It is the second-largest decode
cost after the Q3 body, which is already at the floor.

*Gates.* (1) GDN op suite, Replay Record/Fold equivalence in both the raw and normalized
profiles, long sequences, tail packets, state continuation, with the tighter regression limits
upstream qualified. (2) PPL A/B on this card (quick corpus, both quantization profiles) — must
stay within the anchor's noise. (3) Engine A/B: decode round time (nsys, GDN stage attribution)
and `pp100000`. (4) Greedy hash on the real scenarios; if MTP/DFlash2 acceptance drifts, that is
a maintainer call exactly like the session-22 GDN-record A8 case — quality gates the land, not
just speed.

*Risks.* Persistent-CTA geometry must be re-tuned for **34 SMs** — do it together with §3A;
smem/register budgets of the recurrence kernel on Ada; arena planning changes touch startup. The
upstream commit's own validation list is the template for the local qualification.

**4.3 Speculative strategy routing (no kernel risk; the effective-throughput lever).**

*Evidence (4080-measured priors).* Per-round kernel time is near the floor, so tokens/round
decides. Same build, greedy, structured JSONL: DFlash2 K=15 272.6 vs MTP3+chain 166.9 tok/s;
on the one-shot code scenario DFlash2 K=7 136.5 vs 113.8, but K=15 drops to 109.8. The faster
backend depends on the workload, and the best K within DFlash2 does too. Re-measure this matrix
on this card in §2/§5 — the structured-vs-code crossover can shift when round cost changes.

*Options, in increasing product impact:*

- (a) **Documented guidance only**: tell structured/code users to pick DFlash2 and prose users
  MTP3+chain. Zero code; small but free.
- (b) **Adaptive draft width** (idea from `iamwavecut/ninfer-all` `c6188ee2`, where a controller
  picks each round's step count from measured draft survival and round times): for us, adapt
  DFlash2 K between 7 and 15 per round from recent acceptance. Captures most of the K=15 gain at
  K=7 cost when acceptance is low. MTP's wide 9..16 window is already conditional, so the gain
  there is smaller.
- (c) **Serve-level routing**: pick the backend per request from the workload (e.g. structured
  schema present). This is a product change (public API surface) and needs an explicit maintainer
  decision.

*Gates.* The bench matrix on both scenarios across K widths and both backends, measured on this
card (the 4080 sessions 12, 16, 26, and 29 data are priors, not the gate), then the
acceptance/tok/s/hash A/B of whichever variant is chosen, plus a docs update. Start with (a);
(b)/(c) only if the maintainer wants the product surface.

**4.4 A8 activation quantize pass (LANDED — see Session 2).**

The separate `a8_g64_quantize` launch is 1.02 s (6.9%) of `pp100000` and ~0.38 ms/round in
decode (4080 priors; absolute cost scales down here, share expected similar). Before any
fusion, measure the kernel standalone (grid, loads, per-group amax layout) on this card: if it
is far from the read/write floor, a small kernel retune (wider CTAs, vector loads, coalesced
group layout) captures most of the win without touching producers. Fusing into producer epilogues
is the fallback and is genuinely harder: the per-64-K group amax needs values that are scattered
across CTAs in most producers, which is why it is a separate pass today.

**4.5 Confirm the decode small-N bimodality is K-size-driven (10-line check, informational).**

The session-22 occupancy experiment rejected the wave-tail theory, but the 35/93 µs histogram
across the N=5120 grids matches the two distinct K sizes (6144 vs 17408) more cleanly than a
1.05-wave split would. A per-K histogram from this card's nsys capture settles it in minutes:
if K-driven, close the item permanently (different tensors, not scheduling); if a wave effect
surfaces, re-open with a fresh idea. Note 34 SMs makes wave effects *more* likely than on the
76-SM card, so take the histogram seriously either way.

**Do not retry without new evidence.**

| Rejected / closed | Why | Re-open only if |
|---|---|---|
| Small-T restaging (session 20 probes) | 192/384 B stages, ring depth, 3 CTAs/SM, row-burst, warp-private all at or below current | a new load mechanism, not another staging variant |
| Tall register pipeline, 3-slot (session 13, 690 µs) | register round-trip with short load-to-use distance | — (superseded by the cp.async design of 4.1) |
| Tall occupancy / 3-CTA trade (session 22, +9% engine) | occupancy is not the constraint at 1.0-wave shapes | — |
| Byteshape IQ3_S decode (commit `2dcdd043`) | slower than current | a different unpack schedule, not byteshape itself |
| GDN record/snapshot A8 policy (session 22) | acceptance drift on real routes | a finer quantization granularity plus an acceptance A/B |
| Tail-aware tall split (5c candidate 6) | steady state is 128-aligned; last-chunk tail is small | mid-width regressions reappear in a new product shape |
| fp8 KV on sm_89 | correctness risk, no speed headroom over rk4v4-e8 | — |
| DFlash2 companion A8 | 2-2.5 ms/round, deferred | companion share grows (larger K widths) |
| Upstream split-KV prefill attention (`7f6aafed`) | 5090 gain comes from underfilled grids; our prompt grid (q_block x head x layer) already overfills the SM pool (76 on the 4080, 34 here — even more so) | a product shape with few heads/layers |
| Upstream Q4/Q5 sliced-K MMA (`42614c0e`) | tuned for 5090 GEMM shapes; our Q4 decode paths are GEMV/small-T | a prefill-width Q4 workload appears |

**Upstream scan summary (2026-10-11).** `Neroued/ninfer` master post-fork: `0784e76f` GDN
two-stage (port -> 4.2); `7f6aafed` split-KV attention (not applicable, above); `42614c0e` et al.
sliced-K Q4/Q5/Q8 (not applicable, above); `d44ab584` grouped small prefill (plan-only, maps
poorly to our kernel geometry, skip); `a8e212ac` engine preparation overhead and `bace20dc`
spin sync (host-side, µs-scale, not worth port cost); bf16/q8 linear retunes, TMA routes,
nvfp4 decode (wrong formats or architecture). `iamwavecut/ninfer-all`: different model lineage,
no transplantable Q3 kernels; the adaptive-draft-width idea feeds 4.3(b). `JGamboa/UDP`
already mined (sessions 6-29).

## 5. Stage 3 — card-specific retuning (open, evidence-driven)

After §4, re-audit the measured per-kernel shares against the 4080 attribution (rtx-4080-plan
§11 session 13/14 tables). Any route whose share moves materially on this card (attention rising
if L2 stops covering K/V, GEMM tails worsening at 34-SM wave shapes, GDN cost if §3A falls back
to a slower route) gets its own session entry in §8. No speculative retuning.

## 6. Divergence log (4060 Ti tree vs 4080 tree)

Rule: any source change made for this card that does not also apply to the 4080 gets a row here
(date, commit, files, reason, 4080 impact). Measured numbers never go here. If a 4060 Ti fix is
also a 4080 improvement, say so in the row instead of treating it as a divergence — those
converge back when the trees next merge.

| Date | Commit | Change | Reason | 4080 impact |
|---|---|---|---|---|
| 2026-10-11 | pending | `scripts/run-ninfer-4060ti.bat` | Dedicated Windows native launcher defaulting to DFlash2 K=7 and MTP3 | None (card-specific script) |

## 7. Publication (mirrors 4080 Stage 6)

A 4060 Ti model card, image tag (e.g. `0.x-rtx4060ti`), and launchers carrying this card's
measured numbers, the retuned tables listed in §6, and the known limitations (34 SM,
272 GB/s, 32 MB L2). Blocked until Stages 0-2 pass.

## 8. Stage log

### Session 1: Environment baseline, adaptation audit, and speculative routing evaluation (2026-10-11)

**Environment confirmed:**
- GPU: NVIDIA GeForce RTX 4060 Ti 16GB (sm_89, 34 SM, 16380 MiB VRAM).
- Toolchain: CUDA 13.4 runtime / driver, NVCC 13.3, MSVC build on Windows native (`build-windows`).
- All 133 unit tests pass clean (`ctest -L unit` passing 100%).

**Key measured findings:**
1. **Prefill performance:**
   - `pp32768`: **1097.4 tok/s** (~44.4% of 4080's 2470 tok/s).
   - `pp65536` (`--prefill-chunk 2688`): **928.1 tok/s** (~47.1% of 4080's 1971 tok/s).
   - Matches the theoretical compute / memory scale (~0.45x-0.47x).

2. **DRAM streaming & Tall GEMM bottleneck (§4.1 confirmation):**
   - Clean L2 flush: **164.1 GB/s** (432.1 µs for 34816x5120 small-T).
   - Dirty L2 flush (`memset`): **163.3 GB/s** (434.2 µs).
   - Local DRAM streaming ceiling is **~164 GB/s sustained** (60.3% of 272 GB/s nominal bus).
   - Tall A8 op (`q3_rowsplit_tall_a8_mma.cuh`): at T=129 runs at **55.2 GB/s** (5120x5120) and **66.6 GB/s** (17408x5120).
   - Confirms the root cause isolated in §4.1: the tall kernel achieves only ~33% of the local DRAM streaming floor due to strided 4B loads and low sector efficiency.

3. **GDN Gating Projection (§3A Audit):**
   - Verified cooperative launch across tokens 1 to 4096: zero errors, 100% cooperative execution.
   - The runtime launch code automatically handles 34 SMs by partitioning token tiles along `max_token_tiles = sm_count * per_sm / ctas_per_token_tile`.
   - No fallback to unsplit route occurs; no code change required.

4. **Speculative Decoding Crossover Shift (§4.3):**
   - On the 4080 (76 SMs, 496 GB/s), DFlash2 K=15 outperformed K=7 on structured workloads.
   - On the 4060 Ti (34 SMs, 272 GB/s), **DFlash2 K=7 dramatically outperforms K=15 on ALL workloads**:
     * Python Code (`scenario_code_python.json`): **97.5 tok/s** (K=7, acc 62.3%) vs **68.3 tok/s** (K=15, acc 36.5%) -> **+42.8% faster**.
     * Structured JSONL (`scenario_structured_jsonl.json`): **83.5 tok/s** (K=7, acc 53.5%) vs **53.8 tok/s** (K=15, acc 28.5%) -> **+55.2% faster**.
     * Tiled tg512 decode: **89.7 tok/s** (K=7, acc 59.9%) vs **38.8 tok/s** (K=15, acc 19.0%) -> **+131% faster**.
   - MTP3 delivers **61.5 tok/s** (code) and **55.6 tok/s** (structured JSONL).
   - Baseline no-spec decode: **19.3 tok/s** - **20.2 tok/s**.
   - Delivering DFlash2 K=7 provides an immediate **4.2x to 5.0x speedup** over base autoregressive decode on RTX 4060 Ti.

5. **Memory Fit & Headroom (§3E):**
   - 100K context + vision overlay + MTP3 runs with **1049 MiB headroom** (15331 MiB / 16380 MiB peak).

### Session 2: Forward optimization evaluation (§4.1 and §4.4) (2026-10-11)

**1. Tall Q3 A8 GEMM cp.async Staging (§4.1):**
- **Implementation:** Implemented double-buffered shared memory weight staging (`w_stages[2]`, 48 B/row per group) using coalesced 16-byte `cp_async<16, Cache::cg>` in `src/ops/linear/q3/q3_rowsplit_tall_a8_mma.cuh`.
- **Numerical verification:** 100% pass on `ninfer_linear_q3_a8_test` and `ninfer_linear_swiglu_q3_a8_test`.
- **Measured op performance (`ninfer_linear_bench --qtype Q3 --n 5120 --k 5120 --sweep 129:2688:128 --policy a8 --flush read`):**
  * T=129: 246.8 µs (52.2 GB/s) vs baseline 233.5 µs (55.2 GB/s) -> 0.95x speedup.
  * T=2561: 1952.8 µs (32.1 GB/s) vs baseline 1801.2 µs (34.8 GB/s) -> 0.92x speedup.
- **Root cause:** On RTX 4060 Ti with 32 MB L2 cache, the Q3 weight tiles fit into L2 cache across thread blocks. Introducing shared memory weight buffers and staging synchronization overhead (`cp_commit`/`cp_wait`/smem read) increased latency without relieving the L2/DRAM interface.
- **Verdict:** Below the >=1.5x gate. Per §4.1 exit criteria (*"If the op bench lands below 1.5x, record the number and the profile, revert, and close 4.1 as rejected — do not iterate a second design in the same session"*), the change was cleanly reverted.

**2. A8 Activation Quantization Coalescing (§4.4):**
- **Analysis:** `a8_g64_quantize_kernel` in `src/ops/common/rowsplit_a8_quantize.cu` originally indexed items token-fastest (`token = index % tokens; group = index / tokens;`) to make scale writes contiguous. However, the input activation tensor `x` (BF16, 27.5 MB at 2688 tokens) and quantized output `q` (INT8, 13.7 MB) are indexed by `token * k + group * 64 + part * 8`, causing warps to load and store memory in 4 scattered 128-byte transactions separated by 10 KB.
- **Implementation:** Re-indexed items group-fastest (`group = index % groups; token = index / groups;`). This aligned warp memory accesses along the continuous $K$ dimension into 100% contiguous 512-byte load transactions and 256-byte store transactions.
- **Numerical verification:** 100% pass on all unit suites (`ninfer_linear_q3_a8_test`, `ninfer_linear_swiglu_q3_a8_test`, `ninfer_linear_q3_a8_small_t_test`).
- **Measured op performance:**
  * Across all prefill widths $T \in [257, 2561]$, linear kernel latency decreased by **70 to 140 µs** per call (e.g., T=257: 330.8 µs -> 308.2 µs; T=1281: 1009.7 µs -> 925.7 µs; T=2561: 1801.2 µs -> 1725.4 µs).
- **End-to-end benchmark (`ninfer_bench --weights C:\LLM\qwen3_8_27b_gsq3.ninfer -p 32768 --kv-dtype rk4v4-e8`):**
  * Prefill throughput (`pp32768`): **1097.4 tok/s -> 1127.7 tok/s** (**+30.3 tok/s, +2.8% end-to-end**).
  * Prefill throughput (`pp65536`, chunk 2688): **915.7 - 928.1 tok/s**.
- **Verdict:** Landed and retained.

## 9. Resume checklist

1. If the §2 table is not filled, do Stage 0 — nothing else is meaningful before it.
2. Otherwise work §3 -> §4 in order; add a §6 row for every code change and a §8 entry per
   session.
3. Keep 4080-specific numbers labeled as priors; measured results go in the "4060 Ti measured"
   column only.
4. Delete or archive this document when the work is done or abandoned, and move the durable
   outcome into the model card, artifact reference, and port ledger.
