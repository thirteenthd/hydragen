# HydraGen — high-concurrency MoE inference on 12 GB GPUs

MoE inference on consumer Ampere hardware (RTX 3060 Ti / 3080 Ti, 12 GB VRAM):
file-backed IQ3_XXS routed experts with an adaptive LRU VRAM cache, a static C++
execution worker, copy-on-write KV funding, and host-tier parked-session dedup —
serving a ~125B-A15B-class MoE at **131k context** with true decode concurrency.

Demonstrated on one RTX 3080 Ti (12 GB) + 60 GB host RAM, no multi-GPU, no datacenter card.

## What's in this repo

| Path | Contents |
|---|---|
| `engine/` | our fork of [Strata](https://github.com/Niko1221/Strata) (MIT) — vendored at v0.2.13 with the six-commit CoW chain applied on top, full commit history inside this repo |
| `configs/` | canonical ctx-131k serving config + launcher |
| `bench/` | the gate suite used for every landing: battery (quality oracle), farm (concurrency ladder), park-ladders (22k and 100k depth), CoW arm runner |
| `docs/` | final benchmark report, the 120-section program changelog, engineering laws distilled from the campaign |

## Headline results (all measured, all quality-gated)

| Result | Numbers |
|---|---|
| 128k-context serving on 12 GB | ctx 131,072, KV streamed to pinned host RAM, quality battery identical to control |
| VRAM CoW fork (M-C1c) | +473 expert slots (1663 → 2136), 4-way same-conversation decode 47–50 tok/s each, wall 23.9s → 18.5s |
| Parked-session dedup (M-C2) | 100k-token parked image 2.19 GB → 1.35 GB host RAM (−38.5%); a 4×99k sibling fleet that **does not fit** the parked budget without dedup **fits at 97%** with it; restore 0/4 fail |
| Park-disk tier (M-C3a) | parked conversations survive RAM-cache eviction: 100k-token image read back in ~1 s vs ~44 s re-prefill; 4-deep harness wall 501→472 s, warm decode 8.0→13.1 tok/s |
| Funding architecture | Strata C++ worker replaces the Python runtime, eliminating allocator static-fraction overhead entirely |

Every change passed: battery quality oracle (QC1–QC8, exact-answer checks), farm
concurrency ladder (prime-list correctness at every level), park/restore
verification (byte-level K/V reuse exact to the byte).

## How it works (one paragraph per layer)

**Weights.** IQ3_XXS routed experts live on disk; an adaptive-LRU VRAM cache pulls
micro-bursts from bounded pinned host RAM. Dense/shared weights are native IQ packs.
Experts are read-only — shared across all sessions by construction.

**Execution.** A static C++ worker (`engine/`) owns the whole decode loop: CUDA-graph
captured steps, MTP coupled-draft speculative decode, batched MTP across streams.
No Python in the decode path.

**Long context.** The authoritative KV lives in pinned device-mapped host memory;
VRAM holds a pure page cache with a per-session page table resolved on-device inside
the captured graphs (`kv_stream`). 131k contexts are fundable this way; the tax is
cold admission, not residency.

**Concurrency funding = deleting duplication.** Sessions fork: complete KV pages are
ref-counted, shared-at-fork (never content-hash — same prompt under different prefill
chunk plans produces different KV bytes, so sharing must be lineage-based, verified).
The clock-sweep eviction guard skips ref-counted pages. Parked sessions adopt a
donor's host K/V buffer segments as verified views — only the tail is copied.

**Knobs.** The VRAM funding mechanisms (global arenas, admission fork with page
ref-counting) are compiled into this engine and always on — that is the product.
`STRATA_PARK_COW=1` turns on host-tier parked dedup — recommended, it's on in our
canonical serving config. `STRATA_COW_JOIN_HH` / `STRATA_COW_SHARED` are the
closed-negative experiments, kept default-off for roomier substrates. See `docs/laws/`.

## Install

### Linux

```bash
git clone https://github.com/thirteenthd/hydragen.git
cd hydragen
./install.sh    # needs NVIDIA driver + CUDA toolkit + Python 3.10+
```

`install.sh` finds the CUDA toolkit, provisions `.venv` (cmake + ninja come from pip),
compiles the vendored engine with the upstream `cuda-compile` release flags, and writes
`hydragen-local.json` from the canonical config. The default build arch is sm_86 — the
3060 Ti / 3080 Ti this program was developed on; use e.g.
`HYDRAGEN_CUDA_ARCH=89-real ./install.sh` for RTX 40-series. A source build is the only
option: prebuilt Strata release binaries do not contain the CoW chain. Verified end-to-end:
`install.sh` from a fresh clone of this repo builds a working engine on a 3080 Ti.

### Windows

```powershell
git clone https://github.com/thirteenthd/hydragen.git
cd hydragen
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Prerequisites: NVIDIA driver, a CUDA toolkit (https://developer.nvidia.com/cuda-downloads),
and an MSVC C++ toolchain — the script tells you exactly what is missing and how to get it
(`winget install Microsoft.VisualStudio.2022.BuildTools` with the VC workload). cmake +
ninja again come from pip; Python 3.10+ required. `install.ps1` mirrors `install.sh`
(same presets, sm_86 default, `HYDRAGEN_CUDA_ARCH` override) and writes
`hydragen-local.json` with Windows paths — edit the model/pack paths before serving.

**Model blobs.** The canonical config points at the quant set behind the published
measurements: IQ3_XXS routed-expert pack, native model + PLE GGUFs (Qwen3.8-Flash-Next,
GSQ-RCO quantization), an MTP draft adapter, and an `expert-profile.bin` routing profile
for the adaptive cache. Point the paths in `hydragen-local.json` at your own set.

**Serve:**

```bash
.venv/bin/python engine/serve/server.py --engine strata --config hydragen-local.json
# dashboard + Monitor concurrency panel: http://127.0.0.1:8181/
```

## Reproducing the measurements

```bash
git clone https://github.com/thirteenthd/hydragen.git
# build engine/ (CUDA, Ampere), place IQ3_XXS blobs + MTP drafters per configs/
bash bench/m25iso3.sh canon                # boot canonical
python3 bench/battery_quality_m25.py canon # quality oracle
python3 bench/m25_farm5.py canon           # concurrency ladder
python3 bench/m25_parkladder100k.py 4 canon # park-dedup fleet
python3 bench/m25_conc8_sweep.py --arm natural --C 2  # deep-ctx sweep (client-side timing)
python3 bench/letta_sim_bench.py --C 4 --turns 6      # agent-harness traffic sim
```

## Measured expectations on one 12 GB Ampere card

Deep-context concurrency sweep on the canonical stack (IQ3_XXS qwen4_exp, C++
worker, batch 2, park CoW on, 131k ctx). All timings are **client-side**
(TTFT = first streamed token; decode rate from piece arrival timestamps), and
every arm passed a needle-completeness QC (the model must echo an embedded
unique code). Two traffic shapes: *natural* (C simultaneous unique ~100k-token
requests, zero shared prefix) and *harness* (C sessions x 6 incremental turns
over a shared 66k-token head, like an agent loop). Reproduce with
`bench/m25_conc8_sweep.py`.

Each request is classified from exact reuse counts (`usage`'s
`prompt_tokens_details.cached_tokens`), because aggregate medians hide the
regimes:

- **cold** — turn 1 of a session: the whole ~66-100k prompt is prefilled.
- **warm** — a later turn whose parked prefix survived: only the new
  ~2k tokens are prefilled.
- **deep_warm_reprefill** — a later turn that lost its parked prefix and paid
  (nearly) the whole history again. This is the eviction tax.

### Natural traffic (worst case: zero prefix sharing)

| C | wall (s) | cold TTFT p50/p95 (s) | decode p50 (tok/s/stream) |
|---|---|---|---|
| 1 | 50 | 37.0 / 37.0 | 72.1 |
| 2 | 65 | 36.9 / 36.9 | 43.5 |
| 4 | 233 | 125.2 / 212.7 | 16.8 |
| 8 | 466 | 244.1 / 447.4 | 14.7 |

Every ~100k cold prompt costs ~37 s of prefill; above C=2 the arm is
prefill-serialized (8 arms = 802k prefill tokens in 466 s ~= 1.7k tok/s
aggregate), and TTFT p95 is queue depth, not engine speed.

### Harness traffic, per class (park CoW, RAM parked cache only)

| C | wall (s) | cold TTFT p50 (s) | warm TTFT p50 (s) | warm decode p50 | deep-warm events | tokens wasted on re-prefill |
|---|---|---|---|---|---|---|
| 1 | 130 | 35.9 | 2.1 | 94.0 | 0 | 0 |
| 2 | 148 | 44.4 | 2.6 | 61.6 | 0 | 0 |
| 4 | 501 | 64.9 | 48.0 | 8.0 | 5 of 20 | 505k |
| 8 | 951 | 156.9 | 41.0 | 13.8 | 7 of 40 | 712k |

C=1..2 is the product sweet spot today: warm turns start in ~2-3 s and decode
60-94 tok/s per stream. At C=4/8 the parked RAM cache (8 GiB) can no longer
hold the fleet, eviction turns look like deep-warm re-prefills, and warm decode
collapses.

### Same arms with the park-disk tier on (M-C3a, `STRATA_PARK_DISK`)

| C | wall (s) | warm TTFT p50 (s) | warm decode p50 | deep-warm events | re-prefill tokens |
|---|---|---|---|---|---|
| 4 base -> tier | 501 -> **472** | 48.0 -> **31.5** | 8.0 -> **13.1** | 5 -> 4 | 505k -> 411k |
| 8 base -> tier | 951 -> 1019 | 41.0 -> 43.6 | 13.8 -> **29.1** | 7 -> 21* | 712k -> 716k* |

\*Client-side classification cannot see disk restores: `cached_tokens` stays ~0
when a returning session's image is read back off disk, so those turns are
labeled deep-warm. The engine log shows **6 restores of 98-107k-token images
in 0.8-3.6 s each** inside the C8 window; the genuinely recomputed tokens were
closer to ~116k than 716k. The wall regression at C8 is real however: spill
writes (~15 GiB churn during the arm) contend with expert-blob streaming on
the same nearly-full volume, one cold request starved 437 s server-side, and
the host memory floor dropped to 2.9 GiB.

Honest summary of where this leaves 12 GB + a ~35 B-class routed MoE:

- **One to two deep agent sessions at ~100k context: comfortable.** Warm turns
  2-3 s, decode 60-94 tok/s. Append-only history is mandatory: one inserted
  line mid-history breaks exact prefix reuse and you pay the full ~40 s.
- **Four deep sessions: workable, not interactive.** Warm turns ~30 s TTFT,
  ~13 tok/s decode; the disk tier cuts the eviction tax ~19% and doubles warm
  decode here.
- **Eight deep sessions: capacity physics, tier or no tier.** Prefill demand
  (~1.2M tokens/arm) saturates the card; treat long tails as patient batch.
- Peak single-stream decode (~70-100 tok/s at 100k depth) is unaffected by any
  cache tier; it is VRAM-capped, and closing it is the KV-window campaign.

Open items tracked in `docs/CHANGES.md`: segment-view dedup on disk (M-C3b,
shrinks images ~2x because each parked conversation carries a ~230 MB
constant state floor), tier placement on a quieter volume, and a sweep v4
class that credits disk restores as their own regime.

## Provenance & license

The engine is a fork of **[Strata](https://github.com/Niko1221/Strata)** (MIT, Niko1221) —
vendored at v0.2.13 plus our six-commit CoW chain, with upstream commit history
preserved under `engine/`. See `NOTICE`.
HydraGen code is Apache-2.0 (`LICENSE`); Strata-derived engine code remains MIT.
The full 120-section program changelog (`docs/CHANGES.md`) is the notarized record
of every measurement → decision → commit in this campaign.
