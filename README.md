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
| Deep resume + CoW parks (M-C9/M-C10) | 100k-token harness sessions resume warm turns in 1.7-2.5 s engine-side (100,219 of 102,344 tokens reused); harness C4 wall 501->317 s, C8 951->548 s; disk-tier restores land as shared-view mounts reading 0.1 MiB vs ~2 GiB full copies |
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
python3 bench/m25_conc8_sweep.py --arm natural --C 4  # deep-ctx sweep (client-side timing)
python3 bench/m25_conc8_sweep.py --arm harness --C 8   # agent-harness traffic shape
```

## Measured expectations on one 12 GB Ampere card

Deep-context concurrency sweep on the canonical stack (IQ3_XXS qwen4_exp, C++
worker, batch 2, park CoW + park-disk v2 tier + deep-checkpoint resume, 131k
ctx). All timings are **client-side** (TTFT = first streamed token; decode rate
from piece arrival timestamps), and every arm passed a needle-completeness QC
(the model must echo an embedded unique code). Two traffic shapes: *natural*
(C simultaneous unique ~100k-token requests, zero shared prefix) and *harness*
(C sessions x 6 incremental turns over a shared 66k-token head, like an agent
loop). Reproduce with `bench/m25_conc8_sweep.py --arm natural --C 4` and
`--arm harness --C 8`.

Each request is classified from exact reuse counts (`usage`'s
`prompt_tokens_details.cached_tokens`), because aggregate medians hide the
regimes:

- **cold** — turn 1 of a session: the whole ~66-100k prompt is prefilled.
- **warm** — a later turn whose parked prefix survived (RAM cache or disk
  tier): only the new ~2k tokens are prefilled.
- **deep_warm_reprefill** — a later turn that lost its parked prefix on both
  tiers and paid (nearly) the whole history again. This is the eviction tax.

### Natural traffic (worst case: zero prefix sharing)

| C | wall (s) | cold TTFT p50/p95 (s) | decode p50 (tok/s/stream) |
|---|---|---|---|
| 1 | 50 | 37.0 / 37.0 | 72.1 |
| 2 | 65 | 36.9 / 36.9 | 43.5 |
| 4 | 245 | 168.8 / 225.0 | 15.1 |
| 8 | 496 | 293.5 / 477.3 | 14.4 |

Every ~100k cold prompt costs ~37 s of prefill; above C=2 the arm is
prefill-serialized (8 arms = 802k prefill tokens in 496 s ~= 1.6k tok/s
aggregate), and TTFT p95 is queue depth, not engine speed. The C=1/2 rows are
from the Nov-14 run; the cold path is unchanged since.

### Harness traffic, per class

| C | wall (s) | cold TTFT p50 (s) | warm TTFT p50 (s) | warm decode p50 | deep-warm events | tokens wasted on re-prefill |
|---|---|---|---|---|---|---|
| 1 | 130 | 35.9 | 2.1 | 94.0 | 0 | 0 |
| 2 | 148 | 44.4 | 2.6 | 61.6 | 0 | 0 |
| 4 | 317 | 84.9 | 6.9 | 24.6 | 1 of 20 | ~180k |
| 8 | 548 | 104.8 | 21.2 | 17.9 | 1 of 48 | ~200k |

The C=4/8 rows were re-run Nov-15 on the current stack. The same arms on the
pre-Nov-15 stack: C4 {501 s wall, 48.0 s warm TTFT, 5/20 evictions, 505k tokens
wasted}, C8 {951 s, 41.0 s, 7/40, 712k}. Warm TTFT at C8 is dominated by queue
wait — the engine itself resumes a 100k-deep session in 1.7-2.5 s
(`prompt 102344 tokens = 100219 reused + 2125 read in 2474 ms`), and that
reuse shows up in the client's `cached_tokens`, so the two classifications
agree. In the same log window the tier served 66 restores, **every one a
shared-view mount: 0.1 MiB read for a 98-106k-token image in 1.1-3.2 s**, where
the pre-dedup tier copied ~2 GiB per restore.

Honest summary of where this leaves 12 GB + a ~35 B-class routed MoE:

- **One to two deep agent sessions at ~100k context: comfortable.** Warm turns
  2-3 s, decode 60-94 tok/s per stream. Append-only history is mandatory: one
  inserted line mid-history breaks exact prefix reuse and you pay the full
  ~40 s.
- **Four deep sessions: interactive.** Warm turns ~7 s TTFT at ~25 tok/s
  decode — a week ago the same arm cost ~48 s TTFT and half the decode.
- **Eight deep sessions: workable.** Warm turns ~21 s TTFT (queue-bound at
  this depth), ~18 tok/s decode; cold admission in an 8-way burst still costs
  ~100 s p50 — that is prefill physics, not the cache. Treat long tails as
  patient batch.
- Peak single-stream decode (~70-100 tok/s at 100k depth) is unaffected by any
  tier machinery.
- These are measurements on one hardware class (12 GB Ampere, single NVMe) and
  one quant mix, not a product promise: your mileage scales with prefill
  bandwidth and falls off if history is edited instead of appended.

## Provenance & license

The engine is a fork of **[Strata](https://github.com/Niko1221/Strata)** (MIT, Niko1221) —
vendored at v0.2.13 plus our six-commit CoW chain, with upstream commit history
preserved under `engine/`. See `NOTICE`.
HydraGen code is Apache-2.0 (`LICENSE`); Strata-derived engine code remains MIT.
The full 120-section program changelog (`docs/CHANGES.md`) is the notarized record
of every measurement → decision → commit in this campaign.
