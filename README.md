# HydraGen — high-concurrency MoE inference on 12 GB GPUs

MoE inference on consumer Ampere hardware (RTX 3060 Ti / 3080 Ti, 12 GB VRAM):
file-backed IQ3_XXS routed experts with an adaptive LRU VRAM cache, a static C++
execution worker, copy-on-write KV funding, and host-tier parked-session dedup —
serving a ~125B-A15B-class MoE at **131k context** with true decode concurrency.

Demonstrated on one RTX 3080 Ti (12 GB) + 60 GB host RAM, no multi-GPU, no datacenter card.

## What's in this repo

| Path | Contents |
|---|---|
| `engine/` | git submodule → [hydragen-engine](https://github.com/thirteenthd/hydragen-engine), our fork of [Strata](https://github.com/Niko1221/Strata) (v0.2.13 + the CoW commit chain, branch `hydragen-cow`) |
| `configs/` | canonical ctx-131k serving config + launcher |
| `bench/` | the gate suite used for every landing: battery (quality oracle), farm (concurrency ladder), park-ladders (22k and 100k depth), CoW arm runner |
| `docs/` | final benchmark report, the 116-section program changelog, engineering laws distilled from the campaign |

## Headline results (all measured, all quality-gated)

| Result | Numbers |
|---|---|
| 128k-context serving on 12 GB | ctx 131,072, KV streamed to pinned host RAM, quality battery identical to control |
| VRAM CoW fork (M-C1c) | +473 expert slots (1663 → 2136), 4-way same-conversation decode 47–50 tok/s each, wall 23.9s → 18.5s |
| Parked-session dedup (M-C2) | 100k-token parked image 2.19 GB → 1.35 GB host RAM (−38.5%); a 4×99k sibling fleet that **does not fit** the parked budget without dedup **fits at 97%** with it; restore 0/4 fail |
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

**Knobs** (all default OFF = stock behavior): `STRATA_COW_ARENA=1` +
`STRATA_COW_ARENA_SLOTS` + `STRATA_COW_FORK=1` (VRAM fork),
`STRATA_PARK_COW=1` (park dedup), `STRATA_COW_JOIN_HH` / `STRATA_COW_SHARED`
(opt-in experiments, kept for roomier substrates). See `docs/laws/`.

## Reproducing

```bash
git clone --recurse-submodules https://github.com/thirteenthd/hydragen.git
# build engine/ (CUDA, Ampere), place IQ3_XXS blobs + MTP drafters per configs/
bash bench/m25iso3.sh canon          # boot canonical
python3 bench/battery_quality_m25.py canon   # quality oracle
python3 bench/m25_farm5.py canon     # concurrency ladder
python3 bench/m25_parkladder100k.py 4 canon  # park-dedup fleet
```

## Provenance & license

The engine is a fork of **[Strata](https://github.com/Niko1221/Strata)** (MIT, Niko1221) —
v0.2.13 plus our six-commit CoW chain, applied cleanly upstream. See `NOTICE`.
HydraGen code is Apache-2.0 (`LICENSE`); Strata-derived engine code remains MIT.
The full 116-section program changelog (`docs/CHANGES.md`) is the notarized record
of every measurement → decision → commit in this campaign.
