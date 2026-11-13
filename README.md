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

| traffic | C | requests | total prompt tok | TTFT p50 (s) | TTFT p95 (s) | decode p50 (tok/s/stream) |
|---|---|---|---|---|---|---|
| natural | 1 | 1 | 100k | 37.3 | — | 100.4 |
| natural | 2 | 2 | 200k | 0.3 | 45.1 | 8.6 |
| natural | 4 | 4 | 401k | 45.4 | 91.9 | 8.3 |
| natural | 8 | 8 | 802k | 219.1 | 357.9 | 8.4 |
| harness | 1 | 6 | 607k | 2.5 | 35.8 | **74.0** |
| harness | 2 | 12 | 1.21M | **2.5** | 35.8 | 8.3 |
| harness | 4 | 24 | 2.43M | 41.2 | 95.0 | 1.0 |
| harness | 8 | 48 | 4.86M | 72.6 | 282.7 | 4.1 |

What to expect from 12 GB + a ~35 B-class routed MoE:

- **One deep agent session (harness C1): comfortable.** Warm incremental turns
  at ~100k context start in ~2.5 s (parked-session prefix reuse) and decode at
  70+ tok/s. This is the product sweet spot today.
- **Two concurrent deep sessions: the hard ceiling.** TTFT stays ~2.5 s warm,
  but each stream decodes ~8 tok/s — at two 100k-token contexts the KV window
  and expert cache are several-fold oversubscribed. Fine for patient tools;
  not interactive-chat grade on both streams.
- **C>=4 at 100k context: don't.** TTFT climbs to minutes and decode
  collapses. Keep deep-context fleets at C<=2, or run shorter contexts where
  the published wide-batch ladder applies (`docs/FINAL_REPORT.md`).
- **Cold unique 100k requests are prefill-serialized:** each ~100k prompt
  costs ~37 s of prefill; concurrency past C=2 mostly queues them.

Closing the deep-context concurrency gap is the open M-C campaign (parked-cache
disk tiering + KV-window dedup, tracked in `docs/CHANGES.md`); updated numbers
land as each milestone ships.

## Provenance & license

The engine is a fork of **[Strata](https://github.com/Niko1221/Strata)** (MIT, Niko1221) —
vendored at v0.2.13 plus our six-commit CoW chain, with upstream commit history
preserved under `engine/`. See `NOTICE`.
HydraGen code is Apache-2.0 (`LICENSE`); Strata-derived engine code remains MIT.
The full 120-section program changelog (`docs/CHANGES.md`) is the notarized record
of every measurement → decision → commit in this campaign.
