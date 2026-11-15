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
| VRAM CoW fork (M-C1c)[^m-c1c] | +473 expert slots (1663 → 2136), 4-way same-conversation decode 47–50 tok/s each, wall 23.9s → 18.5s |
| Parked-session dedup (M-C2)[^m-c2] | 100k-token parked image 2.19 GB → 1.35 GB host RAM (−38.5%); a 4×99k sibling fleet that **does not fit** the parked budget without dedup **fits at 97%** with it; restore 0/4 fail |
| Park-disk tier (M-C3a)[^parkdisk] | parked conversations survive RAM-cache eviction: 100k-token image read back in ~1 s vs ~44 s re-prefill; 4-deep harness wall 501→472 s, warm decode 8.0→13.1 tok/s |
| Deep resume + CoW parks (M-C9/M-C10)[^m-c9][^m-c10] | 100k-token harness sessions resume warm turns in 1.7-2.5 s engine-side (100,219 of 102,344 tokens reused); harness C4 wall 501->317 s, C8 951->548 s; disk-tier restores land as shared-view mounts reading 0.1 MiB vs ~2 GiB full copies |
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
The canonical serving stack is wired by the installer into `hydragen-local.json`:
`STRATA_PARK_COW=1` (host-tier parked dedup, CoW parks), a park-disk v2 tier
(`STRATA_PARK_DISK=<dir>` + `STRATA_PARK_DISK_V2=1`) that keeps parked sessions
cross-boot at view-mount cost, and `STRATA_RESUME_DEEPEST=1` + the
`--prompt-cache-tail` arg so a parked session resumes at its own deepest
checkpoint instead of re-reading its tail. `STRATA_COW_JOIN_HH` /
`STRATA_COW_SHARED` are the closed-negative experiments, kept default-off for
roomier substrates. See `docs/laws/`.

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

### Natural traffic (worst case: zero prefix sharing)

| C | wall (s) | cold TTFT p50/p95 (s) | decode p50 (tok/s/stream) |
|---|---|---|---|
| 1 | 50 | 37.0 / 37.0 | 72.1 |
| 2 | 65 | 36.9 / 36.9 | 43.5 |
| 4 | 245 | 168.8 / 225.0 | 15.1 |
| 8 | 496 | 293.5 / 477.3 | 14.4 |

Every ~100k cold prompt costs ~37 s of prefill; above C=2 the arm is
prefill-serialized (8 arms = 802k prefill tokens in 496 s ~= 1.6k tok/s
aggregate), and TTFT p95 is queue depth, not engine speed. No cache enhancement
moves these numbers: zero shared prefix means nothing to reuse. C=1/2 rows from
the first sweep[^sweep1]; C=4/8 from the Nov-15 re-bench[^run10].

### Harness traffic — the same arms, before vs after

Requests are classified from exact reuse counts (`usage`'s
`prompt_tokens_details.cached_tokens`): **cold** (whole prompt prefilled),
**warm** (parked prefix survived; only the new ~2k tokens are prefilled),
**deep_warm_reprefill** (parked prefix lost on every tier; the whole history is
paid again — the eviction tax).

*Nov-14* is the best pre-enhancement stack (RAM parked cache + full-copy disk
tier, no deep resume, no shared-view mounts). *Current* is the canonical stack
above, measured in one fresh-boot run[^run10].

| Metric | C=4 | C=8 |
|---|---|---|
| wall time | 501 s → **317 s** | 951 s → **548 s** |
| warm turn TTFT p50 | 48.0 s → **6.9 s** | 41.0 s → **21.2 s** |
| warm decode p50 | 8.0 → **24.6** tok/s | 13.8 → **17.9** tok/s |
| turns losing their parked prefix | 5 of 20 → **1 of 20** | 7 of 40 → **1 of 48** |
| tokens wasted re-prefilling lost history | 505k → **~180k** | 712k → **~200k** |

Warm TTFT at C=8 is dominated by queue wait: the engine itself goes prompt →
first token in 1.7-2.5 s at 100k depth (`prompt 102344 tokens = 100219 reused +
2125 read in 2474 ms`). Client-side `cached_tokens` matches the engine's reuse
counts, so the two classifications agree. At C=1-2 nothing evicts and nothing
changes: warm turns stay 2-3 s TTFT at 60-94 tok/s decode.

### The physics behind the delta

| Per 100k-token parked image, engine-side | Nov-14 tier | Current tier |
|---|---|---|
| bytes read from NVMe on restore | ~2,083 MiB (full copy) | **0.1 MiB** (shared-view mount) |
| restore time | 1.2-1.6 s | 1.1-3.2 s at 98-106k tokens |
| resume depth after restore | save-time checkpoint (often 0 reused) | **deepest checkpoint ≤ prompt** |
| re-park cost when the fleet churns | full image copy | shared KV, owned tail only |

Four mechanisms, landed as separate milestones in that order:

1. **Idle conversations survive eviction** — the park-disk tier[^parkdisk]
   spills parked KV images to NVMe under RAM pressure instead of dropping
   them; a 100k image reads back in ~1 s instead of costing a ~44 s re-prefill.
2. **Restores cost 0.1 MiB, not 2 GiB** — disk-resident lineage[^lineage]:
   sibling images spill as tiny view descriptors, share one stored parent
   body, and re-enter RAM as shared views; cross-boot durability plus the boot
   fixes that make the store survive restarts[^xdur].
3. **Resume lands at full depth** — deep-checkpoint resume[^m-c9]: a replayed
   session resumes at the deepest stored checkpoint (100,219 of 100,327
   tokens), not the shallow save-time continuity label — a 38.7 s re-prefill
   becomes a 2 s resume.
4. **Re-parks share instead of copy** — the M-C10 CoW-park line[^m-c10] makes
   each re-park pay only its owned tail bytes and pins shared parents against
   eviction while children are live, with zero anchor losses at C=8.

The earlier VRAM-side mechanisms stay as they were: admission fork for
same-conversation batch decode[^m-c1c] and host-tier parked-session
dedup[^m-c2].

### Honest summary on 12 GB + a ~35 B-class routed MoE

- **One to two deep agent sessions at ~100k context: comfortable.** Warm turns
  2-3 s, decode 60-94 tok/s per stream. Append-only history is mandatory: one
  inserted line mid-history breaks exact prefix reuse and you pay the full
  ~40 s.
- **Four deep sessions: interactive.** Warm turns ~7 s TTFT at ~25 tok/s
  decode — the same arm cost ~48 s TTFT and half the decode before the
  tier/resume/CoW stack landed.
- **Eight deep sessions: workable.** Warm turns ~21 s TTFT (queue-bound at
  this depth), ~18 tok/s decode; cold admission in an 8-way burst still costs
  ~100 s p50 — that is prefill physics, not the cache. Treat long tails as
  patient batch.
- Peak single-stream decode (~70-100 tok/s at 100k depth) is unaffected by any
  tier machinery.
- These are measurements on one hardware class (12 GB Ampere, single NVMe) and
  one quant mix, not a product promise: your mileage scales with prefill
  bandwidth and falls off if history is edited instead of appended.

[^sweep1]: [§121](docs/CHANGES.md#121-conc8-deep-sweep-complete-windows-installps1-fully-verified-public-docs-privacy-scrub) — first full sweep; source of the C=1/2 rows.
[^run10]: [§159](docs/CHANGES.md#159-c4c8-deep-re-bench-on-the-canonical-m-c9m-c10c-stack-run-conc8_run10-2026-11-15) — Nov-15 c4/c8 re-bench on the current stack; source of every **bold** number above.
[^parkdisk]: [§123](docs/CHANGES.md#123-m-c3a-landed-park-disk-tier-under-the-parked-cache-cow-m3--c9d6fc44) — tier lands. Content-chunked v2 store: [§128](docs/CHANGES.md#128-m-c3b1-landed-deduped-block-files-for-the-park-disk-tier-strata_park_disk_v2), [§132](docs/CHANGES.md#132-m-c3b1d-dedup-fire-on-real-kv-06-gib-skipped-per-sibling-spill--the-chunker-was-mis-scaled-not-dead).
[^lineage]: [§139](docs/CHANGES.md#139-m-c3b3a-landed-lineage-spill--view-table-entries-kd4---child-files-carry-only-owned-bytes) — lineage spill. [§143](docs/CHANGES.md#143-m-c3b3b-landed-view-mounted-adoption-on-restore-5f1ae5a6---steady-state-ram-for-n-siblings-of-one-head--parent--n-tails) — view-mounted restore. End-to-end proofs: [§153](docs/CHANGES.md#153-run28-proves-the-disk-tier-end-to-end-cross-boot-lineage-restores---13x-275k-token-images-read-01-mib-each-in-260-470-ms-restore_qc_fail0-on-both-ladders-rest-95-108-first-new-restores-since-kd3-landed), [§155](docs/CHANGES.md#155-vm-proved-view-mounted-cow-adoption-works-cross-boot-m-c6a-run33-37-saga--dashboard-regex-desync-fix).
[^xdur]: [§154](docs/CHANGES.md#154-m-c5-two-phase-boot-rebuild-shipped-cow-m3b-7fdb1eff-and-the-cross-restart-gate-passes-the-store-survives-its-own-restart---5657-files-indexed-1-refused-zero-cascade-38-cross-boot-restore-lines-on-boot-b-12-ladder-images-24k-tokens-read-01-mib-each-in-368-435-ms-restore_qc_fail0-both-ladders-battery-green-sescheck-9494) — M-C5 boot rebuild. [§156](docs/CHANGES.md#156-the-boot-segv-saga-v2-default-poisoned-stores-and-m-c8-run42-run53-2026-11-15) — boot SEGV saga and M-C8.
[^m-c9]: [§157](docs/CHANGES.md#157-m-c9-deep-checkpoint-resume--deep-warm-100k-replays-at-2-3-s-run55-run60-2026-11-15) — deep-checkpoint resume; 100k replays at 2-3 s.
[^m-c10]: [§158](docs/CHANGES.md#158-m-c10m-c10bm-c10c-ram-tier-park-anchor-donor-lifetime---run56-crash-closed-cow-parks-run-zero-loss-run61-run64-2026-11-15) — zero-loss CoW parks; run56 crash closed.
[^m-c1c]: [§107](docs/CHANGES.md#107-m-c1bm-c1c-live-slot_use-refcounts--admission-fork-land-at-full-parity--batch-4-farm-sharing-is-real-cow-end-to-end-on-the-canonical-streaming-kv) — admission fork lands at full parity.
[^m-c2]: [§110](docs/CHANGES.md#110-m-c2-host-tier-parked-session-dedup-lands-verify-then-view-on-the-conversationbuffer-segment-directory--sibling-parks-pay-owned-tail-bytes-only-395-mb-vs-577-mb-per-223k-sibling-batteryfarm-green-ladder-restore-06-fail-commit-a53973bf) — host-tier parked-session dedup.

## Provenance & license

The engine is a fork of **[Strata](https://github.com/Niko1221/Strata)** (MIT, Niko1221) —
vendored at v0.2.13 plus our six-commit CoW chain, with upstream commit history
preserved under `engine/`. See `NOTICE`.
HydraGen code is Apache-2.0 (`LICENSE`); Strata-derived engine code remains MIT.
The full 120-section program changelog (`docs/CHANGES.md`) is the notarized record
of every measurement → decision → commit in this campaign.
