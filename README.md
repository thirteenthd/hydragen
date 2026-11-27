# HydraGen — True Multi-Conversation MoE Concurrency on Consumer 12 GB GPUs

### Massive Mixture-of-Experts fleets with a floor-guarded unified VMM pool, deduplicated active VRAM KV, and a lossless 3-tier `VRAM ↔ RAM ↔ SSD` cache hierarchy — built on Strata `v0.2.21`.

![engine: hydragen v0.0.1 (Strata v0.2.21)](https://img.shields.io/badge/engine-hydragen_v0.0.1_(Strata_v0.2.21)-2f81f7) ![context per chat: 131k tokens](https://img.shields.io/badge/context_per_chat-131k_tokens-2f81f7) ![fan-out decode: 164.6 tok/s (2.3x solo)](https://img.shields.io/badge/fan--out_decode-164.6_tok%2Fs_(2.3x_solo)-2f81f7) ![agentic isolation: 0/300 bleed · 0 503s](https://img.shields.io/badge/agentic_isolation-0%2F300_bleed_%C2%B7_0_503s-2ea043) ![3-tier KV: VRAM ↔ RAM ↔ SSD](https://img.shields.io/badge/3--tier_KV-VRAM_%E2%86%94_RAM_%E2%86%94_SSD-2f81f7) [![notarized experiments: 244](https://img.shields.io/badge/notarized_experiments-244_incl._failures-6e7681)](docs/CHANGES.md)

Every number is measured on one **12 GB RTX 3080 Ti** serving a **~40 GB MoE model (`Qwen3.8-Flash-Next IQ3_XXS`)** and quality-gated before it counts. No multi-GPU cluster. No datacenter card. No answer-quality compromise. On 12 GB cards, `v0.0.1` ships with **MTP off by default** to maximize GPU `ExpertCache` residency (`3,699` slots).

---

## Headline Differentiators (On Top of Strata `v0.2.21`)

[Strata](https://github.com/Niko1221/Strata) solves the **model weight** problem by streaming MoE experts from disk/RAM into VRAM. **HydraGen solves the concurrency and long-context conversation memory problem** that breaks multi-session serving on 12 GB GPUs: one 100k-token conversation wants ~2 GB of KV state, and a fleet of parallel agents wants >20 GB on a card whose VRAM is already mostly filled by the model.

HydraGen turns *"doesn't fit"* into true concurrent serving through four tightly integrated systems:

### 1. Unified VMM Pool & Deduplicated Active KV in VRAM (`CowExpertPool` + `CowArena`)
- **Floor-Guarded Unified VMM Pool (`HG_UNIFIED_POOL=1`):** `ExpertCache` and `CowArena` share a single elastic VRAM budget. Under multi-stream load, `CowExpertPool` lends 2 MiB VMM chunks down to an adaptive expert-residency floor, returns freed chunks directly to CUDA with zero chunk hoarding, and dynamically scales prompt prefill chunks (`relend()`) so `ExpertCache` snaps back to full residency (`3,699 / 3,699` slots) when idle.
- **Per-Slot Footprint Compression & Flat Host RSS:** Shared RoPE tables (`share_rope`, **−0.77 GiB**), FP16 indexer pool (`STRATA_IDX_POOL_F16`), and full host-KV release on idle detach (`qsa_state_release`) keep `--batch 8` running with flat host RSS across hundreds of sessions.
- **Admission Fork (`qsa_cow_fork`) & Join-at-Fetch (`COW_JOIN_HH`):** Incoming conversations that share a prefix splice onto resident donor VRAM KV pages via `slot_use` refcounting and strict per-slot CoW lineage isolation (`0 / 300` cross-stream bleed across `3k–40k × C=1..8` stress ladders).
- **Shared-Region Adoption (`STRATA_COW_SHARED` / `HG_COW_SHARED`):** An opt-in arena-tail shared KV pool with owner-tracked lineage hint tables (`sconv`), live-only CAS pinning (`pin_if_live`), and fenced triple revalidation. Proven `8/8` parallel + `8/8` serial under rolling contention, lifting KV streaming VRAM hit from **`80.11%` → `88.30%` (`2048` rolling) → `99.92%` (`8192`)** *(opt-in default `0` on 12 GB to preserve expert-cache slots)*.

### 2. Full 3-Tier KV Cache Park & Restore (`GPU VRAM ↔ System RAM ↔ SSD Storage`)
- **Idle Slot Detach & Park:** Finished/idle batch slots snapshot their state to host RAM and immediately return their VRAM ranges to the unified pool for active conversations.
- **Dynamic LCP Fan-Out Checkpointing (`HG_FANOUT_PREFIX_CKPT`):** Automatically detects shared headers (`≥ 2,048` tokens) across parallel agent branches and checkpoints the divergence boundary on the first pass — cutting fan-out wall time **−18%** and pushing aggregate decode to **`164.6 tok/s` (`2.3×` solo)**.
- **Zero-Copy RAM CoW Dedup (`HG_PARK_COW` + `ParkAnchorGuard`):** Parked conversations share matching prefix KV blocks in host RAM through refcounted segment views (`reused_kv_bytes`) rather than duplicating full snapshots.
- **Self-Maintaining Content-Addressed SSD Tier (`ParkDisk v2/v3` via `HG_PARK_DISK`):** A 16 MiB salted, content-addressed chunk store (`chunks/`) with persistent index rebuild across restarts, lineage parent shielding, **full Windows + Linux parity**, and **automatic housekeeping** (`16 GiB` default budget via `STRATA_PARK_DISK_MIB`, boot-time trim, true LRU promotion on load, superseded-prefix eviction, and oversize-spill guard).
- **Checkpoint-Aware Delta Restore & Deepest Resume (`HG_RESUME_DEEPEST`):** Restoring a parked chat checks both live state and checkpoints (`best_resumable()`), gates synchronous NVMe reads on a `1,024`-token minimum gain, pulls matching 16 MiB salted chunks from live RAM donors when available (`0.1 MiB` read on warm sibling return), and pre-seeds the GPU adaptive expert cache from the conversation's saved routing histogram.

### 3. Rebased on Strata `v0.2.21` + Tuned for Consumer PCIe/VRAM Physics
- **Clean Linear Fork Stack (`0.1.60 → v0.2.21`, 1,748 upstream commits):** Adopted `v0.2.21`'s `engine/` / `terminal/` / `tools/` layout with a 5-layer topical patch stack ([PATCH_STACK.md](docs/PATCH_STACK.md)) so future upstream tags rebase cleanly.
- **16-Row CUDA Decode Windows & Lazy Loading:** Built with `-DSTRATA_WINDOW_MAX_T=16` and `CUDA_MODULE_LOADING=LAZY`, passing the full GPU KV/batch parity suite (`14/14` on RTX 3080 Ti) and full Windows MSVC/CUDA builds cleanly.
- **Batched PCIe Miss-Share Cap (`pcie_batch_frac_cap`, `0.25` default):** Multi-row decode batches funnel every row's missed experts over a single PCIe link. Capping the batched start share at `0.25` (`STRATA_PCIE_FRAC_BATCH_MAX`) instead of solo `0.50–0.55` delivered **+9.8% to +16.2%** throughput across `N=1..8` and fan-out on RTX 3080 Ti with higher VRAM hit%.

### 4. Zero-503 Auto Batch-N Controller & Dynamic Live Monitor
- **Auto Batch-N Concurrency & Clean Queueing:** Separates attached survivorship slots (`--batch M`) from host-RAM- and VRAM-bounded active concurrency (`K_active`) via a damped admission-level `AutoBatchController`. Excess requests queue cleanly on `slot_cv` (**`0` HTTP `503`s**), with opt-in shortest-unread prioritization (`STRATA_ADMIT_SHORT_FIRST`), wait-age anti-starvation, and per-slot memory backoff (`SLOT_MEM_BACKOFF_S=30s`).
- **Opt-In Decode SLA Governor (`HG_DECODE_AUTO=1` / `--min-stream-tps`):** Protects per-stream decode speed with EMA hysteresis, a routed-expert hit-rate physics brake, warmup grace windows, and prefix-affinity cohort packing.
- **Dynamic Live Monitor Dashboard:** Real-time active-slot cards (`Slot 1..N`) with unified phase colors (blue prefill, dark-green restored, light-green decode), `"Restored · RAM"` / `"Restored · Storage"` badges, a Resource Stats configuration strip (context window, decode floor, concurrency cap, KV quant), and real-disk-space ParkDisk utilization bars.

---

## Measured Benchmarks (RTX 3080 Ti 12 GB, ~40 GB `Qwen3.8-Flash-Next IQ3_XXS`)

### `hydragen v0.0.1` Multi-Stream, Isolation & CoW Benchmarks (Arm 16 `LAZY`, `--batch 8`, MTP OFF)

| Benchmark (`modelserver01`, 12 GB) | Baseline / Control | `hydragen v0.0.1` | Gain / Result |
|---|---|---|---|
| **PCIe Batch Cap** (`0.50` → default `0.25`) `N=1/2/4/8` | `64.9 / 65.8 / 64.3 / 59.7` | **`75.4 / 73.8 / 70.6 / 67.2`** *(auto `0.25` `N=8: 69.5`)* | **+9.8% to +16.2%** tok/s |
| **Agentic Fan-Out `share`** (`4` streams, `11.8k` shared header) | `52.11 tok/s` (`46.06s`, `125.9` dec) | **`73.04 tok/s`** (`32.86s`, **`164.6`** dec) | **2.3× solo** (`-29%` wall incl. PCIe cap) |
| **Multi-Stream Agentic Stress** (`3k–40k × C=1..8`, 5 passes) | `5 / 291` bleeds, `51` stalled disk reads | **`0 / 300` bleeds, `0` HTTP `503`s, `1` disk read (`0.76 s`)** | `228 s → 0.76 s` disk wait; flat host RSS |
| **ParkDisk Boot Trim & Housekeeping** (`16 GiB` default cap) | `101 GiB` (`149` `.ses`), `107.9 s` boot | **`15.9 GiB` (`15` `.ses`), `22.0 s` boot** | **+86.7 GiB** reclaimed, **4.9×** faster boot |
| **Rolling Shared KV (`probe222x` `2048` / `8192`)** | `3/4` × 2 corruption (`80.11%` KV hit) | **`8/8` parallel + `8/8` serial** (`88.3%` / **`99.92%`** KV hit) | `0` corruption, `0` OOM |

### Deep Fleet Latency & Prefix Survival (`~100k`-token conversations on 12 GB[^run10])

*Before* = stock RAM cache + full-copy disk tier + shallow resume · *After* = `ParkDisk` + shared-view restores + deep resume + CoW parks (client-side, needle-checked):

| Metric | C=4 Before | C=4 After | C=8 Before | C=8 After |
|---|---|---|---|---|
| Wall time | `501 s` | **`317 s`** | `951 s` | **`548 s`** |
| Warm turn TTFT p50 | `48.0 s` | **`6.9 s`** | `41.0 s` | **`21.2 s`** |
| Warm decode p50 | `8.0 tok/s` | **`24.6 tok/s`** | `13.8 tok/s` | **`17.9 tok/s`** |
| Turns losing parked prefix | `5 of 20` | **`1 of 20`** | `7 of 40` | **`1 of 48`** |
| Tokens wasted re-prefilling | `505k` | **`~180k`** | `712k` | **`~200k`** |

- **1–2 deep agent sessions:** `2–3 s` warm turns, `~60–94 tok/s` per stream.
- **4 deep sessions:** interactive (`~7 s` warm TTFT, `~25 tok/s` vs `~48 s` TTFT before).
- **8 deep sessions:** workable (`~21 s` queue-bound warm TTFT, `~18 tok/s`).
- Append-only history is the contract: editing mid-history forces a full re-prefill.

---

## Where Everything Lives (`v0.2.21` Rebased Layout)

| Path | Purpose |
|---|---|
| `engine/` | Rebased Strata `v0.2.21` (`0.1.67`) engine + HydraGen Unified VMM Pool / CoW / ParkDisk / AutoBatch stack (`STRATA_WINDOW_MAX_T=16`) |
| `terminal/serve/` | OpenAI-compatible server + concurrency scheduler (`server.py`) & live dynamic dashboard (`web/`) |
| `tools/` | Build, packaging (`build_cuda_windows.py` → `hydragen.exe`), and repo checks |
| `configs/` + `bench/` | Canonical serving configs, quality batteries, and `probe222x` / concurrency ladders |
| [docs/PATCH_STACK.md](docs/PATCH_STACK.md) | 5-layer rebase manifest (`control-pre-0221 → Strata v0.2.21`) |
| [docs/UPSTREAM_DELTA.md](docs/UPSTREAM_DELTA.md) | File-by-file delta and `ParkDisk` vs upstream spill comparison |
| [docs/CHANGES.md](docs/CHANGES.md) | Full 244-section engineering ledger (`§1`–`§244`), including negative results |
| [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md), [docs/MODELS.md](docs/MODELS.md) | Layer architecture, env knobs, and one-command model blob setup |

---

## Install & Quick Start

**Requirements:** NVIDIA Ampere GPU with **12 GB+ VRAM** (RTX 3060 Ti / 3080 Ti / 40 / 50 series), **64 GB+ system RAM** (for host-pinned KV streaming & CoW park tier), NVMe SSD, CUDA 12+, and Python 3.10+.

**Linux:**
```bash
git clone https://github.com/thirteenthd/hydragen.git && cd hydragen
./install.sh
CUDA_MODULE_LOADING=LAZY .venv/bin/python terminal/serve/server.py --engine hydragen --config hydragen-local.json
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/thirteenthd/hydragen.git; cd hydragen
.\install.ps1
$env:CUDA_MODULE_LOADING = "LAZY"
.venv\Scripts\python.exe terminal\serve\server.py --engine hydragen --config hydragen-local.json
```

Both installers fetch a SHA-256-verified prebuilt CUDA release binary when compatible or build cleanly from source (`./install.sh --build-from-source` / `.\install.ps1 -BuildFromSource`; default arch `86-real`, override with `HYDRAGEN_CUDA_ARCH`). Point `hydragen-local.json` at your model blobs per [docs/MODELS.md](docs/MODELS.md). Dashboard + OpenAI API open at `http://127.0.0.1:8181/`.

**Living beside upstream Strata:** The engine builds as `hydragen` (`hydragen.exe`), accepts `--engine hydragen` or `--engine strata`, isolates logs/configs/`ParkDisk` inside this repo, and prioritizes `HG_*` environment variables (`HG_UNIFIED_POOL`, `HG_PARK_DISK`, `HG_PARK_COW`, `HG_FANOUT_PREFIX_CKPT`, `HG_COW_SHARED`, ...) before falling back to `STRATA_*`.

---

## Provenance & License

Forked from **[Strata](https://github.com/Niko1221/Strata)** (MIT, Niko1221) — rebased onto **Strata `v0.2.21` (engine `0.1.67`)** as **`hydragen v0.0.1`**. HydraGen-original orchestration and docs are **Apache-2.0** ([LICENSE](LICENSE)); Strata-derived engine code remains **MIT** ([NOTICE](NOTICE)). Every experiment, failure, and parity gate from day one is logged in [docs/CHANGES.md](docs/CHANGES.md) (§1–§244).

[^run10]: [§159](docs/CHANGES.md#159-c4c8-deep-re-bench-on-the-canonical-m-c9m-c10c-stack-run-conc8_run10-2026-11-15) — Nov-15 C4/C8 deep-fleet re-bench; source of the deep-fleet Before/After table. Phases 1–5 rebase, unified pool, isolation fixes, and `hydragen v0.0.1` verification live in `§238`–`§244`.
