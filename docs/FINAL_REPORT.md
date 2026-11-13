# Final Evaluation & Benchmark Report
## High-Concurrency MoE Inference on Constrained NVIDIA Hardware — Hydragen/Strata Program
**Host:** `the reference host` (RTX 3080 Ti, 12 GB VRAM, 60 GB host RAM) · **Model:** Qwen3.8-Flash-Next (MoE, 122B-class routed experts)
**Program window:** 2026-11-04 → 2026-11-12 · **Change log:** `~/hydragen-experimental/CHANGES.md` (115 notarized sections)

---

## 1. Mission

Serve a massive MoE model that does **not** fit in 12 GB, at **real decode concurrency** (4+ simultaneous
streams) with a **64k–128k context** (modern agent-harness floor/recommendation), while keeping routed
experts in a dynamic LRU VRAM cache backed by bounded pinned host RAM — no full-model residency, no
multi-GPU, no data-center hardware.

## 2. Final Serving Architecture (canonical, live)

| Layer | Implementation |
|---|---|
| Front | Strata `serve/server.py` OpenAI-compat, port 8181 |
| Engine | **Strata 0.1.59 C++ static worker** (`strata-host`) — PyTorch/SGLang fully off the serving path |
| Weights | IQ3_XXS routed experts: file-backed + adaptive LRU VRAM expert cache (~2,456 slots / 4 GiB at batch 2) |
| Dense/PLE | native IQ packs (Q5_K dense wire = banked quality floor) |
| Context | **ctx 131,072**; streamed KV: 20,480 cells/layer resident in VRAM, authoritative K/V in pinned host RAM, device-side resolve/evict in CUDA graphs |
| Spec decode | MTP drafter `--spec 4 --coupled-draft --batch-mtp` |
| Parked sessions | host conversation cache (6,144 MiB, 8 slots) with checkpoint restore |
| Quality oracle | `battery_quality_m25.py` 8-QC battery green at **every** landing since §93 |

**Why the substrate changed:** SGLang's Python runtime + PyTorch allocator could not fund CN≥32 graphs on
12 GB (repeated OOM deaths). Adopting Strata's static C++ worker (M25, §93–§95) eliminated the allocator
fog and made every later lever measurable.

## 3. Headline Concurrency — canonical ctx-131k stack

| Rung | Result |
|---|---|
| C1 decode (spec) | **162–172 tok/s** |
| C2 co-resident | ~75–90 tok/s each |
| C4 co-resident | ~80–90 tok/s each (batch-mtp), wall ≈ 20 s for 4×432-tok jobs |
| Cold 22.5k prefill | 2,337–2,400 tok/s; admission doubles decode window latency (checkpoint restore then recovers: `22,535 = 22,528 reused + 7 read`) |
| Expert capacity vs batch | batch 2 → 2,456 slots; batch 4 → 1,663 slots (the VRAM tax) |

The 131k context is fundable **because** KV is streamed: only 1.55 GiB VRAM of K/V per session stays hot;
cold pages read from pinned host RAM at GPU resolve time.

## 4. The Duplication Discovery

With concurrency, the dominant O(sessions) VRAM consumer is **not** experts (already shared read-only) —
it is **private per-session KV windows**. CoW/ref-counted KV converts N private copies into one shared
prefix + N tails. Critical grounding (§103): identical prompts prefilled under different chunk plans are
**not** byte-identical, so dedup must be **share-at-fork lineage**, never content-hash.

## 5. CoW Campaign — three tiers, measured

### 5a. VRAM admission fork (M-C1b/c, §107) — **LANDED, champion**
Global per-layer slot arenas + refcounted page tables; fork at admission costs zero copies.

| arm | expert slots | C1 | C4 co-resident | C4 wall |
|---|---|---|---|---|
| plainb4 (no CoW) | 1,663 | 128.3 | 42.5–45.1 | 23.9 s |
| **forkb4 (CoW)** | **2,136** | **151.2** | **47.1–49.9** | **18.5 s** |
| b4k8 (simulated smaller windows) | 2,109 | 166.6 | 14.6–23.5 (unique) | 52.9 s |

CoW recovers **+473 expert slots** at batch 4 (past the batch-2 window-sacrifice trick) and wins at every
concurrency level. Fork telemetry: 1,233–1,249 pages spliced per admission, zero corruption, battery green.

### 5b. VRAM fetch-time sharing (M-C1d §108, M-C1e §109) — **race-clean, economically dead**
- Join-at-fetch: 8,615 joins/arm, never corrupted output — but every joined page is **un-evictable**
  (PIN LAW: a page table entry naming a slot means live gather can read it), so joins permanently steal
  working set from small windows. Monotonic damage: hh=512 → 19.9 s wall vs 18.5 s champion.
- Per-lineage shared region: 4,237 region fetches, 97.7% of block reads hit VRAM — proving the cold-head
  tax is **resident capacity, not duplicated fetch traffic**. Region slots compete 1:1 with expert slots;
  you cannot buy capacity with capacity. Opt-in, default off.

### 5c. Host-tier parked-session dedup (M-C2, §110) — **LANDED**
Parked conversations are host-RAM images in 16 MiB segments. A park finds the **longest common token
prefix** among parked donors, **verifies the shared range byte-for-byte** (memcmp before mutation;
mismatch → plain copy), then adopts donor segments as **refcounted views** — views bill 0 bytes, the
donor pays once, only the tail is copied. Knob `STRATA_PARK_COW=1` (default OFF).

22.5k-depth gate (parkc4): per-sibling parked image **577.4 → 395.5 MB** (reused 182.0 MB exact),
park wall **164–192 → 58–61 ms**, restore under eviction churn **0/6 fail**, battery+farm green.

100k-depth fleet (4 siblings × 96,998 tokens, ~66.5k shared head, ctx 131k, batch 2):

| metric | COW off (parkc5c) | park CoW (parkc5) |
|---|---|---|
| parked image per sibling | **2.19 GB** full | **1.35 GB owned** (reused 842.8 MB exact: anchor = 66,510-tok head, 66,508 aligned pages × 12,672 B/token) |
| cache math vs 6.4 GiB budget | 4×2.19 = **8.8 GB — cannot fit**, 15 LRU evictions, parked count oscillates 1–2 | 2.19 + 3×1.35 = **6.2 GB — fits at 97%** of budget |
| park wall | 790–810 ms | 294–514 ms (adopt parks ~2× faster) |
| restore after churn | 0/4 fail, ~20 s each (cold-page decode) | 0/4 fail, ~20 s each |
| host MemAvailable floor | 2 GiB | **4 GiB** |

At 100k depth dedup savings grow with head share: 38.5% per sibling here (69% head), exact-to-the-byte
via page-aligned anchors. R-phase continuation parks correctly report no anchor (donor take()'d at
restore) and fall back to full copy — verified, no corruption, 0/4 restore QC fails both arms.

## 6. Quality Regime

- `battery_quality_m25.py`: 8/8 green at every landing; QC6 prime-list directness and QC7 GTIN/FASTA are
  the arm discriminators; QC6 exact `97, 101, 103` held across perm/CoW/park-CoW arms.
- Farm bench (4×22.5k concurrent prime-QC): `qc_primes=True` on all streams in every CoW arm — the
  mechanism gate is telemetry, but the byte-correctness gate is the primes.
- int8 KV banked over fp8_e5m2 (QC6 discriminator); Q5_K dense wire floor (§Phase2).
- Verify-then-view honors §103 by construction: shared bytes are adopted only when provably identical.

## 7. Physics Laws Learned (the durable output)

1. **One-VRAM-budget law:** experts, KV windows, rings, drafters share one budget; every trade *moves*
   poison. Only dedup *deletes* it — but only where it doesn't compete for the scarcest resource.
2. **§103 law:** KV bytes are plan-dependent; dedup must follow fork lineage + verification, not hashes.
3. **FLOOR law:** only complete pages (P = upto/4) may be shared; generated tails diverge.
4. **PIN LAW:** any slot named by a page table is live-readable; evicting it is a corruption race.
   Shared slots inside a small session window are permanent working-set theft.
5. **Capacity-not-traffic law (§109):** at 97.7% VRAM hit rate, duplicated fetch traffic was never the
   cost — resident capacity is; region slots cost expert slots 1:1.
6. **Host-tier law (§110):** host RAM has no PIN-LAW coupling and no expert competition — parked dedup
   pays immediately, 3× faster parks, savings scale with head share.
7. **Device-pointer law:** indexer `pooled` pools are device-resident — CPU memcmp on them SIGSEGVs;
   small anchors hid it (rows = at/idx_block = 0). Gate verification on `!on_device()`.
8. **Runner laws:** kill+launch in script files (pkill self-match), engine telemetry lives in cfg `log`
   file, fuser absent — wait for no process AND no listener.

## 8. Ship State

| What | State |
|---|---|
| Canonical serving (C++ worker, ctx 131k, int8 KV, MTP, LRU experts) | **live on :8181**, battery 8/8 green |
| VRAM CoW fork (`STRATA_COW_ARENA/FORK`) | landed on `cow-m0` branch (commit `93f218da` chain), **opt-in** — recommended next canonical candidate at batch ≥ 4 |
| Join-at-fetch / shared region | opt-in, **default off** (banked race-clean for roomier substrates) |
| Host parked dedup (`STRATA_PARK_COW`) | landed (`a53973bf`), **opt-in** — recommended for parked-fleet / agent-harness workloads |
| SGLang fallback | preserved (`launch_phase2_m11d5cn24c.sh`), prod stacks stopped per user decision |

## 9. Verdict

The original premise — "dedupe experts to get concurrency" — was half right: experts are already shared;
the O(sessions) duplication is KV. On 12 GB, **VRAM-tier dedup pays only at admission time** (forkb4:
+28% expert slots, +10% decode, −23% wall at C4), **fails at fetch/region tier** (capacity physics), and
**pays fully at host tier** (parked dedup: 31.5% bytes at 22.5k, deeper at higher head share).
A 4×100k agent fleet is servable on one 12 GB card: forkb4 funds the live VRAM half, park dedup funds
the parked host half, and the 64k–128k product window holds throughout.
