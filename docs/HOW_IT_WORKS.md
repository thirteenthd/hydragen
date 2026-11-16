# How HydraGen works

The README sells outcomes; this is the engineering layer map. All of it lives in
the vendored `engine/` fork of Strata plus the Python serve layer around it.

**Weights.** IQ3_XXS routed experts live on disk; an adaptive-LRU VRAM cache pulls
micro-bursts from bounded pinned host RAM. Dense/shared weights are native IQ packs.
Experts are read-only — shared across all sessions by construction.

**Expert routing learns on your traffic.** The shipped `engine/data/expert-profile.bin`
is only the boot seed; the VRAM tier re-ranks experts from live accesses while serving.
The installers wire `--expert-profile-save` so that learned ranking persists (written
every `expert_profile_save_min` minutes, default 10) and the next boot re-seeds from
it. The save file is keyed by the model blob name (`expert-profile-learned-<hash>.bin`),
so switching model artifacts can never load a foreign ranking — a profile whose
layer/expert counts don't match the model aborts startup by design upstream.

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
`HG_PARK_COW=1` (host-tier parked dedup, CoW parks), a park-disk v2 tier
(`HG_PARK_DISK=<dir>` + `HG_PARK_DISK_V2=1`, with `HG_PARK_DISK_MIB=65536`
sized for the 64 GB-RAM class — a tier pinned at a larger cap measurably decays
steady-state throughput on such hosts) that keeps parked sessions
cross-boot at view-mount cost, and `HG_RESUME_DEEPEST=1` + the
`--prompt-cache-tail` arg so a parked session resumes at its own deepest
checkpoint instead of re-reading its tail. `STRATA_ADMIT_SHORT_FIRST=8192` gives
short agent turns the prefill lane before big cold prompts (a long prompt that
waited `STRATA_ADMIT_MAX_WAIT_S`, 30 s, counts as short — nothing starves).
`HG_COW_JOIN_HH` /
`HG_COW_SHARED` are the closed-negative experiments, kept default-off for
roomier substrates. Naming rule for knobs: fork-introduced names read `HG_*`
first with the legacy `STRATA_*` spelling as fallback; upstream-derived knobs
(`STRATA_ADMIT_*`, the coalesce family, `STRATA_PARK_TRACE`/`_ASYNC`/
`_VRAM_MIB`, `STRATA_CRASH_BT`) keep plain `STRATA_*` reads.

**Eviction rules.** Donors whose anchored child is still resident are not eviction
candidates (M-C10); the cascade only takes a donor when the departing child was its
last anchored child; an in-flight park holds its anchor through the cache's guard
from mount until put (M-C10c). These rules are what make shared parking safe at
fleet scale — the crash history behind them is notarized in the change log (§158).

See `docs/laws/` for the engineering laws distilled from the campaign, and
`docs/CHANGES.md` for the measurement → decision → commit trail of every mechanism
above.
