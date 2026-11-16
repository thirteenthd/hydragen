# Upstream delta — what HydraGen changed in Strata

The engine in `engine/` is a fork of [Strata](https://github.com/Niko1221/Strata)
(MIT), vendored at its **v0.2.13** release, plus this campaign's chain. This
page classifies every difference against that baseline; the per-arm narrative
with measurements lives in [CHANGES.md](CHANGES.md), and every milestone
marker it mentions is defined in [MILESTONES.md](MILESTONES.md).

## Net-new subsystems (nothing like them in upstream v0.2.13)

- **Park-disk tier** — `ParkDisk`, `STRATA_PARK_DISK*` env family (0 identifier
  hits upstream, 49 now). Evicted conversation images persist across boots:
  v2 writer, per-session store files, budget + keep-free guards, drop-oldest
  pruning, lineage-aware two-phase boot rebuild with revalidation and
  corrupt-file quarantine.
- **Parked-session CoW dedup** — `STRATA_PARK_COW` (0 upstream). At park time,
  a session adopts a donor's shared-prefix K/V segments as verified views
  (verify-then-view, device-resident pools fall back to copy, views bill
  0 bytes); the M-C10 chain keeps RAM-tier view donors pinned while anchored
  children are resident and holds the park anchor through the cache's own
  single-slot guard.
- **Deep checkpoint resume** — `STRATA_RESUME_DEEPEST` (0 upstream). A parked
  session resumes at its own deepest checkpoint instead of re-reading its
  tail, competing with the RAM match at admission (M-C9).

## Re-engineered (the feature exists upstream; the internals were replaced)

- **K/V slot bookkeeping**: per-conversation page ownership → global
  per-layer K/V slot arenas with `slot_base` indirection, per-slot refcounts,
  and clock-sweep holds.
- **Eviction path**: `evict_oldest()` now hands the full victim image to the
  park tier before RAM is freed, and admission with no usable RAM match reads
  the deepest disk prefix back through the normal snapshot-take path.
- **Store boot rebuild**: two-phase, lineage-aware, with a published-parent
  guard (M-C7) and the empty-span SEGV fix (M-C8).
- **serve/web dashboard**: full Concurrency States panel (slot tiles, live
  state, decode/prefill rates, queue), split Conversation-cache (RAM) /
  Park-disk tier cards, and a parser that turns the engine's park-disk log
  lines into `/metrics park_disk` counters.

## Shipped but OFF by default (mechanisms that measured negative, kept for provenance)

- **Join-at-fetch cold-page sharing** (`CowPeer` registry,
  `STRATA_COW_JOIN_HH=0`): race-clean but farm-negative at expert parity —
  cold-head cost is VRAM capacity, not dedup traffic.
- **Per-lineage shared region** (`STRATA_COW_SHARED=0`): floor-guarded region
  sharing, mechanism clean, farm-negative; kept as the M-C1e line.

## Upstream features this campaign measured, enabled, and ships as default

Not our code — worth knowing before crediting or debugging it as ours:

- `STRATA_ADMIT_SHORT_FIRST` / `STRATA_ADMIT_MAX_WAIT_S` (serve-side admit
  ordering) exist in v0.2.13; this campaign A/B-quantified them (warm-turn
  TTFT p50 halved at C8 — CHANGES §164) and the installers ship them on.
- `--prompt-cache-tail`, the prefill coalesce budget (`COALESCE_TOKENS`), and
  crash-backtrace instrumentation (`STRATA_CRASH_BT`) are likewise upstream
  mechanisms the stack wiring turns on.

## Surface-only renames

The installed binary is `hydragen` and serve accepts `--engine hydragen`, so
an upstream Strata install and HydraGen coexist on one machine. Internals —
CMake target, source names, and every upstream-derived `STRATA_*` env var —
deliberately keep upstream identity for mergeability and provenance. The
thirteen fork-introduced env names (park-disk tier family, parked-session CoW,
deep resume, the CoW arena/plumbing experiments) read `HG_*` first — e.g.
`HG_PARK_COW`, `HG_PARK_DISK_MIB`, `HG_RESUME_DEEPEST` — with the legacy
`STRATA_*` spelling honored as a fallback, so a process environment instantly
shows which knobs are ours and which are upstream's.

## Commit granularity in this repo

The early chain (page-table validation probe → arena/refcount plumbing →
parked dedup → park-disk v1 → dashboard arms, 16 commits) is individually
attributed. The deeper chain (M-C3b → M-C10c) lands as one vendored-engine
sync snapshot; every arm in it — including the farm-negative reverts — is
notarized individually in CHANGES.md.
