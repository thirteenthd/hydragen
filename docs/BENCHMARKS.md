# Benchmarks — methodology, deep tables, reproduction

The headline before/after comparison lives in the README; this page holds the
methodology, the remaining arms, and the physics that explain the delta.

## Method

- **Client-side timing only.** TTFT = first streamed token as seen by the client;
  decode rate from piece arrival timestamps. Engine-side timers are never quoted —
  they hide queueing and transport latency.
- **Needle-completeness QC on every arm.** Each prompt embeds a unique code the
  model must echo exactly; an arm that answers fast but wrong is an invalid arm.
- **Per-class classification.** Each request is classified from exact reuse counts
  (`usage`'s `prompt_tokens_details.cached_tokens`), because aggregate medians hide
  the regimes:
  - **cold** — turn 1 of a session: the whole ~66-100k prompt is prefilled.
  - **warm** — a later turn whose parked prefix survived (RAM cache or disk
    tier): only the new ~2k tokens are prefilled.
  - **deep_warm_reprefill** — a later turn that lost its parked prefix on every
    tier and paid (nearly) the whole history again — the eviction tax.
- **Two traffic shapes.** *Natural*: C simultaneous unique ~100k-token requests,
  zero shared prefix (worst case for any cache). *Harness*: C sessions x 6
  incremental turns over a shared 66k-token head (agent-loop traffic).

## Reproduce

```bash
bash bench/m25iso3.sh canon                 # boot canonical
python3 bench/battery_quality_m25.py canon  # quality oracle
python3 bench/m25_farm5.py canon            # concurrency ladder
python3 bench/m25_parkladder100k.py 4 canon # park-dedup fleet
python3 bench/m25_conc8_sweep.py --arm natural --C 4   # deep-ctx sweep
python3 bench/m25_conc8_sweep.py --arm harness --C 8   # agent-harness traffic
```

## Milestone results (all quality-gated)

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

## Natural traffic (worst case: zero prefix sharing)

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

## The physics behind the delta

| Per 100k-token parked image, engine-side | Nov-14 tier | Current tier |
|---|---|---|
| bytes read from NVMe on restore | ~2,083 MiB (full copy) | **0.1 MiB** (shared-view mount) |
| host RAM staged per restore | ~2 GiB straight out of MemAvailable | **~0** |
| resume depth after restore | save-time checkpoint (often 0 reused) | **deepest checkpoint ≤ prompt** |
| re-park cost when the fleet churns | full image copy | shared KV, owned tail only |
| **real cost to recover a lost session to first token** | **~25-40 s**: restore 1.2-1.6 s, then 20-40 s re-prefill because the shallow resume threw most of the restored KV away | **~2-5 s**: restore 1.1-3.2 s + ~2 s resume, nothing thrown away |

The raw restore times overlap (1.2-1.6 s vs 1.1-3.2 s), and that column alone
would make the new tier look slower — two artifacts, both worth stating: the
Nov-14 range comes from quiet ladder tests, the Nov-15 range is measured inside
live C8 sweeps where restores and spill writes contend for the same NVMe. What
actually moves wall time is the bottom row: the old restore was a fast read of
a whole image that then got discarded down to a shallow checkpoint while its 2
GiB of staging pushed the host toward eviction cascades — so turns kept losing
their prefixes (7 of 40 at C=8) and each loss cost a full re-prefill. The
current tier reads 20,000x fewer bytes, stages no RAM, and keeps what it
restores, which is why lost-prefix turns went 7/40 → 1/48 and C=8 wall went
951 s → 548 s.

Warm TTFT at C=8 is dominated by queue wait: the engine itself goes prompt →
first token in 1.7-2.5 s at 100k depth (`prompt 102344 tokens = 100219 reused +
2125 read in 2474 ms`). Client-side `cached_tokens` matches the engine's reuse
counts, so the two classifications agree. At C=1-2 nothing evicts and nothing
changes: warm turns stay 2-3 s TTFT at 60-94 tok/s decode.

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

[^sweep1]: [§121](CHANGES.md#121-conc8-deep-sweep-complete-windows-installps1-fully-verified-public-docs-privacy-scrub) — first full sweep; source of the C=1/2 rows.
[^run10]: [§159](CHANGES.md#159-c4c8-deep-re-bench-on-the-canonical-m-c9m-c10c-stack-run-conc8_run10-2026-11-15) — Nov-15 c4/c8 re-bench on the current stack; source of both columns of the README comparison table.
[^parkdisk]: [§123](CHANGES.md#123-m-c3a-landed-park-disk-tier-under-the-parked-cache-cow-m3--c9d6fc44) — tier lands. Content-chunked v2 store: [§128](CHANGES.md#128-m-c3b1-landed-deduped-block-files-for-the-park-disk-tier-strata_park_disk_v2), [§132](CHANGES.md#132-m-c3b1d-dedup-fire-on-real-kv-06-gib-skipped-per-sibling-spill--the-chunker-was-mis-scaled-not-dead).
[^lineage]: [§139](CHANGES.md#139-m-c3b3a-landed-lineage-spill--view-table-entries-kd4---child-files-carry-only-owned-bytes) — lineage spill. [§143](CHANGES.md#143-m-c3b3b-landed-view-mounted-adoption-on-restore-5f1ae5a6---steady-state-ram-for-n-siblings-of-one-head--parent--n-tails) — view-mounted restore. End-to-end proofs: [§153](CHANGES.md#153-run28-proves-the-disk-tier-end-to-end-cross-boot-lineage-restores---13x-275k-token-images-read-01-mib-each-in-260-470-ms-restore_qc_fail0-on-both-ladders-rest-95-108-first-new-restores-since-kd3-landed), [§155](CHANGES.md#155-vm-proved-view-mounted-cow-adoption-works-cross-boot-m-c6a-run33-37-saga--dashboard-regex-desync-fix).
[^xdur]: [§154](CHANGES.md#154-m-c5-two-phase-boot-rebuild-shipped-cow-m3b-7fdb1eff-and-the-cross-restart-gate-passes-the-store-survives-its-own-restart---5657-files-indexed-1-refused-zero-cascade-38-cross-boot-restore-lines-on-boot-b-12-ladder-images-24k-tokens-read-01-mib-each-in-368-435-ms-restore_qc_fail0-both-ladders-battery-green-sescheck-9494) — M-C5 boot rebuild. [§156](CHANGES.md#156-the-boot-segv-saga-v2-default-poisoned-stores-and-m-c8-run42-run53-2026-11-15) — boot SEGV saga and M-C8.
[^m-c9]: [§157](CHANGES.md#157-m-c9-deep-checkpoint-resume--deep-warm-100k-replays-at-2-3-s-run55-run60-2026-11-15) — deep-checkpoint resume; 100k replays at 2-3 s.
[^m-c10]: [§158](CHANGES.md#158-m-c10m-c10bm-c10c-ram-tier-park-anchor-donor-lifetime---run56-crash-closed-cow-parks-run-zero-loss-run61-run64-2026-11-15) — zero-loss CoW parks; run56 crash closed.
[^m-c1c]: [§107](CHANGES.md#107-m-c1bm-c1c-live-slot_use-refcounts--admission-fork-land-at-full-parity--batch-4-farm-sharing-is-real-cow-end-to-end-on-the-canonical-streaming-kv) — admission fork lands at full parity.
[^m-c2]: [§110](CHANGES.md#110-m-c2-host-tier-parked-session-dedup-lands-verify-then-view-on-the-conversationbuffer-segment-directory--sibling-parks-pay-owned-tail-bytes-only-395-mb-vs-577-mb-per-223k-sibling-batteryfarm-green-ladder-restore-06-fail-commit-a53973bf) — host-tier parked-session dedup.
