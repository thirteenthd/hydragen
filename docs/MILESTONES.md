# Milestones — what every marker means

This program runs on milestone codes: M-C10c, cow-m3b, forkb4, b3a, KD4,
run56, c8. They are referenced across the changelogs and design docs but
were never defined in one place. This page is that place.

Conventions first:

| Token | Meaning |
|---|---|
| `§NNN` | a notarized changelog entry (docs/CHANGES.md) — every claim carries one |
| `runNN` | a labeled benchmark/replay run (they are numbered once and never renumbered, so a `run` cited anywhere is a stable evidence anchor) |
| `trNN` | early transport experiment (pre-milestone era) |
| `KD2/KD3/KD4` | key-dedup design iterations in the disk-tier campaign |
| `c4 / c8` | benchmark concurrency arms (4 and 8 simultaneous sessions) |
| suffix letters (`M-C3b2a`) | sub-arms of one milestone as the design split into experiments |

Status words: **LANDED/SHIP** = in the canonical stack; **CLOSED** = produced
a measured verdict, adopted or not; **GATED OUT / OFF** = measured and
rejected, sometimes kept as opt-in dead code.

## Era A — the substrate ladder (2026-11-07..10)

Goal: get a real MoE model (qwen4_exp) served on one 12 GB Ampere card at
usable speed, torch/SGLang era.

| Marker | Intended outcome | Progression |
|---|---|---|
| M1 | weight export: GGUF → torch-native checkpoints | §20 |
| M2 | real FP8 qwen4_exp served on the SGLang substrate | mini smoke §21, served §22–23 |
| M3 | Triton w8a16 grouped-GEMM MoE kernel | 15x over eager; verdict: staging-bound, GEMM idle §24–25 |
| M4 | expert-affinity placement — offline eval | NO-OP; capacity, not affinity, is the lever §28 |
| M5 | decode-step anatomy + pinned fp8 head | B1 3.5 tok/s §31 |
| M6 | int4 expert wire-diet | GATED OUT at quality gate §35 |
| M7 | true decode concurrency | root cause: a mamba cap silently pinned batch to 1 — concurrency had never been on §37; batching at cap 2–3 §38–39 |
| M8 | prefill physics + shared-prefix attention | RadixAttention 2.03x wall §40 |
| M9 | loader-skip boot utility + IQ blob store | boot 1014 s → 164 s §41; audit/blob store/native grouped kernel §42–44 |
| M10 | donor IQ checkpoint splice | IQ3_XXS experts quality-fatal at context §45–46 |
| M11 | IQ3_S parity tier full native serving | boot FAILED §49 → two root causes fixed (M11N) §50 → CLOSED PASS §51–52 |
| dense-IQ | dense-weight diet search | floor stays Q5_K; canonical wire, 16.1 tps record §59–60 |

## Era B — per-step cost, speculative decode, KV funding (2026-11-10..12)

Goal: close the per-token overhead gap and fund long-context KV on 12 GB.

| Marker | Intended outcome | Progression |
|---|---|---|
| M12a/b/c | EAGLE-MTP stock arm / custom lookup spec / graphed-verify retry | a functional §64, b open §65, c closed under a fork capture-semantics law §70 |
| M19 | graphed speculative-decode revisit | key-split PASS, economics still dead at bs=1 §79b |
| M13-0 | long-context decode wall measurement | CLOSED PASS §66 |
| M13a | fp8_e5m2 KV | NOT quality-safe §67 |
| M13c/d/e | HiCache host-KV tier / tier presets / 128k sizing | promising §68; SHIP DESIGN §69; funds parked sessions §71 |
| M14 | dense IQ diet tiers | IQ4_XS fast but unsafe; Q5_K floor §72; radix preamble already dedups in stock §73 |
| M15 | int8 KV pool | PASS, quality-safe, funds 2x24k simultaneous §74 |
| M16 | HiCache + int8 KV scale-transfer | SHIP DESIGN §75 |
| M17 | int8 GPU pool raise | 3x23.6k simultaneous active §76 |
| M18a/b/c | decode-step island anatomy / chunk-size rungs / H2D meta-arena batching | anatomy §77; HTIER384 adopted §78; new substrate record §79 |
| M20 / M20d-R | positioning benchmark / DFlash2 spec-decode arm | anchor §80; mechanism PASS, economics negative §85 |
| M21 / M22 | user-proposed decode rungs / slot-cache | CN=24 canonical, CN=32 unfunded §82; corruption fix via per-apply sub-batching §86; window enlargement NOT-FEASIBLE on 12 GB §87 |
| M23 | int8-KV vs CN=32 funding face-off | CN=24 stays canonical; int8 KV validated further §88 |
| M24 | borrow native kernels from Strata (A1 GDN/QSA parity, A1b fused GDN decode, A2 expert-affinity extend staging) | A1 PASS §90; A1b mechanism PASS economics NEUTRAL §91; A2 SHIPPED opt-in §92 |
| M25 | Strata C++ execution worker | Phase 1 funding architecture VALIDATED §93; Phase 2 IQ3_XXS parity ADOPTED — the substrate the current product runs on §94 |

## Era C — Strata-fork behavior campaigns (2026-11-12..13)

Goal: find what actually governs concurrency inside the Strata C++ engine.

| Marker | Intended outcome | Progression |
|---|---|---|
| — | concurrency decode anatomy on the C++ worker | `--batch-mtp` law, window-row ceiling §95 |
| — | engine 0.1.59 (v0.2.13) on 12 GB + context-size curve | parity confirmed §96; KV streaming discovered §97 |
| §98–99 | long-context concurrency arm; prefill-isolation sweep | the prefill poison is structural at 131k/cold-22.5k §99 |
| §100 | wide-batch (b4) arm + long-decode ladder + PD-split | b4 CLOSED; PD-split infeasible |
| §101 | `--no-prefill-borrow` | poison immunity REAL, paid in permanent residency (anti-CoW economics) |
| §102 | traffic-learned expert profile re-rank | CLOSED-NEUTRAL: learned top-k ties the shipped profile |

## Era D — the KV copy-on-write campaign, M-C0 → M-C10c (2026-11-13..15)

Goal: kill duplicated KV so 12 GB can hold many deep sessions at once.
This is the chain the shipped product stack stands on.

| Marker | Intended outcome | Progression |
|---|---|---|
| M-C0 | spike: is cross-run KV bit-identical? | NO — so CoW must be share-at-fork, not share-at-restore §103 |
| cow-m1 | page-table indirection kernel-safe? | PASS — CoW needs zero kernel changes §104 |
| M-C1a/a2 | global K/V slot arenas (the indirection substrate) | landed at full parity §106 |
| M-C1b/c | slot refcounts + admission fork | landed at full parity; batch-4 farmed §107 |
| M-C1d | join-at-fetch: dedup duplicate H2D fetches across fork siblings | mechanism race-clean, pinned sharing net-negative §108 (off) |
| M-C1e | per-lineage shared region | race-clean, floor-guarded, farm-negative at expert pressure §109 (off) |
| forkb4 | the batch-4 CoW-fork champion config used as the C1-era measuring baseline | telemetry table §107–108 |
| M-C2 | host-tier parked-session dedup: verify-then-view over the parked cache | LANDED §110; 100k validation: anchors mount at exactly the shared head (66,510 tokens) §111 |
| cow-m2fix | park-COW SIGSEGV fix | anchor re-resolved after admission evictions §118 |
| §115/117 | canonical switched to the CoW build; product default ON | 4 concurrent unique 102k streams measured on 12 GB |
| M-C3 | disk tier under the parked cache: investigate §120, land (M-C3a, cow-m3) §123 | then the long verdict arc: full-copy I/O net-negative at c8 §127; M-C3b1 deduped block files §128 → content-defined chunking DEAD for this workload §129–130 → M-C3b1d chunker fix makes dedup FIRE (0.6 GiB skipped per sibling spill) §132 → KD3 spill-side-only recovers over half the regression §133 |
| M-C3b2a | restored images re-join host-tier CoW (donor-adopted views on restore) | landed §134, adoption DEAD on arrival §136 |
| M-C3b3 | disk-resident lineage: children spill as view segments, restore re-mounts views | design §138; C3b3a spill-side verified (96 view runs/child; 97k-token children spill as 0.1 MiB) §139–141; C3b3b view-mounted adoption on restore §143 |
| floor knob | pin a RAM floor for view donors so restores mount instead of re-copy | run14–20 saga: best C4 470 s, histogram verdict on the binding blocker §144–152 |
| identity dig | why the disk store kept self-destructing | NOT geometry or RAM — a boot-order GC race §151 |
| run28 | first end-to-end cross-boot disk-tier proof | 13x 27.5k-token images restored from a previous boot, 0 geometry diffs §153 |
| M-C5 | two-phase lineage-aware boot rebuild + stable on-disk serials | SHIPPED; cross-restart gate passes §154 |
| M-C6a | view-mounted CoW adoption proven cross-boot (VM PROVED) | run33–37 saga + dashboard visibility fix §155 |
| M-C6b | parent-pin on view-stamping so re-materialization can't drift | run38: parent pinned, restore_qc_fail=0 at 234 restores |
| M-C7 | `write_v2` may only emit a view entry when its parent is disk-published (unpublished spans fall through to content-defined chunks) | verified commit `df09ca77`; ended the refuse+sweep family poisoning seen in run39/41/42 forensics, §156 era |
| M-C8 | boot-rebuild SEGV at true 100k depth | root cause: empty slot vector indexed in read_v2; fix + clean walk (5 files, 7.9 GiB, 0 refused) §156 |
| M-C9 | deep-checkpoint retention + resume: disk image competes with the RAM match and mounts the deepest checkpoint, not the last-saved one | deep warm 100k replays at 2–3 s wall (was ~39 s re-prefill) §157 |
| M-C10 / b / c | RAM-tier park-anchor donor lifetime: donors stop being eviction candidates while an anchored child is resident; stale deque index in the eviction cascade fixed (the run56 SIGSEGV); park holds the anchor through the cache's single-slot guard | run64: 3x4 deep 100k replays under CoW parks, zero anchors lost, zero QC fails §158 |

## Era E — the concurrency pipeline, S0 → dyn-conc P2 (2026-11-18..23)

Goal: raise live concurrency on 12 GB by making batch slots, idle sessions,
and the K/V arenas elastic instead of boot-frozen.

| Marker | Intended outcome | Progression |
|---|---|---|
| S0 | config-only memory-tier levers | CLOSED-NEGATIVE — resident arena beats the mmap file tier when the pack fits RAM §179 |
| S1 | re-cut the fork onto upstream 0.1.60 | LANDED; fork overlay reapplied, battery constants re-notarized §181 |
| S2 | batch slots at capacity | B=4 boots clean on 12 GB; per-stream window decode pays ~35%; prod rule: pair batch>1 with kv-resident or 131k context (ladder record) |
| S2b / S2b-v2 | shared live-KV prefix across slots | mechanism live, RETIRED as a VRAM capacity lever — cannot buy capacity with capacity; M-C1e law holds in the batch era |
| S3 | two-zone VRAM expert arena (pin floor) | VOID + TABLED — the patch guarded a victim loop the bench harness never ran; pinning axes assessed (host-RAM page-locking pays, VRAM victim-locking doesn't) |
| S4 / S4-v2 | idle-session lifecycle: a detached slot hands its arena back, admission re-carves a bigger window | LANDED §184, §200; payoff = admission economics, not decode speed: 115,698/115,705 tokens re-used in 874 ms vs 65 s re-prefill §203 |
| S5a / S5b / S5d | unbuffered parallel park I/O; .bcs block container; per-session expert-hit histogram | LANDED §183, §186, §195 |
| S6 | multi-profile category store | TABLED alongside S3 |
| S7 | VRAM budget trims (idx-pool FP16, kv-resident rungs) | idx-pool F16 +175 expert slots §199; kv4096 fails identical-outputs-at-depth (depth gate) |
| S8 | shared-head fan-out, CoW arm live | fan-out dilution law §201; crash-class forensics §202; prod 24 GB arm NEGATIVE, rolled back §203/§205; CoW arm green on the bench node, warm fan-out escapes dilution §204 |
| S8-v2 | first-fit arena allocator + zombie reclaim, boot sizing clamp, measured detach/re-carve loop | LANDED §206 |
| S8-v3 grids | mech batch×fraction grid; COW_SHARED re-arm under the elastic floor | batch is the survivorship dial, b8_f50 mech shape §214; COW_SHARED stays OFF §215 |
| dyn-conc P2 stage A | VMM arenas, boot floor, mapped-capacity admission | LANDED on arm branch `dynconc-arena` §211 (+ ARENA_CONC demand sizing: +29.7% resident experts) |
| dyn-conc P2 stage B | elastic windows, detach parks chunks in a bank, reservation = spendable claim | LANDED + PROMOTED on the bench mech §212; floor-dial verdict FLOOR=4096, host-RAM cliff removed by soak §213 |
| dyn-conc P2 stage C | the expert cache funds an arena grow above the reservation | implemented, smoke green (first-ever live grows: map 4096→16384+); b8 grid null — §206 clamp + reservation make the fund path unreachable at batch 8, a b4 or above-batch shape is owed §216 |
| M-D2 | trace the b8-elastic battery death family | CLOSED — a silent CoW-view park write: a re-park's rewrite range lands on borrowed view segments in its own reused park image; `has_borrowed` guard + fresh-copy fallback; same-battery re-run: 1 boot / 0 deaths, guard fired once §217 |

Canonical today: M-C11-win2 + S8-v2, with the dyn-conc P2 chain (§211–§216)
on the arm branch `dynconc-arena`, snapshotted onto `m-d-conc` byte-identical
to the ms01 build that measured it; its landing into `main` is owed a ctest +
soak pass on the merged tree; the
mech-arm C1–C8 scaling curve (cold 28.2→98.0, warm C8 142.4 tok/s, §210) is
the concurrency headline on the bench node.

## Decoding the abbreviations you'll still see

- `b2a / b3a / b3b` = M-C3b2a / M-C3b3a / M-C3b3b (restore-rejoin, spill
  lineage, view-mount restore).
- `cow-m1 / cow-m2 / cow-m2fix / cow-m3 / cow-m3b` = branch names for the
  C1 era, M-C2 (+SIGSEGV fix), and the M-C3 disk-tier line.
- `forkb4` = batch-4 CoW-fork champion arm (Era D measuring baseline).
- `CN=24` = the slot-cache depth rung that stayed canonical (M22).
- `HTIER` = pinned host expert hot-tier size in GiB (Era A/B capacity knob).
- `IQ3_XXS / IQ3_S / Q5_K` = quantization classes of the model blobs
  (expert pack and dense wire formats).
- `sync180` = the engine arm tree built from upstream Strata 0.1.60 plus
  the fork overlay (§181 lineage); on the bench node at `~/hydragen-sync180`.
- `S0..S8-v2` = Era E concurrency-ladder markers (Era E table above); the
  concurrency roadmap they come from is the milestone ladder doc.
- `P2 stage A/B/C` = the arena-elasticity stages: VMM arenas + floor (§211),
  elastic windows + bank (§212), cache-funded admission (§216).
- `b4 / b8 / b12` = batch-slot-count arms; `f35 / f50 / f65` = expert-cache
  fraction arms of the mech grid (§214); `sh` = a `STRATA_COW_SHARED` arm
  (OFF per §215).
- `ELASTIC / FLOOR` = `STRATA_COW_ARENA_ELASTIC` / `_FLOOR`, the arena
  elasticity switch and its free-VRAM dial (§211–§213).
- `dynconc-arena` = the bench-node branch carrying §211–§216's engine
  commits; owed a main-tree landing.
