# HydraGen change log



Chronological record of every measurement → decision → commit in the program.

Newest first; section numbers are stable, so §N always refers to the same entry.







## §165 the levers ship as defaults: tier cap + admit ordering on every serving surface (2026-11-16)

Per §164's measured reading, both levers became defaults on all three serving
surfaces (all overridable by env):

- Reference-host config generator: park-disk budget default 163840 -> 65536 MiB,
  `STRATA_ADMIT_SHORT_FIRST` default 0 -> 8192 (`MAX_WAIT_S` stays 30). Boot
  verified: the live serve process's own `/proc/<pid>/environ` shows
  65536 / 8192 / 30, and `/v1` answers with context 131072.
- Public installers: freshly generated `hydragen-local.json` now carries
  `STRATA_PARK_DISK_MIB=65536` and `STRATA_ADMIT_SHORT_FIRST=8192`; the harmful
  163840 default is gone from the repo. `docs/HOW_IT_WORKS.md` carries the same
  guidance in the stack paragraph.
- Second box: container config env patched and the container restarted - the
  engine child's `/proc/<pid>/environ` shows 65536 / 8192, `/v1/models` back to
  HTTP 200. Its 160 GiB tier is still on disk 15 min after boot (unlike the
  §164 pruning pass); the budget bites as save-side drops bring it down.

`bash -n` and the PowerShell AST parser both clean; the written env block was
key-checked against a rewritten `hydragen-local.json`.

## §168 upstream-delta inventory: every v0.2.13 difference classified; lever knobs found to be upstream features (2026-11-16)

The repo carried the delta as history (vendored base merge, 16 individually
named chain commits, one tip sync) and as narrative (CHANGES), but no
consolidated inventory. Added docs/UPSTREAM_DELTA.md: net-new subsystems
(park-disk tier, parked-session CoW dedup, deep checkpoint resume -
identifier-count evidence: STRATA_PARK_DISK / ParkDisk / STRATA_PARK_COW /
RESUME_DEEPEST / CowPeer all 0 hits in the vendored v0.2.13 tree),
re-engineered internals (slot arenas + slot_base indirection, eviction
handoff, store boot rebuild, dashboard arms), shipped-but-off failed
experiments (join-at-fetch CowPeer registry, per-lineage shared region), and
upstream features this campaign measured and enabled.

Finding worth stating plainly: the §164 lever knobs
(STRATA_ADMIT_SHORT_FIRST / STRATA_ADMIT_MAX_WAIT_S, --prompt-cache-tail,
COALESCE_TOKENS, STRATA_CRASH_BT) exist in upstream v0.2.13 (18/18/17/2/14
identifier hits at the vendored base, unchanged across our chain). The
campaign A/B-quantified them, sized them, and ships them on by default; it
did not author them.

Also corrected the README engine line: the M-C3b->M-C10c chain lands as one
vendored-engine sync snapshot (single-parent commit), so granular attribution
covers the early chain and serve/web arms; the per-arm ledger for everything
else lives in this change log.

## §167 README truncation post-mortem: 85 bottom-half lines lost in the §163 edit, restored from git (2026-11-16)

Caught while adding the §166 auto-sizing blurb: the README stood at 39/41
lines, not ~160. The §163 notarization commit (0c5be807b) had accidentally deleted
README lines 38-122 - everything after the first "## The headline" sentence:
the before/after headline table, the "what that means in practice" bullets,
"Why this exists", the docs map, the model-blobs blurb, the fork-coexistence
section, and the [^run10] footnote. The intended §163 README edit was only
the counter bump.

Restored the intact 161-line version from the last clean commit (7beb2eba4, already
scrubbed, no leak exposure), re-applied the later legit edits (counter bumps,
the §166 try-it sentence), and verified the restored sections: table, docs
map, coexistence note, footnote. Counter now 172.

Law adopted: before any notarization commit, check `git diff --stat` - a docs
commit whose README hunk deletes more than the intended edit is malformed,
not notarization.

## §166 installer environment auto-sizing: GPU identity, RAM-sized tier budget, disk advisory (2026-11-16)

`install.sh` and `install.ps1` now detect the machine before writing configs:

- **GPU** via `nvidia-smi` (largest GPU by `memory.total`): echoes name / VRAM /
  compute capability; WARNING under 12000 MiB; WARNING for compute capability
  outside 8.x / 9.x / 1x.x (the build targets sm_80+); NOTE when `nvidia-smi`
  is absent - sizing is skipped, install does not fail.
- **Park-tier budget** = min(total RAM, 131072 MiB), floor 16384 - §164's law
  encoded: a tier budget sized to the RAM class, because a pinned larger cap
  measurably decays steady-state throughput. WARNING below 49152 MiB RAM
  (documented floor is 64 GB).
- **Disk advisory**: NOTE when the model drive's free space is under
  budget + 174080 MiB (model blobs ~160 GiB); advisory, not a gate.

The detected budget flows into `hydragen-local.json`. Probes against real
hardware: RTX 3080 Ti host with 62168 MiB RAM -> 62168 budget; RTX 3060 box
with 64204 MiB -> 64204; Windows host, RTX 3090, 130966 MiB -> 130966
(no clamping hit below 128 GiB). Parser checks clean on both script families.

## §164 park-cap + admit-ordering A/B on the reference host: 1.89x self-decay measured, levers recover it (2026-11-16)

The second-box run2 finding was reproduced on the reference host and then A/B'd. Baseline A0:
stock config (160 GiB tier cap, no admit ordering) on a box whose tier had drifted to its
159 GiB pin - harness C4 wall 316.9 s (run10, smaller tier) -> 598.8 s, a 1.89x decay with
the deep-re-prefill count up 1 -> 5 of 20. Same machine, same traffic: tier state is the
variable. Levers A1 (64 GiB cap + STRATA_ADMIT_SHORT_FIRST=8192, MAX_WAIT default 30 s):

- C4 wall 598.8 -> 371.0 s (-38%, recovering run10's 316.9 s); worst inter-token gap
  45.1 -> 6.1 s.
- C8 warm-turn TTFT p50 25.0 -> 13.5 s (halved), p95 254 -> 227 s; C4 p95 282 -> 104 s.
- Retention under the cap: deep re-prefills 3 -> 2 of 48 at C8 (improved), 5 -> 6 at C4
  (noise-level). Needle QC green all arms.
- Cost as designed: cold TTFT p50 at C8 163 -> 211 s - short turns take the control line;
  a long prompt counts as short after 30 s, so nothing starves.
- Boot with the lower cap performs the pruning pass automatically (tier 159 -> 64 GiB);
  RAM 57/60 GiB used through the sweep, guard pauses 0, min memavail 1.5 GiB.

Attribution: cap drives the stall/wall recovery (page-cache pressure from tier + pack
streams on 60 GiB hosts); admit ordering drives the warm-TTFT tails. Generator now passes
STRATA_PARK_DISK_MIB / STRATA_ADMIT_SHORT_FIRST / STRATA_ADMIT_MAX_WAIT_S through;
canonical default choice pending owner decision. Sweep files: levers/{a0,a1}/ under the
bench dir; runner ms01_levers.sh, log levers_run.log on the reference machine.

## §163 PCIe-flag correction by load test + suite rerun on the second box (2026-11-16)

§162 flagged the 3060 negotiating PCIe gen1 (gen4 capable). Direct diagnostics on the
box settle it: during a 1500-token generation nvidia-smi samples show the link at
**gen4 x16, SM 1920-1957 MHz, 99-100% util** for the whole burst; only the idle
samples read gen1/210 MHz. The engine metric sampler snapshots at idle, hence the
flag. The degraded-link hypothesis is withdrawn; no physical fix was needed.

Revised attribution of the 1.9-2.4x wall gap: ~1.7x VRAM bandwidth (360 vs 608 GB/s),
plus host-RAM working-set pressure at the 64 GB floor - engine RSS + 40 GiB arena +
a 170 GiB park tier competing for one page-cache pool (min_memavail 8.2 GiB through
the sweep; worst 43 s inter-token gap at C8 is consistent with page-cache thrash
under expert-stream re-reads, not link).

Suite rerun (run2, same four arms, warm box this time): measures run-to-run variance
and steady-state tier behavior; results appended to this entry when complete.