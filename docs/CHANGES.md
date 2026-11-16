# HydraGen change log



Chronological record of every measurement → decision → commit in the program.

Newest first; section numbers are stable, so §N always refers to the same entry.







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