# HydraGen change log



Chronological record of every measurement → decision → commit in the program.

Newest first; section numbers are stable, so §N always refers to the same entry.







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

### run2 result (same suite, warm box, tier now pinned at its 160 GiB budget)

| arm | run1 wall | run2 wall |
|---|---|---|
| natural C4 | 548.7 s | 561.7 s |
| natural C8 | 1113.5 s | 876.7 s |
| harness C4 | 607.0 s | **1091.1 s (1.80x)** |
| harness C8 | 1292.4 s | **2200.9 s (1.70x)** |

The rerun was not noise: with the tier grown to its budget cap (du 160 GiB), the host
sits at 57/62 GiB RAM used and only ~11 GiB buff/cache for pack streaming, arena, and
tier I/O at once; every park now fights a drop-oldest eviction. High-churn harness
traffic decays 1.7-1.8x versus its own earlier run while the fewer-session natural arm
IMPROVES on warm tier hits. Steady-state degradation with tier growth is real and
reproducible; the park-disk cap/pruning lever is promoted from hypothesis to measured
effect on 64 GiB hosts. Idle single-stream decode after the sweep: 47 tok/s - the
5-14 tok/s aggregate figures are contention, not the GPU. No config changes applied;
the cap A/B awaits its green light.
## §162 Cross-node replication: full c4/c8 deep suite on a second 12 GB box (RTX 3060, 64 GB RAM) (2026-11-16)

Same sweep, same scripts (m25_conc8_sweep), client-side timing, run from the the 3080 Ti host against a second machine running the public install (port 8080, no api_key, park-disk v2 tier enabled):
- RTX 3060 12 GB + 64 GB host RAM + same 5900X-class host - literally the README minimum spec.
- All four arms passed needle QC (4/4, 8/8, 4/4, 8/8); zero guard pauses.
- RETENTION BETTER than the the 3080 Ti host: 0 of 20 / 0 of 48 turns lost their parked
  prefix (the 3080 Ti host had 1 each), despite the small 6 GiB RAM cache budget forcing
  87 evictions - the disk tier absorbed everything (83 RAM + 76 disk restores, 0 fails).
- SPEED is the story: walls 2.2/2.2/1.9/2.4x the 3080 Ti (549/1114/607/1292 s vs
  245/496/317/548 s); warm TTFT p50 20.5/35.7 s; warm decode p50 14.1/14.2;
  aggregate decode 5.3/5.6 tok/s. ~1.7x of that is VRAM bandwidth (360 vs 608 GB/s)
  as expected; the extra gap and a 43 s worst-case stall track a degraded PCIe link:
  the GPU negotiates gen1 x16 (gen4 capable) - expert-cache host->VRAM refills are
  link-bound. Flagged to the owner for a physical/BIOS check; re-run after fix pending.
- Park-disk showed 43 pre-existing refuses at connect; zero new refuses during the
  sweep - bookkeeping from an earlier boot, not an active tier defect.
- CORRECTION (§163): the gen1 reading was an idle power-management artifact - under load the GPU negotiates gen4 at 1.9 GHz / 99% util. The speed gap stands; suspect is host-RAM working-set pressure at the 64 GB floor.
- Sweep files: conc8_*.{jsonl,summary.json} under the node's bench dir, sweep
  runner + log on the reference machine.


## §161 Installer model-dir + fetch hand-off; fork identity rename at the installed surface; fresh-clone e2e caught two user-path bugs (2026-11-15)



Installer model-location (20850b05f, 56deca47a):
