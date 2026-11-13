# M23: int8 KV vs CN=32 funding laws (CHANGES §88, CLOSED — CN24 stays canonical)

## Laws
1. On ctx2048/4096-token canonical, the KV pool is CAP-BOUND and TINY (~40 MB total,
   "KV Cache is allocated. #tokens: 4096, K 0.02 GB V 0.02 GB"). int8 KV frees only ~20 MB
   there. int8 KV funding only matters on ctx8/65536-pool (long-session) arms (§76).
2. CN=32 island is a HARD funding wall at mem_fraction 0.95 on 12 GB: dies during graph INIT,
   breakable-graph segment instantiate `CUDA error: out of memory`. The
   `markCaptureEnd ... num_active_captures_ > 0 INTERNAL ASSERT` is a CASCADE of that capture-body
   OOM, not an independent transient. The §82 "assert then clean retry" does not repeat with int8.
3. CN=28 + int8: boots (315 MB free), QC 8/8, fairness warm 33.05/41.57 = no gain vs CN24
   (35.71/43.35) within noise band, but DIES on first 6k-char extend probe (108 MB free).
   Confirms §82 rule: island slot-cache growth must stay under ~1/3 of runtime headroom,
   and boot health is NOT sufficient — extend probes are the real gate.
4. HG_KV_INT8=1 (--kv-cache-dtype fp8_e5m2 + shim) is validated NEUTRAL-TO-POSITIVE on canonical
   short-context decode and QC (QC0 exact, QC8 done, fairness 35.71/43.35 warm). Canonical
   launcher stays fp8 cn24c; cn24i is the int8 opt-in. Use int8 KV where pools are large.
5. CN=32 record decode (50.39/44.53) re-opens only with >=16 GB cards or a funded rebalance;
   the weight-diet route is quality-blocked by the Q5_K dense floor (§72).
