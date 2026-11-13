# M18 graph-island staging laws (cs/h2d attack, cap-4 graph arm)

## M21 closure laws (user-proposed rungs, measured)
- _cache_slot split counters: per-miss torch copy_ dispatch ~18us; _blob_host
  preadd ~150-260us on host pinned-tier miss, ~0 on pinned hit.
- cap-4: HG_MOE_CACHE_N=16 per-layer GPU slot cache vs ~25-30 unique experts /
  layer / step => thrash, hit rate ~11%.
- HG_BLOB_FAST=1 (raw cudaMemcpyAsync ctypes replacing copy_): NEUTRAL at
  aggregate fairness. Deep-warm cs floor = async wire bytes (~20ms/step),
  mid-warm spike = preadd file reads. Python dispatch overlaps async wire and
  is NOT the wall; a C-array tracker or DMA stream cannot cut bytes/misses.
- Side-stream overlap h2d-behind-ng is mathematically dead: ng 3.5ms cannot
  hide 17-20ms transfers; hideable window is non-MoE graph remainder (~40ms) +
  prefill only.
- Pitfall: helper injected at module scope inside def _install() nests the
  island def -> island install NameError -> stock fp8 fused-MoE fallback ->
  Ampere fp8e4nv CompilationError at graph capture (symptom unrelated-looking;
  grep island install lines after shimming).
- QC0 direct can slip once (441) then be exact (391): boot-drift class; repeat
  probe before suspecting bit-identical transfer changes.
- Open candidate: CACHE_N 16->32 GPU slot cache (221->442 MB) to cut
  wire/preadd by reuse; GPU headroom ~0.9 GB, KV pool absorbs delta.

## Anatomy oracle
- HG_NATIVE_PROF=1 HG_MOE_PROF=1 HG_STEP_PROF=1 works on the breakable-graph eager island.
- [hg-natprof] lines print per-apply per-layer; group into steps by L=0..47 and sum per step, median across steps.
- Compare only deep-warm fairness (3rd+ run BURST 2); early runs mix cold tier states and cs medians are not comparable across depths.

## Banked laws
- Decode step is HOST-STAGING-bound, not GEMM-bound: ng (grouped GEMM) is ~3-6 ms/step total.
- HTIER pinned tier 256->384: deep-warm cs 116.8 -> 15 ms/step; prevents cs regrowth under sustained mixed traffic; host RAM fine (pinned ~5.3 GB). HTIER 400 banned (host OOM law).
- In-step parallel pread (HG_NATIVE_PARREAD) NEG: threaded submit/pinned-alloc/event overhead >= savings, even gated to >=4 misses. Re-confirms cold-prefetch-negative law.
- h2d per-chunk pageable torch.tensor(...,device) creates cost 29-36 ms/step (5 x 48 layers). Persistent pinned+GPU meta arena (st[ah]/st[ag]) with slice fills + ONE blocking copy = h2d 5-17 ms/step. HG_META_BATCH default 1.
- RACE LAW: pinned meta-arena reuse with non_blocking copy_ races the queued DMA (CPU rewrites arena while GPU reads) -> corrupted expert pointers -> cudaErrorIllegalAddress reported later at an unrelated sync. Use blocking copy_ or buffer rotation; stream-sync before freeing arena on growth.
- copy_ into arena: ag[:o].copy_(ah[:o]) (sized slice), not ag.copy_(ah[:o]) (size mismatch).

## Deep-warm cap-4 burst progression
- eager: 12.4 -> graph: 20.1-20.3 -> +int8 KV parity 20.6 -> +HTIER384: 23.3-25.3 -> +meta-arena: 27.3-35.4 tok/s (record 35.37 lean d5h384c; 34.14 prof).
- apply ~69 ms, cs ~29, h2d ~17, ng ~3.5 at deep warm; cs remains the top cost.

## M22 rung1 slot-cache laws (§82)
- HG_MOE_CACHE_N = per-layer GPU expert slot cache; canonical now 24.
- cap-4 fairness working set ~25-30 unique experts/layer; CACHE_N=32 gives ~45% hit, cs 5.1->1.2, record deep-warm 50.39 tok/s but OOMs the extend pass at mem_fraction 0.95 (+221MB island cache, <1MiB free, 620MB reserved-unallocated).
- expandable_segments:True does not fund it: pool over-reserves 1.43GiB at fraction 0.95, boot OOM; launchers hard-export False at line 6.
- Law: island slot-cache growth must stay under ~1/3 of KV headroom; bigger CACHE_N needs mem_fraction_static or KV-pool retune first.
- M22 rung2 (§83): preset seeding NEUTRAL vs long cold prefill (fan-out saturates tier); flashinfer JIT curand.h landmine; see references/m22-prefill-and-seed-laws.md.
- CN=24 stable: 0.69GB headroom, deep-warm 22-29 with 47.57 spikes; QC0/QC6 gates pass.

## Canonical launcher
- launch_phase2_m11d5h384c.sh = prior canonical (d5 + HG_MOE_HTIER=384 + metabatch shim code, no prof).
- launch_phase2_m11d5cn24c.sh = CURRENT canonical (adds HG_MOE_CACHE_N=24).
- launch_phase2_m11d5h384p.sh adds HG_NATIVE_PROF for anatomy.
