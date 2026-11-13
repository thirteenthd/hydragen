# Cold-staging concurrency laws (closed NEGATIVE probe, CHANGES §63)

- cap-4 fully-cold concurrency (NOPRE_B4) floor ~0.5 tok/s agg on this host is
  NVMe/page-cache BANDWIDTH-bound, not queue-depth-bound: parallelizing decode-miss
  preads (shared ThreadPool(8), all layer misses fired at once) did NOT move NOPRE
  (0.52 -> 0.47/0.48) and cost warm paths (PRE_B4 8.29 -> 7.0, P1 +16 s) via
  pinned-alloc churn. Reverted.
- Why: 43 GB IQ3_XXS blob store + ~17 GB pinned HTIER(256) + ~48 GB PLE mmap
  oversubscribe 60 GB host RAM -> blob reads always hit disk; ~2.6 GB blob
  traffic per merged cold token (4 streams x 48 layers x 10 experts x 1.37 MB).
- Shared-prefix (warm) concurrency at cap-4: PRE_B4 agg 8.07-8.29. Warm decode
  burst agg 22-33 tok/s on graph arm. Do not chase NOPRE with I/O scheduling
  tricks; the levers are smaller wire (quality-blocked), PLE/blob page-cache
  partitioning, or pinned-tier seeding (tested — below).

## Pinned-tier seeding (HG_TIER_SEED + HG_ROUTE_DUMP on the live shims)
- Recipe: boot once with HG_ROUTE_DUMP=<path> — the live shims append 88B
  records (i32 layer, i32 k, 10 ids i32, 10 weights f32; same grammar as the
  offline trace poc-moe/trace-b4.bin, NOT the engine --dump-routing layout) —
  replay target traffic once, aggregate per-layer top-K (K = HTIER_K) ids into
  JSON {layer: [ids]}, boot with HG_TIER_SEED=<json>. Seeder inserts
  RAREST-first so FIFO eviction later frees cold entries, not hot ones.
- Hot sets are workload-shaped: top-256/layer covers 99.4% of the source
  workload's traffic but only ~55% of a different workload (distinct-set
  overlap ~50%). Build seed lists from traffic representative of production;
  seeding with the benchmark's own trace measures the ceiling, not a
  production claim.
- Banked ceiling: battery-self-seed moved NOPRE_B4 0.47 -> 0.60-0.61 and cut
  fresh-boot P1 ~8%; PRE_B4/warm decode unchanged. Partial because tier cap
  256/layer < merged 4-stream distinct union (~400/layer) — misses persist.
- NOPRE_B4 is a cold-PREFILL-throughput metric, not cold-decode concurrency:
  its wall includes ~4x1000-token prefills while agg = 64 decode tokens only
  (0.5 agg ~= 39 tok/s aggregate prefill). Decode-side staging levers move it
  marginally; judge concurrency fixes on PRE_B4 + BURST as well.
- The live bak_preMetal-derived shims ALWAYS parallel-prewarm prefill chunks
  (~8 threads); the HG_PREWARM gate lived only on the abandoned metal path.
  Before A/B-testing an env gate, grep the shim the server actually loads
  (HG_MOE_SHIM) to confirm the gate is on a live code path.
- Seeder vs graph capture: store init runs inside the capture dummy forward,
  so a seeding thread doing pinned-alloc/Event work starts DURING capture and
  poisons it globally (cudaErrorStreamCaptureInvalidated, scheduler dies in
  init_all_cuda_graphs). Gate background CUDA-adjacent work behind a shared
  deadline (monotonic deadline set at first submit, +~30 s), NEVER a per-task
  sleep in a serial executor (48 layers x 30 s = 24 min).
- Always A/B with the SAME tier-warmness: prefix battery run 1 on a fresh boot is
  cold and underestimates PRE_B4 vs a booted+warmed server. Seeding gains are
  first-sweep gains only: a fully-warm re-run on a seeded arm read NOPRE back at
  the unseeded baseline (tier evicted beyond the merged union) — read seeded A/B
  only at matched run index and warmness.

## Slot-thrash and prefill-serialization laws (TCHUNK text-bisect)
- WARN `cache fully in-use` = forced device-slot eviction whenever a chunk's
  unique-expert set exceeds CACHE_N (16); it fires millions of times per battery
  and is a PERFORMANCE-only signal (restaging cost), NOT corruption: a TCHUNK=2
  (no thrash) vs TCHUNK=128 (massive thrash) boot produced IDENTICAL P1 output
  text. Run the chunk-size text bisect before entertaining any correctness claim
  from eviction WARNs; boot-to-boot P1 text variance is greedy tie-break
  nondeterminism, not slot overwrite.
- Prefill wall = single-stream copy->GEMM serialization, not chunk granularity:
  warm P1 ~32 s at BOTH TCHUNK=2 and 128 despite ~5x more H2D bytes at TCHUNK=2
  (which additionally pays ~508 kernel launches/layer cold). Chunk size is the
  staging-bytes lever: unique experts per chunk grows sublinearly with chunk
  tokens, so staging bytes/token fall ~4x at TCHUNK=1024. Long-context
  feasibility rides on this: a 128k prompt at TCHUNK=128 is ~15 TB re-staged vs
  ~4 TB at 1024 — raise TCHUNK before promising any long-context arm.
- Queued design (unbuilt at time of writing): TCHUNK raise + side-stream
  double-buffered staging with rotating slot regions (device slots 2xCACHE_N,
  even/odd chunks alternate, H2D on a side stream gated by events before the
  grouped launch) so H2D overlaps GEMMs; bump metadata capacity
  (_METACAP_E >= TCHUNK*10) and preallocated scratch in the same patch.
  Prefill is graph-disabled, so side-stream staging carries no capture risk;
  decode (T<=16) untouched.
- Killing a queued remote watcher: killing the LOCAL ssh client leaves the REMOTE
  bash loop running — pgrep it on the node and kill by explicit PID. Any regex in
  a remote wait-loop that matches its own full command line (pattern text
  appearing elsewhere in the same line, e.g. loop bodies quoted verbatim) hangs
  forever.
