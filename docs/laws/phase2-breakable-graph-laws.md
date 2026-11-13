# Breakable CUDA-graph decode arm — final laws (CLOSED PASS 2026-11-10)

- Canonical graph arm: `launch_phase2_m11d5h384c.sh` = d5 + `HG_MOE_HTIER=384` (M18b promotion, CHANGES §78). HTIER=256 variant: `launch_phase2_m11d5.sh` (CHANGES §62b). Eager fallback: d2.
- Native sglang PLE is Ampere-dead: Triton gather uses `fp8e4nv`
  (`type fp8e4nv not supported... fp8e4b15/fp8e5`) — diet shim stays required on sm86.
- PLE + graph: with the diet gather (synchronous, ignores native prefetch), the correct
  graph island for `Qwen4ExpPLELayer.start_prefetch` is a NO-OP. Any variant that runs
  the real prefetch and then clears `_prefetch_state` starves PLE context (echo
  garbage on long prompts); leaving state set trips
  `PLE prefetch state was not consumed before reuse` on replay.
- Graph-safe apply shim = pre-metal (fresh-tensor) apply + only: tuple-dispatch bridge,
  eager_on_graph island wrap (HG_NATIVE_GRAPH_ISLAND=1), bcgraph install (HG_BCG_LPO=1).
  Selected via `HG_MOE_SHIM` env in sitecustomize.
- bcgraph: `__dataclass_fields__` iterates names(str) not Fields →
  use `dataclasses.fields()`; wrong form crashes capture with
  `'str' object has no attribute 'name'`.
- Banked: QC full parity; P1 32.27 s clean; PRE_B4 agg 8.07; warm BURST 20.07-20.29,
  peak 29.6 tok/s; NOPRE_B4 0.51 unchanged (cap-4 cold-expert staging bound).
- Any background thread doing CUDA-adjacent work (pin_memory alloc, Event
  record) must not start before boot-time decode-graph capture finishes — it
  poisons capture globally (cudaErrorStreamCaptureInvalidated). Gate such
  threads behind a shared deadline from first submit, not per-task sleeps;
  store init (and thus any lazily-started background work) fires inside the
  capture dummy forward.
- Step anatomy on the graph arm WORKS: `HG_NATIVE_PROF=1` (prof launcher
  launch_phase2_m11d5prof.sh) drives _PROF/_pt_on in
  hydragen_moe_native_graph.py — the line `[hg-natprof] L=.. T=.. K=..
  apply_ms=.. pre=.. cs=.. h2d=.. ng=.. comb=..` fires every 24 applies per
  layer instance, i.e. one line per layer per sampled step. Slice steps at
  L=0 resets; the server log carries binary progress bytes so always
  `grep -a`. HG_STEP_PROF alone emits only "[hg-step] fwd avg" lines.
  MEASURED cap-4 warm (fairness agg 21.14 tok/s, step wall ~190 ms),
  per-step 48-layer sums (median of 5 sampled steps): apply 227.1 /
  cs 150.1 / h2d 46.6 / ng 3.2 / comb 4.0 / pre 4.9 ms.
  LAW: decode stays host STAGING-bound even on the graph arm — expert-blob
  cache-slot staging (cs) is ~80% of island time and grouped-GEMM compute
  is ~free (3 ms/step). The decode speed lever is cs reduction, then h2d
  overlap with ng — never more GEMM.
  CS RUNGS MEASURED (M18b, CHANGES §78): WIN = tier capacity, `HG_MOE_HTIER`
  256->384: deep-warm cs 116.8 -> 15.0 ms/step; at 256 the tier saturates and
  cs regrows to 146-234 under sustained/mixed traffic, at 384 it holds ~15.
  Warm cap-4 bursts 23.3-25.3 vs 23.4 (parity when fully warm, big win
  mid-warm/sustained). Pinned tier RAM 3.5 -> 5.3 GB, host avail ~43 G = safe;
  HTIER=400 stays banned by the host-RAM OOM law. NEG = in-step threaded
  pread of decode-miss blobs (`HG_NATIVE_PARREAD=1`, singleton executor,
  gated >=4 misses — all variants in pyhook/hydragen_moe_native_graph.py,
  default OFF): cs drops but cost shifts into pre (submit/pinned-alloc/event),
  bursts 17.75-18 vs 19.75-21.14 control. Threaded pread only pays at prefill
  miss-set sizes (T > _TCHUNK). Post-adoption warm-ceiling anatomy: ng
  0.3-6 ms, h2d ~29-36 ms (five tiny per-chunk torch.tensor pageable H2D
  creates x 48 layers), island bookkeeping ~40 ms — next lever is batching
  step metadata into one pinned buffer per layer, never GEMM.
- The prof env trio costs ~20% burst speed on the graph arm — never
  compare a profiled burst against lean banked numbers; measure both or
  compare within the profiled run only.
- Measured: cap-4 burst fwd ~100-150 ms/forward => ~25-38 ms per
  stream-step vs ~50 ms single-stream: batching 4 streams is nearly free;
  decode is bandwidth-bound work, not step overhead.
