# M22 rung-3: extend slot-aliasing corruption + expert sub-batching (CHANGES §86)

## The bug (was live for the whole graph-arm era)
- `pyhook/hydragen_moe_native_graph.py` `_cache_slot` eviction: when every cached slot holds an expert
  present in the current window's `rows_c` (ul > CACHE_N), fallback `victim = order[0]` evicts an expert
  ALREADY STAGED for the current GEMM batch. Its `grp_ptr` (captured before) aliases the slot the next
  `_cache_slot` stages into -> expert weights overwritten before the grouped GEMM reads them.
- ul > CACHE_N is ALWAYS true in extend windows (128-token windows, ~150-520 unique experts vs 16-24 slots)
  and common in cap-4 decode (ul ~ up to 40 vs CN=24). `[hg-native] WARN cache fully in-use` = corruption path.
- QC batteries never caught it: content corruption on long prompts is subtle; QC prompts are short.

## Evidence law: prime oracle
- Prompt: ~350 filler sentences + "list every prime between 200 and 260, comma-separated, nothing else", greedy.
- Correct: 211, 223, 227, 229, 233, 239, 241, 251, 257 (9 numbers, nothing else).
- Pre-fix (CN=16 ctx32k): emitted extra out-of-range 263. Post-fix CN=24: exact MATCH.
- Keep max_new_tokens within ctx (ctx2048 arm: 25 filler sentences + 1200-1650 new tokens); think blocks
  eat the budget on this model — assert on `t.split("</think>")[-1]`, and if the answer is cut, shrink filler.

## Fix: per-apply expert sub-batching (HG_EXT_SUB, default 1)
- When len(ul) > _CACHE_N: slice ul into <=CN consecutive groups; per slice run the ORIGINAL per-window body
  (stage -> metabatch -> grouped GEMM -> combine). Slice i's GEMM completes on-stream before slice i+1
  stages -> no slot ever aliases two live experts. Zero extra VRAM. Opt-out HG_EXT_SUB=0 restores old
  (corrupting, fast) behavior.
- Metabatch arena: per-slice cumulative offset `_wa` (never rewrite bytes an in-flight kernel reads).

## CUDA-graph capture law in the island (the real cause behind repeated capture deaths)
- `torch.cuda.current_stream().synchronize()` (or any sync) executed while a stream is capturing invalidates
  the capture; every kernel launched after it fails with `cudaErrorInvalidValue` — surfacing from C++ as
  `native_expert_grouped/swiglu: invalid argument`, or later as illegal memory access. In the island this
  happens when the metabatch arena (`ah/ag`) GROWS during capture: the realloc path calls synchronize.
  Capture runs the island's Python only during warmup (bs1..bsN captures); serving decode = replay (no Python),
  extend = eager — so multi-slice capture code can stay unexercised for many boots and then kill bs4 capture
  the moment `ul > CACHE_N` at that bs. `[hg-native] WARN cache fully in-use` xN during startup warmup =
  the multi-slice capture path is being hit.
- Fix pattern: pre-size the arena at creation with a floor big enough for any captured window
  (16 KiB covers bs<=4 with top-10 routing worst case — this model routes K=10, so decode ul <= bs*10), and make grow-during-capture raise immediately
  (`torch.cuda.is_current_stream_capturing()` guard) instead of silently corrupting the capture.
- Large LAZY allocations during capture poison capture the same way realloc-sync does: with
  `--skip-server-warmup` the FIRST island apply happens during bs-N graph capture, so a first-use
  `torch.empty` of a 100 MB-class persistent buffer in the apply path allocates from the graph pool and
  capture later dies as `Triton Error [CUDA]: an illegal memory access` inside `load_binary`, logged on
  the `Capturing batches (bs=4 avail_mem=...)` banner with no island traceback. Pre-allocate persistent
  island buffers in the shim's install step (runs before pool sizing and before capture), sized from
  env-known maxima (window size; hidden size via env fallback like HG_ISLAND_H=2560), and make any
  grow path assert `not torch.cuda.is_current_stream_capturing()` so undersizing fails loudly at the
  cause instead of corrupting capture.
- Earlier attribution to "allocation order in apply is sacred" was WRONG — hoisting xf/scr/out is a
  red herring and was later shown to belong to a crash-inducing patch family; allocations may move as
  long as no sync/realloc-forced path fires mid-capture, but there is no reason to move them: keep the
  original per-window/per-slice allocation positions and only add the capture-grow guard.
- Debug order when capture dies: count `Scheduler hit` (0 = scheduler died inside C without a Python
  traceback — read the log window before `Sleeping 5 seconds before crash diagnostics` instead);
  CUDA_LAUNCH_BLOCKING=1 boot names the faulting Python line (combine right after _ng = the GEMM faulted).
- C++ `check()` labels LAG: they call `cudaGetLastError()`, so an async fault from an EARLIER kernel is
  blamed on the next launch's label (`swiglu: invalid argument` was actually the `gu` kernel faulting
  async). Re-run under CUDA_LAUNCH_BLOCKING to get the true kernel name before hunting the labeled one.

## Post-fix gates (all passed, canonical launch_phase2_m11d5cn24c.sh + patched island)
- boot HEALTH_OK with clean capture; prime oracle MATCH; battery_quality 8/8 clean (QC0 exact);
  fairness cap-4 33.42 / 39.61 tok/s — no decode regression (ng was ~free; extra launches cost nothing).

## What the fix does NOT do (perf rung = HG_EXT_WIN)
- Wire bytes unchanged: each unique-per-window expert still stages once per window. The staging floor
  per 15k-token chunk is unique-per-chunk (~1400) vs per-window sum (~4190): enlarge extend windows
  (`HG_EXT_WIN`) for ~/3 wire = prefill TTFT 343s -> ~200s. Unblocked because larger windows stay
  correct via slicing.
- CN=96 is unfundable on any current arm (island buf ~660 MB; min viable mem-fraction 0.795).

## HG_EXT_WIN laws (window enlargement on top of the slicer)
- Before changing any window/token cap in the island, grep for EVERY allocation sized from that cap and
  resize all in the same patch. `xq8` (quantized-activation scratch) is sized via `tc = min(T, _TCHUNK)`;
  enlarging the window without fixing `tc` = illegal memory access on the first extend chunk.
  `scr` is sized per-slice from E; cap slice entries during slice construction (`_emax = _TCHUNK*K`)
  so scr never balloons with the window.
- `_ng` arg 5 is `n_tokens` = window rows; the exported wrapper in `hydragen/cuda/hydragen_bridge.cu`
  quantizes ALL n_tokens x_f32 rows into xq8 on EVERY call — a multi-slice window re-quantizes the whole
  window per slice (known redundant cost). Always verify ctypes arg order against that bridge wrapper,
  NOT against iq_kernels.cu's internal `native_expert_grouped` (different order and arg meaning).
- HG_EXT_WIN crash RESOLVED as patch-family, not window size: the combined family (window loop +
  `_emax` entry-capped slice construction + hoisted per-window xf/out alloc + arena-floor/guard) faulted
  at slice i=1 of the FIRST extend window at every HG_EXT_WIN including 128 (pure §86 geometry + patched
  slice construction), while the pure §86 island restored from backup ran 7 clean T=1024 prefill chunks
  on the very same crashing arm. Correct construction: layer ONLY a strict-minimal diff on the last
  known-good island — window-loop step `_win` plus `tc = min(T, max(_TCHUNK, _EXT_WIN))` xq8 sizing —
  and keep §86's `_sd` slice construction and the per-slice body byte-identical. At large windows also
  cap slice ENTRY count (`_EMAX` ~1280: shrink `_kb` while `grp_start[_kb]-grp_start[_k] > _EMAX`) so
  `scr` and per-slice staging stay bounded — the cap alone is innocent; the crash family bundled it with
  hoisted window allocs and a combine restructure. In-test shape (verdict pending): strict-minimal window
  loop + `_EMAX` cap + shared `_PB` arenas pre-allocated at install with capture-grow assert. Never
  enable HG_EXT_WIN on canonical until the minimal arm passes cold-probe TTFT vs the §83 baseline
  (~290-300 s mean for 15k-char code-domain ctx32k probes), anatomy (`hg-natprof` cs/h2d per chunk vs
  the 5.2/10.2 s baseline — expect ~3x drop if re-staging really fell from 4190 to ~1400 ops/chunk),
  AND QC battery + prime oracle.
- Cross-launcher gate law: a known-good code state counts only if its gate was run on the SAME launcher
  shape (ctx, CACHE_N, HTIER, graph/eager) where the crash reproduces — §86 passed on ctx2048/CN24 but
  had never been run on the ctx32k/CN16 anatomy arm. The decisive first bisection move when a patched
  arm faults is: restore the last known-good backup and run it on the crashing arm, before dissecting
  the new patch — it converts "the patch family is broken" from theory into a single boot's fact.
- Debug prints for island faults (proven recipe): `[hg-ng]` (L, Tw, nslice, slice idx, E, arena wa/ah)
  before `_ng`, plus `[hg-ptr]` (grp_ptr offsets min/max vs `st["buf"].numel()`, BB) — in-range pointers
  rule out slot-corruption and point at sizing/patch regression; `sed -n` the bridge wrapper to confirm
  arg semantics. Then env-flag bisect (HG_META_BATCH=0, HG_BLOB_FAST=0) in one serialized boot to split
  staging mechanism from slice logic, and HG_EXT_WIN=128 as the geometry-equals-baseline rung.
- Probe-progress law: under CUDA_LAUNCH_BLOCKING prefill runs ~3x slower (~33 tok/s), so a probe timeout
  produces CRASH=0 runs that never reached the suspect region — always confirm how far it got by counting
  `Prefill batch` lines (each = one 1024-token chunk) and expected island prints before believing a
  clean leg; size probe timeouts to LB-slowed speed, not normal speed.
- Funding law: `out` = Tw x K x H fp32 grows with the window (1024-token window, K=10 = 100 MB per
  call). MEASURED: 100 MB-class transient allocs at per-slice granularity DO OOM at pool end on a tight
  ctx arm (84 MiB free + ~900 MiB reserved-unallocated — the caching allocator cannot regroup 100 MB
  blocks against a full KV/mamba pool; trim of `--max-total-tokens` alone does not save it). Correct
  shape: ONE module-level shared arena pair — `xfp` (max-window rows x H fp32) and `ob`
  (max-window rows x Kcap x H fp32, Kcap=12 for K=10) — reused across slices AND across all 48 layers
  (window-sized buffers stored PER-LAYER in `st` = 48x the bytes = boot OOM; keep them module globals
  and hand the same tensor to every layer). Zero only the `[:_rows]` prefix once per window before the
  slice loop, and move combine to window level after the slice loop (§86 view-sum semantics preserved).
  Do NOT swap `out` from torch.zeros to torch.empty to dodge the memset: combine multiplies ALL K rows
  by w and uninitialized NaN/Inf x 0 = NaN poisons the residual; the once-per-window prefix zero
  (~100 MB memset x 48 layers per chunk) costs well under a second and stays.

## Blast-radius law for staging-path corruption (which verdicts survive)
- Slot-cache corruption touches ONLY routed experts staged through `_cache_slot` (the blob store path).
  Dense/shared weights loaded by `hydragen_dense_iq.py` are static per-weight buffers (`torch.from_file`
  at load; paged-expert resident buffers are stubbed out) — no slot cache, no eviction, never affected.
  Dense-IQ diet verdicts (IQ4_XS reject, Q5_K floor) and KV-quant verdicts stand.
- Adjudication rule for any staging-path correctness bug: short-prompt sequential batteries run decode
  bs=1 where ul = K < CACHE_N, so the eviction fallback never fires — those quality gates remain valid
  even for builds with the bug live. Outputs from extend/prefill or multi-stream decode with ul > CN
  are suspect and must be re-verified on the fixed build.
