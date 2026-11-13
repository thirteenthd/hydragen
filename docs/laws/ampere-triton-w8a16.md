# Ampere (sm_86) Triton w8a16 paged-experts kernel

Verified on 3080 Ti, 2026-11-08: standalone grouped-GEMM bench cos=1.00000 vs fp32 ref,
13-14 us/expert @ M<=16 (~250 GB/s), ~1300x faster than eager torch dequant path.

## Hard walls
- Triton compiler blocks ALL fp8e4nv loads/casts on sm_86: both native fp8 pointer loads
  and `uint8 -> float8e4nv` bitcasts fail ("type fp8e4nv not supported... fp8e4b15/fp8e5").
  Workaround = manual e4m3 bit-decode in-register (validated maxdiff 0.0):
  ```python
  i = u.to(tl.uint32)                 # u = uint8 view of fp8 tensor
  s = (i >> 7) << 31
  e = (i >> 3) & 0xF; m = i & 0x7
  bits = s | ((e + 120) << 23) | (m << 20)
  normal = bits.to(tl.float32, bitcast=True)
  sub = m.to(tl.float32) * 0.001953125
  v = tl.where(e == 0, tl.where(s != 0, -sub, sub), normal)
  ```
  Then `(v * block_scale).to(bf16)` -> `tl.dot` bf16 tensor cores. Pass W pointer as
  `w.view(torch.uint8)`.
- Block-FP8 qwen4_exp layouts: per-expert w13 [1280,2560] fp8 + scale_inv [10,20] f32
  (grid = ceil(N/128) x ceil(K/128), row-major); w2 [2560,640] + [20,5].
- Native paged experts `--paged-experts-num-resident` floor = router top-k (10) — below
  it, startup dies. Under a shimmed apply the K-slot table (K x 5.6 MB x 48 = 2.7 GB at
  K=10) is dead weight — free it in the first shimmed apply: shrink each
  `checkpoint_params` tensor in place to a 1-element Parameter, null
  `executor`/`device_executor`, `gc.collect()`, `torch.cuda.empty_cache()`.
  VERIFY the free by printing `torch.cuda.memory_allocated()` before/after — MEASURED:
  the free DOES reclaim each layer's slot table (~46 MB/layer) but only INCREMENTALLY at
  that layer's first apply, so the full ~2.2 GB is not free until every layer has seen
  one call — the first request runs under full pressure and an ambitious HG_MOE_CACHE_N
  OOMs mid-first-request at the weight-stack. Budget the launch against PRE-free VRAM.
- Server baseline under all shims ~10.3-11.4 GB committed at mem-fraction 0.95; only
  ~200-600 MB slack for GPU-side expert staging. Banked working shape: HG_MOE_CACHE_N=1,
  stack/GEMM in chunks of <=4 experts, --max-total-tokens 2048, --max-running-requests 1.
  Do NOT lower --mem-fraction-static to "make room" on a hybrid (GDN/mamba) model — the
  hybrid state cache is sized from it and 0.85 dies with `Not enough GPU memory for
  hybrid state cache. Computed max_mamba_cache_size=-6`. Reclaim dead allocations (K-slot
  table above) instead.

## Kernel gotchas that bit during dev
- Validate the kernel with a ladder of deterministic micro-probes before any serving
  run: 1 K-block scale=1, then a 2-K-block probe with deliberately unequal scales
  (catches per-block scale-indexing and A-pointer bugs that ONLY appear when
  K > BLOCK_K), then the full shape. Guessing at the full bench wastes cycles.
- `tl.dot(A, W)` needs W as [K,N]: store [N,K] -> `tl.trans`.
- Do NOT advance `a_ptrs` per k-iter AND index with `k0+offs_k` — double-count (symptom:
  wrong only when K > BLOCK_K).
- Variables defined inside triton `for` are not visible after the loop (scope error).
- Mask stores with `offs_m < M` and keep OUT sized E*Mpad exactly; masked rows must not
  spill into the next expert block.

## Shim architecture (hydragen_moe_triton.py, HG_TRITON_MOE=1)
Patches PagedExpertsMoEMethod.apply: host store -> stage routed experts (raw fp8+scale,
5.6 MB/expert) -> per-expert-row A blocks [U,Mpad,H] -> two chunked Triton grouped GEMMs
(VALID counts per expert block) -> silu*up -> scatter index_add with router weights.
Cache eviction must never evict experts in the current call (KeyError crash seen once).
Count routed rows in python and build the VALID tensor in one shot — `.item()` inside the
per-row mapping loop syncs the GPU per row and stalls the scheduler.
Attribution harness: `HG_MOE_PROF=1` env gate accumulates perf_counter spans around
stage / chunk-GEMM / glue, averaged over a modulo window, per decode step. PITFALL: the
counter lives on `self`, which is PER-LAYER — it increments once per decode step per
layer, so a `%48` gate never fires inside a 32-token run; gate on steps (`%8` prints
every 8 steps per layer). BANKED B=1 warm breakdown: stage ~7-8 ms/layer (8 cache
misses/call, ~45 MB staged, pinned-ring memcpy+enqueue at ~5.6 GB/s), GEMM ENQUEUE
0.3 ms, glue ~0 — MoE staging is ~60% of token time; the kernel and python glue are
not the constraint. Next lever = host read path / page-cache protection / hit rate.

## End-to-end serving status (banked)
- HG_TRITON_MOE=1 replaces the eager host-gather MoE (8.35 s/token): 16-token greedy run
  warm 8.9-9.3 s (0.56 s/token, ~15x), cold first run 23.9 s, coherent output, first
  token = M1 oracle target. The microbench speedup reproduces end-to-end.
- MEASURED concurrency (--max-running-requests 4, N=4): B=1 warm 0.66-0.86 s/token;
  4-way distinct-prompt burst wall ~48-50 s for 64 tokens = agg 1.29-1.35 tok/s — NO
  concurrency gain over single stream. Vectorizing the glue (batched index_copy_ gather /
  index_add_ scatter, one H2D for index lists, CH=8) changed wall time ZERO — per-layer
  python op overhead is NOT the constraint. Prime suspect: host store reads (113 GB
  hg-store + 47.7 GiB PLE mmap thrash a 62 GB page cache -> NVMe re-reads every step,
  ~50 MB/layer staged even warm at N=4). Next lever = host read path / page-cache
  protection / hit rate, not kernel or glue.
- Staging: pinned ring (HG_MOE_RING_MB, torch-pinned buffer reused by offset bump) —
  raw libc mlock fails at the default 8 MB memlock ulimit but the pin_memory /
  cudaHostRegister path works. Two ring bugs to pre-empt: `tensor or None` raises on
  tensor truthiness (use an explicit `is None`/`is False` sentinel), and a slice that
  runs past the ring end is silently CLAMPED to a short tensor (shape errors downstream)
  — the bump must reset the offset to 0 whenever the next chunk would overrun.
- HG_MOE_CACHE_N is a VRAM dial: each cached expert ≈ 5.6 MB x N x 48 layers; N=10 OOMs
  at the W2 weight-stack with everything else banked. N=1-4 is the fitting range.
- Server-boot watch/notify patterns must anchor on the exact marker `Scheduler hit an
  exception` — bare `Error|Traceback` fires on benign warnings (TileLang data-race
  WARNING) and produces false CRASH notifications.
- Relaunch discipline: a new launch while an old server still holds context init dies
  with `cuDevicePrimaryCtxRetain: out of memory` — kill the tree, wait, and confirm
  `nvidia-smi` shows ~9 MiB free BEFORE launching. Put `pkill -f "launch[_]server"` in
  its OWN ssh call: in the same compound command as the launch line, the pattern matches
  the wrapper's own command line and the ssh exits 255 having killed itself. Crash
  diagnosis greps the log between `Scheduler hit an exception` and the traceback end;
  each relaunch overwrites sg-real.log, so extract the traceback BEFORE relaunching.
- Full-shim boot to `fired up and ready` takes 15-25 min (host-embed build + 131-shard
  load at ~4 s/shard; `Multi-thread loading shards: NN%` lines are the liveness signal).
  Arm measurement pipelines with a >=30-min wait loop; a `ConnectionRefused` after a
  short wait means the server is still loading, not crashed — check `pgrep -fc launch`,
  `grep -c "Scheduler hit an exception"`, and the progress line before reacting.

## Tier capacity, affinity, and the host-RAM partition (banked verdicts)
- Staging wall is fixed by a BOUNDED PINNED HOST HOT-TIER (`HG_MOE_HTIER=K`): per-layer
  OrderedDict of pinned expert copies between mmap store and device cache; stage =
  mmap -> pinned copy (LRU-evict) or direct H2D from pinned on tier hit. Banked curve
  B=1 warm / B4 agg: naive 12.5s/1.43 -> K=12 9.1s/1.50 -> K=24 8.3s/1.59 (stage
  3.0-3.7 ms/layer). This is the Phase-1 hot-tier lesson reproduced inside SGLang.
- TIER vs PLE-MMAP RAM PARTITION: pinned tier copies and the 47.7 GiB PLE mmap compete
  for the same host RAM — growing the tier reclaims PLE page-cache pages, so per-layer
  staging REGRESSES past a ceiling (K=40: stage 4.4-6.4 ms despite bigger tier). Do not
  scale the tier blindly; free the mmap first (diet below), then scale.
- PLE RAM DIET (HG_PLE_DIET=1): replace the mmap `index_select` gather with pread +
  bounded row-LRU (rows are 160 B fp8; 262k rows = 42 MB cache), then
  `madvise(MADV_DONTNEED)` the whole native shared mapping once and periodic
  `posix_fadvise(FADV_DONTNEED)` on the pread fd (~every 512 gathers) to keep the file
  cache tiny. Frees ~40+ GB host RAM for the expert tier. fadvise alone does NOT drop
  pages that are still mapped — madvise on the mapping address range is required.
- EVALUATE SERVING LEVERS OFFLINE BEFORE PORTING THEM: replay the captured routing
  trace with `scripts/replay_capacity.py` (88B grammar, block-scan for dynamic batch
  windows — never assume fixed rows/window). Banked qwen4_exp verdicts from that replay:
  affinity/batch-grouping union headroom ~0 (natural 19.1 vs optimal 19.0 vs random 19.2
  unique experts per B=4 layer — routing is hot-skewed, random rows already share ~50%),
  so ExpertFlow-style affinity scheduling is NOT worth porting; the lever is tier
  CAPACITY — LRU hits climb .53->.78->.87 at C=24->64->96/layer and B=1/B=4 hits
  CONVERGE for C>=48 (hot-skew provides the sharing, concurrency becomes nearly free);
  C=64 needs ~17 GB pinned host, which is exactly what the PLE diet unlocks.
- DIET + HTIER=64 BANKED (matches the offline prediction): B=1 warm 6.6-8.0 s/16 tok
  (2.42 tok/s best), B=4 aggregate 1.86-1.99 tok/s, stage 0.4-4.4 ms/layer, RAM: 17.2 GB
  pinned tier + diet'd PLE coexist on the 60 GB host. The §-wall past K=24 is gone.
- THE WALL MOVED OUT OF MoE: `HG_STEP_PROF=1` shim wraps ModelRunner.forward with
  sync-wrapped timing, printing `[hg-step] fwd avg/max ms` every 8 calls. Measured B=1
  fwd avg ~376 ms vs MoE stage+gem ~170-200 ms — ~190 ms/step is NON-MoE eager path and
  B=4 agg still sits BELOW B=1 warm because it scales with streams. Itemize the remainder
  with per-item timers before touching MoE again.
- HOST lm_head IS A HIDDEN PER-STEP WALL: the chunked host-head path streams the whole
  ~5 GB vocab matrix (V x H bf16) host->GPU EVERY step (~150 ms at RAM bandwidth).
  Resident mode (HG_HEAD_RESIDENT=1) copies the 2.5 GB head to GPU once and matmul
  on-device. Fund the VRAM by lowering --mem-fraction-static (the KV pool is idle at
  bench sizes) plus the K-slot free — NOTE this interacts with the hybrid-state-cache
  floor above: lower in small steps and watch for the `max_mamba_cache_size=-6` line;
  arm outcome pending, do not quote as banked until [hg-prof] lm_head-resident ms/step
  lands.
