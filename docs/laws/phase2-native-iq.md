# Native Strata IQ grouped MoE serving (M9, the reference host)

Shim: pyhook/hydragen_moe_native.py, HG_MOE_NATIVE=1. Pinned IQ blob tier
(HG_MOE_HTIER experts/layer), device IQ cache (HG_MOE_CACHE_N), token-chunked
strata_vllm_native_expert_grouped (HG_NATIVE_TCHUNK=32), torch combine.
launch_phase2_m9.sh. M9e: m9e_vs_fp8.py (ng vs FP8-checkpoint expert forward).

SERVING-GARBAGE VERDICT (supersedes the M9/M10 'quality-fatal' claims, CHANGES
§49-50): all native IQ battery failures ('!!!!', garbage text) were a SERVE-SHIM
entry-ordering ABI bug, NOT quant quality.
ABI law (the engine's k_group_small convention): native_expert_grouped entries
must be ranked GLOBALLY by (expert_id asc, token-major entry index); groups =
consecutive runs in that order; ent_dst[rank]=original token-major index,
ent_tok=token. A first-seen-group order yields correct output for only ~2/8
groups (rest cos~0) — indistinguishable from model-quality garbage. Benches
using ent_dst==ent_tok (m9e/m9b) hide it completely. Probe the SERVING shape
first — scripts/m11_abi_probe.py builds shared-token multi-group entries at
real shapes and compares per-entry output vs the fp8-checkpoint expert forward;
blame quant tiers only after that probe is clean.
Probe-clean != shim-clean: with the entry-rank fix applied, the in-server
battery still garbled — a residual probe-vs-shim delta is OPEN. Debug pattern
that localizes it: dual-run hook — on the first call of layer 0, BEFORE the
param-freeze block, call the original apply (params still alive) to capture
its output, run the native path on the same dispatch_output, print per-row
cos + norms, and return the original output to keep the server coherent while
measuring (env-gated, HG_NATIVE_DEBUG=1 in the deployed shim).
Tier status: IQ3_S blob fidelity per-entry cos 0.93-0.97 at combos
(18,20)(21,20)(22,42)(23,20) — parity-tier arm still alive. IQ3_XXS per-expert
cos vs fp8 (0.81-0.94, measured by m9e whose single-group layout WAS valid) is
real quant degradation at that tier, but its battery-failure attribution was
the ABI bug below.

BUG #2 (device cache, fixed 2026-11-10): slot allocator must REUSE the victim's
slot id; s=len(order) after mid-list removal re-hands a slot still owned by a
live expert (collision -> wrong expert bytes). Correct for small T (no
 eviction), garbage once prefill thrashes; [hg-native] WARN cache fully in-use
is the tell. Replay gate: HG_NATIVE_DEBUG=1 -> m11_replay.py vs fp8-ckpt fwd.
(cosine_similarity default dim=1 breaks on vectors — pass dim=-1.)
CLOSED VERDICT (after both fixes): native battery PASSES at both tiers.
- FP8-dense + IQ3_XXS blobs: capital battery clean (' Paris, which is also the
  largest city...'); burst 8.25-8.47 (N=12/HT288), 9.08-9.58 with cap-3 +
  N16/HT400 (launch_phase2_m11a2.sh = fastest banked on the node; 9.58 was
  measured on the profiled arm m11o1, tier-policy-fix A/B still pending).
- donor-S dense + native IQ3_S, and FP8-dense + native IQ3_S: PASS, 4.4-7.0.
XXS is battery-viable under FP8 dense context despite the 0.81-0.94 per-expert
micro ladder — per-expert cos is necessary evidence; the serving battery is
the arbiter. XXS is also the cheaper/faster wire (42.9 vs 50.3 GB store).
Best native config (m11a2): q4e-fp8 + hg-store-iqxxs-blobs, CACHE_N=16,
HTIER=400, TCHUNK=32, --max-mamba-cache-size 12 + SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
(cap-3). Raising mamba-cache above 12 fails boot with 'Loaded weights leave no
GPU memory for the KV cache' — mamba-state tax eats pool-end; move the cap via
the SKIP_DECODE_LOCK ratio, not pool size.
cap-2 is already near the cap-3 win on the native path (+7% only). NATPROF + STEP ANATOMY (banked map, supersedes the earlier half-half split):
HG_NATIVE_PROF=1 sync-times apply every 24th call; the print carries
pre/cs/h2d/ng/comb sub-segments (perf_counter inside apply): cs = cache-slot
staging incl. blob host read, h2d = metadata tensor allocs, ng = ctypes
grouped-call issue, comb = torch combine. Solo decode step = 120 ms:
MoE ~84 (avg 1.7 ms/layer), streamed fp8 lm_head 33.4, PLE gather 1-2,
launch/Python gap ~0 => WIRE + kernel bound, not eager-Python; CUDA-graph
capture or glue fusion do NOT pay — cut wire traffic (staging spikes, head
stream overlap) or spec-decode. cs spikes up to 16 ms/layer = pinned-tier
miss forcing a file read (NVMe cold).
PINNED-TIER HOSTING LAW: the host blob tier must (a) LRU-touch on hit
(del ht[e]; ht[e]=b re-inserts) — insertion-FIFO eviction evicts HOT blobs
and reintroduces NVMe reads mid-decode — and (b) recycle evicted pinned
buffers through a free pool instead of fresh torch.empty(pin_memory=True) per
miss (that alloc is a synchronous page-lock). The evict queue must hold
(buffer, cuda.Event) PAIRS and move the BUFFER to the free pool only once
event.query() is true — appending the Event itself to the free list returns
an Event on the next miss: AttributeError 'Event' object has no attribute
'copy_' crashes the scheduler mid-request (grep the full 'Scheduler hit an
exception' traceback with grep -A120 — the shim frame at the tail is the
decisive line; watcher grep chains hide it).
EVICT-PATH BLIND SPOT: decode-only batteries keep unique staged blobs under
HTIER_K, so tier eviction and the recycle pool NEVER RUN in them — decode
numbers can pass with a broken pool. Long prefill (~460 of 512 experts/layer
unique) is what exercises eviction; run at least one battery_prefix prefill
leg after ANY tier/pool change before believing it.
A/B result: LRU touch alone moved B2 9.58 -> 10.57 tok/s (the broken pool
contributed nothing — it had never fired on decode arms).
SCRATCH PREALLOC LAW: strata_vllm_expert_scratch_bytes(E,NFF) scales with
E (entries = TCHUNK x K per chunk); scratch preallocated at decode size
(E=64) must grow ~160 MB on the first long prefill, and reassign-growth
(`st["scr"] = torch.zeros(need)`) allocates the new buffer while the old is
still referenced — double peak, OOM at pool end ('Tried to allocate 160.00
MiB', 1.6+ GiB reserved-unallocated fragmentation). Preallocate at install
for the prefill-chunk bound (max(64, TCHUNK*K)) and on any later growth
del the old buffer + torch.cuda.empty_cache() BEFORE allocating. Lowering
HG_NATIVE_TCHUNK (32->16) halves scratch and never touches decode (T=1).
PROFILER WALL: torch.profiler/CUPTI (HG_STEP_TRACE) segfaults the scheduler
under native ctypes kernels (sticky CUDA_ERROR_ILLEGAL_ADDRESS, 'Requested
callback is not found' -> segfault). Do step anatomy with the CPU-side region
timers; head/PLE [hg-prof] prints gate on HG_MOE_PROF=1 even on native arms.
LOG WINDOW HYGIENE: aggregate per-arm metrics only from the new boot window
— anchor tail from the LAST 'Uvicorn running' line (grep -an | tail -1,
tail -n +L); a dying prior arm's [hg-step]/crash lines contaminate the tail
when kill and relaunch race in one call. awk math on 'key=value' log fields
must strip the key (grep -o 'apply_ms=[0-9.]*' | cut -d= -f2) — awk coerces
'apply_ms=0.9' to 0 silently.
ENGINE-CHOICE LAW: IQ blob kernels and CUDA-graph-stepped decode are portable
INTO the SGLang skeleton; do NOT adopt the whole Strata engine for step speed
on 12 GB — its batch slot costs ~450 MiB row-state (vs ~59 MB SGLang mamba
slot, ~8x cheaper per concurrent stream) and carries the measured batch-2
cannibalization ceiling. Kernels/weights transfer, engine architecture does not.
Extraction law: GGUF->blob extraction needs only the shard(s) holding routed
experts; ngram/dense-only shards never need downloading for expert-only arms.
Loader-skip has ONE manifest per store dir: `--make` for checkpoint A
invalidates checkpoint B's stamp (next boot = CHECKPOINT CHANGED full load) —
re-stamp when switching arms.
Page-cache hazard: ~90 GB GGUF downloads/extraction exhaust reclaimable cache;
the next FULL 131-shard boot can die with scheduler exit -9 = OS OOM killer
(check `free`/dmesg, not just GPU state). Delete donor GGUFs once blobs are
extracted+verified (re-fetchable from HF), or drop caches before heavy boots
(ssh sudo fails 'terminal required' — needs a TTY/root path).

Laws:
- Native kernel == IQ dequant (cos 0.9999). M9b 0.9999 is kernel-consistency,
  NOT fidelity vs FP8 checkpoint. Quantization damage measure = ng-vs-fp8 cos.
- IQ3_XXS donor mix: m9e (valid single-group layout) per-expert output cos vs
  FP8 0.81-0.94 = real quant degradation at that tier; the garbage TEXT in the
  serving battery was the entry-ordering ABI bug above, not this ladder.
- Throughput of native IQ path is real: B2 burst 4.99-6.90 agg tok/s vs 4.32
  banked FP8 cap2 (+15-60%). Wire diet 42.9 GB total vs 113 GB fp8 store.
- dequant_f16 and native_expert_grouped share internal state in the same
  process: calling dequant first makes the next grouped launch fail
  "invalid argument". Never use dequant_f16 as in-process reference after ng;
  compare against FP8 checkpoint tensors on CPU/torch instead.
- iq_row_bytes(type,COLS) = per-row bytes; blob = rows*row_bytes.
- expert_blob_info(type...) ABI never probed; blob layout [gate|up|down] with
  gu rows=2*N_FF over cols=N_EMBD then down rows=N_EMBD cols=N_FF works.
- IQ3_S pack metadata exists (packs/iq3_s/native_experts.txt) but GGUFs absent.
- Probe hygiene: use guard regions inside one big cuda tensor to catch
  out-of-bounds writers; torch.empty outputs turn launch failures into NaN
  references — allocate references with torch.zeros.

DENSE-WEIGHT IQ4_K WIRE (quantize the non-expert Linear layers, serve via
strata iq_mmvq; frees ~3.7 GB VRAM on qwen4_exp, funds cap-4, +35% decode):
- Build: DIY-quantize from the BF16 dense tensors in the FP8 checkpoint (dense
  tensors in q4e-fp8 ARE bf16; only experts/PLE are fp8) with
  quantize_row_q4_K_ref from ~/llama.cpp/build/bin/libggml-base.so. Q4_K
  superblock needs input cols padded to 256 (down_proj n_in=640 violates).
- Instrument AFTER model load_weights returns: walk named_modules and quantize
  each module.weight.data IN MEMORY, then mod.weight=None + swap forward.
  NEVER filter checkpoint tensor names before load — merged modules break
  (ckpt in_proj_qkv+in_proj_z -> module in_proj_qkvz; gate+up -> gate_up_proj;
  qwen4_exp attention projections are FLAT model.layers.N.qkv_proj/o_proj with
  NO self_attn prefix; finalize_fused_in_proj runs at the end of load_weights,
  so post-call instrumentation sees final shapes for free). Suffixes must be
  loose module-name ends ("qkv_proj", "o_proj", "in_proj_qkvz", "out_proj",
  "gate_up_proj", "down_proj", "index_qk_proj") — 204 modules, 1.54 GB wire.
- Return contract: EVERY sglang *ParallelLinear.forward returns (output, bias);
  call sites unpack (Merged gate_up included). A custom IQ forward returning a
  bare tensor dies with 'not enough values to unpack (expected 2, got 1)'.
- ctypes ABI: iq_mmvq/quantize_q8_1_rows from libstrata_iq_kernels.so live under
  MANGLED names (_ZN6strata7kernels...EiPKvS2_PfiiiPv) — nm -D and use the
  mangled strings. iq_mmvq writes fp32: slice the uint8 output buffer as
  y[: T*nout*4].view(torch.float32).view(T, nout). Activations: pad input to
  cols>=ncp zero-filled f32 on device, quantize q8_1 per call, pass the torch
  stream via torch.cuda.current_stream().cuda_stream.
- Cache IQ bytes on disk keyed by module name+shape (from_file on next boot)
  — CPU q4_K quantize of ~5 GB takes minutes per boot otherwise.
- BANKED: cap-4 m11d2 boots+survives where FP8-dense cap-4 OOMed by 12-16 MB;
  B1 16.15 / B2 16.82 tok/s, PRE_B4 7.04 agg. Host RAM: keep HTIER=256 on
  this arm (HTIER=400 hit kernel OOM-killer mid-battery, scheduler exit -9;
  journalctl -k is the diagnostic). QUALITY OPEN: QC3 echo + QC7 wrong-fact
  flagged vs banked FP8 answers — head-to-head same-boot reference battery
  required; fallback knobs = keep qkv/qkvz stacks BF16, or Q5_K tier.
