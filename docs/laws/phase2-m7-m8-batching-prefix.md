# M7/M8 close-out (2026-11-09) — decode batching + prefix caching on hybrid qwen4_exp

## M7: true decode batching (CLOSED)
- Silent cap: mamba state cache caps running batch; slots/req = base 3 + 2 (config extra-buffer tracking). `--mamba-track-interval` / `--mamba-max-states-per-path` are NO-OPS for slots/req. `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` -> ratio 4. `disable_radix_cache` -> ratio 1 but kills prefix caching.
- Ladder (HTIER=96, 16-tok battery): cap1-serial 2.47 -> cap2 4.32 -> cap3 4.16 tok/s. SATURATES at 2-3: union growth (10 -> 12.6 -> 19.1/layer) consumes batching win. Cap-4 not pursued.
- VRAM law: each mamba slot ~59 MB straight off pool-end headroom. pool10 start 0.62 GB free, pool12 0.50 (works), pool16 0.28 = boots then dies at first request (fp8-head 20 MB block maps). Pool-end floor ~0.5 GB.
- Hidden peak fix: fp8 lm_head chunk alloc was 65536 rows (160 MB fp8 + 320 MB bf16 dequant). `HG_HEAD_CHUNK` env, default 32768.
- Watcher discipline: server passes READY+health then dies during first forward; soak 75-90 s past ready, and first battery request is part of the test.

## M8: prefill physics + RadixAttention (CLOSED)
- PREFILL LAW: for UNIFORM routing, union/layer saturates (512*(1-e^(-rows*10/512))): any prompt >~100 tok touches ~all 512 experts/layer -> ~full-set staging per sweep. BUT repetitive content routes NARROW: measured 1205-token repetitive prefill = 87-100 s. Model union on the actual content distribution, never uniform math for repetitive prompts.
- Shim v2 lifetime: per-chunk stage/GEMM/free (peak ~73 MB vs whole-call 2.5 GB+ — prefill OOM gone). `_PREFILL_MODE` (U > 4*CACHED_N) skips device-cache AND pinned-tier ADMISSION (tier still read): sweep experts evicting the 96-entry hot tier dropped decode 0.4 -> 6.5 s/step. S2 side-stream deleted.
- RadixAttention on hybrid (mamba radix, UnifiedRadixCache hybrid_ssm=True): shared 1024-token prefix -> `#cached-token: 1152/1205`, trio prefills ~free. PRE wall 123.9 s vs unique-prompt control 252.2 s = 2.03x at 4 streams (cap 3). Works at --mamba-track-interval default; state checkpoint lands at prefix boundary.
- Battery trap: control prompts sharing all-but-4-tokens prefix silently share the radix hit (first M8 run read 164 vs 142 = meaningless). Diverge control at token ~4 with marker words, keep length identical.
- KV-pool floor for prefix batteries: max-total-tokens must hold the shared prefix ONCE plus every stream's tail (prefix ~1000 tok + 4 streams needs >= 2048); a KV-diet arm at 1024 400s/aborts before any measurement. Battery legs are only valid on arms sized for them — check pool vs prefix x streams before quoting 400s as a crash.
