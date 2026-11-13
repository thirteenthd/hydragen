# M13 long-context KV laws (HiCache / KV diets, 2026-11-10)

## HiCache host KV tier (OPEN PROMISING, CHANGES §68)
- Attach switch: --enable-hierarchical-cache. hicache_ratio/hicache_size
  flags alone attach nothing (registry gate ctx.enable_hierarchical_cache
  -> init_hicache; log proof: "hicache_attached=True" in
  "Tree cache initialized").
- On this hybrid substrate the host tier covers KV AND GDN/mamba state
  ("NPU HiCache Mamba state transfer mode: sync torch fallback").
- Eviction/reload works: 27.4k-tok prompt evicted by a second 27.4k prompt,
  rehit 1.3s / cached=27328 vs 509s cold re-prefill (~400x).
- Quality: within banked oscillation family; QC6 tie-probe oscillates on
  EVERY ctx-shifted arm (91-slip, think-leak) — do not blame hicache for
  QC6 variance; isolate by running QC on a same-ctx non-hicache arm.
- Active in-flight sequences stay GPU-pinned: host tier funds idle/cached
  prefixes (agent sessions between turns), NOT 4x simultaneously-active
  128k streams.
- Fairness cost small: BURST 18.06/19.50 on hicache arm vs 20+ canonical.
- 128k sizing (§71): --context-length 131072 --max-total-tokens 40960
  --hicache-size 8 funds host pool 149,248 KV tokens = 3.67 GB (the GB
  request splits between KV and mamba state; ratio <2 of the request).
  Reload ~18k tok/s host->GPU: 27.4k re-hit 1.5s vs 498s cold. Parked-
  session capacity at size 8 ~= 5.5 x 27k sessions or ~1.1 x 128k sessions;
  RAM headroom allows raising --hicache-size (24 GB free post-run).
- Probe budget law: each cold 27k prefill leg ~500 s — a 3-request
  eviction/rehit probe needs timeout >= 1500 s (900 s killed it mid-chain).
  Re-running the probe on the SAME live server turns cold legs into
  reload measurements for free (radix/host tier keeps the prefixes) —
  but a companion watcher that pgrep-waits then pkills the server must
  not race that rerun: put kill inside the same chain, after the probe.

## KV dtype floors on Ampere sm86
- fp8_e5m2 KV: mechanics fine (SDPA shim dequant patch), pool halves to
  12.8 KiB/token, +7% decode at 29k ctx — but QC6 FAILS (non-prime
  garbage; 2-bit mantissa too coarse). NOT quality-safe.
- fp8_e4m3 KV: triton e4nv unsupported on Ampere (known crash).
- Remaining lane: int8 pool subclass with per-token scales (gather kernel
  already converts pool-dtype -> bf16 scratch; scales-gather wrapper is
  the open design).

## Long-ctx decode economics (M13-0, §66)
- QSA sparse gather is LIVE on stock decode: indexer topk_indices ->
  qwen_sparse_kv_extraction_compact_triton -> packed_k/v (budget 2048) ->
  varlen fn sees ALREADY-GATHERED rows. Decode attention IO constant vs
  ctx (~25 MB/token); no dense-SDPA 1.6 GB/token wall.
- Decode at 29.4k ctx: 14.9-17.1 tok/s vs ~20 short ctx (-16%).
- Prefill ~50-60 tok/s, expert-staging bound (NOT KV bound): 27.4k cold
  prefill 496-509s. Attack = HTIER/seeded tier + pre-prefill of known
  system prompts, not KV work.
- KV pool sizes from residual GPU mem after mamba/buffers; max-total
  request CAPS it (fp8 arm left 0.79 GB unused at request cap).
- Prompt sizing: probe word-count undercounts tokens (~+1 token/marker
  word); HTTP 400 body carries exact oversize token counts.

## Host RAM triangle (binding budget at 128k)
- 60 GB host = PLE mmap ~48 GB + HTIER pinned tier + KV host tier.
  4x128k bf16 KV host tier ~12.5 GB: fits only with tier/PLE tradeoffs.
- GPU side: active 4x128k needs ~2.8 GiB resident KV vs ~0.5-0.8 GB
  headroom at ctx-32k arm -> GPU KV diet (int8) is the funding gate.
