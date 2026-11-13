# Phase 2 long-context / KV budget laws (M13 direction)

## The big correction: QSA sparse gather is ALREADY LIVE in decode
- `qwen_sparse_attn_backend.forward_decode` computes indexer `topk_indices`,
  runs `qwen_sparse_kv_extraction_compact_triton` to gather only
  <= indexer_budget (2048) KV rows into `packed_k`/`packed_v` scratch
  (capacity = batch x topk; cuda-graph sized at max_tokens x topk), then
  calls the varlen fn with the GATHERED rows.
- The HG_QSA_SDPA=1 shim replaces `_resolve_flash_attn_varlen_func` and thus
  receives ALREADY-GATHERED ragged KV (cu_seqlens_k <= topk) — it is NOT
  attending to the full context. Decode attention IO is flat in ctx
  (~12 layers x 2048 rows x ~1 KiB = ~25 MB/token), regardless of ctx.
- Therefore the "dense SDPA 1.6 GB/token wall at 128k" does NOT exist on
  this substrate. Long-context feasibility on 12 GB is purely a KV MEMORY
  budget problem, not attention bandwidth. M13 = int8 KV + host-paged KV +
  Radix sharing; M13b (QSA gather) needs no implementation.
- QSA index-key ring still caps speculative_num_draft_tokens <= 4 (one group
  held) — see phase2-spec-decode-laws.md.

## KV budget numbers (measured/cfg)
- KV cost 25.6 KiB/token/row (12 full-attn layers, 2 KV heads, 256 dim,
  bf16, page-rounded). fp8 KV dtype status on sm_86: e4m3 hits the triton
  fp8e4nv block, but **fp8_e5m2 BOOTS fine** (pool stores as uint8 — see
  M13a section; it dies on quality, not mechanics).
- cap-4 totals: 2k=190 MiB, 8k=786 MiB, 16k=1.5 GiB, 65k=6.2 GiB,
  128k=12.3 GiB (> whole 12 GB card with weights).
- Boot at --context-length 32768 --max-total-tokens 40960: allocator trims
  max_total_num_tokens to ~36.4k (~0.9 GiB KV); fits alongside weights with
  ~11.9 GB total GPU. ctx-32k arm script: launch_phase2_m11d5ctx.sh.
- Pool-size funding law: the requested `--max-total-tokens` CAPS the pool;
  with fp8 KV the same boot hit 40960 with 0.79 GB still free — the trim is
  a residual-memory floor after mamba/buffer costs, so halving KV bytes
  lets you ASK for proportionally more tokens, don't expect auto-doubling.

## Prefill scaling (graph arm, staging-bound)
- Cold 18k-token prefill: 439 s wall = ~41 tok/s (chunked 1024). Cold
  29.4k-token prefill: 496 s = ~59 tok/s. Long prompts are expensive:
  a 29k cold prefill costs ~8 minutes.
- Decode at 29.4k ctx (radix-warm, ignore_eos, 64 tok): 14.9-17.1 tok/s vs
  ~20 short-ctx = -16% only. Confirms flat attention IO; the small delta is
  indexer + paging + pool-walk overhead, not KV bandwidth.
- Radix cache hit makes re-sends of the same 18k prompt cheap (~5 s first,
  ~0.6 s after) — cached_tokens shows in meta_info.

## Long-context probe pitfalls
- Budget synthetic prompts in TOKENS, not words: marker/separator words can
  each cost 3+ tokens, so a word-count target silently overshoots ctx and the
  whole cold-prefill leg dies on an HTTP 400 whose body carries the exact
  "(X tokens) is longer than the model's context length (Y)" counts — read
  the error body, and measure true ptok with a 1-token round before paying a
  multi-minute cold prefill in a ladder.
- Highly repetitive synthetic prompts (same sentence x N) make the model
  emit a single EOS-ish token (ntok=1, empty text) — decode-rate probes
  silently measure nothing. Build prompts from MANY distinct sentences
  (varied word soup + unique markers) and pass ignore_eos in sampling_params.
- Client-side decode timing at long ctx: send prompt cold once (pays
  prefill), then re-send identical prompt with max_new_tokens=64 and
  ignore_eos; warm rounds = decode at ~ctx length with radix hit.
- This server arm logs no "Decode batch" lines; client wall-clock rounds are
  the measurement source. Reusable probe: scripts/probe_longctx.py (varied
  prompt builder + ignore_eos cold/warm decode rounds + A/B/A eviction
  pattern).

## M13a fp8_e5m2 KV — CLOSED NOT QUALITY-SAFE
- `--kv-cache-dtype fp8_e5m2` boots on sm_86 (pool store_dtype=uint8 because
  index_put lacks fp8 impls; 12.8 KiB/token). Mechanics need one shim patch:
  the HG_QSA_SDPA varlen fn must dequant float8_e5m2/e4m3 k/v rows to bf16
  before SDPA (Ampere SDPA rejects fp8 inputs) — patch_sdpa_fp8.py pattern.
- Gain measured: +7% decode at 29.4k ctx (15.9/17.1 vs 14.9/16.8 bf16).
- Quality FAILS the battery: QC6 emitted odd non-primes (91,93,95,97,99) —
  2 mantissa bits in e5m2 are too coarse for K/V values while other QCs
  pass. Tie-family probes (QC6-class enumerations) are the sensitive
  detector for KV dtype diet — but they also oscillate on ctx/pool-shifted
  arms at bf16, so confirm with a control boot without the change before
  blaming the dtype (the ctx-2048 canonical holds the probe exact).
- Remaining KV-diet candidates: custom int8 pool + per-token scales
  (extraction kernel already supports pool-dtype != out-dtype conversion —
  the gather is the natural dequant insertion point; scale-gather ordering
  = cu_k/indices order), or host-paged bf16 KV (below).

## HiCache host KV tier — the built-in host-paging path
- This fork SHIPS HiCache: host RAM tier for the radix KV tree. `--hicache-ratio` /
  `--hicache-size` only SIZE it — attaching requires `--enable-hierarchical-cache`
  (registry gates `init_hicache` on `ctx.enable_hierarchical_cache`); ratio flags
  alone are a silent no-op, confirmed by `hicache_attached=False` in the
  "Tree cache initialized" boot line. Host pool classes in
  mem_cache/memory_pool_host.py, auto-sizing in hicache_auto_size.py,
  unified_cache components include mamba.py and swa.py — hybrid models
  supported. Allocation proof line to grep: "Allocating kv hierarchical KV host
  pool: <N> tokens, <X> GB host memory". Flag grep law re-applies: config wiring
  lives in mem_cache/kv_cache_builder.py, NOT server_args.py — grep of
  server_args.py alone yields false negatives.
- Design caveat to verify per arm: ACTIVE in-flight sequences are typically
  pinned in the GPU pool — the host tier may only extend capacity for
  idle/cached prefixes (the agent-harness pattern: session KV idles between
  turns, reloads next turn), NOT for simultaneously-active 128k streams.
- VERIFIED via A/B/A eviction probe (scripts/probe_longctx.py): 27.4k-token
  prompt A cold-prefills (509 s), prompt B evicts A's radix pages from the GPU
  pool, re-sent A reloads in 1.3 s with cached_tokens ~= full prompt (~400x vs
  re-prefill). The host tier absorbs evicted prefix pages, GDN/mamba states
  included (boot line: "HiCache Mamba state transfer mode: sync torch
  fallback"). HiCache-arm fairness bursts ~18-19.5 tok/s, slightly under the
  canonical 20+ — per-eviction overhead on radix churn.
- QC discrimination law: QC6 oscillates on EVERY ctx-shifted arm (91-slip and
  think-leak both seen on hicache AND non-hicache 32k arms) — before
  attributing a QC6 miss to the arm under test, boot a control WITHOUT the
  change; HiCache restore is bit-identical copy, no dtype-style regression.
- Host RAM budget law still governs: the ~48 GB PLE mmap + pinned expert
  tier compete with any host KV tier in the same ~60 GB; 4x128k bf16 KV
  host-tier ~13.4 GB fits ONLY at reduced HTIER or with PLE diet active.
