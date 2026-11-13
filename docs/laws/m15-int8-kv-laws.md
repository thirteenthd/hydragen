# M15 int8 KV pool laws

## Ship design
- HG_KV_INT8=1 + --kv-cache-dtype fp8_e5m2 (allocation trick only): 1-byte pool bytes reinterpreted as int8, per-(slot,head) fp32 scale side tables.
- Quality-safe 1B KV tier: QC6 direct primes 97/101/103/107/109, QC7 correct, QC0 clean. fp8_e5m2 fails QC6 with composites; int8 does not.
- No decode penalty: fairness warm burst 20.59 tok/s cap-4 agg (graph band 20.07-20.29).

## Shim integration points (pyhook/hydragen_kv_int8.py)
- Write: wrap MHATokenToKVPool.set_kv_buffer (model sends bf16; intercept before pool cast). k_scale/v_scale args exist but unused by model.
- Read: _get_key_buffer/_get_value_buffer return .view(torch.int8) — gather kernels then carry integer values into bf16 scratch; dequant downstream.
- Prefill-with-prefix: wrap pack_qsa_prefill_kv; row order = b-major positions 0..seq_len; scales gathered via req_to_token[req_indices, j] masked by cu_k lens (eager path, mask OK).
- Decode: wrap qwen_sparse_kv_extraction_compact_triton; compact row order = front-contiguous valid entries (expand_qsa_block_indices sorts); stash scale-row scratch in side channel; chain-wrap _resolve_flash_attn_varlen_func to dequant.
- QSA index keys live in a separate pool path (set_index_k_scale_buffer) — untouched by KV hooks.
- move_kv_cache (mamba extra_buffer relocation) must move scale rows with payload.

## CUDA graph capture laws
- Boolean-mask indexing (t[mask]) forces D2H sync => cudaErrorStreamCaptureUnsupported inside capture. Use per-batch index_copy_ scatters (static shapes).
- Preallocate side tables in pool __init__ wrapper; persistent per-buffer scratch dicts; no fresh allocations inside captured islands.
- Extraction garbage-row law: each b writes full [cu_k[b], cu_k[b]+topk) range in b order; b+1 overwrites b's tail; final tail beyond total_valid never read.

## Funding math (3080 Ti 12GB, dense-Q5_K substrate)
- bf16 pool ceiling at ctx32768: max_total_num_tokens 36416. fp8_e5m2 pool: 40960 @0.79G free. int8: 65536 boots @0.49G free, GPU 11674 MiB.
- Residency proven: 2x~24k-token sessions simultaneously active (48k tokens, pool usage 0.73). Impossible on bf16 ceiling.
- 3rd concurrent 24k > pool -> expected LRU radix eviction (cached=0 re-prefill).

## Limits
- Bare HG_KV_INT8 + --enable-hierarchical-cache is UNSAFE: stock host transfers move payload bytes only; scale tables must be made to ride (see M16 below) or reloaded sessions dequant with stale scales.
- Pool bytes identical to e5m2 tier; the win is quality-floor permitting 1B storage vs bf16 2B.

## M16 HiCache + int8 scale-transfer surface (sglang 0.5.23 fork)
- HiCache transfer verbs live in `srt/mem_cache/pool_host/` — NOT cache_controller.py. `pool_host/base.py` HostKVCache: `self.size` = host token capacity, `page_num` derived. Concrete MHA class: `pool_host/mha.py` MHATokenToKVPoolHost with `backup_from_device_all_layer(device_pool, host_indices, device_indices, io_backend)` (whole-pool D->H) and `load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, ...)` (H->D per layer). Wrap these two class methods to carry custom side tables: backup copies scale rows D->H into a per-layer pinned host mirror (sized by self.size — trivial bytes: 12 layers x 205k x heads x fp32 ~= tens of MB), load restores the layer's scale rows after orig, before attention reads.
- Wire the host-class patch through the same meta-path finder that patches the attention backend: match module name `sglang.srt.mem_cache.pool_host.mha` in find_spec, exec_module wrapper, plus eager sweep of sys.modules (module may already be imported by sibling shims).
- The int8 payload also shrinks the HOST tier bytes/token: `--hicache-size 8` allocated 205,376 tokens / 2.52 GB on the int8 arm vs 149,248 / 3.67 GB bf16 (same GB request, boot log line) — ~38% more parked capacity free from the KV diet.
- Hybrid GDN model KV writes reach the inner MHA pool via HybridLinearKVPool.set_kv_buffer -> full_kv_pool.set_kv_buffer (memory_pool.py ~4560): pool-level hooks see real writes; prove it, do not assume it (see proof law below).
- Status as of banking: scale-ride patch written and booted (write path ACTIVE + host pool 205,376 tok confirmed); cold/re-hit correctness gate pending — do not claim M16 PASS until a re-hit session decodes coherently AND backup/LOAD ride lines appear in the log.

## Proof-of-life law for wrapper shims (cost a whole campaign of doubt)
- Every wrapped hot path must carry a one-shot print on first real call (e.g. `[hg-kvint8] write path ACTIVE layer N`, `host BACKUP/LOAD scales ride ...`). A passing QC battery does NOT prove your writer ran — the read views can serve bytes written by the STOCK cast path and quality still looks fine until the scales diverge.
- After ANY multi-region patch of a shim function, re-read the ENTIRE patched function before deploying — a fuzzy string patch can silently delete variable-extraction lines above the changed block; the surviving NameError only fires at first request inside the scheduler and kills it after a full launch cycle. py_compile does NOT catch NameError.

## Ops law (recurring trap)
- pkill -f patterns self-match the ssh command line when the same bare string appears elsewhere in the command (heredocs/filenames). Run kills in dedicated calls or use the quote-split pattern like "sgla""ng".
