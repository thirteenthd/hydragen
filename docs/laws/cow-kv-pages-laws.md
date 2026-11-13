# CoW KV-pages campaign laws (M-C0+)

## M-C0 determinism spike (v0.2.13, closed 2026-11-12)
- Same prompt, different --prefill chunk plans => KV, GDN and PLE state bytes ALL differ (STRATA_STATE_HASH fingerprints kv=/gdn=/ple=). Hidden states themselves are chunk-plan-dependent; adaptive-tier residency swaps during long reads add more run-to-run variance.
- Content-addressed dedup of independently prefilled prefixes: IMPOSSIBLE. Do not design around hash-matching copies.
- CoW is share-at-fork: first session to prefill a token prefix owns pages; later same-prefix sessions REFERENCE them. This extends the engine's EXISTING token-keyed prefix-reuse trust (22528-of-22535 reused across sessions; program-wide QC green), it does not add a numerical assumption.
- Sessions that arrive cold-parallel (no first-mover) keep private copies; farm-shape (shared pinned system prompt) is the sharing case.

## Engine mapping facts (v0.2.13)
- STRATA_STATE_HASH=1 fingerprints every finished request over cells [0,L): gdn/ple/tail/pooled/kv/mtp/stale/dead FNV-64. Gate needs prompt_cache>0 (default 6). STRATA_STATE_HASH_GDN=1 adds per-GDN-layer hashes.
- KV pools are ALREADY PAGED: layout (page*n_head_kv + head) * page_size cells; hash_cells walks pages. Ref-counting is an addition on top, not a paging restructure.
- BUT canonical KV (--kv-resident 20480) runs MODE-1 STREAMING (p.mode==1): promotion/eviction happens ON-DEVICE via the kv_stream map (slot_block/slot_stamp/slot_ref; slot_ref refcounts already exist for eviction pinning). Mode-1 reads resolve via the device map, NOT the host page_table. Host-side fork fast-path alone covers only mode-0 (fully-resident) or mode-2 rings (MTP drafter). True cross-session sharing in the canonical shape needs a device-side global frame allocator + atomic free-list in the kv_stream device functions.
- PERM gate (cow-m1): STRATA_COW_PERMUTE=1 permutes mode-0 identity tables at allocation (layer.cpp ~968); battery green vs control = every mode-0 kernel path honors page_table, zero kernel change needed for paged-mode CoW.
- Elastic VMM-backed shared K/V pools exist (g_kv_elastic, g_pools, --kv-grow path) but are NOT on the canonical path.
- Allocation chain: session_init(g,...,take-lambda) -> s.qsa_state_arena -> qsa_state_init(g, max_cells, base, st, borrow) carves per-session pools; session_bytes() prices arenas; per-session cudaMalloc (generate.cpp ~6764).
- The hybrid-model serve loop (and STATE_HASH block) lives in src/program/generate.cpp (~19544-19830). The serve printfs inside src/qwen35/q36_engine.cu are a different standalone-serve variant that never runs under serve/server.py — do not patch there.
- serve/server.py does NOT forward shell env to the engine: engine env ONLY via config "env" dict.
- QSA state type: strata::core::QsaState; shapes via strata::core::qsa_shapes(g); conv_state_sizes for GDN/PLE sizes.

## Runner law (bit three sessions running)
- pkill -f self-match: bracket-escaping the pattern is NOT enough when the SAME command line legitimately contains the target script's literal filename (e.g. `bash /tmp/m25iso5.sh` later in the line matches pattern m25iso[5]). kill and launch must be SEPARATE ssh calls, or the whole sequence lives in a remote script file.

## Budget arithmetic being attacked (from §98/§100)
- Per live session VRAM: ~0.55 GiB at ctx 131k (kv-resident 20480 fp32 window dominates); batch 4 costs 793 expert slots (2456->1663).
- Shared-prefix fork target: ~0.05 GiB extra per session instead of 0.36-0.55; parked host snapshots (~400 MB @ 22.5k) get the same refcount treatment later (M-C2).
