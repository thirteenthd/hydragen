# M24 Phase A2 affinity-staging laws (§92)

Opt-in HG_EXT_AFF=1 in island; snapshot .a2-working; launcher launch_phase2_m24a2.sh.

Results banked: extend window apply 1187->~517ms prof, staging ops 4416->~420 (10.4x), TTFT steady-tail -20%, cold-first -53%, QC 8/8, prime oracle exact, fairness canonical band.

Kernel-contract laws (native_expert_grouped):
1. ent_dst = GLOBAL row index into out (tokpos*K + slot j), not per-token j. Bare j aliases rows 0..K-1 -> empty/garbage outputs + downstream qsa_indexer illegal access.
2. cap_groups/cap_entries must equal ACTUAL nul/E of the call. grp_start array has nul+1 entries. Oversized cap = OOB group bounds (CAPS_FIX reverted).
3. Entry arrays need contiguous CUDA int32; ent_tok < t1-t0 rows of the call.
4. Any xq8/scr grow that REPLACES st buffers must keep the old ones alive until in-flight _ng drains — pre-size once per window (not per batch) and store in st, or stash old tensors in locals.
5. DBG arms that run affinity then re-run slice on same res DOUBLE-COUNT (rel=0.5/rel2=1.5/rel4=3.5 signature means results are IDENTICAL, not halved — the delta equals the duplicated slice contribution).
6. Batch caps must be env-knobable (HG_AFF_TOK/PAIR/EX) — bisect with tiny caps (8/32/4) at T<=30 windows (decode-sized) vs long windows.

Design laws:
- Affinity batches: experts<=_CACHE_N, tokens<=256, pairs<=512 per apply; fresh xf/out per apply (persistent cross-apply buffers poison — §87 law).
- Decode path (T<=TCHUNK) untouched — graphs safe; affinity only for extend windows.
- combine: res.index_add_(0, _tkg.repeat_interleave(K), out*w_repeated) replaces view-sum.
- coverage counters (aff_E/aff_pairs vs slice path norms) beat max-diff forensics for correctness proof.
