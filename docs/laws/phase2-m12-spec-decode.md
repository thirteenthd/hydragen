# M12 spec decode — EAGLE-MTP on the native IQ substrate (OPEN, eager arm PASS)\n
Status (2026-11-10): stock EAGLE-MTP (M12a) boots and serves on the EAGER decode path
(`launch_phase2_m11d5sp.sh`, `--cuda-graph-backend-decode disabled`) — spec burst
15.7-21.3 tok/s agg cap-4 (spread across boots; vs 12.4 eager non-spec, ~20.3 graph
non-spec), WARM 1.2-3.2 s. Draft quality is REAL: accept len 1.80-2.25, accept rate
0.39-0.62 (sglang decode log). QC battery after the varlen causal fix below matches the
banked non-spec quirk profile (QC7 '>' correct x2 runs, QC6 primes correct, known
few-shot quirks only) — the earlier QC7 'F'/QC6-garbage collapses are GONE and were
NOT PLE history; see root cause under OPEN walls. Prefix battery leg OOM'd (funding,
not correctness): PLE short_conv wanted 20 MiB vs 18.8 free with 331 MiB stranded in
PyTorch reserve fragmentation. LAW: with graphs FULLY OFF (eager+spec arm) set
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — safe and recovers the stranded
reserve; the graph arms still need :False.

## Draft-worker wiring that WORKS (keep as recipe)
- Draft checkpoint: built from Strata sidecar pack tensors (`mtp-manifest.json` per-tensor
equals BF16 originals via numpy.fromfile + torch.frombuffer), SKIPPING every `mlp.experts`
tensor; experts ride a SECOND native IQ blob store (`HG_IQ_BLOB_DIR_MTP`
equals draft blobs, draft layer keyed by NEGATIVE layer id so it cannot collide with
target layer 0; per-store tier cap HG_MOE_HTIER_MTP).
- `load_weights` hook fakes verification for params matching mlp.experts / shared_head /
equals embed_tokens; EAGLE worker shares embed+head via `set_embed_and_head` — copy the
equals head-shim attrs (at least `_hg_host_embed`) onto the draft lm_head so draft logits ride
equals the same IQ head path (lm_head IQ wire shared: 0.41 GB not 2x).
- Draft config MUST preserve the original top-level config WRAPPER (language_model_only:
equals true, overrides NESTED under text_config: num_hidden_layers 1, ple_layer_ids [],
equals layer_types ["full_attention"]). A flattened config parses in plain AutoConfig but
equals dies in sglang get_config (mm_disabled_models path).

## Funding walls hit and their fixes (all real, all silent-ish)
1. sglang sizes a persistent fp32 (rows, vocab) SHARED LOGITS buffer via
equals `ModelRunner.max_shared_logits_buffer_rows()` = max capture_bs x num_draft_tokens;
equals under EAGLE that reached ~1664 rows x 152k vocab x 4B ~= 970 MiB -> capture OOM.
equals Fix: monkeypatch max_shared_logits_buffer_rows -> cap_bs x num_draft_tokens
equals (a few dozen rows). Opt-in gate on HG_MTP.
2. dense-IQ scratch `_ensure` was pre-sized 1024 rows x n_out fp32; draft-extend arrives
equals with T ~= 1664 rows (num_steps x rows) -> 970 MiB grow -> same OOM line. Fix: CHUNK
equals the IQ fwd into row blocks (HG_DENSE_IQ_YROWS, default 256): scratch never grows,
equals also graph-safe (no post-capture realloc). Chunked fwd is numerically identical
3. qwen4_exp has FIVE islands (_breakable_prepare_ple_batch, _ple_prefetch,
equals _qsa_indexer, _ple_forward, _commit_ple_batch) — spec-verify graph warmup runs them
equals with NO piecewise forward context; EACH needs a ctx-guarded no-op/idle fallback or
equals the scheduler dies at boot with `NoneType.forward_batch` (one guard at a time costs
equals a full ~20-min boot cycle — guard all five in one pass).
4. The fork's QSA backend genuinely CONSUMES topk_indices (our shim only replaced the
equals varlen kernel — a zero-bridge fallback silently attends token 0). Warmup-with-no-ctx
equals must run the eager indexer with the CALLER's real forward_batch and register the
result into `_qsa_prefill_topk_bridges` (the fork's capture assert demands a pre-allocated
equals bridge; replay then recomputes live via the island).

## OPEN walls (do not present as solved)
- breakable-graph decode + EAGLE verify crashes at eagle_sample
`target_predict.reshape(bs, draft_token_num)`: verify got 1 logit row instead of
bs x draft_tokens. ROOT-CAUSED (HG_BCG_TRACE=1 trace in pyhook/hydragen_bcgraph.py):
the breakable decode backend's replay plans are DECODE-SHAPED — per-graph size/cap it
passes to _alloc/_output_rows/_slice is bs (1, 4), never bs x num_draft (3, 12), even
though the verify graph CAPTURES with 3/12-row LPOs. Also max_shared_logits_buffer_rows
reads 4 at capture time in the scheduler subproc while eager boot computed 1664 — the
spec bag is timing-sensitive at capture. bcgraph _output_rows fixed to
max(full_logits, next_token_logits) rows (verify LPO keeps next_token_logits at bs rows;
full_logits carries bs x draft) but replay still truncates: fixing fully needs
fork-level capture-plan spec-awareness. DEFERRED; eager+spec is the working rung.
- Spec-arm quality collapse ROOT-CAUSED AND FIXED (2026-11-10): the HG_QSA_SDPA varlen
SDPA shim ran is_causal=False with NO mask — legal only for q==1 blocks (decode). For
verify blocks (q=3, kv=prefill+verify) each verify token attended FUTURE draft tokens:
verify logits poisoned -> echo loops + fact slips (QC7 'F', QC6 garbage) in the spec
arm only, while WARM decode stayed clean. Fix (patch_varlen_causal.py in
pyhook/hydragen_paged_store.py): q==1 -> no mask; q==kv span -> is_causal=True;
1<q<kv -> explicit end-aligned mask (tril suffix). LAW: ragged varlen blocks align the
query span to the END of the kv span; is_causal=True encodes START alignment only — it
is wrong for any q<kv block. The earlier 'PLE history retains rejected-verify rows'
hypothesis was NOT the cause (diet gather is rollback-safe by construction, no
scheduler-side ring state), but auditing reject rollback in every per-position state
remains sound practice for PLE-hybrid spec arms.
- Acceptance rate is the metric for draft-store correctness, never draft logit quality
(rejection sampling preserves the distribution); a draft store serving garbage still
passes output-quality gates.

## Launch essentials (eager spec arm)
`--speculative-algorithm EAGLE --speculative-draft-model-path <draft> --speculative-num-steps 2
equals --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --enable-linear-replayssm-spec`
topk=1 keeps the PLE chain causal (tree branches have no valid causal PLE gather); EAGLE/MTP
equals identity passes the PLE NGRAM gate legitimately — do not touch the gate itself.
