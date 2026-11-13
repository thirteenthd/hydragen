# Phase 2 spec-decode laws (M12a/M12b, closed 2026-11-10)

## Outcome
- M12a stock EAGLE-MTP: functional on eager decode (IQ draft store + accept
  1.8-2.25), QC oscillates, not banked.
- M12b HYBSPEC (CustomSpecAlgo lookup, zero draft model/KV/forward): fully
  functional, PLE-compliant, speed-parity with eager non-spec, QC oscillates.
  CLOSED FUNCTIONAL, NOT WINNING on eager. Graphed verify was retried as
  M12c (2026-11-10) and CLOSED — capture completes but verify graphs emit
  decode-shaped rows; spec decode is terminally closed on this substrate.

## Registration laws (SGLang 0.5.23 fork)
- SpeculativeAlgorithm.register(name, supports_overlap=..., spec_class=...)
  decorator returns factory wrapper; factory returns worker CLASS.
- The scheduler overlap gate reads the supports_overlap ATTRIBUTE from the
  register kwarg; a method override on spec_class is shadowed by the
  instance attr — set the kwarg.
- CustomSpecAlgo lacks the enum-helper surface; the scheduler duck-types
  SpeculativeAlgorithm methods (carries_draft_hidden_states, create_future_map,
  need_topk, is_* family, pad_batch...) on the algo object. Mix them in from
  the enum class at install time.
- CaptureHiddenMode lives in sglang.srt.model_executor.forward_batch_info
  (no sglang.srt.enums in this fork).
- EagleDraftWorkerBase declares abstract stubs draft() and draft_extend();
  V2 flow calls draft + _draft_extend_for_decode/_prefill — stub draft_extend.
- ModelRunner has no .attn_backend attribute; getattr-guard.

## Verify-path laws
- topk=1 fast path: draft_tokens shape (bs, num_steps), num_draft_tokens ==
  steps+1 asserted, parent_list arange(-1,steps-2) x bs via
  _rebuild_topk1_chain_buffers, build_chain_tree consumes
  draft_input.bonus_tokens; first decode after prefill needs
  _draft_extend_for_prefill to return an EagleDraftInput with bonus/topk set
  (create_idle_input + fields works).
- QSA index-key ring holds ONE group: hard ceiling
  speculative_num_draft_tokens <= QSA compress ratio 4 (steps<=3). steps=3
  measured WORSE acceptance than steps=2.
- Graph+spec substrate: the fork's decode graph runner IS spec-aware by
  design — captured_req_width = num_draft_tokens when spec is on, and
  can_run_graph enforces a uniform per-request width (mismatch falls back
  to eager; ragged_verify_mode has its own path). Under spec, every target
  forward is verify-width, so HYBSPEC (all forwards width-3) replays on
  graphs captured at that width. The earlier shape_key.size==bs collision
  analysis was superseded — do not treat decode graphs as spec-hostile.
- CustomSpecAlgo graph-capture gap CLOSED by M12c experiment (2026-11-10,
  final): get_spec_info wrapper (EagleVerifyInput stub) makes capture
  complete; hidden-mode validator requires clamping BOTH si field AND
  batch-level forward_batch.capture_hidden_mode to NULL (validator compares
  the batch field). Then first real verify crashes eagle_sample
  "shape [1,3] invalid for input of size 1": captured verify graphs emit
  DECODE-shaped LPO rows (1 row) while sampler expects bs*draft. Fork
  capture-mode/LPO-row semantics; not shim-reachable. M12c CLOSED.
- Forced-eager-verify fallback is architecturally pointless: under spec,
  every target forward is verify-width, so can_run_graph=False on verify
  means NOTHING ever graphed — equal to the eager arm (M12b).
- Do not re-open graph+spec: three closure angles now fully mapped
  (backend key collision superseded; width-aware capture exists but
  produces decode-shaped verify rows; eager fallback gains nothing).
  Spec decode stays CLOSED NOT WINNING on this substrate.
- Dropping --enable-linear-replayssm-spec on eager+spec collapses
  _profile_available_bytes (rest_memory<=0): mamba spec extra-buffer sizing
  needs replayssm-spec funding. Keep replayssm-spec + fraction 0.95 on
  eager spec rungs.

## Lookup-decoder behavior
- Greedy self-history n-gram lookup only fires on self-repetition; prose
  accept ~0. Single ctx-3 key: accept_len ~1.0. Multi-length fallback
  (ctx 3..1, most-frequent continuation): best 1.45 len / 0.23 rate on
  forced loops; battery 0.04-0.16.
- Eager verify bookkeeping costs roughly one extra decode step per round;
  accept must exceed ~2/round to net a win — unreachable on non-looping
  traffic with a self-lookup decoder.
