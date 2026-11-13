# Prompt-path arms (prefill/TTFT is the agent's real wall: turns run 36k-144k-token prompts)

Prod env already carries `STRATA_PF_FUSED=1` (log: `prompt experts on the fused int8 kernels
(STRATA_PF_FUSED=1, #136)`), `STRATA_PREFILL_CPU_SHARE=auto`-ish defaults, `STRATA_PREFILL_EQUAL=1`,
`STRATA_ROUTE_PRIOR=0.85`. Prompt fusing is therefore NOT a new win.

## Authoritative measured probe

`probe_prompt.py` (scratch dir): streaming request, cold prompt built by concatenating repo source files in
sorted order (rotate the target size per rep to defeat `--prompt-cache-tail`),
prompt tok/s = prompt_tokens / time-to-first-token. The engine's own accounting line is the trustworthy
number, read it from the log:
`strata serve: prompt 37859 tokens = 0 reused + 37859 read in 17878 ms (2117.6 tok/s), ...`
Cold-prompt reference on this 3090 (IQ3_S, batch1, ec5800): **2117.6 tok/s cold read at 37.9k tokens**,
1857 tok/s on a 13k-token marginal read.

## Switch facts from upstream docs (verified wording)

- `STRATA_PF_FUSED_NT=256` — fused prompt-expert kernels with 256 threads / 2 blocks per SM, bitwise, adds
  device code to every CUDA build. Reported one MoE layer of 3,256 tokens at 8.27 ms on a CMP 170HX.
- `STRATA_PF_BATCH=N` — up to N experts per launch (only >128 with the 256-thread kernels), bitwise.
  With NT=256 the reporter saw prompts +7.6% (4K) / +4.9% (16K) / +4.3% (52K). sm_80 measurements, unverified
  on sm_86.
- `STRATA_PF_FUSED_F16=1` (rounding-level), `STRATA_PF_FUSED_WS=1` (needs ~101 KB shared per block; docs say
  sm_80/sm_90 only — sm_86 caps at 100 KB, so expect "not available" on the 3090), `STRATA_GDN_TC=1`
  (122 KB shared per block — out of reach on sm_86 for the same reason).
- `STRATA_MMQ_DIRECT=1` is **not env-only**: it needs a build with `-DSTRATA_MMQ_TAB=ON`, off by default
  because it costs 36-38 MiB VRAM per GPU. Our shipped binary refuses it outright:
  `STRATA_MMQ_DIRECT=1 is ignored: this build has no pointer-table MMQ (CUDA with STRATA_MMQ_TAB only)`.
  This is a source-build lever, not an env lever.
- `STRATA_PLE_COMPACT=1`, `STRATA_PLE_COPY_STREAM=1`, `STRATA_EMBED_VRAM=1`, `STRATA_DEQUANT_SIDE=1` — PLE
  rows go up raw and dequantize on-card on their own stream, embedding table in VRAM, dense dequant on a side
  stream; all bitwise; +2.2 to +2.6% prompts for the four together. Caution on this card: EMBED_VRAM competes
  with the expert cache for the last ~1 GiB.
- `STRATA_EMB_REUSE_ACCOUNT=1` — prompt ~+3.9% on an RTX 3060, changes prompt rounding, no KL published.
- `STRATA_PF_GROUP_SCRATCH=1` / `STRATA_PF_HC_REGION=1` — free 0.3-0.5 GiB of borrowed cache slots, but only
  on the prompt path **on MMQ, not the fused int8 experts** (our path is fused int8, so expect an ignored
  line). Their speed gain only appeared with `STRATA_PREFILL_CPU_SHARE=0`.
- `STRATA_OWNED_PRICE=exact` — the planner prices the prompt path's own buffers by real allocation; bitwise;
  changes how many expert slots the prompt path borrows. Our engine logs
  `the prompt path borrows 2138 CUDA0 cache slots (4.03 GiB)` every start — a residency lever worth one arm.
- Prompt-lookup drafts: upstream's documented agent win is `--lookup-max-t N` (+11 to +14% on file rewrites
  and agent edit turns; on by default at 12 in the 8-row single-GPU serve build; `STRATA_LOOKUP_MAX_T=0`
  disables). Our 0.1.54 help exposes only `--lookup-chain K` (default 0 = off) plus `--lookup-chain-min M`,
  and the log already shows `suffix drafts: 1 windows, 3 of 3 drafts accepted` — verify which knob this build
  actually honours before arming it.

## Arm log

| Arm | cold prompt tok/s (37.9k) | verdict |
| --- | ---: | --- |
| prod defaults (PF_FUSED only) | 2117.6 | reference |
| `PF_BATCH=16` + `PF_FUSED_NT=256` + `MMQ_DIRECT` (ignored) | 2117.6 first rep at 37,859 read | needs baseline A/B before judging; MMQ_DIRECT provably inert |
