# M20d DFlash2 co-edit campaign (§84) — UNMEASURED, contaminated
- Gemini-owned trail on the reference host: DFlash2 block-diffusion draft (incoai/Qwen3.8-Flash-Next-DFlash2
  603M BF16, block 4) on q4e-fp8 + paged experts. Arms m20d2-d8 + launch_matrix_stub/plain/plain2.
- Never produced accept/throughput data. Walls seen:
  - draft funding: target load leaves 0.19 GB free; BF16 draft (1.21 GB) must be funded by
    CACHE_N 24->8 + cap-1 before pool sizing.
  - d2/d3 draft-build OOM; d5 intrinsic ValueError: replayssm-spec + DSPARK/DFLASH requires KDA
    model (Qwen GDN not KDA) — may be bypassable by dropping the flag; d6 flashinfer JIT curand.h;
    d7 best arm ended by orderly shutdown pre-measurement (likely an external restart).
- Contamination: concurrent agents on one GPU, no GPU-clear preamble in their launchers, plus
  user-forced engine restarts. Status UNKNOWN, not dead. Clean serialized re-test = d7 config + measure.
- Economics question open: §79b law says spec loses when per-row cost scales with staging; block-
  diffusion drafts width-4 in one shot which may change that math — worth ONE clean measurement.
- Co-edit law reinforced: every campaign arm must append a CHANGES.md section at run time.

## Clean re-test result (§85, 2026-11-11 late night)
- Mechanism PASS: accept len 3.33-4.00 of 4, rate up to 1.00, cuda graph True (HG_SPECGRAPH key-split
  works with DFLASH too). Draft quality is perfect under greedy; the limiter is row-scaled staging.
- Perf: paired A/B on CN=8/cap-1 diet substrate — plain 15.9-16.5 vs DFLASH 13.9-15.1 deep-warm
  => 0.92x, NEGATIVE. 4-row verify ~3.6x a 1-row step + draft forward. §79b law holds with a perfect draft.
- NOT adopted; canonical stays cn24c. DFlash2 re-opens only when per-row staging cost flattens.
- curand JIT landmine fix (Gemini): ~/curand-pkg + CPATH/NVCC_PREPEND_FLAGS/LD_LIBRARY_PATH exports.
