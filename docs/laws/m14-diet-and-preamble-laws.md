# M14 dense-IQ diet tiers + radix preamble laws (2026-11-11)

## Dense IQ diet verdicts (installed Oct-31 libstrata_iq_kernels.so)
- fmt codes (strata numbering): 12=Q4_K(144B) 13=Q5_K(176B) 14=Q6_K(210B) 21=IQ3_S(110B) 23=IQ4_XS(136B). IQ2_K does not exist as a ggml type. IQ2_S(22) dispatches but old kernel mis-anchors rows (nan). IQ4_NL: fmt29 garbage, fmt27 unsupported.
- GPU mmvq micro-cos: IQ3_S 0.9934, IQ4_XS 0.9976. IQ quantizers need ggml_quantize_init(fmt) before use (GGML_ASSERT kgrid else).
- Per-module override: HG_DENSE_IQ_FMT_MAP="substr:fmt" in hydragen_dense_iq.py (opt-in). _quant returns 5-tuple; second call site = hydragen_paged_store.py lm_head island (HG_HEAD_IQ=1). Keep tuple signatures uniform.
- Same-day face-off: IQ4_XS dense-wide: wire 1.46G GPU 10306MiB, warm bursts +30-45% — but QC7 greedy knowledge miss ('FASTA starts with a semicolon') + QC6 think-shift -> NOT QUALITY-SAFE. IQ3_S se-only: -7% burst, QC6 tie-shift -> not banked. Dense wire floor stays Q5_K canonical.
- Ampere IQ dequant cost law: IQ4_XS nibble grids cheaper than Q5_K; IQ3_S ternary/quinary expensive. Diet is a VRAM-for-speed trade, not free bits.

## Battery oracle laws (M14)
- Fairness FIRST run after boot reads cold wire cache: bursts ramp 7->12->18 on identical canonical; compare second+ runs only.
- Boot-drift noise appears even on canonical same day: WARM echo drift (Berlin/Madrid), QC0 arithmetic slip, QC1 think-empty. Arm discriminators: QC6 prime-list directness, QC7 GTIN/FASTA exactness, QC0 first clause.

## Radix preamble dedup (closed)
- Stock radix longest-prefix dedups byte-identical preambles across sessions: 1574-tok preamble, 4 divergent sessions -> sessions 2-4 cached=1536 (96%, page-64 granularity), first-turn wall 67.5s -> ~5s (~13x).
- Launch-time pre-prefill (one warmup request of exact preamble after READY, one per variant) makes session #1 a hit too (4.9-6.5s). Cost: one prefill per variant, ~40MB pool.
- Harness rule: volatile content (timestamps, per-session blocks) must come AFTER preamble; any early divergence kills the shared prefix.
- No segment dedup of mid/non-prompt-prefix regions in stock radix (exact prefix from token 0 only); that is unified_radix_cache fork surgery.
- Preamble prefill rides the same expert staging: domain-matched preset halves it (M13d law).
