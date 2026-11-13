# M19 graphed-spec revisit laws (updates M12c door)

## KEY-SPLIT LAW (M12c door is open without fork surgery)
BreakableCudaGraphBackend keys graphs by ShapeKey.size only; static-mode
DecodeCudaGraphRunner._capture_graph_size returns bs for BOTH decode (bs rows)
and spec verify (bs*num_draft rows) -> verify graph overwrites the decode graph
at the same key slot (M12c crash: eagle_sample reshape of 1-row logits).
Fix: override _capture_graph_size to return num_tokens (hydragen_specgraph.py,
HG_SPECGRAPH=1, chain-import hook on
sglang.srt.model_executor.runner.decode_cuda_graph_runner). Decode runner has
captured_req_width=1 (key=bs unchanged); verify runner width=num_draft
(key=bs*width, stable per replay since replay recomputes
bs*self.captured_req_width). Verified: 'cuda graph: True' on spec decode
batches, accept len 1.95-2.25, no exceptions.

## Ragged COMPACT is dead for EAGLE+GDN
SGLANG_RAGGED_VERIFY_MODE=compact gates on supports_ragged_verify()
(DSPARK/DFLASH KDA fold-every-commit family only). With
--enable-linear-replayssm-spec the guard is semantic: GDN materializes state,
KDA-family only can ring-write verify. Dropping replayssm-spec instead trips
mamba-spec buffer sizing (rest_memory<=0). Do not pursue compact for EAGLE/HYBSPEC here.

## ECONOMICS LAW (why spec stays dead on this substrate)
Staging-bound decode: unique-expert count ~ width x topk per step and each
unique expert stages ~288KB, so verify(3 rows)+2 draft ~ 3x step cost while
accept_len <= 2.25. Face-off ctx2048: bs=1 spec 13.0-14.6 vs non-spec 20.0
tok/s; cap-4 warm spec 13.6 vs non-spec 27-35 tok/s. Spec only wins when
per-row cost is flat (launch-bound or GEMM-bound substrates). Graphing verify
removes launch overhead but cannot flatten the staging slope.

## Stock DFLASH2 arm — DEAD on GDN targets (same KDA wall, from boot-side too)
The stock DFLASH/DSPARK algorithm rejects non-KDA targets outright:
`ValueError: --enable-linear-replayssm-spec with DSPARK/DFLASH requires a KDA model` at
boot after the draft runner reports ready — GDN (qwen4_exp) is not KDA. Consistent with
the ragged-compact gate above; do not re-attempt stock DFLASH2 on this model family.
Funding/footprint laws if any spec draft worker is ever attempted on this card:
- The draft worker's weights are allocated BEFORE KV-pool sizing — fund them by dieting
  the device slot cache (HG_MOE_CACHE_N) and running cap-1 first; the target alone
  leaves ~0.2 GB free, so a >1 GB BF16 draft has nowhere to land otherwise.
- Spec Triton kernels lazy device-load AFTER serving starts and squeeze a zero-headroom
  GPU (free mem drops toward 0.5 GiB, then OOM). Expect pre-load-at-init or keep margin.
- A 603M-class BF16 draft costs ~1.2 GB; draft model __init__ in srt/models/dflash.py
  can also crash before any of this — check the traceback class in the arm log first.

## Files
- pyhook/hydragen_specgraph.py (HG_SPECGRAPH=1)
- pyhook/hydragen_ragged_verify.py (HG_RAGGED=1; kept, compact path unused)
- launch_phase2_m11d5spm4.sh (graphed EAGLE-MTP arm), spm2/spm3 (compact attempts)
