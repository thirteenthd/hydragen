# §91 M24 A1b — native fused GDN decode laws

- hg_gdn_decode_fused (bridge v2) is pool-native: reads/writes SGLang pool (slots,HV,V,K) directly, zero permute. Parity PASS all 8 dtype combos (fp32 ULP ~1e-7; bf16 out 9.8e-4 state 1.6e-2 = quantum).
- Hook = pyhook/hydragen_gdn_native.py, HG_GDN_NATIVE=1, sitecustomize-gated. Wraps TritonGDNKernel.packed_decode; capture-safe (current stream), 0 fallbacks at bs1+bs4 graph capture, quality 8/8. Keep on disk; NOT canonical.
- DECODE ANATOMY LAW (cn24 canonical, paired profiling): bs4 step ~60 ms = 48 layers x ~1.2 ms island apply (~58 ms). ALL non-MoE torch/triton (Triton GDN+conv+QSA FA+dense) = ~3-6 ms total. GDN/QSA/HC fusion-for-decode ceiling 1-2 ms -> NOT-ECONOMIC on staging-bound substrate (same death as §79b/§85). Do not re-attempt decode-side fusion until island apply share drops.
- Fairness across boot epochs is NOT comparable (41-47 vs 15-20 for same code). Only same-epoch paired hg-step fwd last20 comparisons are authoritative.
- nvcc/ctypes pitfalls: __forceinline__ not __forceinline; templates outside extern "C"; <<<>>> rejects implicit void* args — cast inside typed launcher helper; if/else macros need do{}while(0); __shfl_down full sum only in lane 0 — __shfl_sync broadcast before all-lane use; pkill -f over ssh self-matches — use bracket pattern "sglang[.]launch_server".
- Next lever per §89: A2 prefill expert-staging ring (extend island apply 16.8 s/chunk: h2d 10.2 + cs 5.2 + preadd 3.6).
