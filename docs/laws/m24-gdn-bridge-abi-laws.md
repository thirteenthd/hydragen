# M24 GDN/QSA native-kernel bridge ABI laws (§90)

Laws pinned empirically on the reference host, 2026-11-12. Violating any one silently corrupts
state or explodes outputs — all cost hours to isolate.

## Build
- `build_gdn_kernels.sh`: nvcc `-O3 -use_fast_math -arch=sm_86 -shared -fPIC` over
  `strata/src/kernels/cuda/{gdn,native_gdn,native_qsa}.cu` + `bridge/hydragen_bridge_gdn.cu`.
  Needs only `-Istrata/include`; vendored ggml NOT required for these TUs.
- nvcc is not on the bare ssh PATH: use `/usr/local/cuda/bin/nvcc`.
- `GdnShapes` is passed by const-ref → ctypes needs the flat-ABI shim; do not ctypes the
  C++ signatures directly.

## Kernel contracts (headers under-document these)
1. State layout for BOTH gdn_step and native_gdn_step: `(S_i, h_v, S_j)` — k-dim OUTERMOST,
   head middle, v-dim fastest. SGLang pool is `(slots, HV, V, K)`: bridge permute (2,0,1).
2. `gate` arg is LOG-space; kernel computes `expf(gate)`.
3. `beta` arg is pre-sigmoid.
4. q/k l2norm is an UPSTREAM preprocess for both kernels (native_gdn.cu contains none).
5. Readout scale: ref path caller pre-scales q by `scale*K`; native applies 1/sqrt(S)
   internally (native_gdn.cu:118).
6. Head pairing: strata = cyclic `h % h_k`; SGLang = group `hv // (HV//HK)`. Adapter must
   pre-expand q/k rows to (HV, DK) and pass h_k = HV.
7. Native API throws std::invalid_argument on NULL stream; torch default stream handle is 0 —
   run ctypes calls on a dedicated `torch.cuda.Stream()` and set it current.
8. qsa rms_norm rows are row-major chunks of n_cols with gamma[n_cols]; gate_apply is
   head-major `(n_head, 2*head_dim)` [q|gate] rows.
9. gdn_conv_step conv_state is (C, d_conv-1) channel-major, kernel[0] taps OLDEST.

## SGLang parity target
- Decode GDN: `sglang.kernels.ops.attention.fla.fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode`
  via `gdn_backend.packed_decode` (scale=layer.head_k_dim**-0.5,
  use_qk_l2norm_in_kernel=True). Packed qkv row = [q|k|v]; pool updated IN PLACE
  (h0=ht=initial_state); out shape (B,1,HV,DV).
- `qwen4_exp.py` greps for gdn/gated_delta find nothing — the linear-attn path is wired
  through the hybrid backend, not the model file.

## Parity anchors
- Unit vs torch ref: WORST 1.19e-06 PASS (test_gdn_parity.py).
- 8-step decode vs live Triton: ref path 1.19e-07/4.8e-07, native path 5.96e-08/4.8e-07
  (test_gdn_triton_parity2.py). Any future kernel edit must re-pass these.
- Debug law: when parity fails, triangulate hand-ref vs BOTH sides at 1 step and check
  l2/scale/pairing consistency of the reference against the exact use_l2 flag passed.
