# M10 Option-B donor splice (q4e-iq)

## Donor GGUF laws (llama.cpp fork on the reference host)
- Split GGUF: each shard has its OWN tensor table with PER-SHARD data offsets.
  IQ3_XXS donor: shard1 = 1223 tensors tiling 47.03 GB; shard2 = 1 tensor
  (per_layer_token_embd, 28.80 GB, local offset 0). A merged global-offset
  dict reads expert bytes as garbage (NaN / huge-value dequant).
- This donor's GDN tensors (attn_qkv, attn_gate, ssm_out) are ALREADY in HF
  row order. _LinearAttentionVReorderBase was NOT applied at conversion.
  Never reorder; earlier "0.48/0.045 cos mismatch" was the shard-offset bug.
- Fork enum renumber: 14=q6_K, 21=iq3_s, 23=iq4_xs, 42=q2_0. Query
  ggml_type_size/ggml_blck_size at runtime via libggml, never hardcode.
- iq4_nl rows: ggml_row_size(nx)=nx/32*18 (90B for nx=160) works with
  dequantize_row_iq4_nl at same stride.
- GGUF string KV skip: read u64 len, advance len (do not add 8 twice).

## Builder m10_build_v2.py
- Replaces every non-expert tensor whose donor GGUF twin is quantized
  (dequant via dq_shim2.so -> bf16 bytes); float-class HF bytes kept;
  expert+ngram shards hardlinked (ngram served fp8 from PLE file dir).
- DONE replaced_tensors=302 -> /mnt/docker/hg/q4e-iq.
- Verify law: rebuilt tensor cos vs donor raw = 0.999999 (not vs fp8 ckpt).
- Sanity bars: attn 0.997+, GDN 0.99, shared 0.987+, embd 0.987, float-class 1.0.
- Tensor-name suffix matching: use `name.endswith("attn_gate.weight")`, NEVER
  `split(".")[-1]` (returns bare "weight") — silently no-ops the transform and
  the sanity cos then looks like a layout bug.
- Misc-file copy loop over a checkpoint dir: skip non-file entries (`.cache/`
  raises IsADirectoryError mid-build).
- When a compound ssh command was blocked/denied mid-chain, the scp chained
  before it did NOT run — after user approval, re-scp and grep the remote file
  before rerunning; identical repeat error text = stale remote copy.

## Serving
- launch_phase2_m10.sh = native m9 env + model-path q4e-iq.
- Detached remote daemons over ssh: `cd dir && setsid nohup cmd > log 2>&1 < /dev/null &`
  — plain `nohup ... &` keeps the ssh channel open (call hangs/times out).
- First boot of a new checkpoint dir is FULL (~1000s): loader-skip stamp is
  checkpoint-identity keyed; arm skip for the new dir via --make/--check
  (see phase2-loader-skip.md) once exp/ple classes are verified byte-identical.
- battery_fairness.py remains the quality oracle.

## Banked M10 result
- Whole-model IQ3_XXS donor splice (donor dense+GDN+shared+embd+lm_head, fp8
  ngram kept) is tensor-correct (cos 0.999999 vs donor) but battery STILL FAILS:
  output collapses to repeated '!' — lm_head/logit-collapse signature.
  Donor-context calibration compensation is NOT the whole story; expert-tier
  IQ3_XXS fidelity remains the binding limit.
- Native IQ throughput on donor dense context: 14.5-17.3 agg tok/s B4 burst
  (vs 4.99-6.90 on FP8-context IQ splice, 4.32 FP8 cap2).
