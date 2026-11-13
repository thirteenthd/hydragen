# Upstream Strata is open source — the lever menu it names

Repo `Niko1221/Strata`, MIT, branch `main`. Signed distribution (`bin/strata.exe` + ed25519 manifest)
does NOT mean closed source. Our prod engine component reports `0.1.54`; upstream main is `0.2.5+`.
`thirteenthd/hydragen` is NOT a GitHub fork of Strata (fork: false, parent: none) — our C++
(`csrc/core/native_*`, `csrc/ipc/strata_fast_ipc`, `csrc/paged_attention`, `cuda/hydragen_bridge`) is ours.

## The measured wall is upstream's documented mechanism

`docs/EXPERTDEC.md`: `native_expert_grouped` lives in `src/kernels/cuda/iq_kernels.cu`. Group = one expert
blob; its entries = the window rows that picked it (1..T). The default gate/up kernel re-reads each entry's
q8_1 block from global memory (9 scattered 4-byte loads), so **an extra entry costs ~60% of a whole expert**
(1.6 us vs ~2.6 us per expert+layer) while reading nothing from DRAM. That is the same effect our microbench
found (14-16 us/expert flat vs ~180 GB/s effective on the hit path, ~1.7 rows/blob at N=8).

## Bitwise fix already compiled into our own binary

`STRATA_EXPDEC=1` (opt-in on CUDA; default on gfx12). Full knob family present in `bin/strata.exe`:
`STRATA_EXPDEC_R/_RB/_MINB/_MINBLK/_DR/_DRB/_DOWN/_D20`; `expdec_gu_kernel` / `expdec_down_kernel`
template instantiations are in the binary. Formats covered at n_embd 2560: IQ2_XXS, IQ2_XS, IQ3_XXS,
IQ3_S, IQ2_S (our prod pack is IQ3_S). Output is bit-for-bit identical to the default kernels.

Upstream kernel-level: gate/up -11/-20/-22/-23% at T=1/2/3/4; per-row slope +0.86 -> +0.60 ms; per-entry
1.26 -> 0.64 us (2080 Ti IQ3_S). Their end-to-end was only +1.1% because expert kernels were ~20% of the
critical stage on a 2-GPU layer split — on a single 3090 with 158 slots/layer the expert hit path IS the
critical stage, so the local upside can be much larger.

Counter-evidence, same doc: upstream lists `STRATA_EXPDEC` among "switches that did not pay or were mixed"
in their default sets (0.1.45: 3060 +0.5%, A4000 +1..+5% short context). Measure it locally; do not assume.

### MEASURED on the 3090 (IQ3_S, batch1, ec5800, ctx200k) — REJECTED, not in prod

Paired probe (`probe_paired.py`: identical prompt, max_tokens 300, reasoning budget 64, 2 warm-ups, 10
single-stream reps, 3 N=4 passes, one restart per arm, same box):

| Arm | single-stream median | stdev | N=4 aggregate median |
| --- | ---: | ---: | ---: |
| default kernels | **104.7 tok/s** | 2.8 | 63.1 tok/s |
| `STRATA_EXPDEC=1` | 98.1 tok/s | 3.7 | 61.6 tok/s |

Default kernels ~6.7% faster single-stream (gap ~4x the standard error) and equal-or-better at N=4. The log
proved the kernel actually swapped (`strata expdec: CUDA0 gate/up type 18/21/22: expdec_gu_kernel R 1 MINB 4,
64 rows per block, grid 20 x 40 (cap 40 groups), captured into a graph`; the down projections printed
`not covered: the default kernel runs` because `STRATA_EXPDEC_DOWN` defaults to 0 on CUDA) — the arm was
genuinely tested, not silently inert.

Why it does not pay here: EXPDEC's win is per-entry at T>=3, but our windows carry ~1.7 rows/blob at batch1,
which is the T=1/T=2 regime where Ampere loses (upstream's own RTX 3060 IQ3_S bench: 1.09x SLOWER at T=1,
0.87-0.89x faster only at T=3-4). Reaching T>=3 needs batch slots, and batch2/batch4 are already rejected on
this card. Prod stays on the default kernels.

Bench method rule: the mixed-concurrency `bench_quick.py` is too noisy for sub-10% claims (N=1 swung 111.6 ->
96.3 across passes on one config). Use `probe_paired.py` (same prompt, 10 reps, median + stdev) for any
single-stream decision; stdev was 2.8-3.7, so differences under ~5 tok/s are not resolvable.
Enable line to look for in the service log: `strata expdec: CUDA1 gate/up type 21: expdec_gu_kernel R 1
MINB 4, 16 rows per block, grid ...` (a `not covered: the default kernel runs` line means no effect).

## Injection point

`hydragen.json` has an `"env"` dict — env switches go there, not in the registry. Restarting the
`Hydragen` service (nssm-wrapped `.venv/Scripts/python.exe`) requires elevation; `tools/nssm.exe restart`
and `net stop/start` both fail un-elevated, and UAC prompts must be approved interactively.

## Other switches compiled into our binary (untested locally)

- `STRATA_PF_BATCH=N` — up to N experts per launch on the prompt path (bitwise). Upstream: prompts
  +7.6% (4K) / +4.9% (16K) / +4.3% (52K) with `STRATA_PF_FUSED_NT=256`.
- `STRATA_MMQ_DIRECT=1`, `STRATA_MMQ_GROUP_SORT=1`, `STRATA_MMQ_RESIDENT_SORT_NE`, `STRATA_MMQ_TAB` —
  prompt MMQ reads resident experts in their cache slots; groups sorted by row count. Verify with
  `STRATA_MMQ_DIRECT_LOG=1`. This is the prompt-side twin of the row-coalescing lever.
- `STRATA_GDN_TC=1` (DeltaNet recurrence as a chunked tensor-core scan; needs 122 KB shared/block, doc says
  sm_80/sm_90 only), `STRATA_PF_FUSED_F16=1`, `STRATA_PF_FUSED_WS=1` (101 KB shared; rounding-level KL).
- `STRATA_ROUTE_PRIOR` already used in prod (0.85). `STRATA_ATTN_DAILY`, `STRATA_QSA_ROWS`,
  `STRATA_SIDE_MIXER`, `STRATA_B6_MMVQ_ROWS` are already ON by default on sm_86 — nothing to win there.
- `STRATA_DISJOINT_ADAPT=1` (docs/DISJOINT_EXPERT_CACHE.md): keeps helper-GPU-owned experts out of primary
  promotion candidates; opt-in, off by default. Upstream measured 32K cached request 81.5 -> 94.1 tok/s
  (+15.5%), request time 12.997 -> 11.417 s (-12.2%) on 2x modified RTX 3080 20 GB, IQ3_S, 131K ctx.
  Relevant only if a helper card joins the 3090.

## Source build path (for kernel patches)

```
git clone https://github.com/Niko1221/Strata.git && cd Strata
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target strata -j8
cmake --build build --target expdec_bench        # parity micro-bench, not in default build
```
Blocker on the Windows dev host: cmake 4.4.3 is present, but no `nvcc` and no `cl` on PATH (VS 2022 folder exists,
CUDA Toolkit absent) — install CUDA Toolkit + VS Build Tools first. Version gap (prod 0.1.54 vs main 0.2.5+)
means build upstream side-by-side on a scratch port and measure there before porting our fork patches.

Upstream's own declared unfinished work in this kernel (the real 3.8x grouped-GEMM target):
1. the weight side — `Split<TG>::load` does 8-9 two-byte loads per call/row; a weight tile staged in shared
   memory with 16-byte loads is their stated next step; 2. the T=1 flat cost.

## Pack-shape alternative to lowering bpw

ISTA-DASLab **Coder** pack keeps 256 of each layer's 512 experts (still 10 active per token): coverage per
layer doubles at the same bpw and same slot budget. Gate any claim on the engine's own
`filling the GPU's expert cache (N experts, X GiB)` line — upstream states shard1 29.6 GB / ~23 GB experts,
which is ABOVE the 3090's ~14.4 GiB expert budget, so do not assume it fits.
