# Building the Strata engine from source (the reference host, 2026-11-06)

Notarized for co-editing agents at `~/hydragen-experimental/CHANGES.md`
(mirrored pointers: `hydragen/NOTE-HENSON-EXPERIMENTAL.md`, `strata/NOTE-HENSON-EXPERIMENTAL.md`).

## Layout that survives a fresh session

```
~/hydragen-experimental/
  hydragen/           clone thirteenthd/hydragen  branch feat/modular-hardware-core @ 9d67b01
  strata/             clone Niko1221/Strata main @ 9f7d977d = v0.2.11-3-g9f7d977d (project 0.1.57)
  .venv-build/        cmake==4.4.3 ninja==1.13.2 (pip, user-space)
  control.tar.gz      strata-control-0.1.57-linux-x86_64.tar.gz (v0.2.11), not extracted
  probe_paired.py probe_cold.py probe_rewrite.py + probe_*.json results
  hydragen/bin/strata.local057   my build (95,114,416 B, sha256(16) 5dd8536b51efb659)
  hydragen/hydragen.exp.local057.json  experimental config (untracked; prod hydragen.json stays pristine)
```

Prod trees on BOTH hosts (3090 `C:\Hydragen`, the reference host `~/hydragen`) are
already at `9d67b01`. The only version delta in play is the engine component: shipped
**strata 0.1.55** vs upstream **0.1.57**.

## Toolchain without root (no passwordless sudo on the reference host)

* CUDA 13.1 toolkit already present at `/usr/local/cuda-13.1` (nvcc 13.1.115).
* host gcc 15.2.0 is accepted by nvcc 13.1 — no `--allow-unsupported-compiler` needed.
* cmake/ninja only obtainable user-space: `python3 -m venv .venv-build && pip install cmake==4.4.3 ninja==1.13.2` (matches strata's `requirements-build.txt`).
* GPU RTX 3080 Ti 12 GB, driver 595.91.07, 24 cores; build with `-j6` (RAM is tight, prod resident).

## Recipe

```bash
cd ~/hydragen-experimental/strata
. ../.venv-build/bin/activate
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_CUDA=ON -DSTRATA_MMQ_TAB=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build --target strata -j6     # 179 targets, 0 errors / 12 warnings
```

Configure prints the truth you need: `STRATA_MMQ_TAB on: pointer-table MMQ for
q2_0;iq2_xxs;iq2_xs;iq2_s;iq3_xxs;iq3_s;iq4_nl;iq4_xs;q8_0` (covers IQ3_XXS prod quant),
`trellis MMQ ... overlay on the pinned ggml-cuda in build/ggml-tq-overlay`, `ggml 0.24.0 commit 3cf03257f`.

## Running a built engine under Hydragen

No signature gate: Hydragen just launches `exe` from the config. Copy prod's `bin/` and `lib/`
into the experimental tree (bundled `libcublas*/libcudart.so.13` work for a system-CUDA build too —
verified with a CUDA probe under both lib sets), symlink `models` to the prod target, and **copy
`data/`** — the engine exits with `read_expert_profile: cannot open data/expert-profile-learned.bin`.
Run via a **systemd user unit** (`~/.config/systemd/user/hydragen-exp.service`, mirror of
`hydragen.service` with `--port 8081` + the exp config): `systemctl --user start hydragen-exp`.

## Measured arms (3080 Ti, model id qwen3.8-flash-next-iq3_xxs)

| arm | single tok/s | N=4 agg | cold prompt tok/s |
|---|---|---|---|
| prod 0.1.55 | 80.1 (sd 2.4) | 73.7 | 2220.0 |
| 0.1.57 default | 74.6 / 76.5 warmed (n=8 each) | 61.4 / 66.3 / 68.4 | 2243.2 |
| 0.1.57 + `STRATA_MMQ_DIRECT=1` | — | — | 2246.0 |

1. **Pointer-table MMQ is a resident-heavy lever.** It engaged but read nothing in place:
   `strata prefill: layers 0..48: 0 of 5674 resident experts read in place, 461270 streamed
   (STRATA_MMQ_DIRECT, totals)` — only 2112 of 24576 blobs resident, hit 50-81%. Upstream's own
   precondition is "with every expert resident". So MMQ's payoff is on the **3090** (96-99% hit),
   whose shipped Windows binary has no pointer-table MMQ. VRAM cost 11431 -> 11559 MiB.
2. 0.1.57 decode measured ~4-6% below 0.1.55 and prompt +1% — NOT decisive (different times of
   day, 0.1.57 decode ran after cold prompts borrowed `1852` cache slots). Both engines pre-fill
   the same `2112` experts / 3.43 GiB (44 slots/layer), so capacity was not the variable. Needs a
   paired interleaved A/B before any verdict; if real, bisect `STRATA_SM86_PERF=0` and the
   `prompt chunk auto: 5888` (vs 8192 on 3090) — all switch-controlled.
3. New CLI surface 0.1.55 -> 0.1.57: `--lookup-max-t N`, `--lookup-sizes A,B`, `--draft auto`,
   `--draft-fallback mtp|none` ("the MTP layer's ~0.9 GiB of VRAM goes to the expert cache"),
   `--ram-keep-gib W`. Unmeasured. `--draft-fallback none` is the biggest 12 GB capacity lever found.

## Co-editing etiquette used here

Never leave experiment knobs in a tracked file: `hydragen.json` was restored with
`git checkout --` and the experiment config moved to `hydragen.exp.local057.json`. Keep
`git status --porcelain` free of tracked modifications, add an untracked
`NOTE-HENSON-*.md` pointer in each clone, and write the full notarization at the workspace root
(where it cannot cause a merge conflict). Never run both engines at once on one 12 GB card.
