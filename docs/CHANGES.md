# Notarized change log — hydragen-experimental (the reference host)

**Agent:** the agent (default Hermes profile on the Windows dev host, Telegram @the agent bot)
**Session:** local-1e8183300e3253ed85bcd14a5b5ebf6c062c1c34b02f084c2737d8291b5c31cd
**Date (local, BCDT):** 2026-11-06, ~17:50–18:35
**Purpose:** build the latest open-source Strata engine from source on the reference host and measure it against the shipped 0.1.55 engine, without permanently changing the live coder-serving stack.

> **If you are a co-editing/review agent:** read this file before touching anything below.
> The two git clones are **clean** (no tracked file modified by me). Everything I added is
> untracked and listed in "Files I created" with teardown commands at the bottom.

---

## 1. Layout & provenance

Root: `~/hydragen-experimental/` (2.8 GB)

| Path | What | Provenance |
|---|---|---|
| `hydragen/` | clone of `https://github.com/thirteenthd/hydragen` | branch `feat/modular-hardware-core`, HEAD `9d67b01` |
| `strata/` | clone of `https://github.com/Niko1221/Strata` (MIT) | `main`, HEAD `9f7d977d` = `v0.2.11-3-g9f7d977d`, CMake project version **0.1.57** |
| `.venv-build/` | build toolchain venv (python 3.14) | pip: `cmake==4.4.3`, `ninja==1.13.2` |
| `control.tar.gz` | `strata-control-0.1.57-linux-x86_64.tar.gz` (99,944,994 B) | GitHub release `v0.2.11`, **not extracted yet** |
| `control-0.1.57/` | empty placeholder for that tarball | created, unused |

Key fact discovered: the live trees on **both** hosts (3090 `C:\Hydragen` and the reference host
`~/hydragen`) are already at `9d67b01` = the tip of `feat/modular-hardware-core`.
So the only version delta in play is the **engine component**: shipped `strata 0.1.55` vs
upstream `0.1.57`.

Shipped engine component (prod, unchanged): `bin/engine-component.json` →
`0.1.55 cuda archs sm_75, sm_86, sm_89, sm_120, compute_120`.

## 2. Toolchain on the reference host (no root used)

* CUDA toolkit already present: `/usr/local/cuda-13.1` (nvcc 13.1.115)
* host gcc: 15.2.0 — accepted by nvcc 13.1 **without** `--allow-unsupported-compiler`
* cmake / ninja: user-space via `.venv-build` (no `apt`, **no sudo used anywhere**; the box
  has no passwordless sudo)
* 24 cores, 60 GiB RAM, GPU = RTX 3080 Ti 12 GB, driver 595.91.07

## 3. Exact build recipe (reproducible)

```bash
cd ~/hydragen-experimental/strata
. ../.venv-build/bin/activate
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_CUDA=ON -DSTRATA_MMQ_TAB=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build --target strata -j6        # log was /tmp/strata_build.log
```

Configure evidence: `STRATA_MMQ_TAB on: pointer-table MMQ for q2_0;iq2_xxs;iq2_xs;iq2_s;
iq3_xxs;iq3_s;iq4_nl;iq4_xs;q8_0` (covers our prod quant, IQ3_XXS); `trellis MMQ … overlay on
the pinned ggml-cuda in build/ggml-tq-overlay`; `ggml version 0.24.0`, `ggml commit 3cf03257f`.

Build result: `[179/179] Linking CXX executable strata` — **0 errors, 12 warnings**,
`strata 0.1.57`, 95,114,416 B, sha256(16) `5dd8536b51efb659`.

## 4. Files I created (all untracked; nothing of yours was edited)

Inside `hydragen-experimental/hydragen/`:

| Path | Notes |
|---|---|
| `bin/strata.local057` | my build (95,114,416 B, sha256(16) `5dd8536b51efb659`) |
| `bin/` , `lib/` | **copies** of prod's shipped `bin/`+`lib/` (prod `strata` = 286,548,608 B, sha256(16) `e4406b474c292c8b`, plus `strata-vision`, `strata.fork`, `libhydragen_core.so`, `libpaged_kv.so`, `libstrata_fast_ipc.so`, `libstrata_kq.so`, bundled `libcublas*/libcudart.so.13`) so the experimental tree can boot an engine. `bin/` is gitignored; `lib/` shows as untracked |
| `models` | symlink → `~/.models_backup/Strata-data` (same target prod uses; read-only use) |
| `data/` | copy of prod `data/` (11 MB: draft vocab, expert profiles) — copied because the engine exits without `data/expert-profile-learned.bin` |
| `hydragen.exp.local057.json` | my experimental serve config (see §5) |
| `hydragen.log`, `run.log` | engine/serve logs from the experimental run |

`git status --porcelain` in that clone right now:
```
?? hydragen.exp.local057.json
?? lib/
```
**Tracked-file note:** I briefly left my config in the *tracked* `hydragen.json`. It was
restored with `git checkout -- hydragen.json` at 18:30–18:35, and my config now lives at
`hydragen.exp.local057.json`. Verified clean above.

Outside the clones:

| Path | Notes |
|---|---|
| `~/.config/systemd/user/hydragen-exp.service` | **new** user unit: mirrors prod unit, `WorkingDirectory` = experimental tree, `--port 8081`, `--config …/hydragen.exp.local057.json`, `LD_LIBRARY_PATH=…/hydragen/bin`, `LimitMEMLOCK=infinity`. **Not enabled**, currently `inactive` |
| `/tmp/strata_build.log`, `/tmp/cfg.log`, `/tmp/help_new.txt`, `/tmp/help_prod.txt`, `/tmp/cuda_probe*` | build/config/probe scratch |
| `hydragen-experimental/probe_*.py`, `probe_*.json` | my probe harnesses + results (copied from the 3090 session, patched: `PORT` env var, Linux source-file roots, model id `qwen3.8-flash-next-iq3_xxs`) |

**Not touched:** `~/hydragen/**` (live prod tree), `hydragen.service`,
`strata.service`, `hydragen-actions-runner.service`, the 3090 host's `C:\Hydragen` config.

## 5. Experimental config diff (vs prod `hydragen.json`)

Identical engine args and env as prod, except:
```
exe:  bin/strata.local057        (prod: bin/strata)
port: 8081                       (prod: 8080)
log:  hydragen.log               (prod: hydragen.log in prod dir)
cwd:  .
env:  + STRATA_MMQ_DIRECT=1, STRATA_MMQ_DIRECT_LOG=1   (added for the MMQ arm)
```
Prod args for reference: `--pack models/packs/iq3_xxs --native …/IQ3_XXS/…00001-of-00002.gguf
--ple-gguf …00002-of-00002.gguf --expert-profile data/expert-profile-learned.bin
--expert-cache auto --prefill auto:8192 --spec 4 --mtp models/mtp/rt --max-context 131072
--kv int8 --vision --vram-reserve-mib 800 --pcie-frac 0.20 --spec-min-p 0.45 --coupled-draft
--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --adapt-async 1 --kv-resident 20480
--conversation-cache-mib 6144 --conversation-cache-slots 8 --prompt-cache-tail --batch 2
--batch-mtp --expert-profile-save data/expert-profile-live.bin --expert-profile-save-every 30
--expert-cache-per-layer`

## 6. Measurements (3080 Ti 12 GB, the reference host)

Method: `probe_paired.py` = identical prompt, 300 max_tokens, 64 reasoning-budget tokens,
8–10 single-stream reps + 3 N=4 passes, medians. `probe_cold.py` = matched **disjoint**
~33–38 k-token cold windows, same window order for every arm (no prefix reuse).
All four runs used the *same* engine args; only the binary/env arm changed.

| Arm | single-stream tok/s (median) | N=4 aggregate (median of 3 passes) | cold prompt tok/s (median of 3 windows) |
|---|---|---|---|
| prod **0.1.55** (live service, before swap) | **80.1** (stdev 2.4, n=10) | **73.7** (76.3 / 73.0 / 73.7) | **2220.0** (2204.5 / 2317.7 / 2220.0) |
| **0.1.57** default | 76.1 (stdev 4.2, n=10, cold-cache) → warmed **74.6 / 76.5** (n=8, n=8) | 68.6 cold-cache → warmed 61.4 / 66.3 / 68.4 | **2243.2** (2216.9 / 2301.8 / 2243.2) |
| **0.1.57 + `STRATA_MMQ_DIRECT=1`** | — | — | **2246.0** (2224.2 / 2308.4 / 2246.0) |

Findings:

1. **Pointer-table MMQ engages but gains nothing on this card.** Engine evidence:
   `strata prefill: layers 0..48: 0 of 5674 resident experts read in place, 461270 streamed
   (STRATA_MMQ_DIRECT, totals)` → in-place reads are **zero** because only ~2112 of 24576
   blobs are resident (decode hit rate measured 50–81%). Upstream's own claim is "Faster read-in
   time **with every expert resident**". Prompt rate moved 2220 → 2243 → 2246 tok/s (+1.2%,
   noise band). VRAM cost as documented: 11431 → 11559 MiB.
   **Conclusion: MMQ is a resident-heavy lever — it belongs on the 3090 (96–99 % hit), where the
   shipped Windows binary has no pointer-table MMQ, i.e. exactly the build we just proved out.**
2. **Decode on 0.1.57 looked ~4–6 % slower than 0.1.55** (single 80.1 → 74.6/76.5; N=4 73.7 →
   61–68) and **prompt +1 %**. *Not yet decisive*: the arms were measured at different times and
   the 0.1.57 decode followed cold prompts that borrowed `1852` cache slots
   (`strata serve: the prompt path borrows 1852 CUDA0 cache slots (3.05 GiB)`). Capacity itself
   was **not** the variable: both engines pre-fill `2112` experts / `3.43 GiB` (44 slots/layer).
   Needs a paired, interleaved A/B in one window before any verdict.
3. 0.1.57 defaults printed: `sm_86 (Ampere): 9 exact speed switches on by default … ADAPT_NOWAIT
   ADAPT_LAG CPU_IQ_ALL_EXACT1 SIDE_MIXER ATTN_DAILY GR_UP75 GU_SWIGLU QSA_ROWS B6_MMVQ_ROWS`
   and `prompt chunk auto: 5888 tokens, a 512-slot ring`.
4. New engine surface vs 0.1.55 (from `--help` diff): `--lookup-max-t N`, `--lookup-sizes A,B`,
   `--draft auto`, `--draft-fallback mtp|none` ("the MTP layer's ~0.9 GiB of VRAM goes to the
   expert cache"), `--ram-keep-gib W`. None measured yet.

## 7. Live state right now (as of this log)

* `hydragen.service` — **active**, prod 0.1.55 engine on **:8080**. Verified by smoke test:
  reply `prod ok`, 51 tokens @ 78.9 tok/s, expert cache 81.1 % hit, VRAM 11529/12288 MiB.
* `hydragen-exp.service` — **inactive** (stopped after the measurement window).
* GPU downtime during the experiment: ~30 minutes (18:19 → 18:31 local). Coder delegation was
  offline in that window.
* **Never run both engines at once** — one 12 GB card, each engine takes ~11.5 GB.

## 8. Open items / next steps (not done by me)

1. Paired interleaved A/B: 0.1.55 vs 0.1.57 with matched warm state, decode + cold prompt,
   2–3 passes each (bench noise on this box is ±5 %).
2. If decode regresses: bisect the 0.1.56/0.1.57 default switches (`STRATA_SM86_PERF=0`,
   prompt chunk 5888 vs 8192) — all are switch-controlled, so it is a config question, not a
   code question.
3. Try `--draft auto` / `--draft-fallback none` on Flash-Next to free the MTP layer's ~0.9 GiB
   into the expert cache — biggest capacity lever available on 12 GB.
4. Extract `control.tar.gz` (0.1.57 `strata_control` + tools) and diff its `tools/` against
   prod's, if the pack/tooling side needs the newer scripts.
5. A Windows `sm_86` + `STRATA_MMQ_TAB=ON` build is the pay-off target for the 3090; no MSVC/
   CMake toolchain exists on the 3090 host, so that build needs a VS 2022 install there (or a
   CI build using the repo's `hydragen-actions-runner`).
6. Quality gate for any *pack* change remains `canonical_xcheck.py` (mandatory) — untouched here.

## 9. Teardown / hand-back commands

```bash
# experimental engine (already stopped)
systemctl --user stop hydragen-exp
systemctl --user disable hydragen-exp 2>/dev/null   # if you enabled it
rm ~/.config/systemd/user/hydragen-exp.service && systemctl --user daemon-reload

# remove everything I added (keeps both clones' git history intact)
rm -rf ~/hydragen-experimental
```
Or keep the tree and just reset it to a clean clone:
```bash
cd ~/hydragen-experimental/hydragen
rm -rf lib bin/strata.local057 hydragen.exp.local057.json hydragen.log run.log
git status --porcelain    # expect empty
```

---

## 10. Concurrency arms on the 3080 Ti (2026-11-06, second window, ~19:05-19:30)

Scope set by the user: **12 GB cards are the target, 3090 out of scope**, IQ3_XXS stays
(capacity only from freed VRAM), coder downtime acceptable (route subagents to the local engine).

`bench_quick.py` (N=1/2/4/8, 400 tokens, warmed passes), port 8080 = prod 0.1.55, 8081 = experimental 0.1.57:

| arm | slots/layer | N=1 | N=2 | N=4 | N=8 agg | N=8 per-stream |
|---|---|---|---|---|---|---|
| prod 0.1.55 baseline (2 passes) | 44 (2112) | 76.3 / 78.3 | 58.6 / 59.7 | 57.3 / 62.6 | 56.5 / 62.7 | 13.6 / 13.9 |
| A0 0.1.57 control, same args | 44 (2112) | 74.9 | 56.0 | 56.3 | 57.4 | 12.0 |
| A1 `--batch 4` | **28 (1344)** | 63.1 | 48.0 | 51.3 | 47.7 | 8.8 |
| A2 `--vram-reserve-mib 400` + `--prefill auto:4096` | **49 (2352)** | 76.6 / 77.0 | 60.2 / 59.7 | 59.6 / 62.1 | 61.4 / 60.7 | 14.8 / 13.9 |

Findings:
1. **0.1.57 is version-neutral** on this curve — A0 sits inside the prod baseline band.
2. **Batch >2 is self-defeating on 12 GB.** Each extra batch row costs ~450 MiB of row-state
   VRAM, taken out of the expert cache: batch 4 dropped the cache 2112 -> 1344 slots
   (44 -> 28/layer), hit 72.9/81.8% -> 58.6/68.9%, and every N point fell. **Batch 2 stays.**
3. **Capacity from freed VRAM works but is weak:** reserve 800->400 bought +240 slots
   (+11%) and only +3-6% at N>=2. The planner ignored the `--prefill auto:4096` ceiling
   (`prompt chunk auto: 7168 tokens`), so treat prefill as non-leverable here (same lesson as
   the 3090's auto:12288 no-op). A2's extra headroom is not free: prod has already logged
   `prefill: device buffers for a chunk of 1024 tokens do not fit (46 of 11903 MiB free)`.
4. **The wall is bytes-through-RAM, not scheduling.** At 44 slots/layer with hit 73-82%,
   every token still streams ~18% of 480 routed experts (1.76 MB each) from the RAM copy; at
   N=8 that is ~1.2 GB per step-time, which matches the observed ~60 tok/s aggregate plateau
   (N=2 through N=8 all land 56-63). Adding slots cuts that traffic only linearly; cutting
   bytes/expert cuts it multiplicatively.
5. `--pcie-frac` in the prod arg line is **not a documented flag in 0.1.57** (engine accepted
   it, no effect line printed) — candidate cruft to drop when adopting 0.1.57.

Teardown of this window: prod `hydragen.service` restarted and verified (`back online` smoke
reply), `hydragen-exp` left inactive. Arm configs live only in
`hydragen/hydragen.exp.local057.json` (+ `.a0.bak`); bench JSONs `bench_a*.json` here.

## 11. POC build window (2026-11-06 evening) — poc-moe Phase 1
New workspace: poc-moe/ (untracked, separate from hydragen/ and strata/ clones).
- donor/: RustingStrata kernels.cu + tables generated from its quant/tables.rs;
  donor_all.cu compiles clean standalone (sm_86).
- poc2.cu: our pager+harness (mine). mmap sparse host pool (48x128 experts,
  2,176,000 B blobs = IQ3_XXS gu + IQ4_NL dn, +17% vs pack mean = conservative),
  per-layer LRU/heat slot cache, capped cudaHostRegister window, GemmTile/xrow
  tiles feeding donor gemm_iq3_xxs/gemms + gemm_iq4_nl, grouped (union) vs
  per-pick tiles, skipmiss ceiling arm, event-based copy->compute handoff.
  NO CPU expert compute anywhere (invariant 2 held by construction).
- ABI verified against donor: gemm*/gemms*(tiles,wd,nd,x,x_stride,xrow,y,
  y_stride,rows,n64,act), grid(rows/64,tiles), block 2*GEMM_BM, gemms takes
  cnt<=32 tiles; act 0=store 1=silu(y)*v 2=+=; pack type ids: 16=IQ2_XXS
  17=IQ2_XS 18=IQ3_XXS 20=IQ4_NL 21=IQ3_S 22=IQ2_S 42=q2_0(Strata ext).
- Results (steps=6, zipf0.85, K=44): B=8 grouped 18.5 tok/s agg, hit 0.55,
  449 MiB/token => 8.3 GiB/s PCIe = reproduces banked 8.7 GB/s wall on new
  substrate. skipmiss ceiling: 570 tok/s (K=24) / 350 (K=44) with zero traffic
  => compute is NOT the wall, ~6x headroom over the 63 bar.
- Grouped tiles halve tile count (80->44/layer at B=8) but ~0 net gain while
  traffic-bound. heat policy +3% over lru; zipf .85->.95 +11%.
- Verdict: pipeline works end-to-end GPU-only; POC v1 loses to prod 57-63 at
  B=8 because (a) copies serialized before compute, (b) synthetic independent-
  stream router hits 55% where prod profile-seeded hits 83% at same K.
  Next arms: copy/compute overlap (double-buffer next layer under current),
  stream-affinity router, prefetch on routing-trace popularity.
- hydragen.service stopped for the window, restarted after; /v1/models loaded
  + chat smoke verified (prod 0.1.55 untouched). sweep{1,2,3}.json in poc-moe/.

## 12. POC v2 window (2026-11-06 late) — affinity + pinned uploads
- Binary: poc-moe/poc2 (poc2.cu adds --affinity arm + CLI; v1 binary = poc).
- NOTE for co-editor: local-edit->scp flow must re-apply:
  sed -i 's/donor_all.cuh/donor\/donor_all.cu/' poc2.cu  (scp overwrites it).
- Arms (steps=8, zipf .85, pool 128/layer, pin 2048 = pinned-window uploads):
  * pinned-window vs pageable: +38% (18.8 vs 13.6 tok/s same hit).
  * batch affinity at B=8 K=44: hit 55%->95.2%, traffic 442->48 MiB/token,
    tok/s 13.6->109 (lru) ->130 (heat). THE lever is union compression.
  * K cliff: affinity union ~31 experts/layer; K=24 < union -> thrash 22-25;
    K=44/64 -> 130. Union must fit K.
  * ceiling (skipmiss, zero traffic): 509-570 tok/s -> compute never the wall.
- HEADLINE: B=8 K=44 heat affinity = 130 tok/s aggregate vs banked prod 57-63
  at 44 slots/layer on same 12 GB class. ~2.1x prod at 3x less expert traffic,
  GPU-only compute, <=6.2 GiB pool VRAM. Success bar (>=63 @B8) CLEARED
  under POC assumptions (synthetic cooperative router, no KV/attention/routing
  costs, blobs +17% vs pack mean).
- Not yet met: N=1 (34-42 vs prod 76): naive 3-launch small-tile GEMM path,
  no MTP/spec, no fused routing/attention. Noise between repeat configs ~1.4x
  at low tok/s (clock ramp); steps=16 would tighten.
- hydragen.service stopped for window; restarted; /v1/models loaded, chat 200.

## 13. Affinity validation on REAL routing traces (2026-11-06 night)
- Captured multi-stream trace: hydragen-exp + --batch 4 --dump-routing, 4
  concurrent coder-style sessions (poc-moe/drive_sessions.py) -> trace-b4.bin
  (record = i32 layer, i32 k=10, 10 ids, 10 weights; 88B/rec; batched dispatch
  writes B consecutive recs per layer-step).
- Pager sims (python, calibrated: single-stream LRU K=44 hit .788 vs prod
  measured .73-.83):
  * static top-44 coverage: 0.53  -> polysemantic invariant CONFIRMED on real data.
  * old single-session trace: consec B=8 union 35 (fits K=44), far-pair union
    60 > K=44, LRU batch8 hit .507 vs single-stream .788.
  * new 4-stream trace: union/step 26.5 (p90 33), step persistence .507,
    LRU K=44 .622 cold / .775 warm, K=64 .861.
- POC now replays real traces (poc2.cu --trace --warm --pipe; binary poc2t):
  * B=4 heat K=44 pin4096: 37.9 tok/s hit .766 233 MiB/tok
  * B=4 heat K=64: 56.2 hit .843 156 MiB/tok
  * B=8 heat K=44: 35.6 hit .750 | K=64: 52.8 hit .831 168 MiB/tok
  * pipe (1-layer lookahead): +0.2 at K=64 -> wall is per-blob transfer
    overhead (~50us/blob at 2.2MB), NOT latency. 8.6 GiB/s plateau reproduces
    prod's 8.7 GB/s exactly.
- CORRECTION to section 12 headline: the 95%-hit 130 tok/s affinity arm held
  one union FIXED across all steps (stationary hot set). Real 4-stream traffic
  gives union persistence .507 and hit .75-.84, not .95. Honest verdict:
  GPU-only micro-burst POC MATCHES prod at K=64 (~56 vs 57-63 tok/s) but does
  NOT yet beat it at K=44 (36-38 vs 57-63). Prod's residual edge = profile-
  seeded placement + CPU drain hiding misses (invariant-2 violation by design).
- Remaining levers, ranked: (1) batched/coalesced uploads to amortize the
  ~50us/blob overhead (target 20+ GiB/s; Strata BATCHED_DMA pattern);
  (2) IQ2_XS blobs (-36% bytes at same layout); (3) profile-seeded placement
  (prod-style, allowed: mmap-read-only seeding, still no CPU compute).
- Windows: exp capture window + 2 POC sweep windows; hydragen.service restored
  after each; final check active + /v1/models OK.


## 14. Coalesced-upload test + IQ2 capacity win (2026-11-06 night, tr4–tr6)

Root-cause discovery: `nvidia-smi -q -d PCIE` on the reference host reports
**pcie.link.gen.current = 2, width 16** — the 3080 Ti is running Gen2 x16.
That is the real transfer ceiling (~8 GB/s), which explains why BOTH the
prod engine and the POC plateau at ~8.6 GiB/s.

Batch-DMA arm (`cudaMemcpyBatchAsync`, `--bdma 1`, CUDA 13.1) and an 8 GiB
pinned-tier arm (also `cudaHostRegister`-ing the mmap pool, which works):
**zero effect** — every config converges to exactly 8.60–8.68 GiB/s.
The wall is the link, not per-blob overhead or pageable copies. Lever (1)
is dead on this machine. Lever (2) — fewer bytes/blob — is the lever.

IQ2 arm (`poc2q`, `-DGU_IQ2XS -DDN_Q2_0`: gu IQ2_XS 74 B/256v + down q2_0
72 B/256v = 1.408 MB blob, mirrors the pack's own lighter blob class
gu16/d42 = 1,305,600 B measured; donor implements both types).
Real 4-stream trace replay, heat policy, pin 4096, bdma 1, warm 64:

| B | K=44 | K=64 | K=96 | K=128 |
|---|---:|---:|---:|---:|
| 1 | — | — | 96.5 / .930 | 123.1 / .952 |
| 2 | — | — | 128.1 / .907 | 188.9 / .942 |
| 4 | 54.4 / .766 | 92.0 / .843 | 150.9 / .906 | 218.9 / .942 |
| 8 | 58.4 / .750 | 87.2 / .831 | 143.4 / .898 | 217.2 / .936 |

(tok_s / hit; vram 2.59–7.53 GiB; artifacts `tr5.json`, `tr6.json`,
binary `poc2q`). Hit rises .75→.95 with capacity: **capacity is the lever
and IQ2 bytes-per-slot buys 1.44× more slots per GiB at the same VRAM.**

Honest equal-budget projection: prod's expert budget ~4.8 GiB fits 2112
slots (K=44) at ~2.27 MB blobs; the same 4.8 GiB fits ~3400 IQ2-class
slots (K≈71) → real-trace interpolation ≈ 100 tok/s B=8 aggregate vs
prod measured 57–63: **~+60–90% at identical VRAM, GPU-only, no CPU
compute, mmap pool with 4 GiB bounded pin tier.**

Caveats: harness has no attention/KV/PLE/routing-head/MTP costs (absolute
tok_s optimistic for all arms); full-IQ2 pack requires repack + quality
eval, the current pack is mixed-quant; K=128 arms shown exceed a 12 GB
full-system envelope and demonstrate the curve, not a deployable config.

Prod `hydragen.service` active after every window; `/v1/models` verified.
## 15. PCIe transfer-path root cause — supersedes section 14's "Gen2 wall" (2026-11-06 night, pbench{2,3,4,6,7,8} + tr7/tr8/tr9)

Section 14's root-cause claim ("PCIe Gen2 x16 link ceiling") was WRONG. Corrected diagnosis:

### What the link actually does (`/tmp/pcbench*.cu`, prod untouched)
- back-to-back pinned copies (native `cudaHostAlloc`), 2.24 MB ops: **24.05 GiB/s** — the link
  DOES negotiate Gen4 x16 (idle reads show Gen1; it retrains on demand).
- 64 MB continuous warm: 20.6 GiB/s.
- same pinned copies with 0.3 ms host gaps: 6.15 GiB/s; with 3 ms gaps: 0.67 GiB/s —
  the link downtrains between bursts and pays a retrain ramp. Burstiness DOES matter,
  but it is not the whole story.

### The real wall: pinned transfer was never happening for pool pages
- `cudaHostRegister` on **file-backed (and memfd) mmaps FAILS on this host**:
  `invalid argument` for every variant tried (O_RDONLY/MAP_PRIVATE, O_RDWR/MAP_SHARED,
  4 KB/64 KB/2 MB chunks, MAP_POPULATE, memfd, `LimitMEMLOCK=infinity` systemd-run scope).
  Anonymous pages register fine. Likely a kernel/driver GUP limitation on this
  Ubuntu 26.04 stack — re-check on any new box before designing around mmap pinning.
- Consequence: POC `--pin` tiers and any mmap-pool pin path silently fall back to
  **pageable DMA, measured 9.4-9.5 GiB/s back-to-back**; in-engine plateau 8.6 GiB/s.
  That is why the bdma/pin arms in tr4 were flat — policy changes on a dead axis.
- Keepalive copies (1 MiB/5 ms, tr7 `--linkkeep`): no effect; own gaps exceed the retrain threshold.
- `--pipe 1` (one-layer lookahead, tr8): +13% on the small-burst arm (33.3->37.8 B4K44 IQ3), ~0 elsewhere.
- `--stage 512` (synchronous 4-thread memcpy into native pinned ring + pinned DMA, tr9):
  **worse — 5.4-5.7 GiB/s**; memcpy lands on the critical path. The staging design only works
  when the memcpy overlaps GPU compute (worker-async ring), not synchronous.

### Corrected conclusions
1. There is no hard Gen2 ceiling. The directive premise "PCIe 4.0 ~24-31 GB/s" is REAL but only
   reachable via native-pinned continuous traffic (anonymous pinned ring at ~24 GiB/s).
2. Mandatory transport on this host: mmap pool -> (async, multi-thread) memcpy into a bounded
   native-pinned ring -> continuous pinned DMA. Worker-async staging is the next experiment (tr10).
3. Earlier "transfer levers dead" verdicts (tr4 bdma, pin 4096/8192) were measured on the
   pageable path; re-score them once async staging lands. IQ2 capacity results stand
   (bytes/token is path-independent), but their absolute tok/s may move.
4. Burstiness still matters on top: even pinned, >=0.3 ms inter-op gaps cost 3-4x. Upload
   streams must stay continuous (prefetch depth >= one full layer + coalesced ops).
5. Do NOT chase BIOS/ASPM forcing first: pinned continuous already hits 24 GiB/s without it;
   retrain penalties are an argument for coalescing/prefetch, not for BIOS changes.
6. Add `LimitMEMLOCK=infinity` to serving units regardless (harmless; may be required on other kernels).

Artifacts: `/tmp/pb[678].cu`, `/tmp/pcbench{2,3}.cu`, `tr7.json` (linkkeep), `tr8.json` (pipe),
`tr9.json` (stage). All POC windows restored `hydragen.service`; `/v1/models` verified 200 after.

## 16. tr10 worker-async pinned ring — transport verdict + paper reviews (ExpertFlow / TurboQuant / Hydragen-namesake)

### 16.1 tr10: async staging through anonymous pinned ring
Design: 8 worker threads do PURE memcpy mmap->anonymous-pinned ring slots (no CUDA calls
off-thread; cross-thread event waits deadlocked in v1). Main thread waits per layer, issues
the pinned DMAs itself, records per-slot events, reclaims slots via cudaEventQuery into a
freeSlots pool. Debug path: 3 wedged sweeps traced via /proc/*/wchan + per-stage counters;
root cause was `new std::atomic<int>[N]` UNINITIALIZED under C++17 host default
(value-init of atomics is C++20) — counters read as garbage, waits never satisfied.
Zero-init fixed it (sanity arm then ran first-try: 52.7 tok/s, hit .897).

Real 4-stream trace, heat, bdma=1, stage=512MiB, steps=192 (tok_s | hit | eff GiB/s):

| arm | pipe0 | pipe1 | pageable ref (tr5) |
|---|---|---|---|
| IQ3 B4 K44 | 29.8 .766 6.8 | 32.1 .766 7.3 | 37.9 .766 8.6 |
| IQ3 B4 K71 | 56.3 .860 7.6 | 56.2 .860 7.6 | ~48-50 .860 |
| IQ3 B8 K44 | 28.5 .750 6.9 | 28.6 .750 7.0 | 35.6 .750 |
| IQ3 B8 K71 | 48.9 .850 7.1 | 48.9 .850 7.1 | ~44-46 |
| IQ2 B4 K44 | 54.4 .766 7.5 | 57.7 .766 7.9 | 54.4 .766 |
| IQ2 B4 K71 | 102.8 .860 8.4 | 102.7 .860 8.4 | ~95-100 (K71 interp) |
| IQ2 B8 K44 | 48.5 .750 7.1 | 48.7 .750 7.2 | 58.4 .750 |
| IQ2 B8 K71 | 85.5 .850 7.5 | 85.5 .850 7.5 | ~90 (K71 interp) |

Verdict: the pinned ring works mechanically (no deadlocks, 24 GiB/s DMA achievable per
microbench) but the extra mmap->ring memcpy costs exactly what pinned DMA gains: effective
transport lands 6.8-8.4 GiB/s, i.e. ~= the pageable path (8.6), sometimes worse at B8
(bigger bursts, thread contention). Practical transport wall on this host stays ~8.6 GiB/s
as long as file-backed cudaHostRegister fails. To exploit the 24 GiB/s headroom we would
need to amortize the copy: an anonymous-PINNED host hot tier (copy once on first load,
re-miss = pure DMA at 24 GiB/s). One clean arm, not yet built.

Equal-VRAM headline stands and is now measured through the working ring: IQ2 at
K=71 fits 4.18 GiB expert budget (prod = 4.8 GiB / K=44): B4 102.8 tok/s, B8 85.5
vs prod banked 57-63 B8 => ~+40-50% at equal-to-lower VRAM on the real trace.

### 16.2 Paper reviews (scored on our real trace, not on paper claims)
ExpertFlow (arXiv 2410.17954): (1) PLEC adaptive per-layer quotas: NEUTRAL here
(K44 .741->.736, K71 .830->.834) because heat counters already learn layer skew;
(2) token scheduler / route-similar batching: our trace already shows union
compression (mean 26.5 vs naive 40/step) — Phase-2 scheduler material;
(3) learned routing-path predictor: oracle-prefetch ceiling only ~+3 pts hit on
our trace; prediction hides latency, does not reduce bytes — deferred.
TurboQuant (Google blog): near-optimal online vector quantization for KV, not expert
traffic. Value to us = non-expert VRAM recovery (freed KV GiB -> +IQ2 slots -> hit/
throughput). Phase-2 lever; QSA/hybrid attention shrinks the KV share, measure first.
Hydragen-namesake (arXiv 2402.05099): exact shared-prefix attention decomposition +
inter-sequence query batching. Our stack (despite the name) does NO prefix sharing:
measured 451 MiB/row state tax IS the per-row-KV problem this paper solves. Agent
fleets = its best case (long common system prompts). Directly attacks the batch-
capacity collapse we measured at --batch 4. Phase-2 non-expert-side lever, pairs
with IQ2 on the expert side. Caveat: QSA kernel adaptation; POC harness has no KV path.

### Notarization
Files: poc2.cu now has --stage worker-async ring (default off; stage_mib>0 selects it),
poc2d/poc2t/poc2q rebuilt with it; tr10.json/tr10.err in poc-moe/.
hydragen.json untouched; sweep configs untracked; prod restored + verified.
## §17 — Transport wall broken: file-backed pinning dead-end, pinned hot-tier wins (tr13–tr18)

Source: `poc-moe/poc2.cu` (this commit). Artifacts: `tr17.json` (partial, superseded), `tr18.json` (16 rows, banked), probes `pb9..pb15.cu`. Prod `hydragen.service` intentionally DOWN for POC duration (user-approved 2026-11-06); restore at POC end.

### Root-cause chain for the old ~8.6 GiB/s wall (all four defects real)
1. `cudaHostRegister` refuses `PROT_READ` file mappings — NVIDIA GUP pins with `FOLL_WRITE` (pb10 battery: anon/malloc/static fine; file `PROT_READ` invalid; `PROT_READ|PROT_WRITE` fine).
2. `cudaMemcpyAsync` refuses any copy **spanning two adjacent registrations** (`invalid argument`) even when both halves are registered (pb15: inside-window 24.02 GiB/s ok, straddle invalid). Earlier 2 MiB-granule designs died here.
3. `cudaMemcpyBatchAsync` is broken on CUDA 13.1 + driver 595.91.07 (invalid argument, sticky context poison). Removed from harness.
4. `MAP_PRIVATE` + `FOLL_WRITE` registration is a **CoW-RAM trap**: every pinned page CoW-faults into anon RAM that stays resident after `cudaHostUnregister`; window LRU caps do not bound cumulative RSS (~44 GiB, host thrash — this killed tr16). `MAP_SHARED` avoids CoW but the driver then refuses registration (kernel disallows write-pins on page-cache pages: `pin_fail=13115, pin_ok=0`).
5. GSP-off (user-tested via `NVreg_EnableGpuFirmware=0`) was a red herring — did not change any of the above; may be reverted.

Conclusion: **file-backed cudaHostRegister is a dead end on this Linux/driver stack.** Bounded pinned RAM must come from anonymous pinned memory.

### Final transport design (banked): anonymous pinned hot tier
- `--tier <GiB>`: cudaMallocHost ring, LRU-resident expert blobs keyed by (layer,expert).
- Cold miss: mmap→tier memcpy (once), then pinned H2D. Re-miss: direct pinned H2D from resident slot — 24 GiB/s class, zero bounce.
- Bounded, caps honest, survives process lifetime; matches pcbench2/3 pinned ceilings.
- `PinTier` window machinery (64 MiB lazy windows, boundary-split copies, `pin_ok/pin_fail/pin_used_mib` counters) remains in code but file-backed pins only engage if driver policy changes.

### tr18 results (real 4-stream trace, 192 steps, heat, grouped, pipe0)
| binary | B | K | VRAM | hit | tok/s tier8 | tok/s tier20 |
|---|---|---|---|---|---|---|
| IQ3 | 4 | 44 | 4.28 | .766 | 57.8 | 83.0 |
| IQ3 | 4 | 71 | 6.91 | .860 | 87.4 | 119.7 |
| IQ3 | 8 | 44 | 4.28 | .750 | 54.2 | 82.5 |
| IQ3 | 8 | 71 | 6.91 | .850 | 83.4 | 129.2 |
| IQ2 | 4 | 44 | 2.59 | .766 | 121.4 | 134.1 |
| IQ2 | 4 | 71 | 4.18 | .860 | 175.2 | 190.7 |
| IQ2 | 8 | 44 | 2.59 | .750 | 113.7 | 138.6 |
| IQ2 | 8 | 71 | 4.18 | .850 | 177.9 | 213.1 |

vs tr10 same arms (staging ring, ~8.6 GiB/s wall): IQ2 B4K71 102.8→190.7 (+85%), B8K71 85.5→213.1 (+149%) at identical hit rates. vs prod banked 0.1.55 B=8 aggregate 62.7 tok/s @ ~4.8 GiB expert VRAM: **IQ2 B8 K71 tier20 = 213 tok/s @ 4.18 GiB (+240% at less VRAM)**, cost: 20 GiB pinned host RAM. Micro-bench class harness (no attention/KV path) — capacity/transport proof, not serving numbers.

### Harness notes
- Pool now mapped `PROT_READ|PROT_WRITE, MAP_SHARED` (avoids CoW trap; harness never writes pool). Harmless for tier arms (page-cache reads).
- C++17 pitfall fixed: `std::list::push_back` returns void (use `std::prev(end())`).
- Atomic counter arrays must be value-initialized (`{0}`) — C++17 zero-init trap (tr10 wedge).

## §18 — Equal-budget accounting (tr19, IQ3 = prod quant class, no MTP credit)

Artifacts: `tr19.json` (12 rows: B{1,4,8} x K{28,44} x tier{8,20}GiB). Prod baseline banked earlier (0.1.55, MTP included): B1=78.3 (K44/4.8GiB), B4=62.6, B8=62.7 (K28/2.9GiB — KV row-state collapses K at high batch).

Equal expert-VRAM verdicts:
- B8 @ prod budget K28 (2.73 GiB): tier20 56.4 agg / 7.05 per-stream, hit .623 — ~= prod 62.7/7.8 (prod keeps MTP edge; POC raw). tier8 only 38.8: 8 GiB tier thrashes the ~41 GiB miss footprint at K28.
- B4 @ K44 (4.28 GiB): tier20 83.2 agg / 20.79 per-stream vs prod 62.6 -> +33% at same quant+VRAM.
- B1 @ K44: 58.5 vs prod 78.3 — micro-harness has no MTP/overlap; not engine-fair at low concurrency.
- K44@B8 (82.4) exceeds 12 GiB envelope once prod-measured 451 MiB/row KV floor is added; needs phase-2 KV/shared-prefix work to unlock at B=8.
Tier RAM cost: 8 or 20 GiB pinned host; tier20 needs ~64+ GiB hosts (or 12-16 GiB compromise).

## §19 — Gate-1 shared-prefix KV demo: PASS (kv1.cu, kv1.log)

qwen4exp geometry read from GGUF metadata (Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS shard1): 48 layers, head_count=24, head_count_kv=2, key/value_length=256 -> KV floor 96 KiB/token/row (f16); prod 451 MiB/row = ~4.6K-tok block, matches. (Metadata parse dies on qwen4exp.ple.layer_multipliers object — TODO.)

kv1.cu (fp32 math, ratios dtype-neutral; B=8, S=P/4): naive per-row decode vs Hydragen shared-prefix + online-softmax LSE recombine, one-pass kernel.
- maxdiff = 0.00e+00 at P=2048/4096/8192: prefix/suffix split + LSE recombine is BIT-EXACT vs per-row attention at our geometry.
- VRAM floor (f16, 48L): naive 1.88/3.75/7.50 GiB -> shared 0.56/1.12/2.25 GiB. Freed: 1.31/2.62/5.25 GiB = (B-1)*P*96KiB.
- At prod B=8 4.6K-token rows: KV floor 3.75->1.12 GiB with a shared 4096 prompt; frees 2.62 GiB -> prod expert budget K=28 (2.9 GiB) grows to ~K=68 (5.5 GiB) at equal quant, no IQ2 needed.
- Speed leg: naive my-kernel 6.4-26.9 ms/48L-step vs shared 51-204 ms (0.13-0.20x): single-block-per-head shared kernel is under-parallelized; paper uses batched tensor-core GEMM kernels — phase-2 Triton prerequisite, NOT a gate failure. Bandwidth model: shared cuts step KV traffic 3.3x (15->4.5 GiB fp32 @P8192).

Gate verdict: 451 MiB/row state tax is recoverable exactly; equal-budget B=8 parity (tr19) expected to flip to win once K=28 collapse is removed.

### §19 correction — hybrid architecture (tensor-name audit, shard1)
GGUF tensor audit: 48 layers = 36 x Gated DeltaNet linear (ssm_* + fused attn_qkv) + 12 x full attention (blk.3,7,11,...: attn_q/k/v split, q/k norms, attn_gate, indexer q_proj/k_proj/k_norm) — Qwen3-Next-style 3:1 interleaving. Every layer: hyper-connections (hc_attn_*/hc_ffn_*), MoE (ffn_gate_inp + *_exps + shared exp), PLE tensors (ple_key/value/conv1d/norm_*).
Consequence: KV floor = 12 layers x 1024 B = 24 KiB/token/row (not 96). Prod 451 MiB/row = ~16-19K-token blocks (incl DeltaNet states + indexer + MTP).
Gate-1 corrected capacity numbers (B=8): freed = (B-1)*P*24KiB(+indexer state): P=4096 -> ~0.66 GiB; P=8192 -> ~1.3 GiB. Exactness (maxdiff=0) and mechanism unchanged; PASS still holds, magnitude corrected. kv1.cu speed leg unaffected (per-layer scaling linear).
Model != upstream qwen3_next: custom model file needed (hc_*, PLE, indexer), but ref-sglang (HEAD db5cd6b) has qwen3_5/qwen3_next classes as templates.

## §20 Phase-2 M1 weight export (2026-11-07, the agent)

Goal: get qwen4exp weights into a form SGLang/torch can consume, with experts kept
as raw Strata IQ blobs for our substrate.

Findings:
- `strata` repo = llama.cpp fork (`build/_deps/strata_llamacpp-src`); ggml enum ids
  in the GGUF: Q4_K=12/Q5_K=13/Q6_K=14, IQ2_XXS=16..IQ4_XS=23 (upstream numbering),
  BF16=30, Q2_0=42 (fork-only: {half d; u8 qs[16]}, w=((2bit)-1)*d, qk=64),
  TQ1_0=34/TQ2_0=35 (fork TurboQuant, unused in this file).
- Both fork and 2025-upstream llama.cpp REMOVED public bulk dequant (from_to) —
  wrote shim linking upstream llama.cpp fd27b20 libggml `dequantize_row_*` exports.
- Tensor census (shard1, 1223 tensors): non-expert quants = f32/f16/bf16 +
  Q4_K/Q5_K/Q6_K/IQ3_XXS/IQ3_S/IQ2_S (attn, GDN, ssm_out, shexp, token_embd,
  output). Routers/hc/indexer-proj/PLE = bf16/f32. Experts (144 tensors, all in
  shard1 data region) = IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ4_NL/Q2_0 mixed per layer
  (GSQ-RCO guided).
- Shard2 = single tensor `per_layer_token_embd.weight` [160, 320001536] iq4_nl
  (26.8 GiB PLE ngram embedding table) — kept raw, never fully dequantized.
- GGUF split layout: shard1 carries full table; shard2 has own mini-table.

Artifacts (the reference host):
- ~/hydragen-experimental/deq_dump.cpp + build_deqd3.sh — f32 dump tool
  (links ~/llama.cpp/build/bin libggml; handles q2_0 manually).
- /tmp/q4e_w/ — 1079 non-expert tensors dequantized to f32 (4.948B elems, 19 GB).
- ~/hydragen-experimental/pack_q4e.py — bf16 converter + safetensors packer.
- ~/q4e-m1/ — M1 artifact: 3 bf16 safetensors shards (9.5 GB, all non-experts)
  + manifest.json raw-region map: 144 expert tensors (17.56 GiB, GGUF file offsets)
  + PLE table region. Expert blobs keep native IQ types; our substrate reads raw.

Next: torch eager reference runner for qwen4exp graph (GDN + hyper-connections +
PLE + indexer + MoE with expert streaming), then token-diff correctness gate
against strata binary served with spec-decode off.

Co-edit note: all new files under hydragen-experimental/ and ~/q4e-m1/; no
shared-clone files touched; ~/llama.cpp is a separate upstream clone (fd27b20).

## 20. M1 token parity — PASSED (M1 gate closed)
- Root cause of the last mismatch: the oracle prompt was NOT bare tokens [760,6511,314,9338,369].
  llama-completion prepends the GGUF chat template: token ids [248044, 248045, 846, 198]
  (= BOS, im_start-like, "user", newline) -> full sequence = 9 tokens.
  Identified by nearest-neighbor lookup of EI embres dumps in token_embd (cos 1.0 exact).
- "-b 1" oracle run (q4orclEI) gives clean per-token single-column graphs (g00..g08 = p0..p8).
  Earlier 13-column EF graphs = 4 template + 5 prompt + draft replay; old "11751 Paris" target
  was an artifact of ngram-draft verify columns, not a clean-sequence prediction.
- dbg9 with 9-token prompt, f32 weights (~/q4e_w3 override): my argmax after prompt = 248046
  == EI reference argmax 248046. logits cos 0.892; ladder cos 0.997 at layer 0 decaying to
  ~0.5-0.7 deep (bf16-vs-f32 quantization of IQ3_XXS weights, amplified by 48 layers;
  argmax still exact). Oracle output text: "user\nThe capital of France is\nassistant\n<think>".
- M1 correctness gate: CLOSED. Next: M2 SGLang expert-substrate integration.

## §21 M2-SGLang mini smoke — PASSED (2026-11-07, by the agent)
- Official HF checkpoints found: Qwen/Qwen3.8-Flash-Next (BF16 ~400G, 131 shards) and
  Qwen/Qwen3.8-Flash-Next-FP8 (185.5G, block-FP8 [128,128] dynamic, standard
  quantization_config). Config values match Q4E-CONFIG.md 1:1 (hc_count 4, hc_lowrank 320,
  indexer_budget 2048 ratio 4, eos 248044, layer_types linear x3 + full @4).
- ~/q4e-mini/ = random-weights qwen4_exp (4 layers, 8 experts, tiny ngram vocab) built with
  transformers-native modeling (sgmini.py in ~/hydragen-experimental). Real tokenizer from
  FP8 repo (~/q4e-sg).
- TRAP: both transformers-native and sglang Qwen4ExpTextConfig drop full_attention_interval
  in the full-kwargs parse path, but sglang qwen3_5._refreshes_rope_cos_sin reads it.
  Fix: PYTHONPATH=~/hydragen-experimental/pyhook sitecustomize shim patches both classes

## §21 M2-SGLang mini smoke — PASSED (2026-11-07, by the agent)
- Official HF checkpoints found: Qwen/Qwen3.8-Flash-Next (BF16 ~400G, 131 shards) and
  Qwen/Qwen3.8-Flash-Next-FP8 (185.5G, block-FP8 [128,128] dynamic, standard
  quantization_config). Config values match Q4E-CONFIG.md 1:1 (hc_count 4, hc_lowrank 320,
  indexer_budget 2048 ratio 4, eos 248044, layer_types linear x3 + full every 4th).
- ~/q4e-mini/ = random-weights qwen4_exp (4 layers, 8 experts, tiny ngram vocab) built with
  transformers-native modeling (sgmini.py in ~/hydragen-experimental). Real tokenizer from
  the FP8 repo (staged in ~/q4e-sg).
- TRAP: both transformers-native and sglang Qwen4ExpTextConfig drop full_attention_interval
  in the full-kwargs parse path, but sglang qwen3_5._refreshes_rope_cos_sin reads it.
  Fix: PYTHONPATH=~/hydragen-experimental/pyhook sitecustomize shim patches both classes'
  init to restore the attribute (opt-in env, no site-packages edits, co-edit safe).
- TRAP: sglang JIT kernels need ninja on PATH (it lives in ~/sg-env/bin).
- TRAP: startup warmup crashes in QSA _qsa_write_plan (token_slot_table dim size 0) on the
  mini config; --skip-server-warmup works around it.
- Result: sglang 0.5.23 serves qwen4_exp on the 3080 Ti (port 30099): GDN + QSA +
  hyper-connections + PLE + indexer + FusedMoE all execute; 32/32 concurrent greedy
  requests OK; GPU ~8.1 GiB with mem-fraction 0.5.
- Disk: /home has 138G free; FP8 needs 185.5G. /mnt/docker (229G free) is root-only and
  there is no passwordless sudo. Awaiting user decision on deleting IQ3_S (52G) +
  packs/iq2_xs (35G) from ~/hydragen/models.
## §22 Real FP8 qwen4_exp SGLang substrate — load path (2026-11-08, by the agent)

Real checkpoint `/mnt/docker/hg/q4e-fp8` (185.5 GB, 131 shards) now constructs + loads
on SGLang 0.5.23 on RTX 3080 Ti 12 GB with four opt-in shims in
`~/hydragen-experimental/pyhook/hydragen_paged_store.py` (all env-gated, no site-packages edits):

1. **Vision tower skip**: `--json-model-override-args '{"language_model_only": true}'`.
   The CLI flag `--language-model-only` rejects `Qwen4ExpForConditionalGeneration`
   (allowlist), but the model itself honors the config attribute
   (`qwen3_vl.py` builds no vision, `qwen4_exp.py` skips `visual.*` weights).
2. **PLE ngram table (320,001,536 × 160 fp8 = 47.7 GiB)**: cannot be pinned on this
   60 GB box (pinned ceiling ~32–36 GiB). Native `--ple-offload-backend file` is
   Blackwell/GB10-only. Shim: `HG_PLE_HOSTSIDE=1` → `allocate_ple_host_table` file-backed
   mmap under `SGLANG_QWEN4_PLE_FILE_DIR` + `Qwen4ExpPinnedHostEmbedding.gather` runs
   host-side index_select then H2D copy. Keep `--ple-offload-backend pinned` on the CLI
   to satisfy the startup check; the shim overrides allocation anyway.
3. **Embed (248320×2560 bf16, 1.19 GiB)**: `HG_HOST_EMBED=1` builds the embedding
   numpy-backed on pinned host RAM. `create_weights` calls bare `torch.empty` under the
   device context, which overrides even explicit `device=` args via `__torch_function__`,
   so interception must be allocation-level: swap the `torch` module reference inside
   `unquant`/`parameter` for a numpy-backed proxy (`from_numpy` is context-immune),
   passthrough for sizes outside the embed band so PLE meta-build stays native.
   Gate must use ctor args (`num_embeddings`, `embedding_dim`) — `self.num_indices`
   doesn't exist at wrapper entry.
4. **Post-load staging OOM**: `loader.py:47` name-imports `stage_module_for_post_load`;
   sitecustomize's own `import sglang.srt.model_loader.post_load` pulls `loader.py` in
   before the patch line runs. Must patch `sglang.srt.model_loader.loader` namespace
   too. Wrapper skips staging for modules flagged `_hg_host_embed`
   (unquantized embedding `process_weights_after_loading` is a no-op, safe to skip).

Launch (12 GB config):
```
HG_PAGED_STORE_DIR=~/hg-store HG_PLE_HOSTSIDE=1 HG_HOST_EMBED=1 \
SGLANG_QWEN4_PLE_FILE_DIR=/mnt/docker/hg/ple PYTHONPATH=~/hydragen-experimental/pyhook \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang launch_server --model-path /mnt/docker/hg/q4e-fp8 \
 --json-model-override-args '{"language_model_only": true}' \
 --mem-fraction-static 0.85 --context-length 2048 --max-running-requests 8 \
 --kv-cache-dtype fp8_e4m3 --enable-paged-experts --paged-experts-num-resident 10 \
 --disable-shared-experts-fusion --cuda-graph-backend-decode disabled \
 --ple-offload-backend pinned
```

Banked facts:
- PLE merged runtime table `(320001536, 160)` fp8 = 47.7 GiB (checkpoint stores
  8 shards × 2.5M rows only; runtime materializes full hash space).
- Pinned ceiling on box: 32 GiB OK, 36 GiB FAIL (`ulimit -Hl` 8192 KB is soft;
  driver refuses above ~36 GiB anyway).
- File-backed expert store fills fully: `~/hg-store` ≈ 113 GiB real
  (512 experts × 48 layers × (w13+w2+scales), FP8 block [128,128]).
- Checkpoint FP8 covers experts + PLE only; attention/GDN/head/embed stay BF16
  (hard floor: non-expert weights ≈ 10.4 GiB on GPU; K=10 resident experts ≈ 2.36 GiB
  included in that count; 12 GB card = functional-poc only, decode graphs off).
- Weight load: cold page cache ~14 s/shard (~30 min); warm ~2 s/shard.

## §23 M2-Real: real FP8 qwen4_exp SERVED on 12 GB Ampere (2026-11-08, by the agent)

**Result: the full real checkpoint (186 GB) generates tokens on the reference host (RTX 3080 Ti 12 GB, 60 GB RAM) via SGLang 0.5.23 paged-experts + opt-in pyhook shims. Smoke: "Hi" -> ", what are you" (4 tok, 33.4 s), POST /generate 200 OK.**

Final launch (all via `PYTHONPATH=~/hydragen-experimental/pyhook`, shim file `hydragen_paged_store.py`, no site-packages edits):

```
env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     HG_PAGED_STORE_DIR=~/hg-store   # file-backed expert store (113 GB sparse real)
     HG_PLE_HOSTSIDE=1                               # PLE table -> file mmap + host index_select gather
     HG_HOST_EMBED=1                                 # embed_tokens built/pinned on host, staging skip
     HG_HOST_HEAD=1                                  # lm_head on host, chunked fp32-accum logits
     HG_EAGER_MOE=1                                  # PagedExpertsMoEMethod.apply -> eager host-gather bf16
     HG_QSA_SDPA=1                                   # QSA decode varlen -> torch SDPA (flash_attn cute is Blackwell-only)
     SGLANG_QWEN4_PLE_FILE_DIR=/mnt/docker/hg/ple
args: --json-model-override-args '{"language_model_only": true}' --skip-server-warmup
      --mem-fraction-static 0.95 --context-length 1024 --max-total-tokens 4096 --max-running-requests 2
      --mamba-ssm-dtype bfloat16 (NO --kv-cache-dtype fp8_e4m3: triton fp8e4nv unsupported on sm86)
      --enable-paged-experts --paged-experts-num-resident 10 --disable-shared-experts-fusion
      --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
      --ple-offload-backend pinned   # CLI value only; shim overrides to file backend
```

Ampere walls found (each was one launch cycle, root-caused, fixed in shim):
1. Native `--ple-offload-backend file` refuses on sm86 (device-side file paths = GB10/Blackwell feature). Shim forces file mmap + host gather.
2. Merged PLE host table is (320,001,536, 160) fp8 = **47.7 GiB**; pinned ceiling on box ~32 GiB (36 FAIL). Pinned PLE impossible; mmap+host-gather works.
3. `VocabParallelEmbedding` + `ParallelLMHead` (~1.19 GiB each) starve the KV pool on 12 GB -> host-build via numpy-backed ctor proxy + `_hg_host_embed` flags + `stage_module_for_post_load` skip (patch `loader` binding too - loader imports the fn by name).
4. Block-FP8 triton MoE kernels: `type fp8e4nv not supported in this architecture` on sm86 -> HG_EAGER_MOE host-gather dequant (gate/up half-order: silu(x[:I]) * x[I:]).
5. Same fp8e4nv in QSA KV-extract triton kernel -> just use bf16 KV (--kv-cache-dtype fp8_e4m3 dropped).
6. QSA decode packed attention falls back to `flash_attn.cute` (CUTLASS DSL, Blackwell): `CUDA_ERROR_INVALID_CONFIG` -> HG_QSA_SDPA replaces `_resolve_flash_attn_varlen_func` with per-seq SDPA (GQA repeat_interleave, scale=layer.scaling, is_causal=False since max_seqlen_q=1).
7. eager MoE `.cpu().tolist()` syncs illegal under CUDA graph capture -> both graph backends disabled.

Load time: warm page cache ~9 min for 131 shards (cold ~25 min; store already built). Eager decode ~8 s/token (diagnostic substrate, not perf target). Co-edit note: all hooks env-gated; untouched runs = untouched stock SGLang.
- Capital-of-France greedy 16 tok: ' Paris. Paris is a city. All cities are located on Earth.

Based' — first token Paris = M1 oracle target. M2-Real correctness CONFIRMED.
## §24 M3: Triton w8a16 grouped-GEMM MoE on Ampere — 15x faster than eager (2026-11-08, by the agent)
- New opt-in shim pyhook/hydragen_moe_triton.py (HG_TRITON_MOE=1, HG_MOE_CACHE_N=k): replaces PagedExpertsMoEMethod.apply.
- Kernel: block-FP8 -> register e4m3 decode (manual bit-decode; sm_86 Triton blocks ALL fp8e4nv loads/casts) -> bf16 tl.dot. Validated standalone: cos=1.0 vs fp32 ref, 13.2 us/expert (M=1, N=1280, K=2560), ~250 GB/s. bench: w8a16_bench.py.
- Staging: raw fp8+block scales host store -> per-layer device FIFO cache (N entries) + per-call fresh dict; GEMMs chunked CH=4 experts; A padded Mpad<=128 rows/expert.
- Kernel traps fixed: double-advance of A pointer (initial offs_k AND +=BLOCK_K both), tl.dot needs tl.trans for [N,K] weight layout, loop-local mask var hoisting, eviction must not evict current-call ids (KeyError 71 crash), stack peak must stay chunked (prefill OOM at U=45).
- Memory: paged-experts-num-resident floor = router top-k (10) -> native K=10 slots cost 2.7 GiB even though Triton apply bypasses them; Triton cache N=1 (0.26 GiB) + KV 2048 to fit. mem-fraction 0.85 starves mamba state cache (known-good is 0.95).
- Result (16 tok greedy, capital-of-France): eager MoE 8.35 s/tok -> Triton warm 0.56 s/tok (8.9-9.3 s/16tok), cold-first 1.49 s/tok. Output ' Paris. The capital of Germany is Berlin. The capital of Italy is Rome.' coherent, Paris-first.
- Now bottleneck mix: pageable host->device staging (~50 MB/layer/token miss-dominated at N=1) + per-call python/launch overhead (~12 ms/layer). Next levers: raise N with VRAM plan, pin staging (cudaHostRegister on store pages), fuse chunks into single grouped launch (moe_align-style sorted ids), batch affinity at concurrency>1.
- Launch shape unchanged vs §23 except: HG_TRITON_MOE=1 HG_MOE_CACHE_N=1 (replaces HG_EAGER_MOE=1), --max-total-tokens 2048.

## §25 M3 concurrency verdict: staging-bound, GEMM idle (2026-11-08, by the agent)

B=1 + B=4 measurements on Triton path (HG_TRITON_MOE, ring 192 MiB, K=10, mem 0.95):

| config | B1 warm | B4 wall | agg tok/s |
|---|---|---|---|
| N=4 CH=4 | 10.5 s/16 tok | 49.6 s | 1.29 |
| N=4 CH=8 vectorized gather/scatter | 13.8 | 47.5 | 1.35 |
| N=6 | 12.5 | 44.7 | 1.43 |
| N=8 | OOM at prefill peak (9.8 GB alloc + 160 MB) | — | — |

HG_MOE_PROF=1 per-layer breakdown (warm B=1, 48-layer avg/call):
- stage (host store read + pinned ring copy + H2D enqueue): 7.9-9.3 ms
- gem launch enqueue: 0.3-0.4 ms
- glue: ~0 (noise)
- miss/call ~7-9 of U=10 (LRU cross-step reuse weak: router working set churns per token)

Verdict: decode wall = MoE staging ~1.9 GB/token at ~5.6 GB/s effective (page-cache misses: PLE mmap 47.7 GiB + store 113 GB vs 62 GB RAM). GPU GEMM is 0.3 ms/layer = idle. Cache N growth does NOT amortize naive LRU. Python/vectorization fixes moved nothing (overhead not binding). B=4 gives no win over B=1 on naive substrate (1.43 vs 1.5 tok/s).

Next levers (decision gate): (A) port Phase-1 winner: dedicated host hot-tier (bounded, not page-cache lottery) + batch affinity scheduling — tr18 proved hit .850 at tier20; (B) shared-prefix (Gate-1) attacks prefill cost, orthogonal to this decode wall. Server left live PID 362324.

## §26 M3+ pinned host hot-tier: mechanism proven (2026-11-08, by the agent)

HG_MOE_HTIER=K shim tier (per-layer OrderedDict of pinned expert copies, LRU, sits between mmap store and device cache; H2D direct from pinned, no ring copy).

| config | B1 warm | B4 wall | agg tok/s | stage ms/layer |
|---|---|---|---|---|
| N=4 naive LRU | 12.5 | 44.7 | 1.43 | 7.9-9.3 |
| N=4 + HTIER=12 | 9.1 | 42.6 | 1.50 | 4.8-8.2 |
| N=6 + HTIER=24 (~5.8 GB pinned) | 8.3 | 40.3 | 1.59 | 3.0-3.7 (one 8.4 outlier) |

- Tier converts NVMe-miss staging into pinned-copy+DMA: stage floor now ~3.5 ms/layer = pinned->device DMA of ~45 MB (16 GB/s) — the hard floor without device-resident working set.
- MoE now ~170 ms/token of ~520 ms/token warm B=1; remaining ~350 ms = GDN/SDPA attention layers, host lm_head (1.27 GB/token H2D), sampler/scheduler eager path.
- Host RAM note: pinned tier survives; PLE 47.7 GiB mmap absorbs reclaim pressure. Watch PLE gather latency if tier > ~10 GB.
- B4 outlier: one stream 40 s vs 11 s — scheduler fairness question, not yet investigated.
- Concurrency thesis still open: batch routing affinity (dedup 34 unique->~20 at B=4) untested; that is the Phase-1 ExpertFlow lever not yet ported to serving.
- Device lm_head not viable at load time (peak before slot-free); would need load-after-free sequencing.

Server: N=6+HTIER=24 live PID 363718. Next decision points: (1) HTIER=40, (2) affinity scheduling port, (3) Gate-3 prefix integration (orthogonal prefill axis).

## §27 tier-size curve + RAM partition finding; best config banked (2026-11-08, by the agent)

HTIER sweep (N=6, ring 192, B=1 x3 + B=4 burst, identical prompts):

| HTIER | pinned | B1 warm | B4 wall | agg | stage ms/layer |
|---|---|---|---|---|---|
| 12 | ~2.9 GB | 9.1 | 42.6 | 1.50 | 4.8-8.2 |
| 24 | ~5.8 GB | 8.3 | 40.3 | 1.59 | 3.0-3.7 |
| 40 | ~9.7 GB | 7.5 | 39.8 | 1.61 | 4.4-6.4 REGRESSED |

K=40: free(0)/used 25 GB, buff/cache 51 — pinned tier squeeze shrinks PLE mmap working set; hidden PLE gather NVMe reads grow outside the stage measurement, cancelling tier gains. Optimum ~K=24 on 60 GB host. Expert tier and PLE resident set compete for the same RAM — explicit partitioning (or PLE file-direct reads with smaller mmap) required before scaling tier further.

Note: B=4 aggregate (1.61) < B=1 warm (~2.0 tok/s) — co-decode raises per-step unique experts 10->34, tier hit collapses, NVMe misses dominate (est 6.4 GB/step). Concurrency needs routing-affinity batching or tier x streams sized to 34+churn; naive batching actively hurts.

BEST CONFIG for M3 substrate (leave running as reference): HG_MOE_CACHE_N=6 HG_MOE_HTIER=24 HG_MOE_RING_MB=192, mem 0.95, max-total-tokens 2048, max-running-requests 4. B1 warm 8.3 s/16 tok (1.9 tok/s), B4 agg 1.59, first-token correctness = Paris maintained across all arms. Next: M4 = affinity batching (ExpertFlow lever port) with trace harness re-use.

## §28 M4 offline eval: affinity is a NO-OP; capacity is the lever (2026-11-08, by the agent)

Replayed the 55 MB real-routing trace (C:\Hydragen\traces\route0.bin, 628608x88B records, 4526 full-48-layer blocks, dynamic rows/window {1:1075,2:754,3:607,4:1919,5:10,6:161}) — same arch as qwen4_exp (48 routing layers, top-10/512).

1) B=4 union/layer: natural-first4 19.1 | affinity-optimal subset 19.0 | random 19.2. Affinity grouping has ZERO headroom on real traffic: routing is hot-skewed, random rows already overlap ~50% (union of 4x10 picks = 19 not 40). ExpertFlow affinity scheduling: NOT WORTH PORTING for this model/traffic. Supersedes §27 'next: affinity'.

2) LRU tier capacity sweep (steady-state, churn-inclusive):
| C/layer | B1 hit | B4 hit | B4 staged GB/step |
|---|---|---|---|
| 24 | .531 | .364 | 3.20 |
| 48 | .710 | .676 | 1.63 |
| 64 | .781 | .762 | 1.19 |
| 96 | .872 | .866 | 0.67 |
| 128 | .925 | .922 | 0.39 |

- Hits at B1≈B4 for C>=48: concurrency is nearly FREE once tier covers the hot set — matches Phase-1 'row sharing' thesis via a different mechanism (hot-skew does the sharing, not affinity).
- C=64-96 required for .78-.87 hit = 16.9-25.4 GB pinned host tier at 5.6 MB/expert.
- PREREQUISITE: PLE RAM diet — replace 47.7 GiB full-mmap PLE table (HG_PLE_HOSTSIDE) with bounded LRU + direct pread (PLE per-token working set is tiny); frees ~40 GB RAM for the tier. Without this, HTIER>24 impossible (§27 wall).
- Projected after repartition: stage misses at C=64: B4 1.19 GB/step ≈ 75 ms DMA path -> MoE off critical path; target B1 ~2.5 tok/s, B4 agg ~3 tok/s.

## §29 PLE RAM diet + HTIER=64: new bests (2026-11-08, by the agent)

PLE diet shim (hydragen_ple_diet.py, HG_PLE_DIET=1) live:
- gather -> pread + 262k-row LRU; log: "PLE diet gather installed/active"
- one-shot madvise(DONTNEED) on the 47.7 GiB shared mapping + periodic fadvise DONTNEED
- RAM now: used 32 / shared 22 / available 28 with 17.2 GB pinned tier (HTIER=64) resident. The §27 tier-vs-PLE RAM wall is broken.

HTIER=64 + diet (PID 365926), N=6, ring 192:
- B1_RUN1 55.0 cold; B1_RUN2 9.8; B1_RUN3 6.6 s/16 tok = 2.42 tok/s (prev best 8.3 s)
- B4 wall 34.4 s, agg 1.86 tok/s (prev best 1.59). Text correct (Paris/Berlin/Rome/Madrid).
- hg-prof: miss/call 6-9, stage 2.4-9.0 ms, gem 0.4 — tier hits good on quiet windows, churn outliers remain at B=4.

Remaining wall: B1 warm 412 ms/token; MoE stage+gem ~200 ms -> ~200 ms/token NON-MoE (GDN/SDPA attention, host embed/head, eager sampler/scheduler) now co-dominant. B4 agg (1.86) still BELOW B1 warm (2.42): non-MoE cost per step scales with streams under eager path.
Next: HG_STEP_PROF — wrap ModelRunner.forward timing to split MoE vs non-MoE per step, then attack non-MoE (graph-able attention subgraph / reduce eager overhead).

## §30 non-MoE itemization: fp8 streamed head, resident-head OOM, PLE diet cheap (2026-11-08, by the agent)

HG_STEP_PROF=1 (ModelRunner.forward sync-timed): fwd avg 376-380 ms B=1 decode.

1) Resident head FAILED on VRAM: tied head = V248320 x H2560 bf16 = 1.19 GiB; at 0.95 only 600 MiB free. mem-fraction floor is ~0.87: at 0.78 mamba-state pool rejects (max_mamba_cache_size=-10, 56.11 MB/req). Resident head impossible on 12 GB with this config. (HG_HEAD_RESIDENT code kept.)
2) HG_HEAD_FP8=1 deployed: head e4m3-quantized host-side (per-row scales), streamed 65536-row chunks, dequant->bf16 on device, matmul on device. Wire 0.6 GB/step vs 2.4 GB bf16: lm_head 73.0-73.3 ms/step (was ~150 est bf16 chunked). Zero VRAM cost.
3) PLE diet gather: 0.8 ms/avg call — diet is effectively free.
4) Correctness with fp8 head: first token ' Paris' + Berlin/Rome/Madrid chain unchanged (greedy margins large; caveat logged for marginal tokens).

New bests (HTIER=64 + diet + fp8 head, N=6, ring192, PID 369688):
B1_RUN2 6.0 s/16 tok = 2.67 tok/s (variance run2 6.0 / run3 11.0 — tier warm vs churn); B4_WALL 32.3 s agg 1.98 tok/s.

Budget at 380 ms/step: MoE ~170 (stage~2.6+gem .35+glue x48) + head 73 + PLE ~1 -> ~135 ms UNACCOUNTED = GDN x36 + SDPA x12 + embed + sampler + scheduler eager glue.
Next lever: that 135 ms eager overhead; CUDA graphs blocked by MoE router .cpu() sync -> M5 = graph-compatible MoE (pre-staged experts, no in-forward host sync) or per-module graphlets.

## §31 M5 anatomy + pinned fp8 head: B1 3.5 tok/s (2026-11-09, by the agent)

torch.profiler one-decode-step (HG_STEP_TRACE=1, forwards #40/#72):
- Self CUDA 207 ms/step: aten::copy_ 171 ms (HtoD pinned 102 = tier 1.9GB/token PCIe floor; HtoD PAGEABLE 64 = unpinned fp8 head wq). ALL GEMMs ~28 ms (mm 11 + _hg_gemm_e4m3 9.5 + ampere 5.4 + cutlass 2.4). cat/eltwise/GDN ~15 ms. GEMMs 13% — pure copy-bound.
- aten::copy_ Self CPU 840 ms: pageable HtoD stalls dispatch loop (CPU total 1.144s vs CUDA 207ms).

FIXED: pinned fp8 head buffers (empty_like pin_memory at quant time): lm_head 73 -> 33 ms. Run-to-run variance collapsed (run2/run3 4.7/4.4 vs prior 6-11s band).

N=10 device cache OOM'd (+1.05 GB over N=6; 10.19 allocated, 27 MiB free) even post diet+slot-free. N=6 = hard ceiling on 12 GB w/ static pool 0.95. Banked.

New bests (PID 372826): B1 warm 4.4-4.7 s/16 tok = 3.4-3.6 tok/s; B4 wall 30.8 s agg 2.08 tok/s. Correctness intact (Paris/Berlin/Rome/Madrid).
Budget ~275 ms/token: pinned H2D ~102 (floor at 16 GB/s PCIe) + head 33 + GEMM 28 + attention/glue ~100.
Next levers: (1) overlap tier copies with GEMM via side-stream chunk pipeline in apply (CH=8 chunks already); (2) wire-diet expert quant below fp8 (int4/IQ2 — Phase-1 kernel exists in POC); (3) B4 scheduler fairness (4.3s vs 30.8s streams).

## §32 side-stream staging: no win, kept opt-in off (2026-11-09, by the agent)

HG_MOE_S2=1 (copies on stream 2, per-expert events, chunk GEMMs wait own experts):
- B1_RUN2 4.0 s/16 tok = single-run best (4.0 tok/s), B1_RUN3 10.8 s — variance band persists.
- B4 wall 32.9 s agg 1.95 (vs 30.8/2.08 serial §31). Slightly WORSE at B=4.
- hg-prof stage 5.6-7.8 ms now measures ISSUE time only (completion async) — wall is truth.

Verdict: within-apply overlap can't pay: staging issues all miss-copies up-front and chunk0 waits on the copy queue anyway (copies 2.6 ms/layer >> GEMM 0.35 ms/layer to hide); extra event plumbing costs a bit at B=4. Code stays gated, DEFAULT OFF. Reference config remains §31 (N=6 HTIER=64 diet fp8-head-pinned, no S2): B1 4.4-4.7 s, B4 agg 2.08 tok/s.

Known issue logged: greedy same-prompt outputs differ run-to-run (RUN2 vs RUN3 variants). Cause: on-device index_add_ float combine is non-deterministic in ordering; near-tie argmax flips. Text stays coherent/correct-class; exact reproducibility needs deterministic combine (sort-by-expert or double accumulation) — NOT chased, note for any future strict-parity arm.

Remaining honest levers ranked: (1) wire-diet experts below fp8 (int4 halves 102 ms PCIe floor; Phase-1 IQ2 kernel exists in POC, port = new Triton kernel + store repack), (2) B4 stagger analysis (streams finish 4-33 s apart = partial co-decode; scheduler admits sequentially as slots free), (3) tier K sweep 64->96 (RAM has ~11 GB headroom post-diet; trace says +0.11 hit at C=96).

## §33 HTIER=96: B4 agg 2.47 tok/s (2026-11-09, by the agent)

HTIER=96 (25.4 GB pinned tier; N=6, diet, fp8-head-pinned, no S2):
- B1_RUN2 4.2 s/16 tok; B4_WALL 25.9 s agg 2.47 tok/s (was 30.8/2.08 at HTIER=64). New bests.
- RAM: used 42 / shared 32 / available 18 — tier fits post-diet, no stage regression (4.4-6.8 ms).
- Trace prediction confirmed: hit .76->.87 band at C=96 -> fewer NVMe/store reads per miss; B4 wall -16%.
- Session cumulative: B1 warm 8.3 -> 4.2 s/16 tok (2.0x); B4 agg 1.59 -> 2.47 tok/s (1.55x).
Next: HTIER=128 probe (trace hit .92; +8.5 GB pinned against 18 available — watch §27-style regression; stop at first stage-time rise).

## §34 tier curve closed: HTIER=128 hits the pinned ceiling; 96 is the banked optimum (2026-11-09, by the agent)

HTIER=128 (33.9 GB tier + ring/head ~0.8): stage regressed 10-14 ms + glue 28-32 ms, then CUDA OOM mid-run — torch reports "11.48 GiB in use incl NON-PYTORCH": cudaMallocHost-backed memory counts against the per-process GPU allocation accounting. Pinned total ~34.7 GB crossed the same driver wall measured for the PLE-pinned experiment (32 OK / 36 FAIL). Pinned ceiling, not page-cache pressure, is the binding limit past tier 96.

Tier curve (same battery): HTIER 24 -> 8.3 s B1 / 1.59 agg; 64 -> 6.6 / 1.86; 96 -> 4.2-4.7 / 2.47 (BEST); 128 -> FAIL. Practical pinned budget for tier on this host ~= 26 GB total incl ring+head => HTIER 96-100.
Further hit-rate gains (trace .87 -> .92 at C=128) need SMALLER bytes/expert, not more tier: int4 wire-diet is the next lever.

Reference config re-pinned: N=6 HTIER=96 diet fp8-head-pinned no-S2.
## §35 M6 int4 wire-diet: GATED OUT at quality gate (2026-11-09)
Gate-first: w4_a16 microbench (w4_bench2.py, real layer-0 experts, 16 sampled, CPU exact-dequant reference) run BEFORE any repack.
Results: per (row, K-block-32) symmetric signed-nibble Q4_0-style:
- weight roundtrip rmse/std 0.496, max-abs 0.2285 vs ref std 0.0132 (error > largest weights);
- GEMM-output cos vs exact-dequant fp8: min 0.866 / mean 0.868 (gate >=0.99 FAIL);
- Triton w4 kernel 28.5 us/expert vs 13.2 w8a16 (scalar nibble unpack 2.2x ALU penalty), packed 1.76 MB/expert.
Root cause: e4m3 per-value exponent survives outlier-heavy blocks; uniform 4-bit + per-32 scale does not.
Closing to cos>=0.99 needs ~5.5-6 effective bits (Q4_K/IQ4-class recipes land ~0.93-0.97 est. on this data),
where wire savings vs fp8 ~9bpw largely evaporate. Phase 1 IQ2/IQ3 survived because its quality bar was
argmax-parity chat POC, not 0.99 GEMM fidelity.
VERDICT: int4 wire-diet CLOSED on this checkpoint. fp8 experts are near the quality-safe floor for wire bytes.
Remaining capacity levers after pinned ceiling (S34): scheduler fairness for B=4 wall outliers; or accept
banked best (B1 4.2s/16tok, B4 agg 2.47 tok/s).
## §36 OPS (2026-11-09): Strata prod stack on ms01 stopped+disabled by user decision
systemctl --user stop+disable hydragen.service strata.service (unit files kept; autostart wants-links removed).
Reason: 12 GB GPU + 60 GB RAM needed for Phase-2 SGLang arms; prod held 11.2 GB VRAM + RAM headroom.
hydragen-actions-runner.service and hydragen-exp.service left untouched.
To re-enable prod: systemctl --user enable --now hydragen.service
SGLang reference relaunched via launch_phase2_best.sh (N6/HTIER96 banked config).
## §37 M7 ROOT CAUSE: concurrency was NEVER on — mamba cap silently pinned running batch to 1 (2026-11-09)
Fairness probe (battery_fairness.py, per-token SSE arrivals) shows B4 streams decode STRICTLY SEQUENTIALLY:
S0 first_tok=0.8 last=8.3; S1 first_tok=9.4; S2 first_tok=18.7; S3 first_tok=23.8 — each stream solo,
next starts only after prior finishes. Wall = 4x serial replay, not batched.
Log smoking gun: "max_running_requests is capped to 1 by the mamba state cache (max_mamba_cache_size=7,
5 state slots per request)" (context 1024 -> 5 slots/req; pool sized auto at mamba-full-memory-ratio 0.9).
REINTERPRETATION: all banked B4 agg numbers (1.29 -> 2.47 tok/s) are SERIAL throughput with a hotter tier,
never true decode-batching. Capacity model §28 (B1/B4 hit converge) still valid; batching upside unmeasured.
M7 arm: launch_phase2_m7.sh = best config + --max-mamba-cache-size 20 (20 slots / 5 per req = batch 4).
Cost ~+0.8 GB ssm pool vs prior 0.44 GB; fits via K-slot free (2.2 GB recovered at first apply).
Expect true-batch B4: per-token wire bytes drop ~2.1x (union 19.1/4 = 4.8 vs 10 experts/token/layer, §28 trace math).
## §38 M7 PARTIAL WIN: true decode batching at cap 2 -> agg 4.29 tok/s (2026-11-09)
m7c = best config + --max-mamba-cache-size 10, HG_MOE_CACHE_N=5 (running cap 2: 10//5 slots/req).
Fairness probe burst2: S0+S1 lockstep (first 1.77/2.15s, both done 9.35s), then S2+S3 lockstep (10.44->14.93s).
PAIRED step ~0.28 s vs single ~0.26 s = batch doubles throughput for ~8% cost. Hot-tier paired rate ~3.5 tok/s per stream.
B4 wall 25.9 -> 14.9 s; agg 2.47 -> 4.29 tok/s (still only batch-2!).
m7b lesson: --mamba-max-states-per-path 2 is a NO-OP for slots/req; the 5/req comes from
context_len/track_interval+1. m7d arm: --mamba-track-interval 1024 --max-mamba-cache-size 10 -> 2 slots/req, cap 4.
VRAM law: mamba pool +0.25 GB forces N=6->5 device cache (warmup-forward OOM right after "ready to roll";
watcher must soak 75 s past ready to catch it). pkill -f "launch[_]server" sometimes leaves the scheduler
child (382235) alive holding GPU -> kill tree by PID after pkill, verify nvidia-smi empty before relaunch.
## §39 M7 CLOSED (2026-11-09): true decode batching works; saturates at 2-3 on HTIER=96; head-chunk peak fix
Root chain complete: silent cap 1 (mamba 5 slots/req = base3+extra2) -> SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 (ratio 4).
--mamba-track-interval / --mamba-max-states-per-path: confirmed NO-OPS for slots-per-req.
m7e (pool16, N=5): booted, died req1 at head bf16 chunk = 320 MB single alloc (65536 rows x 2560 bf16) + 160 MB fp8 chunk.
FIX: pyhook HG_HEAD_CHUNK env, CH default 65536->32768 (peak 480->240 MB). Prior arms survived this hidden peak only on pool margin.
VRAM law: each mamba slot ~59 MB off pool-end headroom (pool10: 0.62GB free, pool12: 0.50, pool16: 0.28 -> fatal).
Fairness battery on cap 3 (m7g): B1_WARM2 4.2 s (B=1 no regression). BURST2 warm: trio lockstep done 10.4 s + solo -> wall 15.4 s agg 4.16 tok/s.
cap2 = 4.32, cap3 = 4.16, cap1-serial = 2.47. VERDICT: batching gain +74% then SATURATES at 2-3: union growth per extra stream
(offline: 10 -> 12.6 -> 19.1 unique/layer) consumes the win at HTIER=96. Cap-4 not pursued (+118 MB state tax buys nothing).
Next: M8 RadixAttention shared-prefix arm (Gemini #2, the one live lever).
## §40 M8 CLOSED (2026-11-09): prefill physics + shim v2 + RadixAttention 2.03x wall
Radix works on hybrid qwen4_exp: trio shows #cached-token 1152/1205; PRE_B4 wall 123.9 s vs NOPRE 252.2 s = 2.03x (agg 0.52 vs 0.25 tok/s @4 streams, 1024-token shared prefix).
PREFILL LAW: union/layer saturates ~ all 512 experts for any prompt >~100 tok on UNIFORM traffic (512*(1-e^(-rows*10/512))); repetitive prompts route NARROW -> 1205-tok prefill = 87-100 s, not the 130 GB sweep uniform math predicts. Always model routing union on the ACTUAL content distribution.
SHIM v2 (hydragen_moe_triton.py): per-chunk staging lifetime (stage 8 -> stack -> GEMM -> free; peak 73 MB vs 2.5 GB+; fixes prefill OOM); U x Mpad intermediates per chunk; _PREFILL_MODE (U>4N) skips device-cache AND pinned-tier admission (one-shot sweep experts evicted the hot decode set: decode 0.4 -> 6.5 s/step before fix); S2 side-stream code deleted.
Battery bug banked: control prompts must diverge at token ~4 (marker words) AND match length — first M8 run's "control" shared the prefix by construction (fake 164 vs 142).
Gemini-relayed 5-item audit closed: fairness=M7 root-cause (done), radix=M8 (done, 2.03x), glue=already ruled out (§25/31), chunked-prefill=already on, multi-stream GEMM=already how the kernel works.
## §41 M9 LOADER-SKIP UTILITY: boot 1014s -> 164s (79/131 shards, 130.2 GB skipped), battery-verified
Gate-first loading-time utility for all future arms. pyhook/hydragen_loader_skip.py (v2).

Safety design (law banked the hard way):
- hg-store bytes are BYTE-UNSTABLE across boots (native loader rewrites shared
  torch.from_file mappings; requant churn) -> store hashing CANNOT gate skip.
- v2 fingerprints the CHECKPOINT instead: sha256(config.json)+sha256(index.json)
  +64KiB head/mid/tail of 3 sampled shards; plus store file-set + PLE size sanity.
- Preflight: launch script runs `--check <ckpt_dir>` BEFORE launch; workers trust a
  fresh stamp (<=20 min). Avoids racing concurrent workers against loader refresh.
- HG_SKIP_KINDS=exp,ple bisect knob (per-class skip control).

ROOT CAUSE of first skip-boot degeneracy ('the capital of France is the capital of'):
sglang/srt/layers/moe/paged_experts/method.py create_moe_runner seeds the WHOLE host
store: `host.copy_(getattr(layer,name).data[:1].cpu().expand_as(host))` (padding
defaults per its own comment). With expert shards skipped the module params are
uninitialized -> the seed BROADCASTS GARBAGE THROUGH THE SHARED MMAP, actively
clobbering the verified hg-store bytes. Skipping did not merely omit writes; it
destroyed the very files it relied on. (Full boots self-heal by rewriting everything;
that masked it.)
FIX (shim-wrapped, version-fragile — co-editors beware on sglang upgrades):
HostExpertStore.__init__ parks self.host -> _hg_host, sets host={}; seed loop
iterates zero times; PagedExpertsMoEMethod.create_moe_runner wrapper restores
store.host right after. No seed touches the store under HG_SKIP_LOAD=1.

Banked results (battery_fairness.py = oracle gate: ' Paris'/' Berlin' greedy):
- full boot baseline: 1014s (warm) / 1398s (cold, concurrent-boot contention)
- expert-only skip (KINDS=exp): boot 575s, battery PASS
- full skip (79/131, 130.2 GB): boot 164s cold / 118s warm, battery PASS
- boot-time physics: PLE table re-read (~49.6 GB in 31 files) dominates non-skip
  boot; cold shard reads ~16s/it vs warm ~3.5s/it (60 GB RAM < 130 GB file set).
- m8 launch script now preflights --check and auto-exports HG_SKIP_LOAD=1 on OK.
Method: kill-server + repair-full-boot + stamp-from-verified-store sequence is the
canonical way to re-anchor the manifest after store clobbering.

## 42. M9 donor IQ audit — M9-0 + M9a fidelity gate (2026-11-09 20:30)
- M9-0: libstrata_iq_kernels.so iq_supported/iq_row_bytes respond for 16
  IQ-family types (IQ2_XXS..IQ4_XS + custom 29/30 ~1.75/2.0bpw STRATA class).
  iq_dequant_f32 is a CUDA launcher — illegal memory access on CPU, do not
  ctypes it. ulimit -Hl = 8 MB on the reference host (mlock tier needs limit raise).
- Donor: ~/.models_backup/Strata-data/models/IQ3_XXS/
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS (2 shards, 74 GB) — GGUF arch=qwen4exp,
  EXACTLY the qwen4_exp model (48L, 512 experts, ffn 640, hid 2560).
  Fused expert tensors: blk.L.ffn_{gate,up}_exps dims (2560,640,512) type 17
  (IQ2_XS 2.31bpw, 473600 B/expert), ffn_down_exps (640,2560,512) type 20
  (IQ4_NL 4.5bpw, 921600 B/exp). Total 1.87 MB/expert = 3.0x fp8 wire diet.
  Layout matches torch expert byte order (no transpose).
- dq_shim.c/.so: ctypes CPU dequant built against ~/llama.cpp
  (link ggml-quants.c compiled-in; dequantize_row_* NOT exported from prebuilt
  libs; rpath absolute; preload libggml+libggml-cpu RTLD_GLOBAL).
- M9a (m9a_iq_gate.py): layer-0 16 experts, IQ-donor vs official-FP8-ref GEMM cos:
  gate/up cos 0.938-0.951 (rmse 0.004), down cos 0.989-0.994 (rmse 0.0016).
  MEAN 0.961 MIN 0.9385 -> numeric gate FAIL vs >=0.99 bar.
  BUT: cos >> 0.5 proves SAME WEIGHTS (GSQ-RCO is not a fine-tune), and this
  exact recipe is the Strata production brand (3090 daily-serving precedent).
  int4 comparison: IQ2_XS 0.94 > our int4 0.866.
- Decision: GEMM-cos bar is not the final arbiter; serve-side battery vs fp8
  banked outputs is the quality gate for M9. Wire 3x diet stands.

## 43. M9c IQ blob store + M9b native grouped kernel PASS (2026-11-09 21:10)
- hg-store-iq: per-tensor IQ files, byte-faithful to donor GGUF (types 16/17/18/20/21/22/42
  all present; 42=Q2_0_STRATA unknown to ggml -> llama.cpp shim cannot dequant those,
  Strata dequant_f16 handles them fine). Verified L0 type17/20 + L1 type16 + L35 type18
  byte-identical vs GGUF + sane dequant std. NOTE iq_row_bytes(type, COLS) = per-ROW bytes.
- hg-store-iq-blobs: 48 x L{NN}_blobs.bin, 512 contiguous expert blobs [gate|up|down],
  blob_bytes 1868800 (gu_row 740 x 2*640 rows + d_row 360 x 2560 rows) — EXACTLY matches
  strata expert_blob_info(). 40 GB total. manifest.json has per-layer gu_type/d_type.
  (gate/up types differ per layer but share type within layer; assert enforced.)
- M9b (m9b_native_bench2.py): strata_vllm_native_expert_grouped on 8 blobs / 32 tok:
  NATIVE_GROUPED 5.2 us/call -> 0.65 us/expert (Triton w8a16 was 13.2 us = 20x faster),
  FIDELITY cos 0.9999 vs dequant-ref SwiGLU forward (silu(gate)*up then down).
  ABI: x_q8_1 must be 20B/16elem (undersized buffer = NaNs); grp_ptr[u64 dev ptrs],
  grp_start[i32 prefix], ent_dst=dest entry idx in out[ENTRIES,n_embd], ent_tok=x row;
  kernel quantizes x itself (pass x_f32); scratch=expert_scratch_bytes(cap_entries,n_ff).
- Next: M9c serving shim HG_MOE_BACKEND=native — staged ring holds raw IQ blobs
  (3x smaller slots), device cache holds IQ blobs (1.87MB -> N=24 fits), GEMM = one
  native grouped call per layer per step, combine = strata_vllm_moe_combine_multi.

## 44. M9 native IQ serving battery (2026-11-09, agent the agent)
- Deployed pyhook/hydragen_moe_native.py (HG_MOE_NATIVE=1): pinned IQ-blob tier (HTIER=288, ~26 GB), device IQ cache (HG_MOE_CACHE_N=12, 22 MB VRAM), token-chunked strata_vllm_native_expert_grouped (HG_NATIVE_TCHUNK=32), torch combine. launch_phase2_m9.sh.
- M9b microbench caveat corrected: cos 0.9999 there proved kernel==IQ-dequant consistency, NOT fidelity vs FP8 checkpoint.
- M9e (m9e_vs_fp8.py, ng vs FP8-checkpoint expert forward, same experts/x): (17,20) L0 cos 0.918-0.942; (16,42) L1 cos 0.81-0.84; (22,42) L12 cos 0.84-0.88.
- Serving battery on native IQ boot: throughput real (BURST1 4.99, BURST2 6.90 agg tok/s vs 4.32 banked fp8 cap2) but text GARBAGE (B1_WARM2 = " 3, notelyesian- 4 (0- > )  </"). Battery gate FAILED.
- Verdict: donor IQ3_XXS mix (9 layers IQ2_XXS, 10 IQ2_S, 13 IQ3_S, 10 IQ2_XS, only 6 IQ3_XXS; 24 layers custom-42 down) is too lossy through 48-layer MoE stack. Kernel plumbing correct; quantization error compounds.
- Harness traps: dequant_f16 and native_expert_grouped share state - calling dequant first poisons next grouped launch ("invalid argument"); never mix entry points in one process as reference. OOB pointer arithmetic in probes caused earlier phantom NaNs.
- IQ3_S GGUF recipe metadata exists (packs/iq3_s/native_experts.txt, v3 blob table) but GGUF files are NOT on disk.
- Next decision: (a) close IQ line, bank throughput law; (b) repack gate/up at higher bpw from FP8 checkpoint and re-gate.
## 45. M10 Option-B donor splice: q4e-iq checkpoint BUILT (2026-11-09)
- Builder: m10_build_v2.py + dq_shim2.so (11 ggml dequant cases, runtime ggml_type_size/blck_size).
- Donor: GSQ-RCO IQ3_XXS split GGUF. Split-GGUF law: each shard has its own tensor table with PER-SHARD offsets (shard1=1223 tensors tiling 47.03GB, shard2=1 tensor: per_layer_token_embd, 28.80GB local). Merged global-offset view reads garbage.
- GDN law for this donor: attn_qkv/attn_gate/ssm_out rows are ALREADY HF-order (no _LinearAttentionVReorderBase applied at conversion). Do not reorder.
- iq4_nl (ngram/down_shexp): row bytes = nx/32*18 at 90B stride for nx=160; earlier "NaN/garbage" was the shard-offset bug, not the codec.
- Sanity cos (donor vs HF-fp8 ckpt, float-class=1.0 identity): attn 0.9966-0.9998, GDN 0.9966/0.9898/0.9961, shared 0.9871-0.9959, embd 0.9874.
- Build: /mnt/docker/hg/q4e-iq, DONE replaced_tensors=302; expert+ngram shards hardlinked/kept (fp8 ngram served from PLE file dir; donor IQ4_NL ngram NOT used — disk budget); verify: rebuilt k_proj cos vs donor 0.999999, vs fp8-ckpt 0.9962.
- Launch: launch_phase2_m10.sh = m9 native env (HG_MOE_NATIVE=1, N=12, TCHUNK=32, HTIER=288) + model-path q4e-iq. First boot full (loader-skip stamp stale -> expect ~1000s); battery_fairness.py is the gate.
- disk: /mnt/docker 6.4G free after build; no headroom for extra copies.
## 46. M10 Option-B battery verdict: IQ3_XXS expert tier is context-independent quality-fatal (2026-11-09)
- q4e-iq native boot READY at 1794266976 (full boot ~15.7 min; loader-skip later armed via --make/--check since exp+ple classes are byte-identical to q4e-fp8).
- battery_fairness: FAILED. B1_WARM1/2 = 16x "!". Capital+neutral prompts both -> "!!!!".
- Burst throughput 14.52/17.30 agg tok/s appears inflated by degenerate repetition; do not bank as serving win.
- Splice integrity re-proven: safetensors headers clean (lm_head/embed BF16, ngram F8_E4M3 preserved), rebuilt tensors cos vs donor 0.999999.
- Verdict: M9 (fp8 dense + IQ3_XXS experts) and M10 (donor dense + IQ3_XXS experts) both battery-fail -> the IQ3_XXS routed-expert tier is fatal independent of dense context. Matches HF card: IQ3_XXS is the cheap tier; parity tier is IQ3_S (3.50 bpw / 83.6 GB), NOT on disk (only 1.5 GB metadata pack present).
- Speed law still banked: native IQ grouped-kernel path sustains >=4.99-6.90 tok/s FP8-comparable bursts with 42.9 GB wire vs 113 GB; the substrate plumbing works, the IQ3_XXS weights do not.
## 47. M11: IQ3_S parity tier arm (download started, 2026-11-09)
- User chose option 1: test Strata actual parity tier IQ3_S (3.50 bpw, 83.6 GB) after IQ3_XXS verdict (46).
- Deleted dead IQ3_XXS stores: hg-store-iq (per-tensor) + hg-store-iq-blobs (native blobs). Re-derivable from IQ3_XXS GGUF which remains on disk.
- Download: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF IQ3_S 2 shards (54.8G+28.8G) -> ~/.models_backup/Strata-data/models/_iq3s_dl (hf CLI, setsid). rco-allocation.txt fetched (per-tensor RCO type map).
- Planned flow: builder m10_build_v2.py retargeted (GGPATHS->IQ3_S, DST=/mnt/docker/hg/q4e-iq3s; q4e-iq to be removed for space), blob extract GGUF->native blobs direct, native launch (launch_phase2_m10.sh pattern), battery_fairness gate.
- Disk math: /home 112G free pre-DL; GGUF 84G -> ~28G left; IQ3_S expert blobs ~50G need swap: q4e-iq (44G, dead) to be deleted on /mnt/docker + possible IQ3_XXS GGUF retirement AFTER IQ3_S blob extraction. Verify exact expert bytes from GGUF table before placement.
## 48. M11 IQ3_S substrate built; native boot started (2026-11-09)
- IQ3_S GGUF downloaded exact-size (54.8G+28.8G). hf-download law: passing explicit filenames alongside --include silently drops the globs; pass full repo paths.
- m11_extract_blobs.py: GGUF -> native blobs direct (no per-tensor store): BLOBS_DONE 50.29 GB, 48 layers; combos (18,20)(18,42)(21,20)(21,42)(22,20)(22,42)(23,20). Native .so iq_supported: ALL six types true (mangled C++ symbols; bare names undefined).
- Retarget gotchas: GGUF tensor-skip loop needs 8*nd+4+8 (ty is u32); builder misc-copy skips index.json via ".safetensors" substring -> copy manually.
- m11 dense sanity: attn/GDN/shared all 0.994-0.9998 cos, float-class 1.0 (better than XXS tier).
- q4e-iq3s built: DONE replaced_tensors=302; q4e-iq (dead XXS splice) deleted; IQ3_XXS donor GGUFs deleted by user consent.
- launch_phase2_m11.sh: HG_IQ_BLOB_DIR=hg-store-iq3s-blobs (shim now env-overridable), HTIER 288->256 (50.3 GB wire raises pinned tier cost), full boot PID 423571; battery_fairness gate next.
- Ops law: kill stale server before relaunch - old q4e-iq native server held 11.7 GB VRAM + port 30099; first m11 launch died on it (PID 423472), clean relaunched 423571.
## 49. M11 IQ3_S native battery FAILED; bisection matrix isolates expert path (2026-11-10)
- M11 native (donor-S dense + IQ3_S blobs): battery garbage ("kgclzur", "CapTVP+") - structured garbage, not "!!!!".
- m11_fidelity.py (native ng vs FP8-ckpt expert fwd, store=hg-store-iq3s-blobs): L0 combo(18,20) cos 0.943-0.963 -> blobs byte-faithful (quant-loss band, not corruption); extraction pipeline exonerated.
- M11T (donor-S dense + Triton FP8 experts, launch_phase2_m11t.sh): BATTERY PASS " Paris. Paris is a beautiful city." -> donor dense splice mechanics fine end-to-end in serving.
- Matrix: (fp8,fp8)PASS (donerS,fp8)PASS (fp8,XXS)FAIL (donerXXS,XXS)FAIL (donerS,S)FAIL (fp8,S native) = M11N pending.
- VRAM law update: native device cache cost = N x blob_bytes x 48; IQ3_S blob 2.05 MB vs XXS 1.75 MB -> N=16 costs +0.5 GB pool-end -> lm_head OOM at first request (M7 pool-end floor ~0.5 GB). IQ3_S native config: CACHE_N=12, HTIER=256.
- battery_fairness exit-code trap: piping through head masks python traceback; run unpipeled for real exit code.

## §50 M11N native IQ3_S SERVING FIXED — two root causes (the agent, 2026-11-10)
Battery gate on FP8-dense + native IQ3_S experts now PASSES:
B1_WARM2 ' Paris, which is also the country's larg', burst agg 6.97 tok/s (vs 4.32 FP8 Triton cap2 banked).
Root cause 1 (ABI): entries must be globally ranked (expert asc, token-major idx) like k_group_small;
first-seen-group order corrupted all but coincidentally-aligned groups (probe m11_abi.py).
Root cause 2 (device cache): _cache_slot handed s=len(order) after mid-list victim removal ->
slot collision with a still-cached expert -> wrong expert bytes under eviction (prefill).
Correct at small T (unique<=N) which hid it in microbenches. Fix: reuse victim slot id via free list.
Verification ladder now: m11_abi.py (serving-shape probe) -> HG_NATIVE_DEBUG=1 dump +
m11_replay.py vs fp8-ckpt forward (L0 real-traffic cos 0.996 at decode shape) -> battery_fairness.py.
Shim: hydragen_moe_native_v3.py deployed as pyhook/hydragen_moe_native.py.
iq_kernels.cu source copy added at strata/src/kernels/cuda/ (read-only reference for co-editors).
Flagship donor-S dense + native IQ3_S (launch_phase2_m11.sh) battery in flight.


## §51 M11 CLOSED — IQ3_S parity tier PASSES full native serving (the agent, 2026-11-10)
Flagship donor-S dense (q4e-iq3s) + native IQ3_S blobs (launch_phase2_m11.sh, PID 431911, skip boot):
B1_WARM1 ' Paris.\nThe capital of France is Paris.', B1_WARM2 " Paris, which is also the country's larg".
Bursts: agg 6.22 and 4.44 tok/s. M11N (FP8 dense + native IQ3_S) also PASS: 5.05/6.97 tok/s.
Full 2x2+b matrix now:
  FP8 dense + Triton FP8            PASS  2.47 serial / 4.32 cap2
  donor-S dense + Triton FP8        PASS  (splice integrity proof)
  FP8 dense + native IQ3_S          PASS  5.05-6.97 burst
  donor-S dense + native IQ3_S      PASS  4.44-6.22 burst  <- flagship
Earlier IQ3_XXS failures: both root causes (ABI ranking, cache slot collision) fixed;
XXS re-validation not run (IQ3_S is the quality-recommended tier; XXS optional follow-up).
Native IQ path is now the fastest banked aggregate on this substrate: sub-3bpw routed-expert wire,
50.29 GB IQ3_S blob store, HTIER=256, CACHE_N=12, TCHUNK=32.
Caveats: battery = short greedy prompts only; burst-to-burst variance suggests cache-tier tuning next.


## §52 D: IQ3_XXS re-validated with fixed shim = QUALITY PASS (the agent, 2026-11-10)
FP8 dense (/mnt/docker/hg/q4e-fp8) + native IQ3_XXS blobs (hg-store-iqxxs-blobs, 42.91 GB,
extracted from GGUF shard-1 only; ngram shard never needed for expert-only arm).
launch_phase2_m11x.sh (HG_IQ_BLOB_DIR override, HTIER=288, CACHE_N=12, TCHUNK=32):
B1_WARM1 ' Paris, which is also the largest city i' / B1_WARM2 " Paris, which is also the country's larg".
BURST agg 8.25 and 8.47 tok/s (fastest banked on this substrate); single-stream ~5.5 tok/s.
M9-era "IQ3_XXS quality-fatal" verdict is formally dead: it was the two shim bugs
(entry ranking + cache slot collision), not quant tier.
IQ3_XXS wire floor: 42.91 GB store vs 50.29 GB IQ3_S (-15%), smaller blobs also faster.
Donor-GGUF files for XXS: shard-1 only on disk (~/.models_backup/Strata-data/models/_iqxxs_dl),
shard-2 (ngram) intentionally never downloaded.
Next: B (native batching cap-3 via SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK + mamba-cache 20 +
HG_NATIVE_PROF=1 per-apply sync timing every 24th call).


## §53 B: native cap-3 batching live; step anatomy measured (the agent, 2026-11-10)
launch_phase2_m11b.sh = m11x + HG_NATIVE_PROF=1 + SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
(mamba-cache 20 FAILED boot: mamba state tax starves KV pool -> ValueError 'Loaded weights
leave no GPU memory for the KV cache'; keep 12 + lock-skip ratio 4 => cap-3, same VRAM shape).
BATTERY PASS: ' Paris.\nThe capital of France is Paris.'
BURST2 agg 9.08 tok/s; 3-stream phase = 3x16 tok in 4.27 s (~11.2 tok/s at T=3).
cap2 (m11x) 8.47 -> cap3 9.08 (+7% only; cap2 already captured the win).
NATPROF (T=1 decode): MoE apply 0.5-4.2 ms/layer (median ~0.9; spikes = _blob_host
pread on miss). => MoE ~45-70 ms of ~118 ms step; non-MoE eager overhead the other half.
Gap-to-Strata anatomy: MoE plumbing (tunable: CACHE_N/HTIER/hit-rate) + ~50-60 ms
Python/attention/head floor + no spec decode. Streaming wire itself is NOT the limiter.


## §54 A sweep + C quality-parity first pass (the agent, 2026-11-10)
A (cache tuning on native XXS cap-3): CACHE_N=16 burst 7.29/8.04; +HTIER=400 burst 7.54/9.30,
3-stream phase 3.68s vs 4.27s at cap-1-era config. Diminishing returns; floor is Python step
overhead, not cache. BEST CONFIG = launch_phase2_m11a2.sh (XXS native, CACHE_N=16, HTIER=400,
cap-3, TCHUNK=32): burst agg up to 9.30 tok/s. VRAM 12.08 GB used, RAM shared 21 GB.
C (battery_quality.py, 8 substantive greedy prompts, native-xxs vs fp8-triton-ref):
QC0 391 both OK; QC1-3,5,7 coherent both; QC4 fib code-gen works both.
QC6 (primes 90-110): REF answers 97,101,103,107,109; NATIVE-XXS returned EMPTY -> under
investigation: 5x repeat on XXS, then IQ3_S arm for tier-vs-plumbing separation.
Speed note: reference arm QC took 32-65 s/prompt vs 1-14 s native XXS (~4-6x).
OOM note: full 131-shard ref boot was OOM-killed once under 90 GB page-cache pressure from
XXS GGUF + blob store; freeing pinned RAM from dead arms + relaunch succeeded (no sudo TTY
for drop_caches; consider rm XXS GGUF after extraction, it is now redundant).

## 55. C quality-parity CLOSED 2026-11-10 — no IQ-tier regression; QC6 is a tie-boundary probe (the agent).
- battery_quality 8-prompt parity: native XXS and IQ3_S match FP8-Triton on QC0-QC5, QC7; QC6 flagged then investigated via 5x repeat hammers, scaffolded + raw prompts, across THREE boots.
- SMOKING GUN: the FP8-Triton reference itself flips boot-to-boot on QC6 (boot A: "97, 101, 103, 107, 109"; boot B m11c: stable x5 empty). Stable within-boot, flipped between boots => full-precision model sits on a logit near-tie at that prompt; tie-break is decided by process-level nondeterminism (Triton autotune/reduction order, cf. index_add_ law).
- Native XXS/IQ3_S vary across REPEATS within a boot (weaker arms have wider logit noise, so ties resolve more often) but all outputs are plausible attempts; no consistent IQ-only failure.
- VERDICT: parity holds. Native IQ tiers (IQ3_XXS, IQ3_S) show NO detectable substantive-quality regression vs FP8-Triton on this model class. Battery-design law: NEVER probe parity with near-tie prompts (counting/listing problems, exact arithmetic edges); probe with content-quality + instruction-following prompts, and repeat x5 with per-boot variance expected.
- Tier guidance (final): IQ3_XXS = fastest banked native substrate (burst 9.30 tok/s B+A config, ~43 GB store); IQ3_S = conservative parity tier (50 GB store); both pass battery_fairness + battery_quality.
- m11c reference-arm pitfalls banked in skill: derived arms must re-set HG_TRITON_MOE + HG_MOE_CACHE_N=6 (Triton FP8 experts OOM the 12 GB pool at N>=10), re-stamp loader-skip manifest per model (single shared manifest!), and launch fully detached (setsid) or ssh teardown SIGTERMs the server mid-boot.
- Disk after cleanup: /home 109G free (XXS GGUF deleted post-extraction), /mnt/docker 6.4G free (two checkpoints live: q4e-iq3s, q4e-iqxxs).
- Campaign status: M11 CLOSED, D CLOSED, B CLOSED, A CLOSED, C CLOSED. Phase-2 native-IQ program complete on the reference host.
## 56. Per-step overhead attack (open) — step anatomy + tier-policy fixes (the agent).
- Solo-decode step map (native XXS, CACHE_N=16 HTIER=400, the reference host): wall ~120 ms = MoE apply ~84 ms (avg 1.74/layer, long tail p90 8.3 max 16.6) + lm_head fp8 wire 33.3 ms + PLE gather ~1-2 ms; attention/GDN/HC glue hides inside async overlap (wall ~= GPU busy; NOT launch-gap dominated -> CUDA graphs are NOT the top lever here).
- apply sub-timers (HG_NATIVE_PROF print now splits pre/cs/h2d/ng/comb): staging (cs) dominates apply variance; grouped kernel issue (ng) ~0.1 ms; combine (comb) ~0 (async); metadata H2D ~0.2-0.4 ms/layer.
- ROOT CAUSE of cs spikes: _blob_host pinned-tier eviction was insertion-FIFO (hot blobs evicted by cold ones, causing repeat NVMe reads) AND every miss allocated a fresh pin_memory buffer (page-lock sync). Fixed in pyhook/hydragen_moe_native.py: LRU touch on ht hit + deferred recycle pool for evicted pinned buffers (event-gated, no reuse race). Effect: burst B2 9.58 -> 10.57 tok/s (new banked best), apply p50 1.1 ms.
- VRAM funding cap law (12 GB, this substrate): pool-end headroom ~0.5 GB at fraction 0.95; KV 4096->1024 saves only 0.08 GB; mamba slot 59 MB each (capped pool needed for cap-3); DEVICE CACHE 1 slot = 2.05 MB x 48 layers = 98 MB per +1 of N. N=16->24 needs +0.79 GB — UNFUNDABLE without cutting concurrency. lm_head resident = bf16 1.19 GB (branch does w.to(dev) unquantized) — also unfundable; fp8-resident variant would need a new branch (~0.63 GB, still unfundable at cap-3).
- mem-fraction-static law refinement: static budget = fraction x avail minus post-load; mamba+KV pool must fit inside it — fraction 0.90 starves the 12-slot mamba pool and startup dies with "Loaded weights leave no GPU memory for the KV cache" even though min-viable prints 0.861. Keep 0.95 for cap-3 arms.
- Loader-skip law bites again: any arm launched after a foreign --make silently full-loads (m11o2 boot ran 131 shards, 9.84 GB GPU post-load, then died on the fraction math). ALWAYS re-stamp for the target checkpoint immediately before launch.
- torch.profiler/CUPTI segfaults the scheduler with the native ctypes kernel path (sticky CUDA_ERROR_ILLEGAL_ADDRESS + callback-not-found, then segfault at capture start). Do NOT use HG_STEP_TRACE or /start_profile on native arms; use the [hg-natprof] sub-timers instead.
- Old-arm race hazard: when relaunching over a killed arm, the dying old tree logs SIGQUIT/kill_process_tree lines into the same log AFTER the new boot starts; anchor analysis at the last Uvicorn-running line or crash-window garbage gets aggregated as if it were the new arm.


## 57. Native-XXS prefill funding + radix concurrency proof (2026-11-10, per-step campaign close-out)
- Root-caused three 12 GB prefill funding defects on the native IQ arm:
  1. grouped-kernel scratch grew lazily to ~160 MB on first long prefill at pool end: fixed by boot-time prealloc scr = _sc(max(64, TCHUNK*16), NFF) plus del-then-alloc grow (no double peak).
  2. pinned-buffer recycle pool corruption: eviction queue appended cuda Events where buffers belonged, so b.copy_ hit an Event (AttributeError, scheduler crash mid-battery). Rewritten as _evq list of (buf, event) pairs; buffer moves to freep only when event.query() is true. Note: decode-only batteries never evict (ht stays under HTIER_K=400), so this pool was dead code until a long prefill exercised it.
  3. lm_head vocab-chunk OOM on merged multi-stream extend (2048 rows x 32k vocab x 2B > pool tail): HG_HEAD_CHUNK=8192 in launch_phase2_m11o3.sh.
- TCHUNK 32 -> 16 on this arm (scr bound halves; decode unaffected, T=1 single chunk).
- Prefill is the concurrency wall for unique long prompts on native IQ: battery_prefix (1016-token prefix, B4):
  - P1 solo prefill+decode: 56.8 s (expert union near-full store, NVMe-bound).
  - NOPRE_B4 (4 unique prefills): 98.53 s wall, agg 0.65 tok/s.
  - PRE_B4 (radix shared prefix): 10.64 s wall, agg 6.02 tok/s -> 9.26x wall win.
- Fairness battery after fixes (CACHE_N=16 HTIER=400 cap-3): B2 up to 12.38 tok/s, B1 11.3-11.5 (new banked bests; was 9.30 pre-campaign).
- Files: pyhook/hydragen_moe_native.py (scr prealloc, _evq pool, del-first grow), launch_phase2_m11o3.sh (TCHUNK=16, HEAD_CHUNK=8192, KV 2048).

## 58. Prefill-staging campaign: extend prewarm + host-tier sizing A/B (2026-11-10, open)
- hydragen_moe_native.py extend path now PREWARMs the whole per-layer missing blob set: 8-worker ThreadPoolExecutor os.pread into pinned buffers (pool-borrowed, LRU-evicted via _evq), plus direct ctypes.memmove pread replacing the per-miss 2 MB bytearray + frombuffer double-copy. memoryview(torch tensor) is illegal (no buffer export); use data_ptr + memmove.
- A/B battery_prefix (P1 / PRE_B4 / NOPRE_B4) + fairness B2:
  - baseline (no prewarm, HTIER=400): 56.8 / 10.6 (agg 6.02) / 98.5 (0.65), B2 up to 12.4
  - prewarm, HTIER=400: 44.8 / 11.4 (5.62) / 101.6 (0.63), B2 up to 11.7
  - prewarm, HTIER=256: 46.1 / 12.6 (5.07) / 129.7 (0.49), B2 up to 12.5
- Verdict: prewarm cuts cold prefill 57 -> 45 s (drive-bound ~1 GB/s at QD8, floor of the 45 GB sweep). HTIER=400 confirmed sweet spot: 256 costs NOPRE 28 s; decode indifferent to tier size. Keep HTIER=400 despite host RAM pressure (avail ~0-9 GB with 39 GB pinned + 47 GB PLE mmap; kernel thrashes page cache but stable).
- Hazard banked: killing native-arm servers must sweep wrapper + scheduler PIDs, not just GPU compute apps — a scheduler holding 39 GB pinned RAM survived GPU kill and got OOM-killed when the next boot grew its own tier, crashing the boot mid-battery.
- Files: pyhook/hydragen_moe_native.py (prewarm block ~line 128, _pread_into top-level), launch_phase2_m11o5.sh (HTIER=256 arm, superseded).

### 58b. cap-4 attempt on native IQ arm — CLOSED NOT FUNDED (2026-11-10)
- launch_phase2_m11o6.sh (mamba-cache 16 + lock-skip ratio 4, chunked-prefill 1024, max-running 4):
  - fraction 0.95 + KV 4096: boots, dies at first merged extend (CUDA OOM 16 MiB short, 1.57 GiB reserved-unallocated fragmentation).
  - fraction 0.92 and 0.90 + KV 4096: static budget ValueError (min viable 0.861) — mamba states cost ~90 MB/slot at this config, 16 slots eat ~1.5 GB.
- Verdict: cap-3 is the true 12 GB ceiling for 4x ~1k-token concurrent streams on native IQ + FP8-dense. Row-state cost per extra concurrent stream ~180-360 MB (mamba states + KV) — matches the concurrency thesis: engine-side row state, not weights, is what kills high N on 12 GB.
- Canonical serving config stays launch_phase2_m11o3.sh (HTIER=400, mamba 12, fraction 0.95, KV 2048, TCHUNK=16, HEAD_CHUNK=8192) + prewarm shim = B2 up to 12.4, P1 prefill 45 s, PRE_B4 agg ~6.

## 59. Dense-weight IQ campaign (OPEN, 2026-11-10)
- Dense resident FP8 census (safetensors index): GDN 3.89 GB, ATTENTION 1.25, MLP 1.10 + shared_expert 0.45, lm_head 1.18, EMBED excl PLE ~1.2, router/hc/norms ~1.0. PLE ngram (47.7 GiB) is host-side IQ4_NL already; lm_head streams pinned fp8 (not resident).
- Donor GGUF dense tiers (from §20 census): attn/GDN/ssm_out/shexp/token_embd/output = Q4_K/Q5_K/Q6_K/IQ3_XXS/IQ3_S/IQ2_S (GSQ-mixed), routers/hc/indexer-proj/PLE = bf16/f32. End-state dense wire ~4 GB vs 7.9 FP8 -> frees ~3.8 GB VRAM: funds cap-4 outright + KV/heads headroom.
- Kernel entries exist in libstrata_iq_kernels.so: iq_mmvq (dense matvec, decode), gemm_iq_f16_grouped (prefill), iq_dequant_f16, iq_embed_rows (already live for PLE). Strata serves whole-IQ dense models with these daily (3090 prod) — Ampere-proven.
- Plan: M-A extract donor dense IQ blobs (re-download donor OR quantize from FP8 via ~/llama.cpp ggml quantize_row_* exports — decided per ABI recon); M-B shim wraps dense Linears: skip FP8 weight load, IQ bytes VRAM-resident (dense tensors used every token -> zero cache churn), forward via iq_mmvq at T=1 + gemm_iq at prefill; M-C parity battery rerun (near-tie law) then cap-4 allocation.

## 60. Dense-weight IQ campaign CLOSED — Q5_K dense wire is CANONICAL; cap-4 funded; 16.1 tps B2 (2026-11-10, the agent)
- Substrate: native IQ3_XXS expert blobs + DENSE LINEAR layers served as IQ wire via strata iq_mmvq (fmt 12/13/14 = Q4_K/Q5_K/Q6_K). New canonical arm: launch_phase2_m11d2.sh (HG_DENSE_IQ=1 HG_DENSE_IQ_FMT=13, HTIER=256, mamba-cache 16 -> cap-4, chunked-prefill 1024, max-total-tokens 4096).
- Implementation (pyhook/hydragen_dense_iq.py): quantize module weights IN MEMORY after load_weights (post-merge, post-finalize_fused_in_proj) — immune to checkpoint-vs-model name mapping; per-module disk cache ~/hg-store-dense-iq2/<name>.f<fmt>.bin; q8_1 activation quant + iq_mmvq kernel via ctypes mangled symbols; padded rows to 256 superblock multiples.
- LAWS learned:
  1. ALL sglang *ParallelLinear.forward return (output, bias) tuples — IQ shim must return tuples or unpack errors at call sites (qwen2_moe MLP gate_up, _ = ...).
  2. Model module names are MERGED and FLAT: in_proj_qkvz (qkv+z fused), in_proj_ba, qkv_proj/o_proj live directly at layers.N (NO self_attn. prefix), shared expert = gate_up_proj/down_proj. Suffix-match against the MODEL, not the checkpoint.
  3. Dense tier quality floor = Q5_K: Q4_K dense wire (1.54GB) produced QC7 FASTA slip (> -> ;), B1_WARM1 hallucination continuation, P1 prefix echo; same-boot FP8-dense reference disproved QC3/QC5 flags (reference echoes/errs identically — those are model quirks, NOT regressions; §55 law confirmed again).
  4. Decode GEMV is weight-bandwidth-bound: halving dense wire bytes made decode FASTER despite per-call kernel overhead (B2 12.4 -> 16.1-16.8; Q5_K slightly under Q4_K as expected).
  5. Host-RAM wall: at HTIER=400 the scheduler (PLE tmpfs 47.7G + pinned tier) OOM-killed mid-battery (kernel oom-killer, exit -9). HTIER=256 restores ~14G headroom; with dense-IQ the decode cost is a wash.
- RESULTS (cap-4 Q5_K arm): instrumented 204 modules wire 1.89GB (vs ~5.3GB BF16 -> ~3.4GB freed); max_running_requests=4 confirmed; VRAM 11.6GiB, KV pool 4096; P1_prefix 34.5s CLEAN; PRE_B4 agg 6.13; NOPRE_B4 agg 0.51 (drive-bound); B1 11.8, B2 16.12; quality battery PARITY PASS vs same-boot FP8-dense reference (QC0 391, QC7 exact, QC6 clean primes, WARM1 clean, QC5 same-answer-as-ref).
- cap-4 RETRACTION of §58b verdict: cap-4 was NOT fundable on FP8-dense; dense Q5_K wire funds it (+~3.4GB). Keep §58b row-state anatomy (~90MB/slot) as the why.

## 61. NGRAM spec decode on hybrid GDN + paged experts — CLOSED NOT FEASIBLE on stock paths (2026-11-10)
- launch_phase2_m11d3.sh = d2 + NGRAM (steps 3, topk 1, draft tokens 4) + ReplaySSM-spec + triton linear-attn decode backend.
- Banked law: sglang paged_experts method.py loads as TWO module instances per process (sitecustomize import-hook on builtins.__import__ misses one; importlib bypasses it). Only a sys.meta_path finder wrapping exec_module patches BOTH copies reliably. Opt-in via HG_PAGED_SPEC=1 in sitecustomize (find spec, wrap loader.exec_module, patch check_paged_experts_compat after exec).
- Banked law: spec decode on GDN hybrids expands the mamba pool — fused_entry_bytes("mamba") = host+draft+pad per slot, PLUS reserves intermediate = per_req*(capped_reqs+1)*num_draft_tokens. This ate ~5+ GB and failed the static-budget check at fraction 0.95.
- Countermeasure: --enable-linear-replayssm-spec (Part B #28511) replaces per-draft full-state snapshots with per-slot raw-input windows; requires linear-attn decode backend triton or flashinfer, and linear chain only (eagle-topk <= 1).
- Diagnostic law: print pid + module id + "rebound" identity to detect module-copy splits; three hg-skip installs per boot = preflight stamp, parent launch_server, scheduler.

- Walls hit, in order:
  1. paged_experts validator — bypassable via sys.meta_path exec_module patch (banked above).
  2. mamba draft-state funding: fused_entry_bytes("mamba") widens every slot (host+draft+pad) and reserves intermediate per_req*(capped_reqs+1)*D. --enable-linear-replayssm-spec removes the intermediate reserve on GDN (hybrid_gdn_config resolves for qwen4_exp: Qwen4ExpTextConfig subclasses Qwen3NextConfig); cap-4+spec still unfundable.
  3. HARD GATE qwen4_exp.py:259 NotImplementedError "Qwen4 PLE does not support NGRAM speculation" — PLE per-request n-gram history would be corrupted by NGRAM accept/reject replay. Do NOT bypass: semantic, not validation.
  4. MTP route: the PLE gate allows topk=1 chains, but the FP8 checkpoint has zero mtp tensors (loader skips "mtp") and sglang has no Qwen4 draft worker. NEXTN/EAGLE need an external draft model.
- NGRAM post-init rewrites speculative_eagle_topk to max_bfs_breadth (10); linear-chain spec contracts need --speculative-ngram-max-bfs-breadth 1.
- Next speed rung (unstarted): decode CUDA graphs are disabled on our arms — ctypes blob staging has no capture contract. Capture plan: device-side run construction (GPU sort/unique_consecutive of router ids into fixed-capacity arrays), fixed slot table, eager inter-replay staging patch (contents change, pointers do not). Kills most of the ~100 ms eager step from section 56.
- Canonical arm relaunched clean: launch_phase2_m11d2.sh PID 490173.

## 62. Breakable CUDA-graph decode arm (launch_phase2_m11d5.sh) — OPEN
- Goal: kill eager Python step overhead (~90 ms/step host) via sglang
  breakable CUDA graph backend + eager islands around host-side shims.
- All patches opt-in: HG_NATIVE_GRAPH_ISLAND=1 gates islands in
  pyhook/hydragen_moe_native.py (apply island + graph-replay dispatch bridge),
  pyhook/hydragen_ple_diet.py (start_prefetch island + diet gather island +
  meta_path finder for multi-copy modules), pyhook/hydragen_paged_store.py
  (VocabParallelEmbedding/hg_gather/qsa varlen/chunked lm_head islands),
  pyhook/hydragen_bcgraph.py (LogitsProcessorOutput marshalling for
  BreakableCudaGraphBackend; backend class is BreakableCudaGraphBackend).
- expandable_segments:True splits the graph pool (~1.45 GiB reserved-
  unallocated) -> use expandable_segments:False.
- eager_on_graph islands MUST NOT nest; diet gather inside islanded
  start_prefetch guarded via _hg_inpf flag.
- Graph replay flattens NamedTuple dispatch inputs to tuples; native apply
  bridge restores topk_weights/topk_ids/router_logits positionally.
- sglang native PLE has _breakable_ple_prefetch islands; our diet gather
  must be patched per module COPY (module-copy split law; builtins hook
  missed importlib-loaded copies -> meta_path exec_module finder added).
- Sed-chain pitfall: repeated patches dropped `cls.gather = _dg2` silently
  -> native pinned-PLE fp8e4nv kernel ran and crashed on Ampere. Verify
  replacements land, not just compile.
- Status: capture fixes in flight; no battery/speed banked yet.

## §62b Breakable CUDA-graph decode arm — CLOSED PASS (2026-11-10)

**Root cause of the prefix/prefill `illegal memory access` saga: the metadata-pass
("metal") apply itself.** Persistent per-layer `st` buffers in the metadata apply race
host-ahead-of-GPU under prefill. Bisection ladder (all innocent, then revert=clean):
graph off, CUDA_LAUNCH_BLOCKING (blocking made prefill clean => temporal race not OOB),
prewarm gate, --disable-overlap-schedule, HTIER=0, pinned-metadata ping-pong;
full revert to pre-metal apply => prefix battery CLEAN. The metadata pass is abandoned:
whole-step gain was only ~3% (261->253 ms) vs fresh-tensor apply; not worth the race.

**Second root cause (echo-quality regression on graph arm): the PLE start_prefetch
island cleared `_prefetch_state` while forward() fell back to a stale/zero path.**
Since the diet gather serves PLE rows synchronously and ignores native prefetch,
the graph-safe island is a NO-OP start_prefetch (SPNOOP_OK). Native sglang PLE is
Ampere-dead (Triton fp8e4nv), so diet stays required; PLE_HOSTSIDE=0 under graph
(paged_store _patch_q4e already early-returns under island flag).

**Shim selection (co-edit safe):** sitecustomize honors `HG_MOE_SHIM` (default
`hydragen_moe_native` = pre-metal apply, canonical d2/d5eg). Graph arm uses
`hydragen_moe_native_graph` = pre-metal apply + graph-essential inserts only
(tuple-dispatch bridge, eager_on_graph island wrap, bcgraph install).

**bcgraph rebuild pitfall banked:** iterating `__dataclass_fields__` yields NAMES not
Fields (`'str' object has no attribute 'name'`); use `dataclasses.fields()`.

**Final d5 config (promoted graph arm):** d2 substrate (XXS blobs, dense Q5_K,
HTIER=256, cap-4) + breakable decode graph (bs 1,4), prefill eager, HG_HEAD_IQ=1,
HG_DIET_FINDER=1, HG_PLE_HOSTSIDE=0 HG_PLE_DIET=1, expandable_segments:False,
HG_MOE_SHIM=hydragen_moe_native_graph.

**Banked numbers (PID 544246):**
- QC battery FULL PARITY incl QC6 exact (97,101,103,107,109); QC7/QC2 clean think.
- P1 1016-tok prefill: 32.27 s text= ' Paris. The capital of France is Paris'
- PRE_B4 agg 8.07 (up from 6.13); NOPRE_B4 agg 0.51 (unchanged).
- Fairness warm: BURST1 20.29 / BURST2 20.07; peak warm observed 29.6.
- B1_WARM 1.2-1.5 s/token class-fast; same-boot WARM1 flip quirk persists (known).
- GPU ~11.5 GB, RAM avail ~13 GB.

Canonical promotion: launch_phase2_m11d5.sh is the graph-stepped canonical;
launch_phase2_m11d2.sh remains the conservative eager fallback.

## 63. MTP draft-layer adaptation (sglang native Qwen4ExpForCausalLMMTP) — OPEN
Recon (2026-11-10, the agent):
- §61 verdict "no Qwen4 draft worker" is STALE for sglang 0.5.23: srt/models/qwen4_exp_mtp.py
  (Qwen4ExpForCausalLMMTP subclasses Qwen3_5ForCausalLMMTP; forces num_hidden_layers=1,
  layer_types=[full_attention], ple_layer_ids=[]; Qwen4ExpModel(is_nextn=True); own
  ParallelLMHead at module path model.shared_head.head; _mtp_quant_config decides draft quant).
  qwen4_exp_mtp.py has NO load_weights override — inherits Qwen3_5ForCausalLMMTP.load_weights:
  draft ckpt carries official mtp.* names + "model.embed_tokens.weight" (special-cased);
  non-mtp names pass through.
- ModelConfig: is_draft_model + arch Qwen4ExpForConditionalGeneration -> Qwen4ExpForCausalLMMTP,
  num_nextn_predict_layers=1, layer_types/layers_block_type shrunk for full_attention_layer_ids=[0].
  resolve_spec_hidden_size: DSV4 + Qwen4Exp carry the hc-FLATTENED stream (hidden*hc_mult=10240)
  across the target->draft boundary.
- QSA backend native MTP support: capture_mtp_sparse_indices (draft-extend), should_reuse +
  lookup_mtp_sparse_indices (draft decode steps reuse draft-extend selection).
- PLE gate (qwen4_exp.py:257-264): spec allowed if topk=1; only NGRAM hard-blocked.
  Spec algos: EAGLE / EAGLE3 / FROZEN_KV_MTP / STANDALONE / MULTI_LAYER / NGRAM;
  EAGLE-family = V2 worker, draft writes KV chains -> MTP qwen4 is EAGLE-family (frozen-kv = fallback rung).
- Funding law (§61) stands: fused_entry_bytes("mamba") widens slots host+draft+pad; intermediate
  reserve killed by --enable-linear-replayssm-spec (arg_groups/fields/exec_.py). RETEST cap-4+spec
  funding now that dense-IQ frees ~3.7 GB (old unfundable verdict was BF16-dense).
- Strata MTP assets (~/hydragen/models/mtp): BF16 inventory = official MTP block, 31 tensors 5.2 GB
  (fc_embedding+fc_hidden fusion 2560x2560 ea; ONE layer: full self_attn 48q/2kv head256 + indexer
  (QSA) + HC mixers + 512-expert MoE + shared expert; NO ple, NO embed/head tensors in block);
  tensors/*.bin = raw BF16 per-tensor (mtp-manifest.json has name/shape/dtype/shard/sha256);
  mtp-q2_0.gguf (889 MB: experts q2_0, dense q8_0) + rt/ pack (dense.bin, experts.bin 708 MB,
  draft_vocab.bin = pruned-vocab head subset for the draft logits view).
- Measured Strata prod MTP acceptance (hydragen.log "MTP proposals accepted"): 58.8-74.6%,
  typical ~68% -> chain-3 ~2.4 tokens per verify step. Real target, not wishful.
- q4e-fp8 schema for the draft ckpt: quant_method fp8, activation_scheme dynamic,
  weight_block_size [128,128]; modules_to_not_convert = embed, lm_head, hc mixers, conv1d/in_proj_a...
  Draft ckpt plan: fp8-block ONLY mtp.layers.0.mlp.experts.{gate_up,down}_proj (2.5 GB) +
  fc/attn/shared/gate BF16 (excluded; ~150 MB) + BF16 model.embed_tokens.weight +
  model.shared_head.head.weight copied from q4e-fp8 (1.19 GB each; host-embed/PLE-offload +
  our IQ head class patches apply to draft instances too).
- Shim interaction law: pyhook patches are CLASS-level with _q4e_patched guard -> draft MoE/
  embed/head instances are swept in automatically. hydragen_moe_native apply path MUST name-gate
  draft instances out of the target hg-store (draft has no store entries) — route draft MoE to
  resident fp8 Triton w8a16 (Ampere-proven path), IQ-store draft experts = later rung (0.96 GB).
Ladder:
- M13a: builder m13_build_draft_ckpt.py -> ~/hg-draft-mtp (config.json w/ draft
  modules_to_not_convert + mtp fp8 shards + embed/head). Offline, CPU.
- M13b: eager correctness arm (graphs OFF): d2 substrate, cap-2, ctx 1024, HTIER=128 +
  --speculative-algorithm EAGLE --speculative-draft-model-path ~/hg-draft-mtp
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  --enable-linear-replayssm-spec; HG_PAGED_SPEC=1 kept. Gate: battery text FIRST (§61 law),
  QC + WARM1 + P1 + acceptance telemetry from scheduler logs.
- M13c: graph arm promotion (d5eg + breakable + spec): capture-shape interaction with
  draft-extend graphs; bs-decode 1 4; island coverage for draft path.
- M13d: capacity: draft IQ blob store (quantize mtp experts BF16->IQ3_XXS via ggml quantizer shim;
  512 blobs ~0.96 GB device-resident) + HTIER/cap ladder.
Decision points: EAGLE vs FROZEN_KV_MTP worker; draft MoE runner (resident fp8 Triton vs IQ native);
cap-4 spec funding (retest); HG_QSA_SDPA=1 fallback must not starve MTP sparse-lookup path.

## 63. Cold-staging parallel-prefetch probe — NEGATIVE, reverted (2026-11-10)
- Probe: decode-miss parallel prefetch (shared ThreadPoolExecutor(8), all layer
  misses fired as concurrent preads instead of serial QD1) in
  hydragen_moe_native{,_graph}.py. Compile-clean, booted (PID 564777),
  battery_quality PASS (facts correct).
- Result: NOPRE_B4 agg 0.52 -> 0.47/0.48 (no gain; two boots, two runs each).
  Warm staging regressed slightly: PRE_B4 8.29 -> 6.88/7.11, P1 32.9 -> 49.4 s
  (pinned-alloc churn when freep empty + pmap holds extra pins).
- Law: cap-4 cold staging is NVMe/page-cache BANDWIDTH-bound, not QD-bound.
  43 GB blob store + ~17 GB pinned HTIER + ~48 GB PLE mmap oversubscribe
  60 GB host RAM, so blob reads land on disk at any queue depth; ~2.6 GB
  blob traffic per cold token pair is the wall.
- Reverted; shims back byte-exact to §62b. Baseline re-verified after revert:
  warm burst 27.8/26.7 -> 33.0 agg, PRE_B4 8.07/8.29, NOPRE 0.51/0.52.
- Next candidate levers (untried): trace-based pinned-tier seeding (Phase 1
  real traces), PLE/blob page-cache partitioning, or accept NVMe floor for
  fully-cold concurrency; shared-prefix concurrency stays 7-8.3 agg.

## 64. M12a stock EAGLE-MTP arm — FUNCTIONAL on eager decode (2026-11-10)
Result: EAGLE spec decode with the Strata MTP draft now serves on the native
IQ substrate. launch_phase2_m11d5sp.sh (+ eager variant via
--cuda-graph-backend-decode disabled). Artifacts:
- ~/hg-mtp-draft: draft ckpt from mtp-manifest.json (numpy.fromfile +
  torch.frombuffer; mlp.experts tensors omitted; config keeps original
  top-level wrapper + architectures=Qwen4ExpForCausalLMMTP +
  language_model_only=true + text_config{num_hidden_layers=1,
  ple_layer_ids=[], layer_types=[full_attention]}).
- pyhook/hydragen_mtp.py (HG_MTP=1): fakes load_weights verification for
  mlp.experts/shared_head/embed_params; copies paged-store head attrs in
  set_embed_and_head so draft logits ride the IQ head path.
- Shims: HG_IQ_BLOB_DIR_MTP + HG_MOE_HTIER_MTP draft store keyed by negative
  layer IDs in both native shims (draft layer 0 must not collide with target
  layer 0).
Measured (eager decode, ctx2048, cap-4, steps=2 topk=1 tokens=3):
- accept len 1.80-2.25, accept rate 0.39-0.62 (draft q2_0 experts from
  hg-store-mtp-blobs IQ store; prediction real, not noise).
- B1 WARM1 2.0s WARM2 1.2s; BURST1 16.53, BURST2 19.92-21.28 tok/s
  (eager+spec ~= graph non-spec 20.07-20.29; eager non-spec was 12.4).
- QC battery: profile matches banked non-spec quirk family after the varlen
  causal fix below (QC7 '>' correct both runs, QC0 391 run B, QC6 tie-family).
Critical bug found and fixed:
- hydragen_paged_store.py _hg_varlen SDPA shim ran with is_causal=False and
  no mask. Legal only for q==1 blocks (decode). For spec verify blocks
  (q=3, kv=prefill+verify) verify tokens attended future draft tokens:
  verify logits poisoned, QC collapses (echo loops, QC7 'F' fact slip) in
  the spec arm only. Fixed with end-aligned causal: is_causal when q==kv,
  explicit tril-suffix mask when 1<q<kv. LAW: ragged varlen blocks align q
  to the END of kv; is_causal only encodes START alignment.
Graph+spec interaction (open):
- breakable decode graphs + verify crashed shape '[1,3]' in eagle_sample:
  LPO.next_token_logits stays bs-rows while full_logits is bs*draft_tokens;
  bcgraph _output_rows returned next_token_logits rows and sliced verify
  output to bs. Fixed to max(full_logits, next_token_logits) rows.
- Booting breakable+spec as of this entry; draft graph capture path
  (EAGLEDraftCudaGraphRunner + breakable backend + islands on draft layer)
  untested.
Also notarized: co-edit duplicate heading (two "## 63." sections: MTP recon
by co-editor at 1259, cold-staging at 1313) left as-is; this is 64.

### 64b. breakable-graph + EAGLE verify incompatibility (2026-11-10, OPEN)
Breakable decode backend replay plans are decode-shaped: per-graph size/cap
for verify graphs = bs (not bs*num_draft). Traced with HG_BCG_TRACE=1:
verify captures show ntl=(12,248320)/(3,248320) but backend passes cap=4/1.
max_shared_logits_buffer_rows returned 4 at graph-init in scheduler subproc
while eager boot computed 1664 -> spec bag timing-dependent at capture time.
bcgraph _output_rows fixed to max(full_logits,next_token_logits) rows;
still crashes at verify replay: eagle_sample reshapes 1-element logits to
[bs,3]. Fixing requires fork-level capture-plan spec-awareness (deferred).
Current M12a posture: EAGER decode + EAGLE spec (graph disabled) is the
working rung; eager+spec 16.5-21.3 tok/s ~= graph non-spec 20.07-20.29.

### 64c. breakable + EAGLE verify CLOSED not feasible w/o fork surgery (2026-11-10)
Root mechanism: BreakableCudaGraphBackend keys graphs by shape_key.size = bs
only; verify graphs (3 rows/req for bs=1) capture with cap=bs and overwrite
the decode graph at the same key slot (_outputs[shape_key] collision).
Row growth in my bcgraph wrappers cannot fix a same-key collision.
Fork fix needs tokens_per_req in ShapeKey + per-phase output plans (fork
surgery, deferred).
Funding notes for eager-spec arm: fraction 0.95 = zero headroom (PLE conv
OOM by 20MiB at P1); expandable_segments:True fails static-budget check at
0.95 (message demands >0.872); fraction 0.93 + segments False boots clean,
GPU 11.8 GB at runtime.
Eager-spec battery spread across boots: BURST 12.3-21.3, QC parity oscillates
(QC7 consistently correct; QC2/QC6 tie-family bad in worst boots).
Verdict: eager-spec ~= eager non-spec at cap-4; M12a mechanics banked, perf
gains require graph+spec. Next: M12b HYBSPEC custom lookup (zero draft
forward per hit).

## 65. M12b HYBSPEC custom lookup spec — OPEN (2026-11-10)
pyhook/hydragen_hybspec.py (HG_HYBSPEC=1, via sitecustomize): registers
HYBSPEC via SpeculativeAlgorithm.register(spec_class=HybSpecAlgo) with
supports_overlap=False and has_draft_kv()=False.
Worker HybSpecWorker(EAGLEWorkerV2) + _HybDraftWorker(EagleDraftWorkerBase):
lookup-only draft: greedy longest-match continuation chain over per-request
token history (origin_input_ids+output_ids), MIN_CTX=3 (HG_HYBSPEC_CTX),
topk=1 linear chain, topk1 prealloc parents/score_indices, bonus fill when
no hit. Zero draft model/weights/KV/forward.
_prefill seeds EagleDraftInput via create_idle_input + bonus/topk_index from
next_token_ids; decode extend is no-op (state recomputed from history).
Contract learned (topk=1 chain fast path):
- draft_tokens [bs, steps]; num_draft_tokens == steps+1 asserted.
- parent_list = arange(-1, steps-2) repeat bs; score_indices prealloc.
- bonus_tokens carry the committed last token; verify builds chain tree via
  build_chain_tree when QLEN_ONLY mask + (bs,steps) tokens.
- prepare_for_draft plans the shared write window (kv_loc_plan); same window
  consumed by verify; window cols = num_draft_tokens when translator+topk=1.
launch_phase2_m11d5hb.sh: eager decode (graph disabled), HYBSPEC args, no
MTP/draft path/replaySSM. V1 POC keeps default draft-KV reserve semantics
(has_draft_kv False may under-reserve; revisit if window OOM appears).
Laws banked during HYBSPEC bring-up:
- create_worker overlap gate reads the supports_overlap ATTRIBUTE from the
  register() kwarg, not a method override; spec_class method shadowed by
  instance attr. Register with supports_overlap=True.
- CaptureHiddenMode lives in sglang.srt.model_executor.forward_batch_info
  in this fork (no sglang.srt.enums).
- EagleDraftWorkerBase declares abstract stubs draft() and draft_extend();
  V2 flow calls _draft_extend_for_decode/_prefill; stub draft_extend to
  satisfy ABC.
- ModelRunner has no .attn_backend attribute in this fork; getattr-guard.
- Dropping --enable-linear-replayssm-spec on eager+spec makes
  _profile_available_bytes rest_memory<=0 (mamba spec extra-buffer sizing
  eats the pool budget); replayssm-spec funds it. Keep replayssm-spec and
  fraction 0.95 for eager spec rungs.
Additional HYBSPEC laws:
- QSA index-key ring holds one group: NotImplementedError "Qwen QSA requires
  speculative_num_draft_tokens <= the QSA compress ratio (4)" at first verify
  with num_draft_tokens=6. Hard width ceiling on this substrate: draft<=4
  (steps<=3). steps=3/draft=4 measured accept-rate ~0.02-0.10 and BURST
  10.4/12.9 (regression vs steps=2).
- Lookup with single ctx-3 key gave accept_len ~1.00-1.02. Multi-length
  fallback chain (HG ctx L=3..1 greedy most-frequent continuation) raised
  accept_len to 1.45/rate 0.23 on self-repeating traffic; prose non-looping
  text essentially never hits (by design; self-repetition decoder).
- steps=2/draft=3 is the best HYBSPEC rung so far; bursts ~12.4/15.7
  vs eager non-spec banked 12.4/16.1 => parity, not win yet.

M12b HYBSPEC closing verdict — CLOSED FUNCTIONAL, NOT WINNING (2026-11-10):
Gate on steps=2/draft=3 + ctx 3..1 fallback (PID 662242):
- QC run A: QC0 tie-family slip (238 vs 235), QC7 think-leak but correct
  facts in B; run B: QC0 collapse ("nonsensical"), QC6 partial.
  Oscillation matches all eager-spec arms; below canonical graph-arm
  full parity.
- fairness: WARM1 2.9s WARM2 3.0s; BURST 12.48/14.80 tok/s.
- accept len 1.09-1.32 rate 0.04-0.16 on battery traffic; best-seen 1.45/0.23
  on forced self-repetition.
Verdict: HYBSPEC proves CustomSpecAlgo plugin path end-to-end on qwen4_exp
(zero draft weights, zero draft KV, zero draft forward, PLE-compliant, no
NGRAM gate trip) but eager verify-round bookkeeping + no graph decode makes
it speed-parity with eager non-spec and a clear loss vs breakable-graph
canonical (20.07 B2). Spec decode on this substrate is gated on graphed
verify, which the fork blocks (ShapeKey/GDN-rotation laws). Canonical d5
graph arm restored.
File kept: pyhook/hydragen_hybspec.py (opt-in HG_HYBSPEC=1) +
launch_phase2_m11d5hb.sh for future graph-verify work.

## 66. M13-0 long-ctx decode wall measurement — CLOSED PASS (2026-11-10)
Premise correction: QSA sparse gather IS live on the stock decode path.
qwen_sparse_attn_backend.forward_decode computes indexer topk_indices, runs
qwen_sparse_kv_extraction_compact_triton into packed_k/packed_v scratch
(capacity batch*topk<=budget 2048), and hands the ALREADY-GATHERED rows to
the varlen fn (HG_QSA_SDPA shim included). Decode attention IO is constant
vs ctx (~25 MB/token); the "1.6 GB/token dense wall" does not exist.
Measurement (d5ctx: ctx 32768, pool trimmed to 36416 tok, cap-4, PID 666337):
- 29.4k-token cold prefill: 496s (59 tok/s, staging-bound; repeats
  degenerate prompts EOS early — probe must use varied prompt + ignore_eos).
- warm radix hit: ~0.3s prefill pass.
- decode at 29.4k ctx: 14.9-16.8 tok/s vs ~20 short-ctx = -16%.
- GPU peak 12006 MiB incl 36416-token KV pool.
Verdict: 128k is a KV-funding problem only. M13 ladder = KV memory diet:
M13a fp8_e5m2/int8 KV (halve), M13c host-paged KV (funds the rest).

## 67. M13a fp8_e5m2 KV — CLOSED NOT QUALITY-SAFE (2026-11-10)
Mechanics PASS: kv-cache-dtype fp8_e5m2 boots on Ampere sm86; SDPA shim
dequant fp8->bf16 patch (patch_sdpa_fp8.py) carries attention; probes and
29k ctx probe run clean (15.9/17.1 tok/s = +7% vs bf16 KV at 29.4k ctx).
Pool: 40960 tok capped by request; 12.8 KiB/token confirmed (available_gpu
0.79 GB left at 40960; room existed to ask ~70k).
Quality FAIL: QC6 = "91,93,95,97,99,101,103" (odd non-primes). QC0/QC7 ok.
e5m2 has 2 mantissa bits — KV values too coarse. e4m3 path is the
triton e4nv Ampere-unsupported dtype from the earlier crash.
Verdict: fp8 KV dies on quality; int8 KV (custom pool+shim) or host-paged
bf16 KV remain. Laws:
- The KV pool sizes from residual mem AFTER mamba/buffer costs; max-total
  request caps it, not vice versa.
- Decode at 29.4k ctx is essentially KV-bandwidth-insensitive (fp8 +7%).

## 68. M13c HiCache host KV tier — OPEN PROMISING (2026-11-10)
Switch: --enable-hierarchical-cache (ratio flags alone attach nothing:
registry gate ctx.enable_hierarchical_cache -> init_hicache).
d5ctxh (ctx 32768, --hicache-ratio 2.0): host pool 72896 tok = 1.79 GB;
Tree cache hicache_attached=True; hybrid GDN supported:
"NPU HiCache Mamba state transfer mode: sync torch fallback" (mamba rows
offload via torch sync copies).
Eviction/reload probe (probe_hicache.py, 27.4k-tok prompts):
A cold prefill 508.7s; B cold 494.0s (evicts A from GPU radix);
A rehit 1.3s cached=27328  -> ~400x vs re-prefill. HOST KV RELOAD WORKS.
Quality: 2 QC runs on hicache arm + 1 run on non-hicache 32k arm:
all within banked oscillation family (QC6 tie-probe oscillates on every
ctx-shifted arm: 91-slip twice, think-leak once; clean 391/10-11/GTIN).
No hicache-specific numerics regression observed.
Fairness hicache arm: BURST 18.06/19.50 (slightly under canonical 20+;
per-eviction overhead on radix churn).
Laws:
- Active in-flight sequences stay pinned in GPU KV; the host tier extends
  capacity for idle/cached prefixes (agent-harness sessions between turns),
  NOT simultaneously-active 128k streams.
- GPU pool sizes residual mem (0.44-0.79 GB avail at ctx32k arm);
  active 4x128k needs GPU KV diet first (int8 pool or tier shrink).
- HTTP 400 carries exact oversize message; probe prompts must budget
  ~3 tokens per marker word vs ntok word-count.
Next: ratio sizing for 4x128k host capacity vs PLE mmap RAM budget;
then funding stage: shrink GPU resident KV (int8 pool attempt 2: e4m3fn
is unsupported but plain int8-with-scales via pool subclass remains open).

## 69. M13d tier presets — CLOSED PASS, SHIP DESIGN (2026-11-10)
Preset A/B on identical d5 substrate (top-256/layer seed JSONs):
CODING preset (coder-trace) boot vs BATT preset (battery-trace) boot,
measured on 3 cold non-shared coder prompts (probe_coder_cold.py):
  prompt            CODING    BATT     ratio
  aiohttp bug-hunt  17.3s     32.1s    1.85x
  rust borrow err    5.9s     13.3s    2.25x
  k8s OOM diag      11.8s     11.9s    1.00x (domain overlap)
Radix-warm repeat on BATT boot: 0.7/1.4/4.5s (cached).
Battery-traffic A/B (3 runs each way): NO visible preset effect —
P1 31.9-32.5s / PRE 6.5-6.9 / NOPRE 0.47-0.63 in every config:
warm prefill + NOPRE are floor-bound (staging bandwidth).
Laws:
- Preset effects are INVISIBLE on floor-bound batteries; they only show on
  mid-length cold traffic where expert misses dominate. Choose oracle by
  regime: battery for floors, cold-mid probes for presets.
- Preset selectivity is real and traffic-matched: ~2x cold prefill speedup
  for matching domain; mismatch is not penalized below its own floor.
- Resolves 64 seeding-reproducibility open item: effect existed but the
  battery oracle couldn't see it.
Ship design: tier-seeds/PRESET_<domain>.json shipped with model config;
launcher HYDRAGEN_TIER_PRESET=<domain> resolves HG_TIER_SEED; one preset
resident per boot (HTIER budget: 256/layer ~17GB pinned; presets not
stackable on 60GB host alongside PLE mmap). Calibration = HG_ROUTE_DUMP
over N representative sessions -> rank by weight x tokens -> top-K<=HTIER.

## 70. M12c graphed-verify retry — CLOSED (fork capture-semantics law) (2026-11-10)
Retry of spec+breakable-graph with fixes from 64c (bcgraph _r in place):
  HYBSPEC boots: capture spec_info None -> shim EagleVerifyInput stub
  (get_spec_info wrap); hidden-mode validator: clamp spec_info AND
  forward_batch.capture_hidden_mode to NULL (batch field, not spec field).
  Capture then COMPLETES (width-3 graphs exist). First real verify:
  eagle_sample "shape [1,3] invalid for input of size 1" — captured graph
  emits DECODE-shaped logits rows (1), sampler wants bs*draft (3).
LAW (final, closes spec+graph on this substrate):
  Fork decode-graph runner IS spec-aware (captured_req_width=num_draft,
  uniform-width can_run_graph, width-aware capture batch), but the
  breakable backend capture-mode path yields decode-shaped LPO rows for
  verify graphs; qwen4_exp island lm_head compounds it. Fork-level
  capture-mode surgery required; shim cannot reach it.
  Forced-eager-verify fallback is pointless: in spec flows every target
  forward is verify, so nothing graphed = plain eager arm (= M12b).
Canonical d5 restored. Spec decode remains CLOSED NOT WINNING.

## 71. M13e HiCache 128k sizing — SIZED, FUNDS PARKED SESSIONS (2026-11-10)
launch_phase2_m11d5ctx128.sh: ctx 131072, max-total-tokens 40960,
--enable-hierarchical-cache --hicache-size 8 (PID 742408).
Host pool funded: 149,248 KV tokens = 3.67 GB (8 GB request split KV+mamba).
Probe: A cold 498.3 s (27,358 tok) / B cold ~494 s; A re-hit 1.5 s cached=27328.
Law: host->GPU reload ~18k tok/s; eviction of idle prefix to host works under
hybrid_ssm (mamba state transfers sync-torch fallback).
Sizing: 149k host tokens ~= 5.5 x 27k sessions or ~1.1 x 128k sessions parked.
NOT solved: simultaneous ACTIVE 128k streams (GPU pool 40960 < 131k single session).
Next lever if chased: int8 KV pool (fp8_e5m2 failed QC), or hicache-size 12+
(RAM allowed: 24 GB free post-run; PLE mmap coexists via page cache).

## 72. M14 dense IQ diet tiers — IQ4_XS FAST BUT UNSAFE, IQ3_S SLOW, FLOOR STAYS Q5_K (2026-11-11)
Per-module fmt override shipped: HG_DENSE_IQ_FMT_MAP="substr:fmt" (opt-in).
Crash law: _quant tuple signature must stay uniform; hydragen_paged_store.py
lm_head island (HG_HEAD_IQ=1) is a second _quant call site.
Micro-probes (installed Oct-31 libstrata_iq_kernels.so):
fmt21 IQ3_S rb=110 GPU cos 0.9934 / fmt23 IQ4_XS rb=136 GPU cos 0.9976 /
fmt22 IQ2_S dispatches but row-table mis-anchors (nan) / IQ4_NL: fmt29 garbage,
fmt27 unsupported / no IQ2_K ggml type exists.
Same-day face-off (battery_fairness warm runs + battery_quality):
canonical Q5_K: wire 1.89G GPU 10890MiB burst 12.20/18.43 QC6 direct primes QC7 ok
se21 (se IQ3_S, dense Q5_K): GPU 10832 burst 13.95/17.19 QC7 ok QC6 think-shift
se21iq4 (dense IQ4_XS + se IQ3_S): wire 1.46G GPU 10306 burst 27.45/24.06
QC7 GREEDY WRONG ("FASTA starts with a semicolon", Genomic Type Identifier) QC6 shifted.
Laws:
- IQ tier dequant cost splits on Ampere: IQ4_XS nibble grids CHEAPER than Q5_K
  (+30-45% warm burst), IQ3_S ternary/quinary EXPENSIVE (-7%).
- IQ4_XS dense-wide NOT QUALITY-SAFE: QC7 hard knowledge miss, same family as
  Q4_K 4.5bpw fail. Dense wire floor remains Q5_K.
- IQ3_S se-only: no speed win, QC6 tie-shift -> not banked.
- Fairness first run after boot reads cold wire cache: bursts ramp 7->12->18;
  compare only second+ runs.
- Boot-drift noise today even on canonical: WARM echo drift (Berlin/Madrid),
  QC0 arithmetic slip, QC1 think-empty. Discriminators: QC6 prime list directness,
  QC7 GTIN/FASTA, QC0 first clause.

## 73. M14b radix preamble dedup — CLOSED: STOCK RADIX DEDUPS, LAUNCH PRE-PREFILL COMPLETES IT (2026-11-11)
1,574-token harness-style preamble, 4 sessions with divergent tails, canonical d5.
Baseline (no prep): S0 cold 67.5s cached=0 / S1-S3 4.6-5.2s cached=1536 (96%).
Launch pre-prefill (1 warmup request of exact preamble after READY):
PREP 76.3s once (wire-cold boot) -> ALL sessions 4.9-6.5s cached=1536, session #1 included.
Laws:
- Stock radix longest-prefix handles multi-session preamble dedup: byte-identical
  preamble from token 0 -> 96% prefix hit (1536/1594, page-64 granularity), ~13x
  first-turn wall cut for every session after the first.
- Harness requirement: everything volatile (timestamps, per-session memory blocks)
  must sit AFTER the preamble; divergence before kills the shared prefix.
- Launch-time pre-prefill removes the cold cost from session #1; costs one prefill
  per preamble variant, radix node holds ~40MB of 40960-token pool.
- NOT supported: segment dedup of non-preamble regions (exact longest-prefix from
  token 0 only) — would be fork surgery in unified_radix_cache.
- Preamble + preset interaction: preamble tokens route through the same expert
  staging; a domain-matched preset makes pre-prefill itself ~2x faster (M13d law).

## 74. M15 int8 KV pool — CLOSED PASS, QUALITY-SAFE + FUNDS 2x24k SIMULTANEOUS
- Shim: pyhook/hydragen_kv_int8.py (opt-in HG_KV_INT8=1 + --kv-cache-dtype fp8_e5m2 for 1-byte pool allocation).
- Design: pool bytes reinterpreted as int8; per-(slot,head) fp32 scale side tables; write hook on MHATokenToKVPool.set_kv_buffer; int8 views from _get_key_buffer/_get_value_buffer; dequant after pack_qsa_prefill_kv (prefix prefill) and via extraction side-channel + chained varlen wrapper (decode); move_kv_cache relocates scale rows; scale tables preallocated at pool init.
- Capture law: boolean-mask indexing syncs => breaks graph capture (cudaErrorStreamCaptureUnsupported). Per-batch index_copy_ scatters + persistent per-buffer scratch are capture-safe.
- Quality: QC battery PASS. QC6 direct 97,101,103,107,109 (fp8_e5m2 emitted composites; int8 does not). QC7 FASTA/GTIN correct. QC0 17x23=391 clean. Think-preamble drift = same-day canonical boot noise family.
- Speed: fairness warm burst 20.59 tok/s cap-4 agg (band with graph arm 20.07-20.29). No decode penalty from quant/dequant ops at these sizes.
- Funding: ctx32768 max-total-tokens 65536 boots (bf16 ceiling was 36416; e5m2 pool arm maxed 40960 at 0.79G free; int8@65536 0.49G free, GPU 11674 MiB).
- Residency proof: two ~24k-token sessions concurrently active = 48k tokens, full token usage 0.73 of 65536, both completed; impossible on bf16 pool (48k > 36416). Third concurrent 24k exceeds pool -> expected LRU radix eviction.
- Laws: int8 row-scale KV is the quality-safe 1B KV tier; fp8_e5m2 mantissa too coarse, int8 7-bit row-scale safe.
- LIMITS: NOT supported with HiCache host KV (scale tables do not ride host transfers) — do not combine HG_KV_INT8=1 with --enable-hierarchical-cache until scale transfer is patched.
- Next levers: pool ~78k if headroom allowed; hicache+int8 scale-transfer patch; 4x long-context still needs >pool or hicache scales.

## 75. M16 HiCache + int8 KV scale-transfer — CLOSED PASS, SHIP DESIGN
- Shim-only extension of pyhook/hydragen_kv_int8.py; no fork edits.
- Wraps MHATokenToKVPoolHost.backup_from_device_all_layer / load_to_device_per_layer
  (pool_host/mha.py): fp32 per-(slot,head) scale rows ride the same host transfer
  via pinned host mirror lists _hg_sk_h/_hg_sv_h sized host_pool.size.
- Law: HiCache physical transfer path passes CUDA-resident index tensors;
  normalize host_indices to CPU and device_indices to the pool device before
  mirror indexing (indices should be either on cpu or same device error).
- One-shot proof logs: "write path ACTIVE", "host BACKUP scales ride", "host LOAD scales ride".
- Funded face-off (int8 + --enable-hierarchical-cache --hicache-size 8, ctx131072, pool 40960):
  A cold 510.3s / B cold 478.8s / A rehit 1.3s cached=27328 — funding law holds.
- Host tier with int8 payload: 205376 tokens / 2.52 GB (vs bf16 149248 / 3.67 GB) = +38pct parked capacity.
- Correctness oracle (probe_m16_correct.py): 64-token greedy continuation on
  host-restored 26k-token KV is TOKEN-IDENTICAL to cold reference decode (MATCH PASS).
- QC battery on combined arm: QC7 correct, QC0 17x239=4063 exact, QC5 clean,
  QC6 direct-16 window 97,101,103 = canonical; 91 slips are boot-drift noise class,
  "Is 91 prime?" answers not-prime correctly.
- Combined addressable KV at ctx131072: GPU 40960 int8 + host 205376 int8 = 246k tokens.
- Launchers: launch_phase2_m11d5ctx8h.sh (int8+HiCache long-context arm).

## 76. M17 int8 GPU pool bump — CLOSED PASS, 3x23.6k SIMULTANEOUS ACTIVE
- launch_phase2_m11d5ctx8x.sh: ctx32768 + HG_KV_INT8 + requested --max-total-tokens 78848.
- Pool auto-sized to 70720 (mem_fraction_static 0.95 cap; 0.38 GB free; GPU 11784 MiB).
  Requested 78848 NOT reached: static budget is the wall, not KV bytes.
- Funding face-off (probe_fund3 3-way concurrent): 3x17007 ptok, TOTAL wall=1440.9,
  identical per-leg walls, zero evictions -> 51k co-resident.
- probe_fund pair at 23.8k: both legs wall=805.8 co-resident (47.6k) then third leg
  1208.7 cold-by-design -> 3x23.6k fits 70720 sequentially-coexistent.
- Funding law updated: int8 pool funds 3 sessions ~= pool/3 tokens each (70720 -> 3x23.5k max,
  or 3x17k comfortably). M15 65536 arm funded 2x23.8k.

## 77. M18a decode-step island anatomy (graph arm, cap-4) — MEASURED, OPEN campaign
- HG_NATIVE_PROF=1 works on the graph eager island ([hg-natprof] every 24 applies/layer).
- cap-4 warm burst (fairness run2): agg 21.14 tok/s, step wall ~190 ms.
- Per-step 48-layer sums (median of 5 sampled steps):
  apply 227.1 ms (island wall incl waits) | cs 150.1 | h2d 46.6 | ng 3.2 | comb 4.0 | pre 4.9.
- LAW: decode step is HOST-STAGING-bound, not GEMM-bound (ng ~3 ms/step total).
  cs = expert-blob cache-slot staging (LRU host lookup + slot evict/recycle + misses) is
  ~80pct of island time; h2d ~25pct.
- Next rung: cs reduction (resident-tier hit rate at K=10 cap-4 working set, slot recycle
  fast-path, or per-layer slot table precomputation), then h2d overlap with ng.

## 78. M18b cs-reduction rungs — PARREAD NEG, HTIER384 ADOPTED
- In-step parallel pread (HG_NATIVE_PARREAD): threaded misses per apply adds ~3 ms/apply
  submit/alloc/event overhead; net NEG even gated to >=4 misses (17.75-18 vs 19.75-21.14 burst;
  cs shifted into pre, total unchanged). Re-confirms 63 NEG for in-step form. Code remains, OFF by default.
- HTIER pinned tier 256->384 (HG_MOE_HTIER): steady deep-warm cs 116.8 -> 15.0 ms/step
  (-89pct); warm bursts 23.28-25.25 vs control 23.36 (parity when fully warm); big win in
  mid-warm/sustained states (tier saturation prevents cs regrowth). Host RAM fine (available 43 G;
  pinned tier ~5.3 GB vs 3.5 GB). 400 remains banned (earlier host OOM law).
- New canonical: launch_phase2_m11d5h384c.sh (d5 + HG_MOE_HTIER=384). Old d5 (256) kept for A/B.
- Fully-warm cap-4 ceiling ~23-25 tok/s; remaining step cost now h2d ~29-36 ms
  (five per-chunk torch.tensor H2D creates x 48 layers) + island bookkeeping ~40 ms.

## 79. M18c h2d meta-arena batching — CLOSED PASS, NEW SUBSTRATE RECORD
- Five per-chunk pageable torch.tensor(...,device) H2D creates replaced by one persistent
  pinned+GPU meta arena (st[ah]/st[ag]) with slice fills and ONE blocking copy per chunk.
  HG_META_BATCH default 1; old path kept under HG_META_BATCH=0.
- First cut used non_blocking copy -> CPU arena reuse raced the queued DMA:
  corrupted grp_ptrs -> cudaErrorIllegalAddress mid-QC. LAW: pinned arena reuse requires
  synchronous copy_ (or rotation); growth must stream-sync before freeing old buffers.
- h2d 29-36 -> 5-17 ms/step. Deep-warm cap-4 bursts 27.3-34.1 tok/s (prior band 20-25,
  old peak 29.6). apply ~69 ms, cs ~29, ng ~3.5 at deep warm.
- QC gate: QC7 exact, QC5 exact; QC0/QC6 boot-drift families unchanged.
- Canonical now: launch_phase2_m11d5h384c.sh (d5 + HTIER384 + metabatch code).
- cs remains the top island cost; next rung is cs at mid-warm (tier fill policy / seed hits).

## 79b. M19 graphed spec revisit — KEY-SPLIT PASS (M12c door open), ECONOMICS STILL DEAD
- MECHANISM: hydragen_specgraph.py (HG_SPECGRAPH=1) overrides
  DecodeCudaGraphRunner._capture_graph_size -> num_tokens. Decode runner
  width=1 keys at bs; verify runner keys at bs*num_draft: the M12c
  breakable-backend ShapeKey collision (verify graph overwriting decode graph
  at same slot) is eliminated WITHOUT fork surgery or ragged mode.
  Verify replay confirmed on graph: cuda graph True, accept len 1.95-2.25.
- Ragged COMPACT path is a dead end for EAGLE on this model:
  replayssm-spec+compact guard requires the KDA fold-every-commit family
  (DSPARK/DFLASH); GDN materializes instead (semantic, not cosmetic).
  Removing replayssm-spec instead trips the mamba-spec buffer sizing wall
  (rest_memory<=0 law reconfirmed).
- Face-off (ctx2048 sp substrate, HTIER256):
  bs=1 single-stream: spec 13.0-14.6 tok/s vs canonical non-spec 20.0 (accept 1.5-1.65 essay / 1.95-2.25 battery).
  cap-4 fairness warm: spec 13.6 vs canonical 27-35.
- LAW: on a staging-bound substrate, verify cost scales ~linearly with width
  (unique-expert count ~ width x K, and each row stages), so accept gains
  (<=2.25) cannot offset width cost (2 drafts + 3-row verify ~ 3x step).
  Spec decode wins only when step cost is amortizable across rows
  (launch-bound or GEMM-bound). Graphed verify removes launch overhead but
  cannot change the staging slope. CLOSED NOT WINNING; hooks kept for
  future substrates where per-row cost is flat.

## 80. M20 Strata public-benchmark positioning (bench anchor)
- Anchor: Niko1221/Strata community bench, RTX 3060 12GB (i7-12700, 64GB DDR4),
  Strata 0.1.41, Qwen3.8-Flash-Next IQ3_XXS (42.9GB pack), MTP on, int8 KV,
  20GiB resident expert pool, parallel 1/2. (bench/results/2026-10-09-community-rtx-3060-0141)
- IQ3_XXS decode: 26.0-27.2 tok/s flat 128->128k ctx (24 at 260k).
- IQ3_XXS prefill: 681 @4k, 883 @32k, 866 @128k tok/s; TTFT 32k = 37s.
- IQ3_XXS concurrency agg (512in/128out one-shot bursts, ctx8192):
  1c=16.0 2c=21.0 4c=17.4 (par1); 22.4/23.1 (par2).
- MTP accept: IQ2_XS 1892/2566=74%, IQ3_S 1769/2611=68%; MTP gain IQ2_XS 35.9->44.1.
- Ours (the reference host 3080 Ti, IQ3_XXS wire + Q5_K dense, no spec):
  decode single 20.0 warm (peak 29.6), cap-4 warm agg 27-35.4, cold prefill ~55-60 tok/s.
- POSITIONING:
  decode gap: Strata step ~3x cheaper (26 tok/s vs our 20 on faster GPU);
  prefill gap: ~15x (883 vs ~55-60 long-ctx) = LARGEST catch-up target;
  concurrency: we lead warm sustained (35 vs 23 aggregate at 4c) and funding
  (70.7k GPU int8 KV + 205k parked host + radix dedup vs 512MiB/4-slot conv cache);
  their MTP accept 68-74% >> our EAGLE 39-62% (real trained MTP head).
- IMPLICATION: their decode economics survive speculation BECAUSE step cost is low;
  our M19 spec verdict is a step-cost law, not an MTP-futility law.
  Catch-up priorities reorder: (1) prefill staging pipeline, (2) step cost to ~40ms band,
  (3) then spec/MTP re-opens.

## 81. M21 user-proposed decode rungs (async H2D overlap + slot-cache fast-path) — MEASURED, NEUTRAL, BANKED LAWS
- Instrumented _cache_slot split counters (H/M/copy/preadd) in the graph island:
  per-miss torch copy_ dispatch ~18 us; _blob_host preadd ~150-260 us when the
  host pinned tier misses (NVMe page-cache/file read), ~0 on pinned hits.
  cap-4: 16-slot per-layer GPU cache thrashes (~25-30 unique/layer vs 16 slots),
  hit rate ~11%, cumulative M ~100/layer.
- Implemented HG_BLOB_FAST=1 (default): raw cudaMemcpyAsync ctypes replaces the
  per-expert copy_ dispatch. SYNTAX/scope bug found (helper injected inside
  _install() scope -> island fell back to stock fp8 fused-MoE -> Ampere
  fp8e4nv capture crash; fixed placement).
- A/B fairness (deep-warm cap-4 agg): fast ON 27.74/23.94 (prof arm peak 36.43
  new record but within run variance); fast OFF 28.71/24.11, 26.93/31.91.
  Verdict NEUTRAL at aggregate. Hook kept default-on (reduces cs window, no harm).
- LAW: blob copies are already async-from-pinned; wire bytes (~20 ms/step at
  cap-4 with 16-slot thrash) is the cs floor at deep-warm, and preadd/file-read
  is the mid-warm spike. Python dispatch (18 us x ~768/step ~ 14 ms) overlaps
  with the async wire and is NOT the wall. A dedicated DMA stream or a C-array
  tracker cannot reduce wire bytes or host-tier misses.
- LAW2: side-stream overlap for h2d behind ng is mathematically dead: ng=3.5 ms
  cannot hide 17-20 ms of transfers. The hideable window is the non-MoE graph
  remainder (~40 ms/step) and prefill, not GEMM.
- NEXT REAL RUNGS (bytes/fetch-bound): larger per-layer GPU slot cache (GPU
  budget ~221 MB at 16 slots limits this), preset-assisted host-tier seeding
  (kills preadd spikes), and prefill staging pipeline (15x Strata gap, §80).

## 82. M22 rung1 CACHE_N slot cache — CLOSED: CN=24 canonical, CN=32 record-but-unfunded
- HG_MOE_CACHE_N is the per-layer GPU expert slot cache (canonical was 16, ~11% hit at cap-4).
- cap-4 fairness working set is ~25-30 unique experts/layer: CN=32 ~= full residency.
- CN=32 (prof arm): hit rate 45%, cs mean 5.1->1.2, preadd /4, deep-warm BURST record 50.39 and 44.53 tok/s (vs 35.4 prior). But +221MB island cache leaves <1MiB free at mem_fraction 0.95: extend-pass CUDA OOM (2-20MiB allocs fail; 620MB reserved-unallocated fragmentation).
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True does NOT fund it: breaks pool sizing at 0.95 (1.43 GiB over-reserved, boot OOM). Launchers hard-export False (line 6).
- CN=24 (launch_phase2_m11d5cn24c.sh, NEW CANONICAL): stable headroom 0.69GB, pool 4096, deep-warm 22-29 with 47.57 spike; QC0 exact, QC6 direct exact x2, no composites. Promoted.
- Law: slot cache must stay <= ~1/3 of KV headroom; funding CN=32 needs mem_fraction retune or KV pool cut.

## 83. M22 rung2 preset host-tier seeding vs cold long-prefill — CLOSED NEUTRAL
- A/B on ctx32k HTIER-384 arms, paired 60k-char novel-domain (code) windows, greedy:
  TIER_SEED_QWEN4EXP mean TTFT 299.6 s vs TIER_SEED_BATT 309.7 s over seeds 1-4 (~3%, noise).
- Law: long cold prefill (~15k tok) saturates the pinned tier by expert fan-out across chunks
  ("cache fully in-use N=32" warnings); seed rank order is irrelevant once every expert is touched.
  Preset gain only exists for mid-length cold sessions (M13d ~2x law) — NOT for prefill-bound prompts.
- Tooling laws: /v1/chat/completions with non-zero temp triggers flashinfer sampling JIT which
  FAILS on this box (curand.h missing for CUDA 13.1) and kills the scheduler; probe_cold_g.py uses
  temperature:0. TARGET_CHARS 130000 > ctx32k => HTTP 400; use 60000 for ctx32k arms.
- Consequence: the 15x prefill gap vs Strata is NOT a tier-seeding problem. It is chunked-prefill
  pipeline + dense/compute cost. Rung 3 (prefill pipeline) is the only remaining lever.

## 84. CO-EDIT NOTE: Gemini DFlash2 campaign (m20d2-d8 + matrix arms) — UNMEASURED, results lost
Trail owned by the co-editing agent (Gemini), NOT notarized when run. Reconstructed from logs 21:53-23:22.
- Design: DFlash2 block-diffusion spec decode (incoai/Qwen3.8-Flash-Next-DFlash2 603M BF16, block 4)
  replacing MTP, on the q4e-fp8 substrate with paged experts (separate line from IQ3_XXS canonical).
- Funding law (theirs, valid): target alone ends weight load with 0.19 GB free (11.26 GB footprint);
  1.21 GB BF16 draft worker lands BEFORE pool sizing -> requires CN 24->8 (-1.44 GB) + cap-1 diet.
- Arm fates: d2/d3 = draft-build OOM in dflash.py __init__ (64 MiB alloc, 58 MB free).
  d5 = DFLASH runner ready, then intrinsic ValueError: replayssm-spec + DSPARK/DFLASH needs KDA
  model; Qwen GDN is not KDA (same wall as §70/§79b M12c). d6 = flashinfer JIT curand.h landmine.
  d7 = DEEPEST: serving started, sampling pre-warm OK, Triton dflash kernels lazy-loaded post-start
  (free mem squeezed to 0.55 GiB) -> ended in orderly shutdown WITHOUT any accept/throughput data.
- CONTAMINATION CAVEATS (per user): (1) forced engine restarts from outside may have cut arms short
  — d7 especially looks interrupted-mid-campaign, not failed; (2) both agents tested on the same GPU
  concurrently; these launchers carry no GPU-clear preamble, so boot-vs-boot contention is possible.
  Do NOT re-run any of this campaign to "confirm" the crashes before a clean serialized re-test.
- Notarization violation flagged: none of this appeared in CHANGES.md while it ran. All agents on this
  box must append a CHANGES.md section per campaign arm. Empty launch_phase2_m20d9.sh and a 22:45
  edit to pyhook/hydragen_dense_iq.py are also from this line and still unreviewed.
- Verdict: DFlash2 status = UNKNOWN, not dead. Open questions for a clean re-test: does block-diffusion
  bypass the KDA/replayssm requirement (d5 error may be fixable by disabling replayssm-spec flag),
  and does it survive §79b staging-bound economics. Launchers ready: reuse d7 config + measure.


## 85. M20d-R DFlash2 clean serialized re-test — MECHANISM PASS, ECONOMICS STILL NEGATIVE (bs=1)
- Paired A/B on identical CN=8/cap-1/paged-experts diet substrate (launch_matrix_plain2 vs launch_phase2_m20df1 = m20d7 clone + GPU-clear preamble, new log).
- plain baseline bs=1 warm: 15.9-16.5 tok/s. DFLASH warm: 11.0-11.2 (short probes), 13.9-15.1 (deep-warm 512 tok).
- Acceptance is the best in campaign history: accept len 3.33-4.00 of 4, rate 0.78-1.00 (deep-warm 4.00/1.00), cuda graph True via HG_SPECGRAPH key-split (works with DFLASH, not just EAGLE).
- Law: 4-row block verify costs ~3.6x a 1-row step on staging-bound substrate (+ 603M BF16 diffusion draft forward + fused KV materialization). 4/4 accept yields only 0.92x net vs plain. The §79b economics law holds even with a perfect draft.
- DFlash2 re-opens ONLY when per-row staging cost flattens (working-set-resident GPU slots / non-diet CN). At accept 1.0 the draft quality is not the limiter; row-scaling is.
- Reusable: curand JIT fix = ~/curand-pkg + CPATH/NVCC_PREPEND_FLAGS/LD_LIBRARY_PATH exports (Gemini contribution). Draft model at ~/models/Qwen3.8-Flash-Next-DFlash2. HG_STUB_RESIDENT + paged-experts line is a Gemini substrate variant, separate from canonical cn24c.
- Verdict: NOT adopted. Canonical remains launch_phase2_m11d5cn24c.sh.
## 86. M22 rung3-CRITICAL: extend slot-aliasing corruption fix via per-apply expert sub-batching (HG_EXT_SUB) — CLOSED FIX LANDED, canonical

Discovery (rung-3 anatomy, ctx32k PROF arm): long cold prefill island apply = 16.8 s per 1024-token
chunk (h2d 10.2 + cs 5.2 + preadd 3.6, ng compute only 0.17 s). While reading `_cache_slot`, found the
eviction fallback: when every cached slot holds an expert in the current block's `rows_c` (ul > CACHE_N,
which is ALWAYS true in extend blocks: ~150-520 unique experts per 128-token window vs 16-32 slots),
victim = `order[0]` evicts an expert ALREADY STAGED for the current GEMM batch while its `grp_ptr` still
aliases that slot, which the next `_cache_slot` overwrites before the GEMM runs. The
`WARN cache fully in-use` spam during prefill was this path. Expert weights were being corrupted in-place
for every window where ul > CACHE_N (prefill AND cap-4 decode with CN=24).

Evidence: prime oracle (list primes 200-260 behind filler prompt, greedy). Pre-fix runs produced
`211,223,227,229,233,239,241,251,257,263` (263 = out-of-range artifact). Post-fix run:
`211, 223, 227, 229, 233, 239, 241, 251, 257` exact MATCH vs reference.

Fix (pyhook/hydragen_moe_native_graph.py, default ON, opt-out `HG_EXT_SUB=0`):
when `len(ul) > _CACHE_N`, slice `ul` into `<=_CACHE_N` consecutive expert groups; per slice run the
ORIGINAL per-window body unchanged (staging -> metabatch -> grouped GEMM -> combine). Correct by
construction: slice i's GEMM completes on-stream before slice i+1 stages into those slots; a slot can
never alias two experts inside one launch. Metabatch arena now advances `_wa` per slice (no cross-slice
overwrite of in-flight kernel args). Zero extra VRAM. Backup: hydragen_moe_native_graph.py.bak-extsub.

Landing pitfall (cost ~1 h): first patch hoisted `xf/scr/out` allocations above the staging block; that
reorder alone broke CUDA-graph capture with `native_expert_grouped/swiglu: invalid argument` / illegal
access EVEN with HG_EXT_SUB=0. Second patch kept the original body byte-identical (list slicing + arena
offsets only) and booted clean. Law: island apply edits must preserve allocation ORDER exactly; never
hoist allocations across staging/capture boundaries.

Gates (CN=24 canonical launch_phase2_m11d5cn24c.sh, patched island):
- boot HEALTH_OK, graph capture clean.
- prime oracle MATCH (above).
- battery_quality 8/8 clean, QC0 first clause exact.
- fairness cap-4: 33.42 / 39.61 tok/s (no decode regression; slices fire at ul~40>24, ng cost invisible).

Decode staging math unchanged (wire per unique-per-window unchanged); the wire-reduction rung
(bigger extend windows HG_EXT_WIN / expert-outer staging) remains OPEN and now unblocked by correctness.
CN=96 unfundable on all arms (island buf ~660 MB; ctx32k OOM, ctx2048 boot OOM; mem-fraction 0.82 boot
refused: min viable 0.795). CN=48 booted? NOT re-tested post-minimal-patch; not needed for canonical.
## 87. M22 rung3b HG_EXT_WIN window enlargement — CLOSED NOT-FEASIBLE on 12 GB canonical

Goal: enlarge extend/prefill apply window (default _TCHUNK=128 -> opt-in HG_EXT_WIN up to 1024)
so each unique expert stages once per large window instead of re-staging per 128-token block.

Implementation trail (island hydragen_moe_native_graph.py):
- min-persist replay (`_PB` shared arenas, window-level zero/copy): ILLEGAL ACCESS at init.
  persistent reuse of xf/out buffers ACROSS _ng calls is poison (mechanism not pinned;
  pointer offsets and arena sizes verified in-bounds). Empirical law: fresh allocator blocks
  per slice are safe and effectively free via torch block reuse.
- fresh out per slice + window loop (`.bak-working` -> later `bak-window-exp`): boots and gates
  green on canonical: QC 8/8, fairness 16.44/38.57, control 6k probes 45.2/44.3/40.0 s TTFT.
- WIN=512 canonical: CUDA OOM (50 MB window buffers, 768 MB reserved-UNALLOCATED fragmentation).
- bucketed _rows + WIN=384: seed0 TTFT 54.7 s vs control 45.2 s (WORSE), then fragmentation
  OOM at _short_conv (20 MB, 826 MB reserved-unallocated). Slice overhead (13-17 slices per
  384-window vs 6 per 128) ate the staging-bytes saved before memory could fund it.

Laws:
1. Window buffers scale as win*K*H*4 (384->37 MB, 1024->100 MB per slice); on 12 GB at
   mem_fraction 0.95 this fragments the runtime pool to death (20 MB conv1d failures while
   ~800 MB sits reserved-unallocated). Do not retry window>256 without KV-pool shrink.
2. Economics already negative at 6k prompt: more slices = more Python/apply overhead per
   chunk; staging-byte reduction from one-stage-per-expert did NOT win at WIN=384
   (54.7 vs 45.2 s TTFT). Re-opens only on boxes with >= 2 GB runtime headroom AND long
   prefill where h2d bytes dominate; the M-churn anatomy (4190 ops vs ~1400 unique) is
   real but is better attacked via CN funding, not window size.
3. ctx8/q4e-fp8 65536-pool arm with pure §86 island runs 60k cold probes fine
   (seed0 TTFT 737 s at HTIER256, no OOM); window-patch island family died there.
   When a crash has NO Python traceback and null-byte log flood: read the lines above
   "Sleeping 5 seconds before crash diagnostics" in sg-real.log — the real class
   (OOM/illegal) is usually 3-8 lines above it.
4. probe_cold_g.py TARGET_CHARS now env-overridable (TARGET_CHARS=6000 for ctx2048 pairs).

State restored: island = PURE §86 (.bak-extwin), canonical launch_phase2_m11d5cn24c.sh
revalidated HEALTH_OK, QC 8/8, fairness 15.73/20.60 warm band. Snapshots:
.bak-working (fresh-out window form), .bak-window-exp (bucketed+combine-fix),
.bak-extwin (pure §86 = canonical), .bak-minpersist, .bak-tch, .bak-allpatch, .bak-extsub.
## 88. M23 int8 KV vs CN=32 funding face-off — CLOSED: CN=24 stays canonical; int8 KV validated on canonical

Premise: §82 CN=32 holds the decode record (50.39/44.53) but died unfunded (+221 MB island).
Hypothesis: int8 KV (HG_KV_INT8=1, §74/§75) frees the funding.

Funding reality discovered: on the ctx2048/4096-token canonical the KV pool is TINY —
"KV Cache is allocated. #tokens: 4096, K 0.02 GB, V 0.02 GB". Pool is cap-bound, not
budget-bound; int8 KV frees only ~20 MB there. int8 KV funding matters on the ctx8/65536-pool
arm (long sessions, §76), not the short-context canonical.

Arms (launch_phase2_m11d5cn24i.sh / cn32i.sh / cn28i.sh):
- CN24+int8: HEALTH_OK, 635 MB free, QC 8/8 (QC0 exact), fairness cold 9.86/12.87,
  warm 35.71/43.35 — decode NEUTRAL-TO-POSITIVE vs fp8 CN24. Validated as opt-in on canonical.
- CN32+int8: died in graph INIT both boots: breakable-graph segment instantiate hit
  `CUDA error: out of memory` inside capture (hydragen_kv_int8.py:114 frame); the
  `markCaptureEnd ... no captures in progress` ASSERT is the secondary cascade, not the cause.
  §82's "transient assert + clean retry" did NOT repeat — CN32 is a hard funding wall at 0.95.
- CN28+int8: boots (315 MB free), QC 8/8, fairness warm 33.05/41.57 (NO gain vs CN24i in noise
  band), then DIED on first 6k-char extend probe: Scheduler exception, 108 MB free.
  Confirms §82 law: island slot-cache growth must stay < ~1/3 runtime headroom.

Verdict:
1. CN=24 remains canonical on 12 GB. CN=32 record decode re-opens only on >=16 GB cards or a
   funded KV/weight rebalance — M14 dense floor (Q5_K) blocks the weight route (§72).
2. HG_KV_INT8=1 is validated neutral-to-positive on canonical decode+QC; canonical launcher
   unchanged (fp8), cn24i kept as the int8 opt-in arm; use int8 KV for long-session ctx arms.
3. Law: markCaptureEnd INTERNAL ASSERT at graph init = capture-body CUDA OOM cascade; look for
   `CUDA error: out of memory` at instantiate frames below it, not the assert itself.

Canonical launch_phase2_m11d5cn24c.sh restored, HEALTH_OK, 679 MB free.
## 89. M24-KICKOFF: native kernel borrowing from Strata (Antigravity proposal) — ACCEPTED WITH CORRECTIONS

Proposal reviewed from Antigravity. Files real: strata/src/kernels/cuda/ has fused_gdn,
native_gdn(.preprocess), native_qsa(+indexer,+score), qsa_(decode|prompt)_attn, native_rope,
native_router, native_gr_*, moe prompt kernels; strata/src/split/src/ has gdn_core/qsa/hc/
moe_mma_iq/prefill .cu(h). third_party/ggml is vendored (vecdotq headers present).

CORRECTIONS to the proposal's VRAM audit (live cn24c boot log 07:28, authoritative):
- Load weight end: "mem usage=11.26 GB", avail 11.45 -> 0.19. NOT "~2.9 GB resident".
- Mamba cache: conv 0.04 + ssm 0.90 = 0.94 GB (claim 1.44). Graph bs[1,4] capture: 0.36 GB
  (claim ~1.25). Post-capture avail 0.69 GB (the real headroom, matches §82/§88).
- There is NO "~3.85 GB allocator slack". Observed reserved-UNALLOCATED spikes were
  620-863 MB at OOM moments (§82/§87/§88), not a permanent 3.85 GB pool. Do NOT budget
  work around freeing 3.85 GB; the wall is the 11.26 GB weight region + 0.69 GB real headroom.
- split/ is a SEPARATE multi-card EXECUTABLE (CMake: strata-split-N, one per card count,
  line-protocol driven, anonymous-namespace kernels hardcoded to session layouts, needs
  llama.cpp ggml-cuda headers). "compile split .cuh into a .so" is not a drop-in.
  The MAIN engine kernels (src/kernels/cuda/native_*.cu, include/strata/kernels/*.hpp) are
  free functions in namespace strata::kernels with raw-pointer signatures
  (native_gdn_step(float* state, q,k,v,...), native_qsa_* etc.) — THAT is the borrow source,
  same as §60 iq_kernels precedent.

Phase-0 measurement (cn24cp arm, HG_NATIVE_PROF+HG_STEP_PROF):
- [hg-step] decode fwd avg 71 ms (prof-inflated; lean 50 ms), extend fwd 127 ms avg.
- Decode island apply avg 1.35 ms/layer (cs .72 h2d .4 ng .04) = ~27-31 ms of a ~50 ms lean
  step (per §77/§81 laws). => NON-MoE (GDN/QSA/HC/PLE/router torch+triton) = ~20-25 ms/step,
  i.e. ~40-50% of decode. Fused-kernel borrow ceiling ~= decode 50 -> 35 ms = +40%.
- Prefill remains island-staging dominated (16.8 s/chunk anatomy §83) -> the 15x gap is
  expert STAGING PIPELINE quality, not GDN/QSA torch ops.

ADOPTED SEQUENCE (supersedes proposal ordering):
A1. GDN/QSA/HC native bridge for DECODE: hydragen_bridge_gdn.cu + libstrata_gdn_kernels.so +
    opt-in pyhook/hydragen_gdn_native.py (HG_GDN_NATIVE=1/QSA/HC flags separate), static
    ping-pong scratch INSIDE the bridge (Strata design, safe outside capture). Target:
    decode step -30-40%.
A2. Prefill expert-staging ring port (pf_common/prefill.cu design: batched 64/128-row
    per-expert prompt jobs, pinned ring, async): attacks the 15x prefill gap AND may revive
    HG_EXT_WIN (window buffers live in the C++ ring, not torch's allocator — §87 poison is
    graph-capture + torch pool, not the C++ arena).
A3. Only then revisit CN=32 funding via scratch savings.
A4. Spec decode re-open stays behind A1+A2 (laws §79b/§85: per-row cost must flatten).

Co-edit declaration (file ownership for §89, all NEW names):
pyhook/hydragen_gdn_native.py, bridge/hydragen_bridge_gdn.cu, build_gdn_kernels.sh,
launch_phase2_m24g*.sh, logs/nohup_m24*.out, references m24-*.md.
Antigravity: do NOT touch these; stay off pyhook/hydragen_moe_native_graph.py (canonical
.bak-extwin restored) while A1 runs.

## 90. M24 Phase A1: libstrata_gdn_kernels.so build + GDN/QSA parity — PASS (the agent, 2026-11-12)

Ownership (as declared in §89): bridge/hydragen_bridge_gdn.cu, build_gdn_kernels.sh,
libstrata_gdn_kernels.so, test_gdn_parity.py, test_gdn_triton_parity2.py — the agent.

### Inventory (main-engine single-card kernels)
- src/kernels/cuda/gdn.cu          — reference GDN layer ops (self-contained, 248 L).
- src/kernels/cuda/native_gdn.cu   — pinned llama.cpp-derived fused GDN step (warp-shuffle).
- src/kernels/cuda/native_qsa.cu   — QSA rms_norm_weighted / rms_norm_grouped / gate_apply.
- include/strata/kernels/{gdn,native_gdn,native_qsa,native_qsa_indexer}.hpp — contracts.
- Vendored third_party/ggml NOT needed by these three TUs (only gdn/native_gdn/native_qsa
  includes + strata/core/geometry.hpp) — build is clean without ggml-cuda.

### Build
- build_gdn_kernels.sh: /usr/local/cuda/bin/nvcc -O3 -use_fast_math -arch=sm_86 -shared -fPIC
  gdn.cu native_gdn.cu native_qsa.cu bridge/hydragen_bridge_gdn.cu -> libstrata_gdn_kernels.so
- Status: COMPILED first try (1,175,328 B ELF .so). Exports hg_* flat C ABI (9 entry points).
- Flat-ABI shim required because GdnShapes is passed by const-ref (not ctypes-friendly).

### ABI contracts pinned empirically (these were NOT fully documented in headers)
1. strata ref gdn_step state layout is (S_i, h_v, S_j): k-dim OUTERMOST, head middle,
   v-dim fastest — NOT per-head (h,k,v). native_gdn_step same.
2. gate argument is LOG-space (kernel computes expf(gate) internally).
3. beta argument is pre-sigmoid (caller gates; gdn_beta_gate is the gating op).
4. l2norm of q,k is an UPSTREAM preprocess for BOTH kernels (native_gdn.cu contains none;
   the "fused l2" in split gdn_core.cuh is the multi-card engine only).
5. Readout scale: ref path = caller pre-scales q (scale*K); native path = 1/sqrt(S) internal
   (native_gdn.cu:118).
6. Head pairing: strata indexes k/q rows by h % h_k (CYCLIC). SGLang pairs hv->h by GROUP
   (hv // (HV//HK)). BRIDGE ADAPTER: pre-expand q/k rows to (HV, DK) with group mapping and
   present h_k = HV so the modulo becomes identity.
7. Native API requires a NON-NULL cuda stream (torch default stream handle 0 throws
   std::invalid_argument); use a dedicated torch.cuda.Stream().
8. qsa rms_norm_weighted/grouped: row-major rows of n_cols with gamma[n_cols]; gate_apply:
   attn (n_head, head_dim), q_full (n_head, 2*head_dim) rows = [q|gate], head-major.
9. gdn_conv_step conv_state is (C, d_conv-1) channel-major, oldest-first per channel row;
   kernel[0] taps OLDEST, new input appended at last tap.

### Parity results
- Unit parity (test_gdn_parity.py vs independent torch reference, 12 checks):
  WORST 1.192e-06 PASS (step ref/native state+out, l2, beta, conv+shift+append, outnorm,
  qsa_rms, qsa_gate).
- Triton parity (test_gdn_triton_parity2.py vs LIVE SGLang kernel
  sglang.kernels.ops.attention.fla.fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode,
  production convention use_qk_l2norm_in_kernel=True, scale=DK^-0.5, HK=16 HV=48 DK=DV=128,
  8 chained decode steps, in-place state pool both sides):
    native=0 (gdn_step):      max out 1.192e-07, max state 4.768e-07
    native=1 (native_gdn_step): max out 5.960e-08, max state 4.768e-07
  PASS < 5e-3 (ULP-level). No drift over 8 steps.
- SGLang decode path located: gdn_backend.py packed_decode -> fla.fused_recurrent packed
  kernel (state pool (slots,HV,V,K) in-place; packed qkv row [q|k|v]).
- Canonical server untouched during A1: H_200 after tests, 11524 MiB used.

### Next (A1b)
pyhook/hydragen_gdn_native.py wiring with HG_GDN_NATIVE opt-in per §89 plan; the adapter
must do (K,HV,V) state permute + group-pairing expansion + l2 preprocess. QSA/HC wiring is
separate (qsa decode/prompt attn kernels not yet exported in the shim — only norm/gate ops
are, since qsa attention itself in SGLang is a Triton FA variant, not the strata QSA path).
## 91. M24 Phase A1b: native fused GDN decode wired live — mechanism PASS, economics NEUTRAL (the agent, 2026-11-12)

Artifacts (the agent-owned; co-edit only with CHANGES entry):
- bridge/hydragen_bridge_gdn.cu v2: added hg_gdn_decode_fused — pool-native fused decode kernel,
  grid(B*HV, V/4), one warp per v-column, i-dim 32-strided coalesced; fuses q/k l2norm+scale,
  softplus/-exp(A_log) gating, sigmoid beta, group pairing i_h=hv/(HV//HK), in-place state IO,
  dead-slot zero path; templated pool x act x slot-idx (fp32/bf16 x fp32/bf16 x int32/int64).
  Zero permute: operates directly on SGLang pool (slots,HV,V,K).
- test_gdn_fused_parity.py: PASS all 8 dtype combos vs live Triton packed_decode, 8 chained steps:
  fp32/fp32 out 5.96e-08 state 7.15e-07; bf16 worst out 9.77e-04 state 1.56e-02 (bf16 quantum).
- pyhook/hydragen_gdn_native.py (HG_GDN_NATIVE=1, sitecustomize-gated): wraps
  TritonGDNKernel.packed_decode; guards ReplaySSM/dtype/shape/stride -> Triton fallback;
  derives HK from qkv_dim assuming K==V (holds: DK=DV=128); launches on current stream
  (graph-capture-safe); counters log native hits / fallback reason.
- launch_phase2_m24g1.sh (= cn24c + HG_GDN_NATIVE=1), launch_phase2_m24g1cp.sh (+ prof).
- sitecustomize.py: appended one gated import block (§91 comment). Canonical launch_phase2_m11d5cn24c.sh unchanged.

Build/ABI lessons: nvcc wants __forceinline__ (not MSVC __forceinline); templates must live
OUTSIDE extern "C"; void*->typed pointer needs explicit casts inside a non-template launcher
helper (<<<>>> will not implicitly convert void*); macro bodies with if/else need do{...}while(0);
shfl_down reduction full-sum lives in lane 0 only — broadcast with __shfl_sync before use
(caught by parity: state err 1.86 -> 7e-7).

Live boot (m24g1 lean): H_200; "[gdn-native] installed" x4 procs; native hits=64 triton=0
through bs1+bs4 breakable-graph capture; 693 MiB free (canonical ~679); no exceptions.

Quality (m24g1): battery_quality 8/8; QC6 direct exact primes 97,101,103,107,109; QC7 correct;
QC2 haiku / QC4 fib / QC5 fraction clean. QC0/QC1 think-preambles = known canonical boot drift.

Fairness (m24g1 lean): 41.13/41.28 then 45.53/47.39 warm. Do NOT compare across boot epochs:
paired profiling same-epoch below is authoritative.

Paired profiling A/B (m24g1cp vs cn24cp, identical probe sequence SS x3 + fairness):
- island apply T=1: native 1.19 vs canon 1.22 ms/layer (island untouched, as designed)
- step fwd bs4 deep-warm last20: native 63.5 vs canon 60.0 ms -> NEUTRAL (drift band)
- single-stream SS3: 20.5 vs 22.1 tok/s -> NEUTRAL
- fairness prof: 36.16/42.63 vs 42.77/37.76 -> NEUTRAL overlap

LAW (decode-step anatomy, cn24 canonical): bs4 decode step ~60 ms ≈ 48 layers x ~1.2 ms island
apply ≈ 58 ms; ALL non-MoE torch/triton (incl. Triton GDN packed decode + conv + QSA FA + dense)
shares ~3-6 ms total. GDN fusion ceiling is therefore ~1-2 ms/step (2-3%), below boot drift.
Closes GDN/QSA/HC fusion-for-decode as NOT-ECONOMIC on the staging-bound substrate (same death
as M19/§79b/§85: decode is island-staging-bound; attack island apply h2d/cs/preadd or prefill,
not the ~5% non-MoE remainder). Hook + kernel stay on disk, opt-in HG_GDN_NATIVE=1, verified
capture-safe and quality-neutral.

Next per §89 plan: A2 prefill expert-staging ring (extend windows: island apply 16.8 s/1024-tok
chunk, h2d 10.2 + cs 5.2 + preadd 3.6 — that IS the prefill gap).

Canonical restored: launch_phase2_m11d5cn24c.sh relaunched (PID in logs/sg-real.log).
## 92. M24 Phase A2: expert-affinity extend staging (HG_EXT_AFF) — SHIPPED OPT-IN; prefill island ~2.3x, staging ops 10.4x cut (the agent, 2026-11-12)

- Island: pyhook/hydragen_moe_native_graph.py, opt-in HG_EXT_AFF=1 (default 0, slice path untouched). Knobs: HG_AFF_TOK (batch token cap, 256), HG_AFF_PAIR (pair cap, 512), HG_AFF_EX (expert cap, _CACHE_N). Launcher launch_phase2_m24a2.sh = canonical + HG_EXT_AFF=1. Snapshot: .a2-working (135-line diff vs .bak-a2prof).
- Mechanism: sort window (token,slot) pairs by expert, batch into applies (experts<=CN, tokens<=TOK, pairs<=PAIR), gather xg via index_select, one fresh xf/out per apply, combine via res.index_add_. Same pair compute as slice path, but each expert staged once per window.
- Measured (T=1024 extend window, prof arm): staging ops M 4416 -> ~420 (10.4x), window apply 1187 -> ~517 ms (prof tax included).
- TTFT 6000-char greedy: A2 lean 32.1/34.6/34.7 vs same-day canonical lean 68.9/44.6/39.9; steady-tail 32.1 vs 39.9 = -20%; cold-first-seed 32.1 vs 68.9 = -53% (seed-order/wire-cache effect).
- Quality: battery_quality 8/8 QC_DONE, no QC5/6/7 echo degeneration; LONG 1262-token prime oracle EXACT "211, 223, 227, 229, 233, 239, 241, 251, 257". Fairness warm 39.3/40.5->44.6 (canonical band).
- Decode untouched: T<=TCHUNK keeps §86 slice path; graphs unaffected.
- LAWS banked (bug hunt, this section):
  1. native_expert_grouped entry semantics: out row = ent_dst (GLOBAL row of out, not per-token slot; per-token j must be remapped tokpos*K + j). Writing bare j aliases every token into rows 0..K-1 -> coherent-but-empty or garbage outputs, downstream qsa_indexer illegal access.
  2. cap_groups/cap_entries must equal ACTUAL nul/E of the call (grp_start has nul+1 entries; larger cap = OOB group bounds).
  3. Any scratch/xq8 growth that swaps st buffers must be pre-sized OUTSIDE the apply loop (freed block while prior _ng still in flight = corruption); keep-alive store in st avoids stream-allocator races.
  4. DBG arms that re-run the slice path after affinity DOUBLE-COUNT res by construction (rel=0.5/rel2=1.5/rel4=3.5 signature = identical results, not half-results).
  5. _METAB arena: values must be python ints for torch.tensor mixes; np.int64 conversion works but dump hid the real issue — trust coverage counters (aff_E vs slc_E) over max-diff forensics.
- Not adopted canonical yet: soak + co-edit serialization pending; A2 stays opt-in. §93 (M25) next: C++ execution worker (libhydragen_core.so / bin/strata) to escape PyTorch mem_fraction_static hold and fund CN=32/48 properly.
## 93. M25 Phase 1: Strata C++ execution worker booted on 12 GB — funding architecture VALIDATED (IQ2_XS smoke) (the agent, 2026-11-12)

- Direction (user-ordered): stop hunting VRAM inside PyTorch; adopt Strata's static C++ worker (build/strata + serve/server.py) to escape mem_fraction_static entirely.
- Assembled on the reference host from local artifacts only:
  - engine: ~/hydragen-experimental/strata/build/strata (95 MB, Nov 6 build of vendored tree)
  - front: serve/server.py --engine strata --config (OpenAI /v1/chat/completions, tokenizer from pack, port 8181)
  - pack: tools/iq_pack.py on ~/.models_backup IQ2_XS gguf -> ~/strata-packs/iq2_xs (index-only; experts stay in the GGUF, indexed by native_experts.txt)
  - IQ packs REQUIRE --spec T>=2 (MTP always on): used backup mtp/rt, --spec 4 --coupled-draft --spec-min-p 0.45
- Boot accounting (12 GB, vram-reserve 768, kv int8, batch 2, conversation-cache 6144MiB/8 slots):
  - expert cache auto -> 3343 slots = 4.50 GiB resident GPU experts (pre-filled from profile at 23.9 GB/s)
  - batch: 2 slot sessions 0.56 GiB each; prefill ring borrows 2949 slots; MTP draft 839 MiB (675 experts)
  - 799 MiB free with EVERYTHING loaded. No PyTorch hold at all.
  - expert arena = 33 GiB host file-cache (buffered, cudaHostRegister ok, THP skipped)
- Concurrency ladder (greedy, thinking off, 128 tok): WARM decode 40.1 tok/s (MTP accepted 3/3); C2 agg 72.6 (41.2/45.9 per-stream); C3 agg 100.2 (one stream 130.5 MTP burst); C4 agg 87.6 (~48-50 each). True 2-seq co-decode windows verified ("batch windows of up to 2 sequences, layers [0,48)").
- Contrast with SGLang substrate: 3343 resident experts vs CN=24; MTP spec ALIVE here (draft experts resident) vs dead-EAGLE there; conversation-cache parking vs KV-pool eviction.
- CAVEATS: IQ2_XS is a boot vehicle, far below Q5_K/quality floor; §80 anchor (agg 46 at c2) was IQ3_XXS on a loaded system — not directly comparable. Prefill rate on this boot unmeasured beyond 28-token prompts (81.8 tok/s = ramp-dominated).
- Next (M25.2 decision point): parity quant choice — IQ3_XXS (matches §80 anchor, 76 GB download from ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) vs IQ3_S (matches our canonical quality tier). /home has 92G free: either fits alone, both together do not.
- Config kept: /tmp/strata-m25-iq2.json (server up on :8181 while M25 active; SGLang canonical parked).
## 94. M25 Phase 2: IQ3_XXS C++ worker parity run — ADOPTED as funding architecture; prefill 13x, decode 2-2.5x, QC 8/8 (the agent, 2026-11-12)

- Artifacts: IQ3_XXS shards downloaded ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (47,039,860,096 + 28,800,138,432 = byte-exact to fingerprint), 11 min @ ~72 MB/s, to ~/.models_backup/Strata-data/models/IQ3_XXS/IQ3_XXS/. Existing backup pack packs/iq3_xxs reused as-is (index-only; shards resolved beside --native).
- Config: /tmp/strata-m25-iq3xxs.json = same 12 GB line as iq2 (vram-reserve 768, kv int8, batch 2, conv-cache 6144/8, spec4 coupled mtp, max-context 32768, adapt 160/round async).
- Boot accounting: arena 39.97 GiB host file-cache @3.22 GiB/s; expert cache auto -> 2410 slots / 3.95 GiB (pre-filled from profile 24.3 GB/s); batch slots 0.56 GiB x2; prefill ring borrows 2168 slots; 790 MiB free with everything.
- Gates (m25_gates.py, thinking off, greedy):
  - QC 8/8; QC6 exact "97, 101, 103"; QC7 exact; per-QC 0.4-1.5 s (vs SGLang 2.6-15.3 s).
  - TTFT fresh-seed 6000-char (1300 tok): 3.0/2.5/2.4 s = 684-732 tok/s prefill — reproduces the §80 anchor band 681-883 from open components; vs SGLang+A2 32.1 s = ~13x.
  - Ladder: C1 decode 87 tok/s (MTP burst), C2 62.3 agg (32.6/37.1), C3 76.3, C4 74.2 (~37-47/stream). vs CN24+A2 single 20-22 and cap4 agg 36-45.
- §80 anchor replay note: anchor C2 agg 46 vs today 62 — anchor ran 131k-context budgets (fewer expert slots); 32k ctx + int8 KV + profile prefill is faster at equal quant.
- LAW: pkill/pgrep -f patterns self-match when the SAME ssh command line also contains the literal target string (launcher half) — bracket trick insufficient; put kill+launch in a script FILE so targets never appear in any live cmdline.
- Decision: C++ worker (build/strata + serve) ADOPTED for the high-concurrency massive-MoE campaign on 12 GB. SGLang CN24+A2 stays as fallback line; island/A2 work superseded for serving but A2 laws port conceptually to any Python-front substrate.
- Open follow-ups: 128k-ctx kv-resident stress (deferred by user gate), IQ3_S upgrade path (needs /home headroom: delete IQ2 backup first, user call), expert-profile re-ranking from OUR traffic corpora vs shipped profile, /home at 20 GB free = disk hygiene needed before any further model pulls.

## 95. Concurrency decode anatomy on the q36 C++ worker — --batch-mtp law, window-row ceiling, DFlash2 scope

Decode-bound ladder (512-token answers, fresh A/B prompts, temperature 0) on IQ3_XXS C++ worker, the reference host, single 12 GB card:

```
arm                 C1      C2      C3      C4
no --batch-mtp      91.1    73.0    80.0    73.9   (batch windows carry ZERO MTP proposals; solo keeps spec)
--batch-mtp         92.5    74.0    80.9    90.9   (C4 +23%)
--batch-mtp tuned*  89.6    75.8    85.8    80.9
IQ2_XS smoke (93)   ~90     72.6    100.2   87.6   (3343 slots = 33% residency)
```

* tuned = --pcie-frac 0.5, --adapt-swaps 400, --vram-reserve-mib 512.

Laws banked:

1. `--batch-mtp` is REQUIRED for spec in batched decode. Without it, only solo (one-row-window) streams get MTP proposals; batch-decoded streams decode specless. C2 aggregate (2 specless ~38) < C1 (one spec'd 93) — the "C2 worse than C1" signature is spec attribution, NOT PCIe contention.
2. serve-layer per-request `draft_n_accepted` under-reports on batch windows (shows 0 while the engine accepted). Trust the engine telemetry line `strata batch: MTP proposals accepted X of Y` — observed 78.4-84.0% acceptance with bmtp.
3. Window-row ceiling: `kVerify = 8` (compile-time), batch windows of up to 2 sequences for layers [0,48); bmtp adds slot proposals into the same 8 rows (room = kVerify - S, greedy slots in turn). Aggregate decode is capped by window rate: 36-43 ms per 8-row window -> ~25 windows/s -> ~80-97 tok/s aggregate ceiling regardless of C. C4 ~= C1 with bmtp; never meaningfully >1x on IQ3_XXS.
4. Window time anatomy (C4): CPU experts 14-29 ms + CUDA wait 12-20 ms + PCIe 1-4 ms. PCIe is NOT the wall (2.5-11% of routed entries ride PCIe). Tuning (pcie-frac/adapt-swaps) moves entries from CPU-spill to PCIe-staged but the SUM stays flat: the true variable is expert residency. IQ3_XXS = 2410/10240 (23.5%); IQ2_XS = 3343/10240 (33%) is the only ladder that exceeded 1x solo (C3 100.2).
5. DFlash2 scope: implemented only in the split engine (`strata-split-N --draft dflash2 --draft-model PACK`, two cards) and vendored llamacpp dflash.cpp. The q36 single-GPU engine has ZERO dflash references (grep). Documented economics: +5-14% greedy code/JSON, slower on sampled text, agent turns, and at 4 users; Auto never selects it. Consistent with SGLang-side §84/§85 DFlash2 closure. Not a lever for this substrate.
6. Double-server trap: a killed-but-not-dead old serve keeps port 8181 and answers HEALTH_200 while the new config never serves. After any swap verify the live ENGINE cmdline actually carries the flags (`tr '\0' '\n' < /proc/$(pgrep -x strata)/cmdline`), and kill serve + engine by name in a script FILE.
7. Concurrency value on one 12 GB card is prefill overlap + conversation parking (8 slots), not decode aggregate. Decode aggregate scales past 1x only via residency (IQ2 tier), profile rerank (next lever), or the documented two-card split-engine path.

Artifacts: /tmp/m25_ladder2.py, /tmp/strata-m25-tuned.json, /tmp/m25clean.sh, /tmp/m25tuned.sh (the reference host). Configs: ~/hydragen-experimental/strata-iq3xxs-m25.json (bmtp now included).

## 96. Strata 0.1.59 (v0.2.13) built on 12 GB + DFlash2 arm — parity confirmed, engine self-warns

Built v0.2.13 engine on the reference host: git worktree ~/strata-0213, cmake -G Ninja -DSTRATA_ENABLE_CUDA=ON (default OFF; without it there is NO 'strata' target), ninja+cmake from ~/hydragen-experimental/.venv-build/bin, -DCMAKE_CUDA_ARCHITECTURES=86-real -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc, target strata. -> strata 0.1.59, BUILD_OK.

DFlash2 packs fetched from v0.2.13 assets (sha256 OK), kept at ~/.models_backup/Strata-data/dflash2/: q4_0-bf16sel (498 MB) and q4_0-selq4 (324 MB; selector codebooks Q4_0 = 174 MiB less VRAM, same drafts/text). Wire-up: --draft dflash2 --draft-model PACK + keep --mtp DIR (draft_vocab only; MTP layer itself NOT loaded) + --batch-mtp enables STRATA_BATCH_DF2 slot drafters (85 MiB private K/V ring each).

Decode-bound ladder (512 tok, fresh sessions, temp 0), IQ3_XXS, batch 2:

```
arm                            C1     C2     C3     C4
0.1.57 bmtp                    92.5   74.0   80.9   90.9
0.1.57 bmtp tuned              89.6   75.8   85.8   80.9
0.1.59 bmtp tuned MTP          89.3   77.2   84.9   80.7   (no regression; fixes #5096/#4844/#5531 free)
0.1.59 bmtp tuned DFlash2      93.6   69.2   81.7   90.2   (best C1 seen; C2 dips)
```

DFlash2 facts on 12 GB:
- Engine prints on boot: "--draft dflash2: on a 12 GB NVIDIA card the MTP drafter was faster in most measurements (code prompts at a window of 6 gained 3%); continuing as asked".
- drafter = 591 MiB VRAM Q4_0 + 178 MiB draft head; replaces MTP layer (839 MiB) -> expert cache 2410 -> 2598 slots (+188), decode hit-rate 88-89%.
- Batch windows: "DFlash2 slots: N rows deferred (no spare window row)" — batch drafts only ride SPARE rows; the kVerify=8 row budget is the same hard wall DFlash2 cannot widen.
- Fresh-session penalty: "context rows of the window are not the drafter's to give... drafts poorly for the first rounds (answers the same)" — every ladder stream is a new conversation, which is DFlash2's worst case; steady logged sessions would draft better.
- "MTP proposals accepted 688 of 333 (206.6%)" telemetry lines in df2 runs: counter mislabels; do not read as >100%.

VERDICT: DFlash2 is REAL on 12 GB with 0.2.13 (user's read was right) but PARITY with MTP on our mixed prose/code ladder, exactly as Strata's own per-card measurements say (Auto keeps MTP on 12 GB; DFlash2 default only on cards with all experts resident / specific quants). Not adopted; MTP stays. 0.1.59 adopted for the free engine fixes. Configs: /tmp/strata-m25-{mtp,df2}0213.json. The one card's decode aggregate stays ~= 1x single-stream specced: row-budget-limited, confirmed independently by DFlash2 deferral telemetry.

## 97. Context-size curve on 0.1.59 (MTP bmtp tuned, IQ3_XXS, batch 2) — KV streaming discovered

```
ctx       expert slots  C1      C2      C3      C4
8192      2392          106.9   88.1    96.6    90.5
32768     ~2410         89.3    77.2    84.9    80.7
131072    1856          90.6    74.7    95.9    80.2
```

Facts banked:
1. 128k ctx boots HEALTHY on 12 GB: engine enables KV STREAMING automatically: 20480 of 131072 cells per QSA layer stay in VRAM, the rest of K/V lives in 1.55 GiB pinned host RAM. GPU KV cost does NOT scale linearly with max-context on this path (earlier 0.56 GiB/slot linear estimate applies to <=32k full-reserve only).
2. Slots at 131k: 1856 (-23% vs 32k) — the KV window + pinned pool is funded partly from expert cache; ladder still only -5 to -8% vs 32k.
3. 8192 gave the program best single-stream: 106.9-109.1 tok/s; the win at short ctx is attention/decode-path cost, not just slots (8k had ~equal slots to 32k but +14%).
4. expert-cache auto slots are internally capped (~2400 on this card regardless of ctx when free grows: 8k had 6.14 GiB free -> 2392 not ~3100); more free VRAM does not buy slots 1:1 — auto-sizing + profile/ring constraints.
5. UNTESTED DIMENSION: the ladder is short-prompt (~0.5k). The real agent-harness cost of 128k lands in KV window thrash: sessions beyond 20480 cells/page out K/V reads over PCIe per decode token; parking many 128k conversations in the 6 GiB conversation cache overflows (~2.2 GiB each). Need a long-context concurrency arm (>=20k-token prompts, C2-C4) to measure it.

## 98. Long-context concurrency arm (ctx 131k, 22.5k-token sessions, 0.1.59 MTP bmtp tuned)

Design: 4 sessions, ~22535 tokens each (shared ~1.1k-token runbook head + unique 900-line body), 512-token greedy answers, C1->C4 on the live 131k stack. NOTE: repeated levels re-ran the same prompts (cache-hit confound); cold-vs-warm separated via engine telemetry, not request layout.

Ladder: probe 22535 tok prefill 9.2s = 2451 tok/s. C1 57.6 agg (cached, short answer). C2 9.0 (one cached stream 123.3 + one cold-prefill stream 11.7). C3 13.6 [68.4, 39.9, 139.1]. C4 16.5 [12.2, 42.9, 122.7, 48.4].

Findings banked:
1. LONG-CONTEXT DECODE IS CHEAP. Cached sessions decode at 22.5k ctx at 122-152 tok/s (spec'd, drafts accepted 95.7-97.8%) and 40-68 tok/s in batch windows — at or above short-context levels. KV streaming: 74-97% of block reads hit VRAM (20480-cell window vs 22.5k ctx; only 45-100 MiB/period from pinned RAM). The "long context = slow decode" fear is unfunded on this engine.
2. PREFIX REUSE IS ALIVE AT SCALE: "prompt 22535 = 22528 reused + 7 read in 240 ms"; "restored 22528 tokens (checkpoint) in 58.3 ms"; conversation cache reports reused_kv_bytes=343,863,168 — KV BYTES move with the conversation, not recomputed. At this scale Strata delivers RadixAttention's compute half; only the storage half (shared ref-counted blocks) remains SGLang-only.
3. PARKING MATHS: parked 22535-token conversation = ~394-437 MB (int8 KV + snapshot overhead); cache bytes grew 1.4->4.4 GiB with 5-8 parked sessions, byte-cap evictions fired gracefully (evictions=6). 6 GiB conv cache ~= 14 sessions at 22k, ~2 at 128k — a dial (--conversation-cache-mib), not a cliff.
4. THE REAL LONG-CONTEXT COST IS ADMISSION: a cold 22.5k prefill inside batch windows poisons co-resident streams: windows stretched to 53.5-56.4 ms (vs 25-28 warm), routed VRAM hits fell to 21% (CPU experts 45-47%), and co-running streams dropped to 11.7-42.9 tok/s. Chunked prefill exists ("prompt read in 2 parts, slots decoding 1273 ms between them") but the interleaving budget is naive.
5. Prefill scales: 2408-2812 tok/s at 22.5k (vs ~700 at 1.3k) — the prefill ring batch-amortizes; 128k-token cold prefill ~= 45-55 s.
6. Verdict on the RadixAttention repricing question: NOT repriced upward for this architecture. What to fix instead: prefill/decode window scheduling (isolate cold prefills from decode windows), and conv-cache byte budget for 64-128k sessions.

## 99. Prefill-isolation config sweep CLOSED — the poison is structural (ctx 131k, cold 22.5k sessions, ladder4 fresh-prompt-per-stream)

```
arm                                   C1     C2     C3     C4 wall   co-resident cold decode
base (--prefill auto:16384)           9.3    18.9   28.6   38.0 s   11.5-12.0 tok/s
--prefill 8192                        9.2    18.6   28.1   37.3 s   12.1-12.6
--prefill 4096                       12.3    24.8   37.2   49.7 s   10.0-17.7
STRATA_COALESCE_PREFILL=1             9.4    18.9   28.8   38.3 s   7.4-11.9
STRATA_PREFILL_LEND_PCT=50           24.1    48.5   72.5   96.6 s   18-19 (prefill collapsed 2450->243-973)
```

Laws banked:
1. No config knob isolates decode from cold-prefill poisoning. Chunk size, yield granularity (implicit via chunk), serve-level coalesce, and lend-fraction are all flat or negative. p8192 is free-but-neutral; p4096 buys ~+40% co-resident speed at −30% wall.
2. Prefill THROUGHPUT is ring-coupled: the prefill ring is the MMQ expert-staging rate limiter; shrinking it (LEND_PCT=50 -> ~1205 slots) multiplies streaming waves and collapses read rate 2.5-5x, stretching the poisoned period over the whole read. Poison duration ~= prompt_tokens / prefill_rate, and prefill_rate collapses with staging.
3. Engine default lend on this card = 90% of expert-cache slots (STRATA_PREFILL_LEND_PCT default; kAutoLendPct pinned_share>=0.9 ? 90 : 85; g_lend_keep floor 128 slots).
4. The fix is ENGINE POLICY, not config: (a) pin top-N hot decode slots against prefill-ring eviction while KEEPING ring size, (b) refill-then-decode ordering, or (c) the serve-level PD-disaggregation primitive spotted in usage: server.py --split-serve PREFILL,DECODE (unexplored).
5. Swap law: serve holds :8181 for ~5-15 s after TERM (graceful drain). Kill loop must WAIT for curl-refused before relaunch, else the next arm silently probes the old server or dies on bind.
6. kVerify=8 is `static constexpr int kVerify = 8` (q36_engine.cu:1165) — widening is one constant + buffer sizing, pending after config-level batch arms price the row ceiling.

## 100. Wide-batch arm (b4) CLOSED + long-decode ladder5 baseline + PD-split infeasible — the 12 GB one-budget law

ladder5 (long decode: 432 tokens/stream, primes 200-900, cold 22.5k prompts, ctx 131k):
```
arm                          C1              C2               C3                    C4
base5 (batch 2 spec4 bmtp)   172.0 spec      [99.4, 28.9]     [185.2, 29.0, 29.2]   [29.4, 28.8, 29.1, 98.1]
b4 (batch 4 spec4 bmtp)      142.5 spec      [102.8, 19.4]    [18.3, 85.3, 10.8]    [17.9, 10.4, 78.0, 9.1]
b4x (batch 4 spec 0)         IMPOSSIBLE: native IQ packs require --spec T>=2 at boot (engine exit 2)
```
Boot forensics: batch 2 -> expert cache 2456 slots; batch 4 -> 1663 slots (-32%). 4 slot sessions + 4 batch-mtp drafters (50 MiB each) ate 1.27 GiB of expert budget BEFORE any token decoded. b4 co-resident streams: 9-19 tok/s (vs 29 at batch 2). C1 solo also fell 172->142.5 from the squeeze.

Laws banked:
1. THE 12 GB ONE-BUDGET LAW: expert cache, batch slots, prefill ring, KV-resident window, drafters, output head all draw from one VRAM budget. Every measured attempt to trade between them (LEND_PCT, chunk, batch width, int8 KV, KV streaming) moves the wall, never lowers it. §95/§97/§99/§100 together = closed.
2. Wide continuous batching does NOT beat "2 users + drafts" on 12 GB IQ3_XXS: slots cost experts at ~0.45 GiB each; kVerify widening (8->16) is moot until slots get cheap (they'd need KV + drafter state near-zero).
3. --split-serve PREFILL,DECODE (serve-level PD disaggregation, docs/SPLIT_SERVE.md) is TWO-CARD only (two engine loads). the reference host has one 12 GB card -> infeasible here; it is the right primitive on a 2-card box.
4. Long answers amortize MTP much better: spec decode 172-185 tok/s sustained on 432-token answers (accepted 321-324 of ~330) vs ~90-130 on short answers. Program best C1: 185.2.
5. Prefill poison is residency arithmetic, not policy: the engine already lends TAIL (coldest) slots first ("list the pairs from the highest slot down, as the borrowing takes them"), refills after the read, and yields by STRATA_BATCH_YIELD_CHUNKS=8. A 22.5k-token stream's routing footprint simply does not fit in the ~12% residency that survives its own cold read. Product mitigation = checkpoint reuse makes it first-turn-only (admission already serializes cold reads at ~2400 tok/s = 9.5 s per 22.5k).
6. Next lever (pre-agreed): expert-profile re-rank from our corpora. Economics now clear: it raises warm hit AND shrinks the decode-time footprint that must survive cold reads (compact hot sets decode 29 tok/s at batch2; the squeeze sensitivity in b4 shows footprint-vs-residency is the live axis).

## 101. --no-prefill-borrow arms: poison immunity is REAL, paid in permanent residency (anti-CoW lesson)

```
arm (ctx 131k, ladder5 432-tok answers)      expert slots   C1 spec   cold co-resident   C4 wall
base5 (borrow default)                        2456           172.0     12.0 (poisoned)    50.4 s
npb4 (--no-prefill-borrow --prefill 4096)     736  (1.21GiB)  83.5      23.2-26.6 stable   69.0 s
npb  (--no-prefill-borrow, auto:16384)        verify-window instantiate OOM -> all requests 400, engine inert
```

Laws:
1. --no-prefill-borrow carves the prompt chunk buffers + ring PERMANENTLY. At --prefill 4096 the carve costs 1720 expert slots (2456->736). It does buy what it claims: co-resident decode during cold 22.5k reads holds 23-26 tok/s instead of cratering to 12 (zero cache interference). C1 spec decode collapses 172->83.5 from 736-slot residency (7%).
2. Bare --no-prefill-borrow with auto:16384 on 12 GB = engine survives but verify windows cannot instantiate ("batch instantiate: out of memory", every request HTTP 400). Never combine no-borrow with big chunks on 12 GB.
3. Verdict: not adoptable on 12 GB (permanent tax > transient poison), but it PROVES the poison mechanism is cache interference, and it is the anti-CoW contrast: npb removes interference by separating budgets and pays experts; CoW removes interference by deleting KV duplication and can create budget. If/when CoW KV pages land, per-slot KV cost drops ~0.36->~0.05 GiB for shared-prefix sessions and this whole trade flips positive.

## 102. Expert-profile re-rank: CLOSED-NEUTRAL (learned top-2456 ties shipped, even in-sample)

Toolchain found (good to know exists): tools/make_profile.py (STRP v1, 48x512=24576 ranked pairs; --reorder = traces decide top), serve-side --expert-profile-save PATH (#477: saves residents+routing counts on QUIT / every N min), --dump-routing trace format.
Learn loop: canon + --expert-profile-save, drove ladder5 + 3x6-turn code-review sessions + QA burst + coding prompts + battery; QUIT saved data/hg-learned.bin (196,632 B = full 24576-pair re-rank, no coverage loss).
A/B (ladder5, ctx 131k):
```
arm                     C1 spec   C2      C3      C4 wall   battery QC
shipped profile         174.1     24.1    37.2    50.1 s    8/8
learned profile         175.1     24.3    37.3    50.1 s    8/8
base5 earlier           172.0     22.9    35.9    50.4 s    -
```
Laws:
1. On this engine the expert-profile only sets the BOOT PRE-LOAD; the adaptive tier (adapt-every 1, swaps 400, decay 0.97) converges to the conversation's hot set within the first windows, so profile re-rank is a statistical no-op in steady state AND at first-request (shipped top already overlaps our corpus's hot set).
2. Expert hot-set residency was hypothesized as "the remaining scarce axis" — measured NEUTRAL. The remaining walls are pure budget (slots vs experts, §100) and rows (kVerify=8, §95). CoW KV pages + kVerify widen stand as the only engine levers with real headroom.
3. --expert-profile-save is kept as a restart-warmup affordance, not an adoption.

## 103. M-C0 determinism spike: cross-run KV is NOT bit-identical; CoW must be share-at-fork (engine's own token-prefix reuse already vouches for it)

Native tooling found (use this, don't re-patch): STRATA_STATE_HASH=1 (config env dict only; server.py does NOT pass shell env to the engine) fingerprints every finished request: gdn/ple/tail/pooled/kv/mtp/stale/dead FNV-64 over [0,L), page-aware (page_size, per-head ranges). Hybrid-model serve loop + fingerprint live in src/program/generate.cpp (~19544-19650); the q36_engine.cu serve printfs are a DIFFERENT standalone-serve variant that never runs under serve/server.py.
Spike (cow worktree clean v0.2.13, same 9952-token prompt as first request after boot, max_tokens=1, plans auto:16384 / 4096 / 8192):
```
plan          read tok/s   kv=              gdn=             ple=
auto:16384    2337         a4df868dd62ecf7b  a1f9eacffa5264cb 0f8dfbf41b70f4cb
4096          1763         b89f178bcecdbb39  bf94853f861d4ac3 64db0c02e2dc8600
8192          2378         c5a066692a7adb9d  12876e0195779855 862fa5b0a84e0308
```
All differ (kv AND gdn AND ple: chunk plans change hidden states, not just attention K/V; adaptive-tier residency changes mid-read add another source of run-to-run variance).
Laws:
1. Content-addressed KV dedup (hash-match independently prefilled prefixes) is IMPOSSIBLE on this engine. Dead end, closed.
2. CoW design is therefore share-at-fork: the FIRST session to prefill a token prefix owns the pages; later sessions with the same token prefix REFERENCE them (refcount + private append pages). No new numerical bet: serve already reuses prefix bytes across sessions by token key (22528-of-22535 reused; checkpoint restores; QC gates green program-wide). CoW extends the existing trust to VRAM page ownership, saving the ~0.36 GiB copy per forked session.
3. Fork discipline consequence: sessions that arrive cold-parallel (no first-mover) each prefill their own copy (different numerics); sharing only helps sessions forked after a prefix exists. For the agent-farm shape (shared system/repo prompt pinned at session 0) that is exactly the dominant case.
4. M-C1 target files updated: generate.cpp serve/admission + core QSA KV pools (page_size already exists in the fingerprinting loop = paged layout already in use: (pg*n_head_kv+hd)*page_size cells - refcounting is the addition, not paging).
5. Serve-env law: engine env ONLY via config "env" dict; a runner injecting env must write the config json, never export before nohup.

## 104. cow-m1 PERM validation: page_table indirection is kernel-safe — CoW needs zero kernel changes

cow worktree (clean v0.2.13 + one patch): STRATA_COW_PERMUTE=1 swaps adjacent pages in the mode-0 identity table at allocation (core/layer.cpp ~968). All QSA pools boot permuted; serve battery (battery_quality_m25) PASSes vs identity control on the same binary: QC6 97,101,103 exact both arms, QC4/QC5/QC7 match.
Law: every hot kernel path (kv_append_q8/q4, kv_gather_step, qsa selection attn, verify, snapshot) resolves logical->physical through st.page_table with NO identity short-circuit anywhere in the host call path (mode-0 just fills identity; kernels always receive the real table). Therefore ref-counted cross-session page sharing = host-side allocator + admission plumbing ONLY; GPU code is untouched.
Remaining M-C1 surgery map:
- copy machinery: generate.cpp copy_to_slot/copy_to_main/copy_session_between (~14373), direct_slot_copy (14372, device2device by state table), conversation_kv_copy (core/conversation_copy.hpp, shim over 14390), conversation KV exchange buffers (536).
- fork fast-path insertion point: conversation_kv_copy callers at admission: instead of copying pages [0, upto), re-point to.page_table entries at from physical pages + refcount; copy only tail pages.
- allocator: add per-arena frame refcounts; free = deref; last-ref frees. Appends must allocate private frames (allocate-on-write for the boundary page).
Next gate after implementation: battery + ladder farm bench (N forked sessions sharing a 22k prefix), boot accounting must show expert slots near canonical with batch 4.

## 105. b4k8 economics bound (batch 4 + --kv-resident 8192): the KV window is first-order; CoW prize and target metric are now concrete

ladder5 (432-tok answers, cold 22.5k prompts, ctx 131k):
```
arm                              expert slots   C1 spec   C4 co-resident   C4 wall
base5  batch2 kv20480            2456           172.0     29.4/28.8/29.1   50.4 s
b4     batch4 kv20480            1663           142.5     17.9/10.4/9.1    ~66 s
b4k8   batch4 kv8192             2109           166.6     14.6/10.8/23.5   52.9 s
```
b4k8 boot: "--batch-mtp: 4 slot drafters (50 MiB each); 4.91 GiB free"; auto 1595 -> cache 2109 slots 3.46 GiB; prompt chunk auto clamped to 5888 tokens / 512-slot ring at batch 4.
Laws:
1. Halving the per-session resident KV window at batch 4 returns +446 expert slots (1663->2109) and C1 142.5->166.6. The KV-resident window is a first-order, reclaimable VRAM cost; the arithmetic CoW claims (per-session window bytes) is real, not speculative.
2. But window shrink taxes long sessions in proportion to streamed fraction: 22.5k stream vs 8192-cell window = ~64% attention reads from pinned RAM; C4 cold co-resident stays poisoned (10-23 tok/s). b4k8 = the NO-CoW lower bound: same VRAM reclaimed, but paid in per-session window quality.
3. CoW strictly dominates b4k8 for the farm shape: shared 20480-cell prefix window (bytes freed WITHOUT shrinking any session's window) + N private tails. M-C1 success metric is now concrete: batch4 boot >= 2456 expert slots (as if one session paid the window) and C4 co-resident >= 29 tok/s (base5 unpoisoned rate).
4. Implementation shape re-confirmed (cow-m0/m1 findings): mode-1 streaming map allocates resident slots ON-DEVICE (kv_stream map, slot_ref refcounts exist for eviction). True cross-session sharing needs a device-side global frame free-list + atomic pop in kv_stream promote/evict (2-3 device fns) + host arena wiring. PERM validation (§104) proved read/write paths are table-indirect safe; the remaining risk is purely allocator promotion.
5. Stale-arm law extended: a config FILENAME does not identify a running arm; read the engine's live cmdline (--batch/--kv-resident via /proc/PID/cmdline) before trusting :8181. First b4k8 attempt booted base under the b4k8 filename (runner regenerates config from base when EXTRA is empty).
## 106. M-C1a/M-C1a2 global K/V slot arenas land at full parity: per-layer arenas + map.slot_base are graph- and battery-safe, and the private-pool carve is fully reclaimable

The kv_stream CoW blueprint's allocation half is now real code in `~/strata-cow`
(cow-m1a commit + m1a2 patch).

### M-C1a (plumbing, double-counted)
- `KvStreamMap.slot_base`: local slot j names arena page `slot_base + j`; set once per state at
  init, graph-baked. Device resolve: page table stores GLOBAL names (`sl + slot_base`), copy dst
  offsets by `sl + m.slot_base`, stamp/ref stay LOCAL arrays, and out-of-range entries are already
  skipped for stamp/ref (that skip IS the future CoW eviction-safety primitive).
- One arena per QSA ordinal: 4 cudaMallocs (k/v codes int8, k/v scales uint16), lazily allocated
  on the first streamed state. **n_qsa_layers = 12 on this model** (not 16 as earlier assumed).
- Env: `STRATA_COW_ARENA=1` + `STRATA_COW_ARENA_SLOTS=<slots/layer>` (config env dict, as always).
- Gate run (arena 25600, double-counted): expert cache fell to 1707 experts / 2.81 GiB —
  exactly arena bytes (25600 x 4224 B x 12 = 1.27 GiB ~= -749 expert slots) — the allocation is
  real and lands BEFORE expert-cache auto sizing; battery stayed green.

### M-C1a2 (pricing exclusion + range-correct consumers)
- `qsa_state_bytes` excludes the int8 streamed pool for arena states; `qsa_state_init` skips the
  four private takes; `QsaState.cow_codes/cow_scale` carry element offsets of the state's range;
  `conversation_snapshot pools()` returns range-offset device pointers; `qsa_state_zero` zeroes
  only the state's range (the M-C1a zc/zs expressions were missing *head_dim on codes — fixed by
  the stride fields; harmless pre-fix because fresh sessions never read past their cursor).
- Parity gate vs base5 (arena 15360 = 3 sessions exactly; boot never printed "no room", so
  canonical serve creates exactly 3 streamed mode-1 sessions):

| metric | base5 canonical | m1a2 arena |
|---|---|---|
| expert cache | 2456 slots | **2451 experts / 4.02 GiB** |
| C1 spec | 172.0 | 164.2 (accepted 324, boot variance band) |
| C4 co-resident | 29.4/28.8/29.1 | **29.0/28.9/29.1** |
| C4 wall | 50.4 s | **50.4 s** |
| battery | QC_DONE | QC_DONE, QC6 97/101/103 exact |

### Laws banked
- Per-layer arenas + slot_base indirection need NO CUDA-graph changes: tables are baked per
  session and the resolve kernel offsets are constants once captured.
- Session count on canonical serve = 3 streamed states (1 main + 2 batch slots at --batch 2).
- Pages >= a page-aligned shared prefix are never written after sharing: appends only touch the
  newest page, and fork boundaries are block-aligned (checkpoint reuse is 4-token aligned).
  **The farm shape therefore needs no append copy-on-write at all** — only complete, immutable
  pages are ever forked.
- Next (M-C1b): per-layer device `slot_use[]` + clock-sweep skip of `use>1`, admission fork as a
  host-side D2H/H2D page-table splice + batched use bumps, and the session-reset release hook
  (decrement out-of-range entries before kv_stream_reset memsets the map).

Files: ~/strata-cow/src/core/layer.cpp, src/kernels/cuda/kv_stream.cu, src/core/session.cpp,
src/core/conversation_snapshot.cpp, include/strata/*; runner /tmp/m25cowrun3.sh; log
~/hydragen-experimental/logs/cow-m1a2.log.

## 107. M-C1b/M-C1c live: slot_use refcounts + admission fork land at full parity — batch-4 farm sharing is REAL (CoW end-to-end on the canonical streaming KV)

The share-at-fork CoW primitive is now live on the Strata C++ worker's canonical mode-1
streaming KV (kv_stream), with zero GPU-kernel changes to read/write paths:

### Mechanism (all env-gated; canonical path untouched when off)
- `KvStreamMap.slot_use`: one refcount array per QSA-ordinal arena (device).
  Clock-sweep eviction (resolve_kernel step 2) skips any slot with `use > 1`
  (`cow_hold`), bounded sweep leaves the miss unresolved on host read -> graceful stall-free.
- Step-1 residency marking range-checks: entries naming another session's arena range
  (forked pages) are looked through, never stamped/ref'd locally; the owner's clock governs.
- `qsa_cow_fork(from, to, upto)`: host-side table splice. Reads the source's live page table
  (D2H), drops entries outside the source range (aliased/cold -> -1), bumps `use[]` over the
  source range (batched H2D), splices `to.page_table[0..P)` (H2D). Zero new kernels.
- FLOOR law: `P = upto/4` — only COMPLETE pages are forked. The partial tail page must stay
  cold: two sessions appending different tokens into one shared page would trample each other.
- Release law: `qsa_cow_release` runs in `refresh_residency` BEFORE `kv_stream_reset`:
  a session lets go of foreign-range entries it held when its own map resets. Owners never
  release other holders' refs; holders release their own. Parked sessions keep pinning shared
  prefix pages (use>1) — that's a feature: resume keeps the prefix resident.
- Range law: per-ordinal cumulative `g_cow_next[]`, NOT `ctr * p.slots` — with
  `STRATA_COW_SLOT_CELLS` the first (main) session keeps the full window and every later
  session gets a smaller one; ranges must stay disjoint under shrink.
- Base-0 law: `slot_use` is baked for every arena state — session 0's range legitimately
  starts at slot_base 0; keying the bake on `cow_sb != 0` silently no-ops every main-sourced
  fork (the farm shape's main case).
- Observability: engine stderr lands in the config `log` file
  (~/hydragen-experimental/logs/strata-m25-iq3xxs.log), NOT the runner nohup log
  (server.py: stderr=self.log, stdout=PIPE).

### Gates
- fork2 (b2, arena 15360, fork on, pre base-0 fix): battery QC green, farm green —
  fork wiring fires but shares 0 pages (the base-0 bug, caught by the log:
  "points at 0 of 5860").
- fork3 (b2, post base-0 fix): battery green, farm C1 163.3 / C2 85.5+91.0 /
  C3 84.5+186.8+80.6 / C4 [81.5, 81.8, 85.7, 91.2] agg 87.7 (parity with fork2 89.0 —
  b2 windows fit everything alone, sharing is free-but-uneventful there);
  fork telemetry: 529-2569 pages per admission (resident-only fork degrades gracefully:
  main's window holds the prompt TAIL, the cold head streams from host per session).
- expert cache held at 2451 across the arena boot — private-pool carve fully reclaimable,
  pricing exact (b2 arena 15360 = 0.73 GiB printed at boot).

### b4 economics (farm shape = 4 concurrent sessions of one 22.5k-token conversation, ladder5 answers)
| arm | expert slots | C1 spec | C2 | C3 | C4 co-resident | C4 wall |
|---|---|---|---|---|---|---|
| plainb4 (no CoW)      | 1663  | 128.3 | 75.1/80.2       | 52.4/54.5/57.7       | 42.5/43.5/44.5/45.1 | 23.9 s |
| forkb4 (CoW fork)     | 2136  | 151.2 | 88.7/81.7       | 64.7/58.6/60.7       | 49.9/48.9/47.1/47.7 | 18.5 s |
| b4k8 (§105 sim floor) | 2109  | 166.6 | —               | —                    | 14.6/10.8/23.5 (unique) | 52.9 s |
| base5 (§100, b2)      | 2456  | 172.0 | —               | —                    | 29.4/28.8/29.1 (unique) | 50.4 s |

forkb4 boot: arena 9216 slots = 0.44 GiB printed, expert 2136 (3.50 GiB); fork telemetry
1233-1249 pages per b4 admission (slot windows 4096 cells = 1024 slots each).
QC battery green on every arm (QC6 97,101,103 exact; farm qc_primes True on all streams).

### Law banked
- CoW share-at-fork is live on the canonical streaming KV: batch 4 recovers +473 expert
  slots (1663->2136, above the b4k8 2109 floor) AND beats plain b4 at every concurrency
  level (C4 agg 72.2->93.2, wall 23.9->18.5 s). Quality untouched.
- Remaining C4 poison is the COLD HEAD: ~4600 prefix pages outside main's resident window
  thrash each slot's 1024-slot window independently (x4 duplicated fetch churn).
  Next lever (M-C1d): join-at-fetch — when several fork-siblings miss the same cold block,
  fetch once, point all their tables at it (use>1 pins it; sweep guard already honors that).
  The entire 22.5k prefix is only ~297 MB across 12 layers; the budget exists — it is the
  per-session window policy that refuses to share fetched cold pages.
- Slot-window shrink (STRATA_COW_SLOT_CELLS) is the funding knob: main keeps the full
  --kv-resident window, later sessions get the small one; ranges disjoint via per-ordinal
  cumulative counter.
## 108. M-C1d join-at-fetch: the mechanism is proven race-clean, but pinned sharing is net-negative on small windows — the real prize needs a lineage shared region (M-C1e)

Goal: stop duplicate host->VRAM fetches of the same cold prefix block across forked batch siblings (forkb4 farm left ~4600 cold prefix pages thrashing their own 1024-slot windows each).

Mechanism landed on `cow-m0` (`~/strata-cow`):
- `CowPeer` registry (device array, per streamed arena-backed state): `conv` (fork lineage, single 4 B mutation law), `hhead` (join headroom), `ord/base/n_slots/n_blocks`, `pt` pointer.
- resolve step 3 cold miss: scan same-ordinal same-lineage peers; if peer table names block b at global slot s: join = headroom gate -> unconditional `atomicAdd(slot_use)` with prev>=1 gate -> revalidate peer entry -> point own table at s; stale joins roll back. ANY number of joiners per page.
- `fork_join_kernel` + `cow_release_kernel` moved fork/refcount RMW on-device (no host RMW racing device joins).
- telemetry: `jctr` per arena -> "cow: joins-at-fetch served so far: N" in engine log.
- env: `STRATA_COW_JOIN_HH=<n>` seeds per-range join headroom (0 = joins OFF = exact forkb4 behavior).

Gate table (farm = m25_farm5, batch4, arena 9216, slot-cells 4096):

| arm | joins | C1 | C2 | C3 | C4 decode | C4 wall | farm QC | battery |
|---|---|---|---|---|---|---|---|---|
| forkb4 (M-C1bc, joins off) | 0 | 151.2 | 88.7/81.7 | 64.7/58.6/60.7 | 47.1-49.9 | 18.5 s | all true | green (QC6 97,101,103) |
| forkb4d (uncapped joins) | 10,855 | 151.6 | 69.9/75.6 | 71.4/67.1/17.8 | 59.0/38.1/19.0/59.9 | 23.2 s | C2+ false (streams stalled->truncated) | green |
| forkb4d3 (hh = n_slots/2) | 8,615 | 151.6 | 77.1/71.4 | 53.4/51.9/56.4 | 40.7-43.1 | 19.9 s | all true | green |
| forkb4d_hh256 | ~4.5k | 152.1 | 84.6/61.2 | 58.3/67.2/65.1 | 32.7/36.5/38.2/36.6 | 22.9 s | all true | green |

Laws banked:
1. JOIN IS MECHANICALLY SOUND: joins-at-fetch served 8.6-10.9k shared cold pages across siblings with zero corruption — battery QC6 exact and farm prime-QC green even when joins stall streams (damage is scheduling, never bytes).
2. PIN LAW (why joins lose): a joined page is un-evictable by its owner (clock-sweep-hold is load-bearing: a peer table entry naming a slot means a live gather can read it cross-stream; evicting any named slot is a corruption race, not just policy). Every join therefore steals working set from the fetcher's window. With sibling windows = 1024 slots, sharing damage scales monotonically with the cap: 512 -> 41-43 tok/s, uncapped -> stream death. Joining into small windows is net-negative; the mechanism only pays into ranges with big slack.
3. HEADROOM KNOB: `STRATA_COW_JOIN_HH` default 0 => joins off => canonical stays exactly forkb4. hhead refunds are illegal mid-epoch (fork pins never spent headroom; over-refund decays the cap) — budget only refreshes at reset (8 B conv+hhead store).
4. The cold-prefix arithmetic still holds (22.5k prefix = 4600 pages = 233 MB across 12 layers), but the share must live OUTSIDE any session window: M-C1e = per-lineage SHARED REGION in the arena (dedicated slots, fetched once, page tables point into it, eviction legal only when refcount == 0 i.e. no live namer -> no cross-stream gather can be in flight -> race-free by construction). Sizing fits today's arena: 4608 shared + 2048 main + 3x1024 siblings ~= 9216 slots. That converts the farm's 4x duplicate cold-head streaming into one pass without pinning anyone's window.

Next: commit `cow-m1d` (knob default 0), canonical restore, then M-C1e shared-region design gate.
## 109. M-C1e per-lineage shared region: race-clean and floor-guarded, but farm-negative at expert parity — the cold-head tax is resident capacity, not duplicated fetch traffic; the CoW campaign closes at M-C1c/forkb4

Built the lineage shared region (commit `93f218da`, branch `cow-m0`): arena tail region
`[arena_slots, arena_slots+STRATA_COW_SHARED)` owned by no session window, per-lineage hint
tables (`lin_all`, 16 rolling tabs per arena, block -> region slot, -2 fetching), region-local
clock arrays (`sblk/sst/srf`), a region sweep inside `resolve_kernel` whose candidate predicate
is `slot_use == 0` — the PIN LAW boundary by construction: no live table names the slot, so no
gather can be in flight when it is claimed (`CAS(use, 0, -2)`). Adoption is the M-C1d prev-gated
add + revalidate protocol; publish order is refcount, entry, fence, hint, so a joiner that sees
the hint always finds a pinnable refcount. All three gates battery-green with QC6 exact.

New laws banked the hard way:

- FLOOR LAW: fork siblings generate diverging tails, so blocks at or above the fork floor
  (`P = upto / 4`, complete pages) are NOT byte-identical across the lineage. Sharing them (join,
  splice, region adoption) adopts another session's divergent bytes = corruption. The patch adds
  a per-registry-entry `floor` lowered by every fork and guards EVERY sharing path with it. This
  also closes a latent M-C1d hazard: its join/splice had no floor check and was only saved by the
  farm's uniform fork shape.
- SPLICE LAW (tightened): the admission splice now drops every foreign entry to cold; a table may
  only name slots the holder or the fork-join kernel counted a ref for. Siblings re-adopt region
  pages at resolve in one step. Closes the un-pinned foreign-name eviction race.
- WINDOW FLOOR: a streamed slot window must be >= the resolve block width (1024 slots, i.e.
  STRATA_COW_SLOT_CELLS >= 4096); the engine refuses boot below that (`512 slots is fewer than
  the resolve block (1024)`).

Gate table (farm5 = 22.5k-cell sessions, 432-tok prime answers):

| arm | slots (private+region) | expert slots | C1 | C4 decode | C4 wall | farm QC | battery |
|---|---|---|---|---|---|---|---|
| forkb4 (champion, region off) | 9216+0 | 2136 | 151.2 | 47.1-49.9 | 18.5 s | all true | green |
| e1 region 4608 | 9216+4608 | 1969 | 141.0 | 39.0-42.0 | 21.7 s | all true | green |
| e2 region 5632 | 6656+5632 | 2026 | - | sibling window 512 < 1024: engine refused farm leg (503) | - | - | battery green, farm refused |
| e3 region 3072 at parity | 6144+3072 | 2126 | 147.1 | 40.0-43.1 | 20.3 s | all true | green |

Telemetry: e1 served 4,237 region fetches and ~17k adoptions; e3 3,392 fetches / 14k adoptions —
the mechanism works exactly as designed (one fetch per lineage-page, siblings adopt, zero
corruption, QC prime checks all true, battery 8/8 QC6 exact `97, 101, 103`).

Economics verdict: at expert parity (e3) the region still LOSES to forkb4 (C4 40-43 vs 47-50,
wall 20.3 vs 18.5 s), and paying VRAM for the region (e1) loses more. The engine's own counters
are the smoking gun: 97.7% of block reads hit VRAM and only 16.8 MiB came from host RAM across
the whole farm — duplicated cold-head FETCH TRAFFIC was never the cost. The cost is resident
CAPACITY: forkb4's admission fork already converts prefix duplication into window coverage, and
every region slot competes 1:1 with an expert slot for the same VRAM. You cannot buy capacity
with capacity. Sharing beats duplication only when the shared set fits the region outright; on
12 GB it does not fit alongside windows and experts.

Campaign close: the CoW KV campaign lands at M-C1c — admission fork (forkb4): +473 expert slots
(1663->2136), C4 decode 42.5-45.1 -> 47.1-49.9, C4 wall 23.9 -> 18.5 s, race-clean, battery-green
at every step. M-C1d (pinned join-at-fetch) and M-C1e (lineage shared region) stay committed
opt-in with defaults OFF (`STRATA_COW_JOIN_HH=0`, `STRATA_COW_SHARED=0`), their mechanisms
proven race-clean for any future substrate with real VRAM headroom (bigger cards, or host-tier
parked-session CoW where host RAM dedup is the actual prize — no VRAM competition there).
Canonical restored; `battery_quality_m25` 8/8.
## 110. M-C2 host-tier parked-session dedup LANDS: verify-then-view on the ConversationBuffer segment directory — sibling parks pay owned-tail bytes only (395 MB vs 577 MB per 22.3k sibling); battery+farm green, ladder restore 0/6 fail; commit a53973bf

The last lever of the CoW campaign is the only one that never touches VRAM: parked conversations
are host-RAM ConversationBuffer images (16 MiB segments), and a fleet of sibling subagent
conversations sharing a system-prompt head duplicates that head's K/V once per sibling.

Mechanism (`STRATA_PARK_COW=1`, default 0 = today exactly):
- Donor search is a **longest-common-token-prefix scan** over parked entries — `ConversationCache::best()`
  is whole-conversation-prefix by design and never matches siblings.
- Adopt is **verify-then-view**: `memcmp` the session pool bytes against the donor's segments over the
  shared range BEFORE any mutation; mismatch leaves the buffer untouched and the plain copy runs.
  Honors §103 (heads prefilled under different chunk plans are not byte-identical).
- Views hold the donor's `store` shared_ptr with `cap = 0`: they **bill 0 bytes** — the cache budget
  charges shared bytes once, to the donor. Write visits into borrowed segments fail closed.
  Restore/verify/disk-save read through `visit()` and needed zero changes.
- Views flatten to the original donor store (view-of-view adopts the underlying store) — multi-generation
  sibling chains share one copy.

Pitfall banked (two SIGSEGVs): the **indexer `pooled` pool is device-resident** (and `kv_mode 0` pools
can be too) — CPU `memcmp` on them segfaults; small anchors hid it because pooled rows = at/idx_block = 0.
Adopt is now gated `!on_device(src[i])`; device-resident parts fall back to the copy path.

Gate (parkc4 arm, batch 2, ctx 131k):
- battery QC_DONE green; farm C1-C4 all `qc_primes=True` (C1 162.4, C4 agg 86.8) — zero serving regression.
- park-ladder (6 siblings sharing a 14,360-token head, 22,348 tokens each):
  | metric | COW off | park CoW |
  |---|---|---|
  | per-sibling parked image | 577.4 MB | **395.5 MB owned** (reused_kv_bytes = 181,969,920 = 14,360 tok x 12,672 B — exact) |
  | park wall | 164-192 ms | **58-61 ms** (3x less copy work) |
  | ladder host-RAM delta (6 parks) | ~3.4 GB naive | **646 MiB** |
  | restore after eviction churn | — | **restore_qc_fail = 0/6** |
- 31.5% parked-byte saving per sibling at 64% head share; the prize scales with head share:
  a real agent fleet (one 20k head, hundreds of sessions) converges to head-once + tails.

Economics: unlike M-C1d/e (VRAM: pinned names stole working set; region slots competed 1:1 with
experts), host-tier views have zero VRAM interaction — pure host-RAM and park-CPU win.
Eviction law holds: donor eviction frees only its own bytes; views keep the store alive by refcount,
so shared bytes are never double-freed and never double-billed.

The CoW campaign is now complete across all three tiers:
M-C1c forkb4 (VRAM admission fork, banked champion §107), M-C1d/e (VRAM fetch/region sharing,
closed negative §108/§109), M-C2 (host parked dedup, LANDED). All opt-in, canonical default unchanged.
                                                                                                                                                                                                                                                                                                                                                                   