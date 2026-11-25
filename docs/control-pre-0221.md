# control-pre-0221 — the frozen baseline before the v0.2.21 rebase

Phase 0, item P0.1. Captured 2026-11-25T02:10Z. This file is the reference that
Phase 1's parity arms are measured against. The annotated tag
`control-pre-0221` points at the **source** baseline; the ms01 binaries below are
the **runtime** baseline and are not in version control, so both are recorded.

## 1. Source baseline

| item | value |
|---|---|
| tag | `control-pre-0221` → `654b078097aa4c00229ef708ca67119a2e5dbb51` |
| commit | `654b078` "§236: P2-D1 arms on ms01 …" (committed 2026-11-24T17:01:40-07:00) |
| branches at that commit | `main` == `m-d-conc` (P0.1's merge was already done) |
| pushed | `origin/main` = `https://github.com/thirteenthd/hydragen.git` |
| vendored engine | upstream Strata `38f63bc5` (engine **0.1.60**) + fork overlay, landed at `f660d2f` (§181) |
| tree integrity | `git ls-files engine` = **7,332**; `git ls-files` = **7,403** |
| fork overlay since §181 | 52 commits, **30 touching `engine/`** (P0.2's patch-stack manifest) |

## 2. Rebase base (P0.0)

Fetched into `C:/DevOps/hydragen-sync180` (`upstream` = `https://github.com/Niko1221/Strata`);
the tag and its objects were **not present locally before P0.0** (that clone's
`upstream/main` was at `c77572806` = v0.2.15). `v0.2.16`–`v0.2.20` arrived with it.

| item | value |
|---|---|
| tag object | `0f8a6b78c5e37ef84bf8ad8acecf60a44cf0618a` (annotated) |
| commit | `d46f5f268f8ed797e83a5c63a69bffbd7a581cf6` |
| **tree** | **`ac94d63dca450fd38be5c65bf67337c450566428`** |
| engine version | `project(strata VERSION 0.1.67 LANGUAGES CXX)` |
| commits `38f63bc5..v0.2.21` | **1,748** |
| root layout | `app/ terminal/ tools/ docs/ engine/` (restructured at 0.2.16) |

`v0.2.21.1` (`2af30cca`) bumps Control to 0.1.68 only; engine bytes identical — a
free later bump, not part of this base.

## 3. Runtime baseline — ms01 (RTX 3080 Ti, 12,288 MiB, 60 GiB RAM, CUDA 13.1, clock = UTC)

Serving at capture: `serve/server.py --engine strata --config strata-mechdbg.json
--port 8181` → `build-s180/hydragen --serve … --batch 4`, up since
**2026-11-25T00:01:08Z** (the §236 battery's final `mechdbg_restart.sh`).

| binary | md5 | bytes | mtime (UTC) | built from |
|---|---|---|---|---|
| `/home/thirteenthd/hydragen-sync180/build-s180/hydragen` | `e5ada445f1d5f1a3967343c4f9b3b30f` | 52,225,184 | 2026-11-24 18:50:40 | ms01 clone `hydragen-sync180` @ `3665a350` "dyn-conc P2 stage B" |
| `/home/thirteenthd/hydragen-p2d1/build-p2d1/hydragen` | `6d7a1714c4cd6c83fc98c51b3504fdf7` | 52,234,352 | 2026-11-24 21:46:58 | `/home/thirteenthd/hydragen-p2d1` — **not a git repository** (working tree) |

**Provenance gap, recorded deliberately.** `3665a350` does not exist in the public
repo (`git cat-file -t 3665a350` → "Not a valid object name"): ms01's experimental
clone carries its own private history, and its content is what the public tree
bundled into `4c39aa5`. So:

- the binary serving :8181 is **pre-P2-D1** (no `a417b8d` content) and is **not a
  build of `654b078`**;
- `build-p2d1` has no VCS provenance at all — identity is md5 + mtime only.

Consequence for Phase 1: a true parity control must be **built from
`control-pre-0221`**, not re-used from `build-s180`. §236's A0-vs-A1 parity remains
valid as a same-shape binary comparison; it is not a comparison against the tag.

### Build configuration (both build dirs, from `CMakeCache.txt`)

```
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=86-real
STRATA_WINDOW_MAX_T:STRING=auto          # property STRINGS = auto;8;12;16
```

`auto` resolves to **8 rows for CUDA** (`engine/CMakeLists.txt:127-146`: HIP 16,
CUDA 8, because CUDA loads every kernel at start under `CUDA_MODULE_LOADING=EAGER`
and "the wide kernels would take VRAM from the expert cache of every card"). The
configure-time line is `Strata: verify / batch windows hold up to 8 rows`; the
runtime line in the mech log reads
`strata verify: batch windows of up to 4 sequences (layers [0, 48)), one-row windows keep their rows`.
`include/strata/kernels/window_limits.hpp` pins `kVerifyNarrowT = 8` and
`kWindowMaxT = STRATA_VERIFY_MAX_T`, with the rule that "a window of up to
kVerifyNarrowT rows runs EXACTLY the code it ran at 8" — which is what makes the
8-row arm a byte-identity control (§177) and the 16-row arm a separate arm.

At `--batch 8` an 8-row build has `spare = win_cap − nact = 0`: no draft rows, so
speculation is dead and `--batch 8 --batch-mtp` clamps to 4 slots. This is why
decision 3 builds both arms.

## 4. Mech posture (`/home/thirteenthd/hydragen-experimental/strata-mechdbg.json`, 2,708 B, mtime 2026-11-24 20:43)

`exe` = `build-s180/hydragen`, `port` 8181, `parallel` 4, `gpu` 0, `host` 0.0.0.0,
`cwd` `~/hydragen-experimental/strata`, `slot_save_path` `~/hydragen-experimental/slot-saves`,
`model_name` `qwen3.8-flash-next-iq3_xxs-m25`.

Env block, verbatim (20 entries):

```
STRATA_PREFILL_CPU_SHARE=auto      STRATA_PF_FUSED=1
STRATA_PREFILL_EQUAL=1             STRATA_ROUTE_PRIOR=0.8
STRATA_ROUTE_TAIL_SKIP=0           STRATA_PARK_COW=1
STRATA_PARK_DISK=/home/thirteenthd/hydragen-experimental/park-disk
STRATA_PARK_DISK_V2=1              STRATA_COW_ARENA=1
STRATA_COW_ARENA_SLOTS=18432       STRATA_COW_SLOT_CELLS=8192
STRATA_COW_FORK=1                  STRATA_COW_JOIN_HH=8
STRATA_SLOT_DETACH_MS=15000        STRATA_SLOT_KV_GROW=1
STRATA_BATCH_YIELD_CHUNKS=0        STRATA_COW_ARENA_ELASTIC=1
STRATA_COW_ARENA_FLOOR=4096        STRATA_COW_DEBUG=1
STRATA_COW_SHARED=0
```

**`STRATA_COW_CACHE_FUND` is absent** → `g_cow_cache_fund = false`
(`generate.cpp:306`; set only at `:4679-4680` from `=='1'`). The control posture
runs **CACHE_FUND=0**, so decision 4's Stage-C prune is behaviour-preserving on
this baseline.

Args, verbatim:

```
--pack ~/.models_backup/Strata-data/packs/iq3_xxs
--native ~/.models_backup/Strata-data/models/IQ3_XXS/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
--ple-gguf ~/.models_backup/Strata-data/models/IQ3_XXS/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
--expert-profile ~/hydragen-experimental/expert-profile-learned-7292092bd2.bin
--prefill auto:16384
--spec 4 --coupled-draft --spec-min-p 0.45
--max-context 131072
--kv int8
--vram-reserve-mib 512
--pcie-frac 0.5
--adapt-every 1 --adapt-swaps 400 --adapt-decay 0.97 --adapt-async 1
--conversation-cache-slots 8
--prompt-cache-tail
--expert-profile-save ~/hydragen-experimental/expert-profile-learned-7292092bd2.bin
--session-save-reclaim
--conversation-cache-mib 16384
--conversation-cache-min-free-mib 2048
--kv-resident 8192
--batch 4
```

## 5. Live state at capture (not frozen by the tag)

- A stale `p2d1_arms.sh` wrapper is still alive on ms01 (PIDs 4177197/4177198,
  elapsed 9,364 s at capture, started ≈2026-11-24T21:33Z) although the battery
  finished — last arm log `logs/p2d1/A4-server.log` 2026-11-25 00:00, mech
  restarted 00:01:08. **Not killed** (no side effects without a decision); clean
  it up before the next battery, and `pgrep -af hydragen` before any rerun.
- Host RAM at capture: 60 GiB total, 49 used, **10 available** against
  `--conversation-cache-mib 16384` + `min-free 2048`. Thin; relevant to any
  admit-floor arm.

## 6. What the tag does *not* freeze

ms01's binaries, `strata-mechdbg.json`, the park-disk tier
(`~/hydragen-experimental/park-disk`), the learned expert profile
(`expert-profile-learned-7292092bd2.bin`) and the slot-save tree all live outside
the repo. Any Phase 1 claim of "same posture" must re-state them from this file.
