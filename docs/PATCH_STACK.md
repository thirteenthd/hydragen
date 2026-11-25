# Patch-Stack Manifest — `control-pre-0221` → `v0.2.21`

Phase 0, item **P0.2**. Inventory of every HydraGen fork patch carried on top of upstream Strata `0.1.60` (`38f63bc5`), how the 30 post-§181 engine commits map onto the 5-layer Phase 1 replay stack against **Strata `v0.2.21` (`d46f5f26`, tree `ac94d63d`, engine `0.1.67`)**, and how the snapshot commit `4c39aa5` and pruned experiments are handled.

Baseline: `control-pre-0221` (`654b078`, see [control-pre-0221.md](control-pre-0221.md)) · Exit gates: [PHASE1_GATES.md](PHASE1_GATES.md) · Touchpoint diff & Layer 1 overlap: [UPSTREAM_DELTA.md](UPSTREAM_DELTA.md) · Layout move: [LAYOUT_MIGRATION_0221.md](LAYOUT_MIGRATION_0221.md).

---

## 1. Locked Policy & Sync Law

1. **Never `git merge` upstream.** HydraGen's public history is intentionally unrelated to `Niko1221/Strata` (empty `merge-base`). Sync uses a clean upstream `v0.2.21` tree + linear topical fork-commit stack so future `v0.2.22+` bumps can use `git rebase --onto <newtag> <oldtag> fork/stack`.
2. **Layout:** Adopt upstream's `0.2.16` root layout (`engine/`, `terminal/`, `tools/`), keeping upstream `docs/` (plus `AGENTS.md` / `CONTRIBUTING.md`) confined under `engine/docs/` to prevent root `docs/` collisions. Keep `app/` (`engine/apps/`, 1,018 tracked files at `control-pre-0221` / 1,282 at `v0.2.21`) **vendored and unbuilt**. Keep `.github/` dropped (`ebea5fe`).
3. **`STRATA_WINDOW_MAX_T`:** Build both Phase 1 arms from the same rebased source — **8-row** (`auto` / parity control vs `control-pre-0221`) and **16-row** (`-DSTRATA_WINDOW_MAX_T=16` + `CUDA_MODULE_LOADING=LAZY`).
4. **`CACHE_FUND` (asymmetric cut):** **Prune Stage C** (`cow_fund`, `cow_retake`, `g_cow_funder`); **keep Stage C.5** (`generate.cpp:8571-8575`) opt-in and **unbundle** `qsa_cow_arena_set_budget` (`:8575`) so `CACHE_FUND=1` does not zero the grow budget without a funder.
5. **`COW_SHARED` vs `COW_JOIN_HH`:** **Prune `STRATA_COW_SHARED` / `g_cow_shared`** (Depth 1 in Phase 1: host knob, setter, tail allocation; map fields pinned null/0 per `PHASE1_GATES.md` P0.3). **Keep `CowPeer` / `STRATA_COW_JOIN_HH`** (`JOIN_HH=8` in the control posture).
6. **§235 status:** `5074249` ("§235: adoption probe") **touched zero engine files**; the tail-slot pin fix is unwritten and remains a hard prerequisite on Phase 4, excluded from the Phase 1 replay stack.

---

## 2. Phase 1 Replay Stack (5 Topical Layers)

All fork content (both the `38f63bc5..f660d2f` §181 foundation and the 30 `f660d2f..654b078` engine commits below) replays onto clean `v0.2.21` in five dependency-ordered commits, with `ctest` run after each layer:

| Layer | Name | Scope & Modules |
|---|---|---|
| **L0** | Identity, Env & Build | `hg_env.hpp` (`HG_*` → `STRATA_*`), `engine/cmake/strata_gpu_engine.cmake` (`OUTPUT_NAME "hydragen"`), `tools/packaging/build_cuda_windows.py` (`hydragen.exe` staging), root `.gitignore` (`/models/` anchor), Windows runner helpers (`run_*.bat`) |
| **L1** | Park-Disk Tier & Host CoW | `conversation_park_disk.{hpp,cpp}` (v1/v2/v3 BCS, S5a `pd_pread_direct` + Windows stub, S5c-v3 delta restore, S5d histogram), `park_disk_boot_test.cpp`, `HG_PARK_COW` + `ParkAnchorGuard` stack (`conversation_cache.hpp`, `conversation_snapshot.{hpp,cpp}`, `conversation_buffer.hpp` incl. M-D2 `has_borrowed()`), `HG_RESUME_DEEPEST`, S5d histogram (`expert_source.{hpp,cpp}`, `generate.cpp`), `conversation_file.{hpp,cpp}`, `conversation_cache_test.cpp` |
| **L2** | Shared RoPE, Elastic VMM Arena & FP16 Indexer | `session.{hpp,cpp}` (`with_rope` / `share_rope` / `owns_rope`), S7-3 `idx_pool_f16` (`qsa.hpp`, `qsa.cu`, `qsa_select.cu`, `native_qsa_indexer.cu`, `conversation_state.cpp`, `conversation_snapshot.cpp`, `prefill.cpp`, `layer.cpp`), S8-v2 + dyn-conc VMM `CowArena` (`layer.{hpp,cpp}`, `kv_stream.{hpp,cu}`, `verify.{hpp,cpp}`, `generate.cpp`: `elastic`, `floor`, `trim`, `bank`, `cow_elastic_attach`, `pools_budget.shrink` + host-RAM admit, `COW_JOIN_HH`, `COW_FORK_HEADROOM`, §234 unmapped tail, Stage C.5 unbundled; **no** `COW_SHARED` / Stage C `cow_fund`) |
| **L3** | Auto Batch-$N$ & Serve Admission/Backoff | `include/strata/program/decode_governor.hpp`, `tests/program/decode_governor_test.cpp`, `engine/CMakeLists.txt` test target, pre-prefill `BGEN` check + `--min-stream-tps` (`generate.cpp`), `terminal/serve/server.py` (`SLOT_MEM_BACKOFF_S` / `_mem_backoff`, `decode_governor_args`, `configure_decode_governor_env`, `STRATA_YIELD_*` unread ranking) |
| **L4** | Dashboard & Telemetry | `terminal/serve/server.py` (`ConvCacheLog`, `/slots`, `/metrics park_disk`), `terminal/serve/web/{app.js,app.css,index.html}` (Concurrency States panel, split RAM/Park-disk cards, zero-state bar fix) |

---

## 3. Complete Manifest — 30 Engine Commits (`f660d2f..654b078`)

Oldest-first from `f660d2f` (§181 `0.1.60` sync snapshot). Commits between `f660d2f` and `654b078` that touched only `docs/`, `bench/`, or `tools/` outside `engine/` are omitted (all 22 non-engine commits remain in `main` history).

| # | Commit | Topic | Engine Files Touched | Test Target | Verdict & Layer |
|---|---|---|---|---|---|
| 1 | `643471d` | §183 S5a `O_DIRECT` park-restore blob reads (Linux opt-in, Windows stub) | `src/core/conversation_park_disk.cpp` | `park_disk_boot_test` | **keep** → L1 |
| 2 | `ec10027` | §184 `/slots` `state`/`held_tokens`/`idle_s` telemetry | `serve/server.py` | serve smoke | **re-port** → L4 |
| 3 | `59811de` | §185 park-disk bar zero state (grey track when empty) | `serve/web/app.js` | dashboard UI | **re-port** → L4 |
| 4 | `0ccaab9` | §186 park-disk container v3 (BCS content-addressed chunks + salted lineage) | `conversation_file.{hpp,cpp}`, `conversation_park_disk.cpp`, `park_disk_boot_test.cpp` | `park_disk_boot_test` (41 checks) | **keep** → L1 |
| 5 | `a5f170e` | serve #656b `BYIELD` & short-first UNREAD-token ranking | `serve/server.py` | serve unit/admit | **re-port** → L3 (note: `STRATA_ADMIT_SHORT_FIRST` already in `v0.2.21`; port UNREAD delta) |
| 6 | `044f170` | serve #656c `STRATA_YIELD_AFTER_S` wait-age-gated yield | `serve/server.py` | serve unit/admit | **re-port** → L3 |
| 7 | `3b1422e` | S5c audit + `v3lin` boot test coverage | `src/core/park_disk_boot_test.cpp` | `park_disk_boot_test` (51 checks) | **keep** → L1 |
| 8 | `46f194c` | core §191 `ConversationCache` guard stack (`ParkAnchorGuard`) | `conversation_cache.hpp`, `conversation_cache_test.cpp`, `generate.cpp` | `conversation_cache_test` | **keep** → L1 |
| 9 | `7e9f12c` | engine §192 Windows stub for `pd_pread_direct` | `src/core/conversation_park_disk.cpp` | Windows CUDA build + `park_disk_boot_test` | **keep** → L1 |
| 10 | `7e611ef` | engine §194 S5c-v3 salt-to-donor delta restore | `conversation_park_disk.{hpp,cpp}`, `park_disk_boot_test.cpp`, `generate.cpp` | `park_disk_boot_test` (56 checks) | **keep** → L1 |
| 11 | `7675927` | S5d park manifest expert-hit histogram + deep-resume pre-stage | `conversation_cache.hpp`, `conversation_park_disk.{hpp,cpp}`, `expert_source.{hpp,cpp}`, `park_disk_boot_test.cpp`, `generate.cpp` | `park_disk_boot_test` | **keep** → L1 |
| 12 | `ecbb690` | S4-v2 idle conversation detach from batch slots | `src/program/generate.cpp` | E2E idle detach | **keep** → L2 |
| 13 | `093b121` | S5d-v2 prewarm seeds adaptive tier usage counts | `src/program/generate.cpp` | E2E deep-resume | **keep** → L1 |
| 14 | `5f81be1` | S7-3 FP16 pooled indexer-key table (`STRATA_IDX_POOL_F16`) | `qsa.hpp`, `conversation_snapshot.cpp`, `conversation_state.cpp`, `layer.cpp`, `native_qsa_indexer.cu`, `qsa.cu`, `qsa_select.cu`, `prefill.cpp`, `generate.cpp` | `qsa_parity`, `qsa_topk_parity` | **keep** → L2 (verify against `v0.2.21` `qsa_select.cu` churn) |
| 15 | `ffbb52d` | KV-GROW detached slot arena return + re-carve | `verify.{hpp,cpp}`, `generate.cpp` | G4 / detach-recarve | **keep** → L2 |
| 16 | `884f351` | §202 silent batch-window death `ERR` backfills | `src/program/generate.cpp` | E2E error logs | **keep** → L2 |
| 17 | `a621b4d` | §203 packaging stages fork's `hydragen.exe` | `packaging/build_cuda_windows.py` | Windows packaging | **keep** → L0 (`tools/packaging/build_cuda_windows.py`) |
| 18 | `068a7cf` | S8-v2 first-fit arena allocator + zombie reclaim + boot clamp | `layer.{hpp,cpp}`, `session.cpp`, `generate.cpp` | G5 + detach loop | **keep** → L2 |
| 19 | `25a7d1a` | §207 restore `mtmd/models/` swallowed by `.gitignore` | `third_party/strata-llama/tools/mtmd/models/*` (45 files) | `git ls-files` G6 | **superseded** by `/models/` anchor (P0.5) |
| 20 | `29ae08c` | §208 restore `src/models/` swallowed by `.gitignore` | `third_party/strata-llama/src/models/*` (156 files) | `git ls-files` G6 | **superseded** by `/models/` anchor (P0.5) |
| 21 | `ebea5fe` | Drop upstream `.github/` surface (23 files) | `.github/*` | `git ls-files` G6 | **keep dropped** (Decision #5) |
| 22 | `2269997` | M-C11-win Windows runner helpers | `run_link_once.bat`, `run_parktest_once.bat`, `run_parkv2_cuda_build.bat`, `run_parkv2_tests.bat` | Windows build/test | **keep** → L0 (fix `strata.exe` → `hydragen.exe` in `run_parkv2_cuda_build.bat`) |
| 23 | `4c39aa5` | dyn-conc engine snapshot (+727/−31 across 5 engine files) | `conversation_buffer.hpp`, `layer.{hpp,cpp}`, `verify.cpp`, `generate.cpp` | G1/G3/G5 | **split (4-way)** — see §4 below |
| 24 | `0a3d105` | M-D2 `has_borrowed()` guard + drop-reuse-and-copy-fresh fallback | `conversation_buffer.hpp`, `conversation_snapshot.cpp`, `generate.cpp` | G4/G5 re-park | **keep** → L1 |
| 25 | `6270e23` | §219 P2-C.5 reservation-collateral (`CACHE_FUND=1` boot unlock) | `src/program/generate.cpp` | G8 (`CACHE_FUND` A/B) | **split / unbundle** → L2 (keep Stage C.5 opt-in, unbundle `:8575` grow budget) |
| 26 | `c172460` | `STRATA_COW_DEBUG` per-layer arena event log | `src/core/layer.cpp` | log check | **keep opt-in** → L2 (note: live ms01 posture sets `=1`) |
| 27 | `7cb7f1e` | §222 CoW fork fixes (`state_zero` fork-pin release, `STRATA_COW_FORK_HEADROOM` 512, lineage floor) | `layer.cpp`, `kv_stream.cu` | G5 (`probe222x.py`) | **keep** → L2 |
| 28 | `d25af27` | §224-225 identity arms rejected + fork-carrier scoreboard / hint gate | `layer.{hpp,cpp}`, `kv_stream.cu`, `generate.cpp` | G5 | **prune** closed-negative `arm-227` scaffolding; keep structural scoreboard/gate fixes → L2 |
| 29 | `feed840` | §234 VMM unmapped tail (Slot-0 aliasing fix) + `lin` wiring; reverts `arm-227` kill | `layer.cpp`, `kv_stream.cu` | G5 (`4/4+4/4` seeds) | **keep** → L2 (under `COW_SHARED` Depth 1 prune) |
| 30 | `a417b8d` | P2-D1 full-auto batch-$N$: `share_rope`/`with_rope`/`owns_rope`, elastic attach + `conversation_memory_admit`, `DecodeGovernor`, `server.py` 30s backoff + `min_stream_tps` | `CMakeLists.txt`, `session.{hpp,cpp}`, `decode_governor.hpp`, `decode_governor_test.cpp`, `generate.cpp`, `serve/server.py` | `decode_governor_test` + G4/G7 | **keep** → L2 (`share_rope`, `cow_elastic_attach`, host-RAM admit) + L3 (`DecodeGovernor`, `server.py` backoff) |

### Explicit Non-Replayed Items
- **`5074249` (§235 adoption probe):** Zero engine files modified; tail-slot pin fix is unwritten and remains a Phase 4 hard prerequisite.
- **`engine/mc3_patch.py` (pre-`f660d2f` artifact):** Obsolete §181-era text patcher superseded by in-tree CMake registration; **prune**.

---

## 4. Four-Way Decomposition of Snapshot Commit `4c39aa5`

`4c39aa5` ("dyn-conc engine snapshot", 7 files total, 5 engine files: `+727 / -31`) bundled five experiments from the private `dynconc-arena` line and cannot be replayed monolithically. During Phase 1 it decomposes cleanly into four topical hunks inside **L2**:

| Split | Content & Origin Commits | Files | Verdict |
|---|---|---|---|
| **Split A** | VMM-backed `CowArena` (`vmm_reserve`/`map`/`unmap`), `STRATA_COW_ARENA_FLOOR` boot floor, mapped-capacity admission (`qsa_cow_arena_mapped`/`demand`) — from `71f52ff2` | `layer.{hpp,cpp}`, `generate.cpp` | **keep** (L2) |
| **Split B** | Elastic arena windows (`STRATA_COW_ARENA_ELASTIC`, `_BANK_MIB`, `_MIN_FREE_MIB`), detach chunks parked in bank + `trim()`, reservation-as-spendable-claim (`qsa_cow_arena_set_budget` / `give_back`), uncarved slot `max_cells == 0` & `drop_batch_graphs()` — from `51c0ad3c`+`0f467b4c`+`3665a350` | `layer.{hpp,cpp}`, `verify.cpp`, `generate.cpp` | **keep** (L2) |
| **Split C** | Cache-funded arena admission (`STRATA_COW_CACHE_FUND=1` Stage C): `cow_fund` (`generate.cpp:16063`), `cow_retake` (`:16166`), `set_funder` wiring (`:16222-16223`), `cow_retake()` call (`:16256`), `g_cow_funder` (`layer.cpp:1184/1195/1565-1568`) | `layer.{hpp,cpp}`, `generate.cpp` | **prune** (Decision #4; §218/§219 funder never fired) |
| **Split D** | Follow-through from `29f10bf4` (`ARENA_CONC` demand sizing + S8-v2 allocator/clamp follow-through) + `conversation_buffer.hpp` arena helper | `conversation_buffer.hpp`, `layer.cpp`, `generate.cpp` | **keep** (L2) |

---

## 5. Prune & Unbundle Summary (Pre-Replay Checklist)

| Item | Exact Action in Phase 1 | Gate |
|---|---|---|
| `STRATA_COW_SHARED` / `g_cow_shared` | **Depth 1 prune** (delete env read `generate.cpp:4691`, setter `layer.hpp:341`, `g_cow_shared` & tail-alloc `layer.cpp:837,839,890,1416,1441-1471,1486,1658-1662`; pin `st.map` fields null/0) | `PHASE1_GATES.md` P0.3 separability test (`COW_JOIN_HH=8` joins > 0, needle 3/3) |
| `CowPeer` / `STRATA_COW_JOIN_HH` | **Keep** (`JOIN_HH=8` in mech posture; gated on arena + `slot_use`, independent of `g_cow_shared`) | G5 + P0.3 |
| Stage C `cow_fund` / `cow_retake` | **Prune** all Stage C funder machinery from `generate.cpp` and `layer.{hpp,cpp}` | G8 |
| Stage C.5 `generate.cpp:8571-8575` | **Keep opt-in + unbundle:** keep boot reservation collateral (`+0` for `cow_win` when `CACHE_FUND=1`), change `:8575` so `qsa_cow_arena_set_budget` stays funded (`cow_reserve`) instead of zeroing without a funder | G8 (`CACHE_FUND=0` matches control; `CACHE_FUND=1` unlocks +395 slots & grows succeed) |
| `mc3_patch.py` | **Prune** dead §181 script | G6 |
| `STRATA_COW_DEBUG` | **Keep opt-in** (disabled by default; live ms01 mech posture sets `=1`) | Control log parity |
