# Upstream delta — HydraGen vs Strata `v0.2.21` (`0.1.67`)

> **Status (Phase 0, P0.4 audit):** HydraGen's vendored engine is currently at upstream Strata **`0.1.60` (`38f63bc5`, app `v0.2.14`)**, frozen at `control-pre-0221` (`654b078`). This document classifies the complete fork overlay against `38f63bc5` and audits every touchpoint + Layer 1 overlap against the Phase 1 rebase target **`v0.2.21` (`d46f5f26`, tree `ac94d63d`, engine `0.1.67`)**.
>
> Doc-drift cleanup note: `README.md:84,173` and `NOTICE` still say "v0.2.13 baseline" (stale since §181) and will be updated at the Phase 1 `v0.0.1` cut.

Per-arm evidence lives in [CHANGES.md](CHANGES.md), milestone markers in [MILESTONES.md](MILESTONES.md), the frozen baseline in [control-pre-0221.md](control-pre-0221.md), the commit replay manifest in [PATCH_STACK.md](PATCH_STACK.md), and the Phase 1 pass rules in [PHASE1_GATES.md](PHASE1_GATES.md).

---

## 1. Hunk Classification Legend (for Phase 1 Replay)

- **(a) Still needed:** Fork-owned behavior with no upstream `v0.2.21` equivalent (or upstream's equivalent is inert on Windows / incompatible with `--batch` + streaming KV); must be re-ported onto `v0.2.21`.
- **(b) Upstream `v0.2.21` covers:** Upstream added an equivalent mechanism or default; adopt upstream or limit the fork delta to the remaining gap.
- **(c) Dead / pruned:** Retired or superseded experiment removed during Phase 1 per locked Phase 0 decisions (`COW_SHARED` Depth 1, Stage C `cow_fund`/`cow_retake`, `mc3_patch.py`, `arm-227` scaffolding).

---

## 2. Touchpoint Collision Audit (`38f63bc5` → `HEAD` vs `38f63bc5` → `v0.2.21`)

Exhaustive `git` blob comparison across `engine/` found **52 modified files**, **11 fork-only files**, and **23 deleted `.github/` files** (`ebea5fe`). Of the 52 modified files, **21 are CRLF/mode-only or minor `0.1.60`-sync test-fixture deltas**; the table below audits every substantive touchpoint and fork-only file:

| Current Fork Path (`engine/...`) | `v0.2.21` Target Path | Fork Delta (`vs 38f63bc5`) | Upstream Churn (`0160→0221`) | Class & Phase 1 Plan |
|---|---|---|---|---|
| `src/program/generate.cpp` | `engine/src/program/generate.cpp` | **+1,033 / −41** (94 hunks) | +3,206 / −518 | **(a)** `ParkDisk` wiring, S5d hist, S4-v2 detach, Stage C.5 unbundled, `DecodeGovernor`, `BGEN` admit/attach, `share_rope`, `COW_ARENA`/`JOIN_HH`; **(b)** upstream `--kv-grow` / `low_vram_rules` / `conversation_spill` coexist; **(c)** prune `STRATA_COW_SHARED` env read (`:4691`) & Stage C `cow_fund`/`cow_retake` (`:16039–16256`) |
| `src/core/layer.cpp` | `engine/src/core/layer.cpp` | **+845 / −14** (18 hunks) | **+9 / −0** | **(a)** `CowArena` VMM/elastic/bank/floor, `CowPeer`/`JOIN_HH`, S7-3 `idx_pool_f16`, S8-v2 first-fit + zombie reclaim, §222/`state_zero` + §234 unmapped tail; **(c)** prune `g_cow_shared` tail allocation (`:837,839,890,1416,1441–1471,1486,1658–1662`) & `g_cow_funder` (`:1184,1195,1565–1568`) |
| `include/strata/core/layer.hpp` | `engine/include/strata/core/layer.hpp` | +79 / −1 | +8 / −0 | **(a)** arena/elastic/bank/join setters; **(c)** prune `qsa_set_cow_shared` (`:341`) & funder setter |
| `src/kernels/cuda/kv_stream.cu` | `engine/src/kernels/cuda/kv_stream.cu` | +234 / −17 (15 hunks) | +101 / −3 | **(a)** `slot_base` indirection, `slot_use` refcounts, `cow_fork_join_kernel`, §234 `n_shared > 0 && lin_all != nullptr` gate (keep Depth 1 kernel fields untouched in Phase 1) |
| `include/strata/kernels/kv_stream.hpp` | `engine/include/strata/kernels/kv_stream.hpp` | +56 / −1 | +18 / −0 | **(a)** `CowPeer`, `slot_base`, `slot_use`, `join_ctr` |
| `src/core/conversation_snapshot.cpp` | `engine/src/core/conversation_snapshot.cpp` | +139 / −8 | +477 / −0 | **(a)** `HG_PARK_COW` anchor verify-then-view (`conversation_kv_share_views`/`_prefix`), `conversation_park_join_for`, S7-3 f16 pool conversion (verify against upstream `STRATA_PARK_ASYNC`/layer-split additions) |
| `include/strata/core/conversation_snapshot.hpp` | `engine/include/strata/core/conversation_snapshot.hpp` | +17 / −2 | — | **(a)** `anchor` parameter on `conversation_kv_save`, `share_views`/`share_prefix` declarations |
| `include/strata/core/conversation_cache.hpp` | `engine/include/strata/core/conversation_cache.hpp` | +109 / −14 | +127 / −7 | **(a)** `view_donor`, `s5d_hist`, `spill_hook`, `common_best`, `ParkAnchorGuard` stack (keep separate from upstream `park_pool_policy.hpp`) |
| `include/strata/core/conversation_buffer.hpp` | `engine/include/strata/core/conversation_buffer.hpp` | +128 / −11 | +55 / −2 | **(a)** segment views (`adopt_shared`) + M-D2 `has_borrowed()` guard |
| `src/core/conversation_file.cpp` / `.hpp` | `engine/src/core/conversation_file.{cpp,hpp}` | +123 / −6, +25 / −3 | +58 / −1, +10 / −0 | **(a)** BCS v3 container helpers + Windows/POSIX lock/salt support |
| `src/core/conversation_state.cpp` | `engine/src/core/conversation_state.cpp` | +21 / −4 | +212 / −28 | **(a)** S7-3 `idx_pool_f16` snapshot compatibility flag (`+32`) |
| `src/core/session.cpp` / `.hpp` | `engine/src/core/session.{cpp,hpp}` | +18 / −8, +3 / −2 | **identical (0)** | **(a)** `with_rope` / `share_rope` / `owns_rope` — zero upstream conflict |
| `src/kernels/cuda/qsa.cu` / `include/.../qsa.hpp` | `engine/src/kernels/cuda/qsa.cu` / `engine/include/.../qsa.hpp` | +71 / −12, +26 / −0 | **identical (0)** | **(a)** S7-3 `idx_pool_f16` — zero upstream conflict |
| `src/kernels/cuda/native_qsa_indexer.cu` / `qsa_select.cu` | `engine/src/kernels/cuda/...` | +100/−49, +55/−22 | +11/−4, +129/−18 | **(a)** S7-3 `idx_pool_f16` (`qsa_select.cu` needs careful 3-way merge) |
| `src/core/verify.cpp` / `.hpp` | `engine/src/core/verify.{cpp,hpp}` | +14 / −0, +5 / −0 | +874 / −151, +35 / −3 | **(a)** only 2 small hunks (`max_cells == 0` unattached slot guard, `drop_batch_graphs()`); neither exists in `v0.2.21` |
| `src/core/expert_source.cpp` / `.hpp` | `engine/src/core/expert_source.{cpp,hpp}` | +10 / −0, +6 / −0 | +601 / −84, +45 / −9 | **(a)** S5d per-slot `hist_ptrs`/`hist_n` recording (+10 lines) |
| `src/prefill/prefill.cpp` | `engine/src/prefill/prefill.cpp` | +2 / −1 | +953 / −210 | **(a)** S7-3 `s.idx_pool_f16 == 0` debug check guard (+2 lines) |
| `src/core/conversation_cache_test.cpp` | `engine/src/core/conversation_cache_test.cpp` | +23 / −0 | +492 / −6 | **(a)** §191 guard-stack nesting test |
| `cmake/strata_gpu_engine.cmake` | `engine/cmake/strata_gpu_engine.cmake` | +6 / −1 | +79 / −5 | **(a)** **critical:** `OUTPUT_NAME "hydragen"` + `conversation_park_disk.cpp` + `park_disk_boot_test` target |
| `CMakeLists.txt` | `engine/CMakeLists.txt` | +25 / −0 | +273 / −57 | **(a)** `decode_governor_test` registration (+ check `STRATA_WINDOW_MAX_T` lines) |
| `serve/server.py` | `terminal/serve/server.py` | +284 / −74 | +2,125 / −2,864 | **(b)** `STRATA_ADMIT_SHORT_FIRST` already in `v0.2.21` (`:450,454,1808,1931`); **(a)** re-port UNREAD ranking, `STRATA_YIELD_*`, `SLOT_MEM_BACKOFF_S` / `_mem_backoff`, `DecodeGovernor` arg/env forwarding, `ConvCacheLog`, `/slots`, `/metrics park_disk`. **G9 watch:** `:4460` still returns 501 for `slot_action` with `--batch` |
| `serve/web/{app.js,app.css,index.html}` | `terminal/serve/web/{app.js,app.css,index.html}` | +223/−5, +265/−0, +29/−18 | +324 / −51 (`app.js`) | **(a)** re-port Concurrency States panel, split RAM/Park-disk cards, zero-state bar fix |
| `packaging/build_cuda_windows.py` | `tools/packaging/build_cuda_windows.py` | +5 / −2 | +14 / −13 | **(b)** upstream updated `ENGINE_SRC = REPO / "engine"` & `PACKAGING`; **(a)** re-apply `["hydragen.exe", "strata.exe"]` staging hunk |
| *Fork-only (11 files)* | `engine/...` | new | — | **(a)** `hg_env.hpp` (27), `conversation_park_disk.{hpp,cpp}` (246+1,637), `park_disk_boot_test.cpp` (579), `decode_governor.hpp` (258), `decode_governor_test.cpp` (262), `run_*.bat` (4 files); **(c)** prune `mc3_patch.py` |

> **Other upstream `v0.2.21` headers to adopt rather than reinvent (Layer 2+):** `low_vram_rules.hpp` (≤13 GiB reserve 550 MiB, Windows WDDM floor/late-alloc rules, opt-in `STRATA_TINY_CACHE=1`), `spill_gate.hpp`, `ssd_stream_scheduler.hpp`, `prompt_checkpoint_grid.hpp`. Note: upstream `--kv-grow` (`kv_grow_rules.hpp`) requires `o.kv_resident <= 0 && o.batch == 0` (`generate.cpp:7721`), so it does **not** cover our batched KV-streaming elastic arena.

---

## 3. Layer 1 Overlap Matrix — Fork Park-Disk / CoW vs Upstream `v0.2.21` Spill Tier

`v0.2.21` ships `--conversation-spill-dir / -mib / --conversation-spill-persist` (`conversation_spill.hpp`), `park_pool_policy.hpp`, `STRATA_CKPT_POOL` / `_READY`, `STRATA_PARK_ASYNC`, and `STRATA_PARK_SEGPOOL_MIB`. They solve only part of HydraGen's Layer 1 chain and have a critical **Windows parity gap**:

| Capability | HydraGen Layer 1 (`ParkDisk` + `HG_PARK_COW` + `HG_RESUME_DEEPEST`) | Upstream `v0.2.21` (`conversation_spill.hpp` + `park_pool_policy.hpp`) | Verdict |
|---|---|---|---|
| **Cross-boot persistent disk tier** | `HG_PARK_DISK` + v2/v3 BCS index rebuild across restarts | `--conversation-spill-persist` (`fp-<16 hex>` fingerprint folder); ephemeral PID-nonce folders cleaned on next boot | **(b) partial overlap** — upstream still uses full per-session files without block dedup |
| **Windows park/spill parity** | **Full Windows support:** `ParkDisk` v2/v3, `HG_PARK_COW`, buffered `pd_pread_direct` stub (`7e9f12c`) | `STRATA_PARK_SEGPOOL_MIB` (`generate.cpp:534-541`) is **Linux-only — "inert where the OS layer is not written - Windows, macOS"**; `conversation_spill.hpp` works on Windows via `CreateFile` lock, but without CoW or chunk dedup | **(a) Fork required** — Windows `:8080` is a primary target |
| **Content-addressed chunk dedup** | BCS v3 (`0ccaab9`) 16 MiB `PdKey` chunk store (`chunks/`), refcounted blob deletion | None — each spilled session is a standalone `STRSES` file | **(a) Fork required** |
| **Lineage-aware boot rebuild & published-parent guard** | Two-phase `scan_v2_keys` + `view_parents` / `view_refs` + M-C7 published-parent rule + quarantine-in-place for foreign files | Per-file `STRSES` validation on restore; no cross-file view lineage or chunk refcount shielding | **(a) Fork required** |
| **RAM-tier parked CoW dedup** | `HG_PARK_COW` (`common_best` + verify-then-view `adopt_shared`), `ParkAnchorGuard` donor stack (§191), M-D2 `has_borrowed()` guard | `STRATA_CKPT_POOL` / `STRATA_PARK_ASYNC` (Linux segment pool + ready checkpoint reuse, no cross-sibling prefix view sharing) | **(a) Fork required** (adopt `STRATA_CKPT_POOL` only where non-conflicting) |
| **Salt-to-donor delta restore (S5c-v3)** | `read_v2` pulls matching 16 MiB salted blocks from guarded RAM donors instead of NVMe (`7e611ef`) | None — restores full file from disk | **(a) Fork required** |
| **Deepest-checkpoint resume + S5d prewarm** | `HG_RESUME_DEEPEST` competes with RAM match; S5d per-conversation expert histogram pre-seeds cache (`7675927`, `093b121`) | Upstream already saves periodic checkpoints (`prompt_checkpoint_grid.hpp`), no cross-tier deepest resume or S5d histogram | **(a) Fork required** |

**Layer 1 conclusion:** Upstream's `conversation_spill.hpp` cannot replace `ParkDisk` v3 / `HG_PARK_COW` / `S5c` / `S5d` (missing chunk dedup, RAM CoW views, S5c/S5d, and Windows `STRATA_PARK_SEGPOOL_MIB` is inert). Replay Layer 1 intact in Phase 1; keep upstream's `--conversation-spill-*` flags untouched and off by default unless evaluated post-rebase.

---

## 4. Surface-Only & Env-Prefix Rules

- Installed binary: `hydragen` (`hydragen.exe`); serve flag: `--engine hydragen`.
- Fork-introduced knobs read `HG_*` first via `include/strata/core/hg_env.hpp` (`hg_getenv`), falling back to `STRATA_*`. Upstream-derived knobs keep plain `STRATA_*` names.
