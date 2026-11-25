# Layout Migration Inventory — `0.1.60` (Flat) → `v0.2.21` (`0.2.16` Split)

Phase 0, item **P0.5**. Upstream Strata restructured its repository root at `v0.2.16` into `engine/`, `terminal/`, `tools/`, `app/`, `docs/`. HydraGen currently vendors the old pre-`0.2.16` layout inside `engine/` while owning HydraGen-specific files at the repo root (`install.sh`, `install.ps1`, `configs/`, `bench/`, `docs/`, `tools/`).

---

## 1. Directory Mapping & Exact File Counts

Baseline at `control-pre-0221` (`654b078`): `git ls-files engine` = **7,332**, repo total = **7,403** (`4bf3603` after P0.1 docs = **7,405**).
Target `v0.2.21` (`d46f5f26`, tree `ac94d63d`): **8,456** tracked files total (`engine/`: **2,898**, `terminal/`: **1,522**, `tools/`: **165**, `app/`: **1,282**, `docs/`: **2,547**, `.github/`: **33**, root: **9**).

| Current HydraGen Path | `v0.2.21` Target Path | Phase 1 Rule |
|---|---|---|
| `engine/{CMakeLists.txt,CMakePresets.json,cmake/,include/,src/,tests/,third_party/,metal/,spark/,sycl/,bench/}` | `engine/` (pure C++/GPU build tree; `v0.2.21` has 2,898 files; bulk of old `engine/bench` moved to `docs/bench-results`) | **Adopt at root `engine/`** + add fork-only `run_*.bat` helpers & `hg_env.hpp` / `park_disk` / `decode_governor` |
| `engine/{serve/,strata_control/,strata_cli/,data/,integrations/,bin/,requirements*.txt,setup.py,strata.py,conftest.py,tests/terminal/}` | `terminal/` (`v0.2.21`: 1,522 files — `serve/` 306, `strata_control/` 98, `strata_cli/` 43, `data/` 24, `tools/` 530, `tests/` 456) | **Adopt at root `terminal/`** (re-port `serve/server.py` + `serve/web/` fork overlay) |
| `engine/{packaging/,docker/,research/,BENCH.bat,bench.sh}` | `tools/` (`v0.2.21`: 165 files — `packaging/` 41, `docker/` 15, `layout/` 21, `research/` 34, `squant/` 14) | **Merge with HydraGen root `tools/`** (`needle_battery.py`, `nssm.exe`, `run_hydragen_service.py` — **0 filename collisions verified**) |
| `engine/apps/{desktop,host}` (1,018 files) | `app/{desktop,host}` (`v0.2.21`: 1,282 files — `desktop/` 114, `host/` 1,168) | **Keep vendored-unbuilt** (Decision #2: land at `app/` or keep unbuilt at `app/` to preserve `v0.2.21` subtree counts) |
| `engine/docs/` (128 files) + `engine/{AGENTS.md,CONTRIBUTING.md}` | `engine/docs/` (**NOT** root `docs/`) | **Confine to `engine/docs/`** — root `docs/` owns HydraGen's `MODELS.md`, `HOW_IT_WORKS.md`, `BENCHMARKS.md`, `CHANGES.md`, `MILESTONES.md`, `laws/`; landing upstream `docs/` at root causes **two exact-name overwrites** (`MODELS.md`, `HOW_IT_WORKS.md`) plus `benchmarks/` vs `BENCHMARKS.md` |
| `engine/.github/` (deleted in `ebea5fe`, 23 files) | `.github/` (`v0.2.21`: 33 files) | **Keep dropped** (Decision #5) |
| `engine/setup.sh` | `setup.sh` (at root in `v0.2.21`; `SETUP.bat` removed upstream, replaced by `START-HERE.bat` / `terminal/setup.py`) | Place upstream `setup.sh` if needed without overwriting HydraGen `README.md` / `LICENSE` / `.gitignore` |
| `engine/mc3_patch.py` | — | **Delete** (obsolete §181 script) |

---

## 2. Permanent `.gitignore` Fix & Tree-Integrity Gate (G6)

In §207 (`25a7d1a`) and §208 (`29ae08c`), root `.gitignore:6` (`models/`) silently excluded 45 `mtmd/models/` files and 156 `src/models/` files.

- **Root `.gitignore` edit (Phase 1 L0):**
  - Replace line 6 `models/` with anchored `/models/`.
  - Delete the two brittle negation lines (`!engine/third_party/strata-llama/tools/mtmd/models/` and `!engine/third_party/strata-llama/src/models/`) — they become redundant once `/models/` is anchored.
  - Verified impact: only `engine/docs/media/models` (3 small already-tracked files, ~9 KB) sits inside nested `models/` dirs, and `204` files under `engine/third_party/.../models/` are already tracked.
- **G6 Subtree Count Gate:** After checking out upstream `v0.2.21` subtrees (`engine/`, `terminal/`, `tools/`, `app/`), verify `git ls-files <subtree>` matches `git -C C:/DevOps/hydragen-sync180 ls-tree -r --name-only v0.2.21 -- <subtree>` **before** applying any fork commits or trusting any diff.

---

## 3. Consumer Path Inventory (Must Update in Phase 1)

Every HydraGen script/config referencing the old flat `engine/` layout:

| File | Current Lines & Old Path | New `0.2.16` Layout Path | Notes |
|---|---|---|---|
| `install.sh` | `:14` `engine/requirements-build.txt` | `terminal/requirements-build.txt` | pip build deps |
| `install.sh` | `:30,183,239` `engine/data/expert-profile.bin` | `terminal/data/expert-profile.bin` | `--expert-profile` default seed |
| `install.sh` | `:31,234,237` `engine/setup.sh` / `engine/SETUP.bat` | `setup.sh` / `terminal/setup.py` | model downloader instructions |
| `install.sh` | `:141` `-r engine/requirements.txt -r engine/requirements-build.txt` | `-r terminal/requirements.txt -r terminal/requirements-build.txt` | venv install |
| `install.sh` | `:144` comment `packaging/build_cuda_linux.sh` | `tools/packaging/build_cuda_linux.sh` | comment hygiene |
| `install.sh` | `:148,159,160,161` `cmake -S engine -B engine/build` → `engine/build/hydragen` | **unchanged** (`engine/` stays C++ root) | verify `BIN="$PWD/engine/build/hydragen"` |
| `install.sh` | `:241` `engine/serve/server.py` | `terminal/serve/server.py` | serve invocation |
| `install.ps1` | `:22,162` `engine/requirements.txt`, `engine/requirements-build.txt` | `terminal/requirements.txt`, `terminal/requirements-build.txt` | venv install |
| `install.ps1` | `:25,244,247` `SETUP.bat` / `engine\SETUP.bat` | `terminal\setup.py` (or `START-HERE.bat`) | upstream removed `SETUP.bat` in `0.2.x` |
| `install.ps1` | `:170,182,184,185` `cmake -S engine -B engine/build` → `engine\build\hydragen.exe` | **unchanged** (`engine/` stays C++ root) | — |
| `install.ps1` | `:203,236` `engine\data\expert-profile.bin` | `terminal\data\expert-profile.bin` | `--expert-profile` default seed |
| `install.ps1` | `:252` `engine\serve\server.py` | `terminal\serve\server.py` | serve invocation |
| `tools/run_hydragen_service.py` | `:4,159` `engine/serve/server.py` | `terminal/serve/server.py` | NSSM Windows `:8080` service wrapper |
| `tools/packaging/build_cuda_windows.py` | old `engine/packaging/build_cuda_windows.py` (`REPO = parents[1]`) | upstream `v0.2.21` already has `REPO = parents[2]`, `ENGINE_SRC = REPO / "engine"`, `PACKAGING = REPO / "tools" / "packaging"` | re-apply only the `hydragen.exe` staging hunk (`:180–186`) |
| `engine/run_*.bat` (`2269997`) | `cd /d C:\DevOps\HydraGen\engine`, `build\cuda-win\hydragen.exe` | **unchanged** (`engine/` stays C++ root) | fix `run_parkv2_cuda_build.bat` `strata.exe` → `hydragen.exe` |
| `configs/strata-iq3{s,xxs}-m25-0213-ctx131k.json` | `:2` `"exe": "~/hydragen/engine/build/hydragen"` | **unchanged** (`engine/build/hydragen`) | check `--expert-profile` path if referenced |
| `README.md` | `:84,104,155,163,174` `engine/` layout + `engine/serve/server.py` + `v0.2.13` baseline text | update to `terminal/serve/server.py` + `v0.2.21` (`0.1.67`) at `v0.0.1` cut | update alongside `NOTICE` |
