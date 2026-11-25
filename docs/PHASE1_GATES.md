# Phase 1 gates + the COW_SHARED separability spec

Phase 0, items P0.3 and P0.7. Written before any Phase 1 code, so the pass rules
exist before the work does. Baseline: `control-pre-0221` (`654b078`), rebase base
`v0.2.21` (`d46f5f26`, tree `ac94d63d`, engine 0.1.67). Runtime reference:
`docs/control-pre-0221.md`.

---

## P0.3 — COW_SHARED separability spec

### What the knob is

`STRATA_COW_SHARED` appends a **shared region** of `g_cow_shared` slots per QSA
ordinal to each CoW arena, plus a device-side hint-table tail (`lin`, `sblk`,
`sst`, `srf`, `sctr`). Sites at `654b078`:

| file | lines | content |
|---|---|---|
| `src/program/generate.cpp` | 4691 | `hg_getenv("STRATA_COW_SHARED")` |
| `include/strata/core/layer.hpp` | 341 | `void qsa_set_cow_shared(int64_t)` |
| `src/core/layer.cpp` | 837, 839 | `g_cow_shared = 0`, `qsa_set_cow_shared` |
| `src/core/layer.cpp` | 890 | lineage wiring early-out (`g_cow_shared <= 0 → return`) |
| `src/core/layer.cpp` | 1416 | `T = g_cow_arena_slots + g_cow_shared` (arena span) |
| `src/core/layer.cpp` | 1441-1450 | `a.off[8..10]` = the hint-table tail + fetch counter |
| `src/core/layer.cpp` | 1460, 1467, 1469-1471 | map the tail only when `g_cow_shared > 0`; memset it |
| `src/core/layer.cpp` | 1486 | boot line prints `(N shared region)` |
| `src/core/layer.cpp` | 1658-1662 | bake `lin_all/sblk/sst/srf/shared_base/n_shared/sh_ctr` into `st.map` |

### It is already dead at runtime

The live posture runs `STRATA_COW_SHARED=0` (§4 of `control-pre-0221.md`). With
`g_cow_shared == 0`: `T == g_cow_arena_slots`, `a.off[8..10]` collapse to zero
size, the tail stays unmapped and null (`layer.cpp:1460`, the §234 note), and
`st.map.n_shared = 0`. So deleting the knob removes a path that never executes on
the control baseline.

### Joins do not depend on it — proven in code, not just measured

The join-at-fetch registry (`CowPeer`, `g_cow_reg`, `st.map.reg`, `st.map.join_ctr`)
is gated at `layer.cpp:1653` on the **arena** condition
(`g_cow_arena && p.mode == 1 && st.kv_int8 && !st.kv_q4 && !g_cow_arenas.empty()`)
and registration at `:1665` on `g_cow_reg && st.map.slot_use != nullptr`.
**Neither is keyed on `g_cow_shared`.** `pe->hhead` comes from `g_cow_join_hh`
(`:1670`), and `pe->floor = 0x7FFFFFFF` — "no floor until a fork lineage sets one".
This matches §204, which measured `joins-at-fetch 0 → 80` with the boot line
reading `(0 shared region)`.

Therefore `STRATA_COW_JOIN_HH` **stays armed** (`JOIN_HH=8` in the posture): it is
the only thing that ever made sharing engage, and it survives the prune.

### Two prune depths — Phase 1 does depth 1 only

- **Depth 1 (Phase 1, in the rebase):** delete the env read, the setter, the
  `g_cow_shared` uses, and the region/tail allocation. Keep the `st.map` fields
  pinned to null/0. No kernel-side change; behaviour byte-identical, because those
  fields are already null/0 on the control baseline.
- **Depth 2 (deferred, own task):** delete the kernel-side map fields
  (`lin_all/sblk/sst/srf/sh_ctr/shared_base/n_shared`) and the lineage/shared-fetch
  code that reads them. That touches the verify/fetch kernels and needs the parity
  suite; it does not belong inside a rebase.

### §235 is not fixed by this prune

§235's corruption is an arena **slot pin-law** failure (entry committed at window
*k*, slot freed at *k+1*, gathered at *k+2*; the tail-slot pin law broken for
adopted entries). That lives in the `slot_use` / `ablk[L] == b` identity path,
which the COW_SHARED prune does not touch. The prune is hygiene. §235's own law —
"keep `COW_SHARED=0` until the tail-slot pin fix exists" — stays in force either
way, and the pin fix remains a hard gate on Phase 4.

### Separability test (run on ms01, §204 arm-2 shape)

Boot with the prune applied and `COW_JOIN_HH=8`, `COW_SHARED` unset:

```
STRATA_COW_ARENA=1 STRATA_COW_ARENA_SLOTS=9216 STRATA_COW_SLOT_CELLS=4096 \
STRATA_COW_FORK=1 STRATA_COW_JOIN_HH=8 --kv-resident 4096 --kv int8
```

Pass requires all four:

1. boot line reports the arena with **no shared region** and no allocation error;
2. `joins-at-fetch served so far:` reaches **> 0** (arm 2 measured `0 → 9 → 24`);
3. `the admission fork points at N of M prefix pages` reports **N > 0**
   (arm 2 measured 1,023 of 28,927);
4. needle **3/3** at depths 18,995 / 46,993 / 96,994.

Any failure means the prune reached past the region into the lineage path — revert
to depth 1 with the map fields kept, and re-file depth 2 separately.

---

## P0.7 — Phase 1 exit gates

Every gate states: command, pass rule, what to record. Upstream's own rules from
`engine/AGENTS.md` bind the reporting: new features opt-in with the default path
byte-identical; build every backend a file touches; change a default only with
measurements; mark read-only findings as unconfirmed static analysis. Our arms are
**CUDA / arch 86 only** — HIP and SYCL go in a stated "not run" column.

### G1 — build + ctest

```
cmake -B build -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real \
      -DSTRATA_WINDOW_MAX_T=8            # and a second tree with =16
ctest --test-dir build
```
Pass: CPU suite 100%. GPU-parity failures must be **re-notarized against
`v0.2.21`**, not inherited from §181's 322/352 (22 pre-existing + 10 new nvfp4/q36
was the 0.1.60 shape; the list is expected to differ). Record the full failing-test
list with the reason for each.

### G2 — §177 token identity

Same-binary identity is the law; cross-binary parity is near-identical, not
token-identical. Run the QC battery (QC0–QC7) on the rebuilt-from-tag binary
against the `control-pre-0221` binary at identical args.
Pass: byte-identical on QC0–QC6, forks only where a fork point is expected (QC7
forked at 0.1.60). Record per-line byte comparison, not "looks the same".

### G3 — upstream parity suite (new; §181 did not run it)

`kv_stream_parity`, `kv_window_parity`, `kv_hybrid_parity`, `kv_q4_parity`,
`kv_q8_parity`, `kv_steps_parity`, `kv_host_dma_parity`, `verify_batch_parity`,
`batch_rows_parity`, `batch_groups_test`.
Pass: all green on the CUDA/arch-86 build before any fork overlay commit that
touches KV, verify or batch rows is accepted.

### G4 — traffic probes, both shapes

- **Divergent** (`~/hydragen-experimental/bench_quick.py`, 8 unrelated one-line
  prompts, `max_tokens 400`, N = 1/2/4/8) — keeps §236 comparable.
- **Agentic** (`m25_fanout3.py` shape: one long shared head + C workers with short
  unique tails, forced decode length) — the product shape the campaign is about.

Record per arm: aggregate tok/s, per-stream tok/s, `agg(N)/agg(1)`, expert-cache
hit %, PCIe expert fraction, `MemAvailable`, refusals (`ERR BGEN`). The two curves
are the deliverable; a single number is not.

### G5 — sharing-path correctness (the §223 gate)

`probe222x.py parallel`, seeded pairs, qc = both numeric marks echoed **and**
`finish=stop`. Arena-on (`COW_ARENA=1 COW_FORK=1 JOIN_HH=8`) must **not** score
1/4 — §223 measured 1/4 and 1/4 on seeds 8300/8310 against 8/8 with
`COW_ARENA=0`. This gate rides the agentic arm of G4, because the payoff path and
the corruption path are the same path.

### G6 — tree integrity across the layout move

```
git ls-files engine | wc -l      # 7,332 at control-pre-0221
git ls-files | wc -l             # 7,403
```
Pass: after any checkout of the 0.2.16 layout, the per-subtree counts match
upstream's own `git ls-files` for the same subtree **before** any diff is trusted.
Root `.gitignore` anchors to `/models/` and the two dead negations are deleted;
`engine/docs/` (upstream docs, incl. `MODELS.md`, `HOW_IT_WORKS.md`,
`benchmarks/`) never lands at root `docs/` — two exact-name collisions verified
against our own `docs/`. §207/§208 swallowed 156 source files twice; this gate is
what stops a third time.

### G7 — row-cap arms (decision 3)

Two builds from the same source: `-DSTRATA_WINDOW_MAX_T=8` (parity control) and
`=16` with `CUDA_MODULE_LOADING=LAZY` (the engine's `setenv(...,"EAGER",0)` at
`generate.cpp:3536-3540` respects a pre-set value, so wide kernels pay only when a
wide window forms).
Pass: both boot; record the CMake line `Strata: verify / batch windows hold up to
N rows`, the runtime `strata verify: batch windows of up to N sequences` line, GPU
MiB at boot, and at `--batch 8` whether draft rows exist (`spare = win_cap − nact`).
The 16-row arm's VRAM cost on 12 GB is **unmeasured** — that is the number this
gate produces.

### G8 — CACHE_FUND unbundle (decision 4)

Prune Stage C (`cow_fund` `generate.cpp:16063`, `cow_retake` `:16166`, the
`set_funder` wiring `:16222-16223`, the `cow_retake()` call `:16256`, and
`layer.cpp:1184/1195/1565-1568` `g_cow_funder`). Keep Stage C.5 (`:8571-8575`)
opt-in, and **unbundle** the boot-pricing effect from the grow-budget zeroing at
`:8575` (`qsa_cow_arena_set_budget(g_cow_cache_fund ? 0 : cow_reserve)`) so that
with the funder gone a grow is funded from new VRAM — upstream's `kv_grow`
fallback semantics — instead of hitting a zero budget with nobody to pay it.
Pass: with `CACHE_FUND` unset (the control posture) the boot reservation and expert
cache match `control-pre-0221` exactly; with `CACHE_FUND=1`, §219's unlock
reproduces (reservation 3106 → 2434 MiB, cache 1758 → 2153 slots) **and** an arena
grow still succeeds rather than refusing.

### G9 — upstream restriction check, not assumption

`terminal/serve/server.py:4460` in v0.2.21: `"slot save/restore is not available
with parallel requests (\"parallel\" / --batch)"`. Our posture runs `--batch 4`
with `slot_save_path` set and the park/detach tier armed. Determine by test
whether upstream's restriction disables our park/restore path, and record which
side of the seam the fork's path sits on. Do not assume either way.

### ms01 hygiene (applies to every gate)

- `:8181` binary-ownership check before any arm — confirm which exe is serving
  (`pgrep -af hydragen`), per `control-pre-0221.md` §3.
- `pgrep` before any rerun: a timed-out ssh leaves jobs running (a stale
  `p2d1_arms.sh` wrapper was alive 2.6 h after its battery finished).
- Build tooling lives in `/home/thirteenthd/hydragen/.venv/bin` (ninja/cmake).
- mech backups are `.bak-*`; revert = `cp .bak` + `bash ~/mechdbg_restart.sh`.
