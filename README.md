# HydraGen — high-concurrency MoE inference on 12 GB GPUs

### Eight deep conversations. 131,000 tokens each. On the GPU already in your desktop.

![context per chat: 131k tokens](https://img.shields.io/badge/context_per_chat-131k_tokens-2f81f7) ![concurrent 100k-token chats: 8](https://img.shields.io/badge/concurrent_100k_chats-8_measured-2f81f7) ![warm resume at 100k depth: ~2 s](https://img.shields.io/badge/warm_resume_at_100k-~2_s-2f81f7) ![idle chat return: 0.1 MiB](https://img.shields.io/badge/idle_chat_return-0.1_MiB-2f81f7) [![notarized experiments: 173](https://img.shields.io/badge/notarized_experiments-173_incl._failures-6e7681)](docs/CHANGES.md)

Every number measured on one RTX 3080 Ti and quality-gated before it counted;
the naive tier moves ~2 GiB to return an idle chat. No multi-GPU. No
datacenter card. No answer-quality compromise.

**Requirements:** NVIDIA Ampere GPU with **12 GB+ VRAM** (RTX 3060 Ti / 3080
Ti class) and **64 GB+ system RAM** — conversation memory streams to pinned
host RAM — plus an NVMe, CUDA toolkit 11+/12, and Python 3.10+.

**Try it** — the installer builds the engine from source, detects your GPU
(model, VRAM, compute capability) and system RAM, sizes the park-tier budget to
your machine, and writes a default `hydragen-local.json` (localhost, canonical
stack pre-wired):

Linux:

```bash
git clone https://github.com/thirteenthd/hydragen.git && cd hydragen
./install.sh
.venv/bin/python engine/serve/server.py --engine hydragen --config hydragen-local.json
```

Windows (PowerShell):

```powershell
git clone https://github.com/thirteenthd/hydragen.git; cd hydragen
.\install.ps1
.venv\Scripts\python.exe engine\serve\server.py --engine hydragen --config hydragen-local.json
```

OpenAI-compatible API + live concurrency dashboard → `http://127.0.0.1:8181/`
(point the config's model-blob paths at your own model set first — see
Install notes).

## The headline

One fleet of agent sessions, each holding a ~100k-token conversation, on one
RTX 3080 Ti (12 GB). Same benchmark, run **before** and **after** HydraGen's
concurrency features: *Before* is the stock approach (RAM cache + full-copy
disk tier, shallow resume); *After* is this repo's stack (park-disk tier +
shared-view restores + deep resume + CoW parks). Every number client-side,
every arm needle-checked[^run10].

| Metric | C=4 before | C=4 after | C=8 before | C=8 after |
|---|---|---|---|---|
| wall time | 501 s | **317 s** | 951 s | **548 s** |
| warm turn TTFT p50 | 48.0 s | **6.9 s** | 41.0 s | **21.2 s** |
| warm decode p50 | 8.0 tok/s | **24.6 tok/s** | 13.8 tok/s | **17.9 tok/s** |
| turns losing their parked prefix | 5 of 20 | **1 of 20** | 7 of 40 | **1 of 48** |
| tokens wasted re-prefilling lost history | 505k | **~180k** | 712k | **~200k** |

What that means in practice, at ~100k-token depth on a 12 GB card:

- **One to two deep agent sessions: comfortable.** Warm turns 2-3 s, decode
  60-94 tok/s per stream.
- **Four deep sessions: interactive.** ~7 s warm turns at ~25 tok/s decode —
  the same arm cost ~48 s per warm turn before these features landed.
- **Eight deep sessions: workable.** ~21 s warm turns (queue-bound at this
  depth), ~18 tok/s decode. Cold admission in an 8-way burst still costs
  ~100 s — that is prefill physics, not a cache failure.
- Peak single-stream decode (~70-100 tok/s at 100k depth) is unaffected by any
  tier machinery.
- Append-only history is the contract: edit mid-history and you pay the full
  ~40 s re-prefill. These are measurements on one hardware class (12 GB
  Ampere, single NVMe), not a product promise.

## Why this exists

**The goal is more simultaneous conversations, not one faster request.** Many
users (or many cooperating agents) each keeping a long, growing chat — tens of
thousands of tokens deep — on one small consumer graphics card. The datacenter
answer (buy a bigger card, or several) is off the table by design.

A model of this class doesn't fit in 12 GB to begin with — that half is solved
by streaming model pieces in from disk as they are needed. The real squeeze is
*conversation memory*: one 100k-token chat wants about 2 GB of fast memory, a
dozen want over 20 GB, and the card has 12 GB total — shared with the model
itself.

**Our answer: never duplicate what you already have.** Chats that share
content — the same instructions, the same agent's earlier turns, a forked
conversation — share that stored memory instead of paying for it once per
chat. Idle chats are put away as small deltas that point at what they have in
common with their siblings, first into system memory, then onto disk. A
returning chat comes back for 0.1 MiB of reading instead of ~2 GB, because its
shared parts were never actually thrown away — and it resumes at the deepest
point it had remembered, not by rereading its own history. Those rules are
what turn "this fleet does not fit" into "this fleet fits at 97%", and make
four and eight deep chats interactive instead of impossible.

**How is that possible?** Session forks with ref-counted memory pages, a
cross-boot disk tier, restores that re-attach shared parts instead of copying,
deepest-checkpoint resume, and eviction rules that refuse to drop anything
still in use — plus the closed-negative experiments we publish too. Built on
top of [Strata](https://github.com/Niko1221/Strata), which already streams
disk-backed MoE experts; our contribution is concurrency funded by deleting
duplication. The full layer map: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Where everything lives

| Read this | For |
|---|---|
| [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | the layers, the knobs, the eviction rules |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | methodology, every measurement table, how to reproduce them |
| [docs/MODELS.md](docs/MODELS.md) | model files: what to get, where it comes from, one-command fetch via Strata setup |
| [docs/CHANGES.md](docs/CHANGES.md) | the journey: 168 notarized entries, newest first — reads bottom-up from day one |
| [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) | the Nov-12 program snapshot (pre-disk-tier era) |
| [docs/laws/](docs/laws/) | engineering laws distilled the hard way |
| `engine/` | our fork of [Strata](https://github.com/Niko1221/Strata) (MIT): v0.2.13 + the full M-C1→M-C10 chain (tip: M-C10c); per-piece classification in [docs/UPSTREAM_DELTA.md](docs/UPSTREAM_DELTA.md) |
| [docs/UPSTREAM_DELTA.md](docs/UPSTREAM_DELTA.md) | what we re-engineered, what is net-new vs upstream Strata, and which shipped defaults are upstream features we measured and enabled |
| `configs/` + `bench/` | canonical ctx-131k serving config + launcher, and the gate suite every landing must pass (quality battery, concurrency ladder, park-ladders) |

## Install notes

Prerequisites: NVIDIA driver + CUDA toolkit + Python 3.10+ (Windows adds an
MSVC C++ toolchain; both installers tell you exactly what is missing and how to
get it). The installers provision `.venv` (cmake + ninja from pip), compile the
vendored engine with upstream release flags, and write `hydragen-local.json`
from the canonical config. Default build arch is sm_86 — the 3060 Ti / 3080 Ti
this program was developed on; override with `HYDRAGEN_CUDA_ARCH` (e.g.
`89-real` for RTX 40-series). A source build is the only option: prebuilt
Strata release binaries do not contain this engine's chain. Verified
end-to-end: both installers from a fresh clone build a working engine on a
3080 Ti.

**Model blobs.** The canonical config serves Qwen3.8-Flash-Next with
ISTA-DASLab's GSQ-RCO IQ3_XXS quant (~76 GB GGUFs + a built expert pack;
license: Qwen Community). The installer rewrites all model paths to a
directory you choose (`--model-dir DIR` / `-ModelDir DIR`), and the vendored
Strata setup can fetch, verify (SHA-256, resumable) and process everything for
that directory in one command. Full sources and commands:
[docs/MODELS.md](docs/MODELS.md). The expert-routing profile already ships in
this repo.

**Living beside upstream Strata.** HydraGen's engine is a fork of Strata, and it
is built to coexist with an upstream install on the same machine. The binary
installs as `engine/build/hydragen` (`hydragen.exe`), not `strata`, so `ps`,
PATH and task managers never go ambiguous; the serve layer accepts both
`--engine strata` and `--engine hydragen`. Nothing runtime is shared: configs,
logs and the park-disk tier all live inside this repo's folder. Only *model
data* is shareable — the quant packs follow Strata's data-dir layout, so a
`--model-dir` (or `--gguf-dir`) pointing at an existing Strata data folder
reuses its files without re-downloading. One port rule: both dashboards
default to 8181, so run the second with `--port 8182`. The `STRATA_*`
environment variables and everything inside `engine/` keep their upstream
names on purpose — that is what keeps the fork mergeable with Strata and keeps
provenance legible (see Provenance & license).

## Provenance & license

The engine is a fork of **[Strata](https://github.com/Niko1221/Strata)** (MIT,
Niko1221) — vendored at v0.2.13 plus our full M-C1→M-C10 chain (tip: M-C10c),
upstream commit history preserved under `engine/` (see `NOTICE`). HydraGen
code is Apache-2.0 (`LICENSE`); Strata-derived engine code remains MIT. The
167-section change log (`docs/CHANGES.md`) is the notarized record of every
measurement → decision → commit in this campaign — including the ideas that
measured badly and were reverted.

[^run10]: [§159](docs/CHANGES.md#159-c4c8-deep-re-bench-on-the-canonical-m-c9m-c10c-stack-run-conc8_run10-2026-11-15) — Nov-15 c4/c8 re-bench; source of the *after* columns, with the *before* columns from the pre-feature run it compares against.
