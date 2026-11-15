# HydraGen — high-concurrency MoE inference on 12 GB GPUs

### Eight deep conversations. 131,000 tokens each. On the GPU already in your desktop.

| | |
|---|---|
| **131k** | context per conversation — on one 12 GB consumer card |
| **8** | simultaneous 100k-token chats, measured and quality-gated |
| **~2 s** | warm-turn resume at 100k depth, prompt → first token |
| **0.1 MiB** | to bring an idle conversation back (the naive tier moves ~2 GiB) |
| **164** | notarized experiments — failures included — [the whole journey](docs/CHANGES.md) |

No multi-GPU. No datacenter card. No answer-quality compromise — every run is
checked against an exact-answer battery before its numbers count.

**Try it:**

```bash
git clone https://github.com/thirteenthd/hydragen.git && cd hydragen
./install.sh        # Linux · Windows: powershell -ExecutionPolicy Bypass -File .\install.ps1
.venv/bin/python engine/serve/server.py --engine strata --config hydragen-local.json
# OpenAI-compatible API + live concurrency dashboard → http://127.0.0.1:8181/
```

## The headline

One fleet of agent sessions, each holding a ~100k-token conversation, on one
RTX 3080 Ti (12 GB). Same benchmark, two stacks: *Nov-14* is the standard
approach; *Current* is HydraGen's stack. Every number client-side, every arm
needle-checked[^run10].

| Metric | C=4 | C=8 |
|---|---|---|
| wall time | 501 s → **317 s** | 951 s → **548 s** |
| warm turn TTFT p50 | 48.0 s → **6.9 s** | 41.0 s → **21.2 s** |
| warm decode p50 | 8.0 → **24.6** tok/s | 13.8 → **17.9** tok/s |
| turns losing their parked prefix | 5 of 20 → **1 of 20** | 7 of 40 → **1 of 48** |
| tokens wasted re-prefilling lost history | 505k → **~180k** | 712k → **~200k** |

What that means in practice, at ~100k-token depth on a 12 GB card:

- **One to two deep agent sessions: comfortable.** Warm turns 2-3 s, decode
  60-94 tok/s per stream.
- **Four deep sessions: interactive.** ~7 s warm turns at ~25 tok/s decode —
  the same arm cost ~48 s per warm turn a week ago.
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
| [docs/CHANGES.md](docs/CHANGES.md) | the journey: 164 notarized entries, newest first — reads bottom-up from day one |
| [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) | the Nov-12 program snapshot (pre-disk-tier era) |
| [docs/laws/](docs/laws/) | engineering laws distilled the hard way |
| `engine/` | our fork of [Strata](https://github.com/Niko1221/Strata) (MIT): v0.2.13 + the full M-C1→M-C10 chain (tip: M-C10c), commit history preserved |
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

**Model blobs.** The canonical config expects an IQ3_XXS routed-expert pack,
native model + PLE GGUFs (Qwen3.8-Flash-Next, GSQ-RCO quantization), an MTP
draft adapter, and an `expert-profile.bin` routing profile for the adaptive
cache — point the paths in `hydragen-local.json` at your own set.

## Provenance & license

The engine is a fork of **[Strata](https://github.com/Niko1221/Strata)** (MIT,
Niko1221) — vendored at v0.2.13 plus our full M-C1→M-C10 chain (tip: M-C10c),
upstream commit history preserved under `engine/` (see `NOTICE`). HydraGen
code is Apache-2.0 (`LICENSE`); Strata-derived engine code remains MIT. The
164-section change log (`docs/CHANGES.md`) is the notarized record of every
measurement → decision → commit in this campaign — including the ideas that
measured badly and were reverted.

[^run10]: [§159](docs/CHANGES.md#159-c4c8-deep-re-bench-on-the-canonical-m-c9m-c10c-stack-run-conc8_run10-2026-11-15) — Nov-15 c4/c8 re-bench on the current stack; source of both columns of the table above.
