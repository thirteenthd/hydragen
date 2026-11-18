# Model files — what to get and where it comes from

The model itself is not part of this repo (its license travels with the
weights): HydraGen serves **Qwen/Qwen3.8-Flash-Next** with
**ISTA-DASLab's GSQ-RCO quantization** — the same files behind every published
measurement here. The installers already point `hydragen-local.json` at the
layout below; you choose the root (`./install.sh --model-dir DIR` /
`.\install.ps1 -ModelDir DIR`, default `<repo>/models`).

| Config slot | File (under your model dir) | Size | Source |
|---|---|---|---|
| `--native` | `models/IQ3_XXS/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | ~47 GB | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), `IQ3_XXS/` (routed-expert weights shard) |
| `--ple-gguf` | `models/IQ3_XXS/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` | ~29 GB | same repo (n-gram/PLE table shard — streamed from disk, not loaded into RAM) |
| `--pack` | `packs/iq3_xxs/` | ~75 GB | built by Strata's setup from those GGUFs (`engine/tools/iq_pack.py`) — not a standalone download |
| `--mtp` | `mtp/rt/` | ~5 GB | fetched by Strata's setup (`engine/tools/mtp_fetch.py`; range-reads the MTP tensors from the BF16 checkpoint, SHA-256-verified) |
| `--expert-profile` | — | 196 KB | **ships in this repo**: `engine/data/expert-profile.bin` (boot seed only; the installers wire `--expert-profile-save` so the engine re-seeds from what your own traffic learned — see HOW_IT_WORKS) |

The installers default to the **IQ3_XXS** tier. `--model IQ3_S` / `-Model IQ3_S` wires
the same layout for the 3.5-bit tier: `models/IQ3_S/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-*`
shards (shard 1 ~55 GB; the PLE shard is byte-identical across all tiers, ~29 GB) and a pack
built under `packs/iq3_s/`. IQ3_S is the best-quality tier (matches the full model on the
published benchmarks), about 15% slower than IQ3_XXS, and setup documents a 64 GB RAM floor.

## Easiest path: let Strata's setup fetch and verify everything

The vendored engine carries Strata's own setup, which downloads the pinned
GGUFs over 16 connections, resumes interrupted downloads, checks every file's
SHA-256 against pinned values, then builds the pack and fetches the MTP draft
into the same data dir:

(The pack build is also what exports the model's `tokenizer/` folder.  If serve
exits with *the model's tokenizer is missing*, the pack step has not run to
completion — the error prints the exact repair command, including a fast
tokenizer-only export for when the pack itself is already complete.)

```bash
# Linux
bash engine/setup.sh --model IQ3_XXS --data-dir <your-model-dir>
```

```bat
:: Windows
cmd /c engine\SETUP.bat --model IQ3_XXS --data-dir D:\models
```

Use the same directory you gave the installer (`--model-dir`), and the config
already points at the result. Details of the download machinery (connection
count via `STRATA_DL_CONNECTIONS`, pinned revisions, file checks):
`engine/docs/DETAILS.md`, section "Downloading the model files yourself".

## Already have the GGUFs, or want your own downloader

Download both `IQ3_XXS` shards keeping their original file names, then let
setup verify and process them without re-downloading:

```bash
uvx hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
    --include "IQ3_XXS/*" --local-dir /data/gguf
bash engine/setup.sh --gguf-dir /data/gguf
```

(`uvx` comes from `pip install -U uv`; plain `hf` works too via
`pip install -U huggingface_hub`. A Hugging Face login raises anonymous rate
limits: `uvx hf auth login` — Strata never reads or sends your token.)

## Licenses

Weights derive from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
through GSQ-RCO quantization and carry the Qwen Community License 1.0 — see
the quantization repo's card. Engine provenance: `NOTICE` and
`docs/../README.md` (Provenance & license).
