# Hydragen: High-Concurrency Tiered MoE Inference Engine

A standalone, modular Python inference package and server (`hydragen`) that combines **multi-slot continuous batching, prefix caching, and Triton Fused MoE slot-indirection** with **tiered expert hot-swapping, pinned host-RAM expert arena offloading, GSQ sub-4-bit quantization (`IQ3_XXS` / `IQ2_XS` / `Q2_0_STRATA`), and 4-layer Multi-Token Prediction (MTP) speculative decoding**—purpose-built to serve massive Mixture-of-Experts (MoE) models (such as `Qwen3.8-Flash-Next`, 176.9B parameters, 512 experts/layer across 48 layers at `128k` context) on a single **12 GB–24 GB consumer GPU** (`NVIDIA RTX 3080 Ti`, `RTX 3090/4090`, `RTX 4070 Ti+`).

![Hydragen Live Web Monitor](docs/assets/monitor-overview.png)

---

## 1. The Idea & Motivation

Running large Mixture-of-Experts (MoE) models (such as `Qwen3.8-Flash-Next`, ~41 GiB in `GSQ-RCO-IQ3_XXS` quantization with 512 experts per layer across 48 layers, top-10 routing) on a single **12 GB VRAM GPU** presents a fundamental challenge between two worlds:

1. **Standard vLLM:**
   - Excels at continuous batching, PagedAttention KV prefix caching, and high-throughput Triton GPU kernels.
   - However, standard vLLM assumes all model weights reside statically in GPU VRAM and requires standard HF/SafeTensors formats—making a 41 GiB MoE model impossible to load on a 12 GB card without slow, uncoordinated PCIe thrashing.
2. **Standard Single-Stream Tiered Engines:**
   - Excels at tiered MoE weight management: locking the 41 GiB expert pack into pinned System RAM, caching the hottest ~2,265 expert slots (`3.7 GB`) in VRAM, streaming missing routed experts asynchronously over PCIe Gen4 DMA, and using 4-layer Multi-Token Prediction (MTP) draft heads.
   - However, single-stream execution creates severe head-of-line blocking when autonomous agents (e.g., Letta, Hermes, Cursor, subagent teams) issue concurrent requests, and lacks dynamic multi-slot scheduling, expert-locality request reordering, agent loop guardrails for third-party system prompts, and real-time per-slot telemetry.

### The Hydragen Architecture
Hydragen bridges this gap as an independent, modular inference platform with a clean **PEP 621 Python package layout (`src/hydragen/`)** and one-click standalone installer:

- **vLLM General & Quantization Plugin (`src/hydragen/vllm_plugin.py`):**
  - Registered via `[project.entry-points."vllm.general_plugins"]` (`hydragen_gguf`).
  - Registers `@register_quantization_config("hydragen_gguf")` and `@register_quantization_config("strata_gguf")` (`StrataPagedMoEMethod(FusedMoEMethodBase)`) automatically whenever `vllm` initializes—requiring **zero edits** to site-packages.
- **Zero-Dequantization Dual-Kernel Architecture (`src/hydragen/expert_manager.py`, `src/hydragen/triton_moe.py`):**
  - Stores raw compressed `[gate | up | down]` GGUF blocks (`~1.61 MB/expert` instead of `9.83 MB/expert` in BF16/FP16 — a **5.8x VRAM & PCIe bandwidth reduction**) directly in `PagedExpertManager.vram_blob_cache` (`[num_slots, max_blob_bytes] uint8`).
  - **Native INT8 `dp4a` Grouped Engine (`native_gguf_paged_moe_forward`):** Automatically leveraged for high-speed production execution. Quantizes activation vectors once via `quantize_q8_1_rows` and executes Gate + Up + `SwiGLU` + Down directly in INT8 against the compressed `IQ2_XXS`/`IQ2_XS`/`IQ3_XXS`/`IQ3_S`/`IQ2_S`/`IQ4_NL`/`Q2_0_STRATA` expert slots on GPU Tensor Cores—with **zero intermediate BF16/FP16 weight buffer in VRAM**.
  - **In-Kernel Triton JIT Register Unpacker (`@triton.jit def fused_moe_gguf_down_slot_kernel`):** Provides dynamic register-level unpacking for specialized non-linear blocks (`IQ4_NL`, 32 weights/18B and `Q2_0_STRATA`, 64 weights/18B) directly inside the compute `K` loop before `tl.dot`, completely eliminating weight allocations in GPU memory.
- **Universal CUDA GSQ Dequantization & Routing Bridge (`src/hydragen/gguf_loader.py`, `src/hydragen/server/libstrata_iq_kernels.so`):**
  - Links `iq_kernels.cu`, `dequant_bf16.cu`, `route_prior.cu`, and `router_top10.cu` into dynamically located runtime libraries (`sm_86`+).
  - Supports **all 48 layers** of `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS`, including standard IQ quants (`IQ2_XXS=16`, `IQ2_XS=17`, `IQ3_XXS=18`, `IQ4_NL=20`, `IQ3_S=21`, `IQ2_S=22`) and custom `Q2_0_STRATA=42` (`ffn_down_exps` in odd layers 1..27), plus GPU-accelerated cache-aware top-10 routing (`route_prior_top10`).
- **Live PagedExpertManager & Multi-Slot Scheduler (`src/hydragen/expert_manager.py`, `src/hydragen/engine.py`):**
  - Manages dynamic 3-projection (`gate`, `up`, `down`) FP16 VRAM expert slots (`[num_slots, N, K]`), LRU eviction, async CUDA DMA stream transfers, and `ExpertLocalityPolicy` waiting-queue request reordering.
  - Supports continuous batching across parallel slots (`--batch 2`, `4`), interactive prompt chunk yielding (`BYIELD`), small-cap reasoning budget guards, and dynamic solo MTP promotion.

---

## 2. Execution & Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                   OpenAI / Anthropic / Native Endpoints                    │
│          (/v1/chat/completions, /v1/messages, /v1/responses, /metrics)     │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│              Hydragen Multi-Slot Engine (src/hydragen/engine.py)           │
│  • vLLM Plugin Registration (@register_quantization_config)                │
│  • ExpertLocalityPolicy Queue Reordering (maximizes shared active experts) │
│  • 2x–4x Concurrent Batch Slots (128k ctx) + Short-Prompt Preemption       │
│  • Cross-Turn Prefix Matcher (longest prefix & <|im_start|> checkpoints)   │
│  • Dynamic Solo-Path Promotion (engages 4-layer MTP when 1 slot is active) │
│  • Agent Loop Guardrails (STRATA_ROUTE_PRIOR=0.8 + </think> wrap recovery) │
└──────────────────┬──────────────────────────────────────┬──────────────────┘
                   │                                      │
                   ▼                                      ▼
┌─────────────────────────────────────┐ ┌────────────────────────────────────┐
│  Triton PagedMoE & CUDA Bridge      │ │  Resident Tiered MoE Worker        │
│  • fused_moe_indexed_slot_kernel    │ │  • 2,265 VRAM Expert Slots (3.7GB) │
│    (Triton tl.dot + slot_mapping)   │ │  • 40 GiB Paged Host-RAM Arena     │
│  • libstrata_iq_kernels.so / .dll   │ │  • Async PCIe Gen4 x16 DMA         │
│    (IQ2/IQ3/IQ4 + Q2_0_STRATA=42    │ │  • 4-Layer MTP Speculative Heads   │
│     + route_prior_top10 on GPU)     │ │  • Full 48-Layer Attention + KV    │
│  • PagedExpertManager (LRU + DMA)   │ │                                    │
└─────────────────────────────────────┘ └────────────────────────────────────┘
```

### Key Engineering Capabilities

#### 1. Modular Package Architecture (`src/hydragen/`, `pyproject.toml`)
- Fully packaged as a standard Python package (`pip install -e .`), exposing the `hydragen` CLI entrypoint and the `vllm.general_plugins` hook.
- In [`src/hydragen/triton_moe.py`](src/hydragen/triton_moe.py), performs two-stage Tensor Core matrix multiplications (`gate`/`up` + `SwiGLU` + `down`) directly over compact physical VRAM expert caches (`[num_slots, N, K]`) indexed by GPU `expert_to_slot[layer, expert_id]` tables.
- In [`src/hydragen/gguf_loader.py`](src/hydragen/gguf_loader.py), dynamically binds GPU dequantization across all 48 layers—including `Q2_0_STRATA` (18 bytes / 64 weights)—plus GPU `route_prior_top10` routing and token embedding dequantization.

#### 2. Concurrency at 128k Context Without VRAM Overflow
- Tuned KV cache and VRAM reservation (`--vram-reserve-mib 1152`) to support **`max_context: 131072` (128k tokens)** alongside **`--batch 2`** and live Triton PagedMoE expert slots on 12 GB VRAM while keeping **~2,265 expert slots** resident in VRAM.
- **Expert-Locality Queue Reordering (`ExpertLocalityPolicy.rank_candidates`):** Scores waiting requests by Jaccard overlap with currently resident VRAM experts so co-scheduled requests maximize expert cache hits.
- **Smart Slot Selection (`pick_slot`):** Routes incoming multi-turn requests to the idle slot holding the longest matching token prefix or `<|im_start|>` (`TURN_ID = 248045`) turn checkpoint, preserving idle slot KV caches across turns.
- **Head-of-Line Preemption (`BYIELD`):** If a long prompt read is running in one slot and a shorter interactive request arrives, the scheduler yields the long read at the next chunk boundary so the shorter request admits immediately.
- **Dynamic Solo MTP Promotion & Speculative Strategy:**
  - When multiple batch slots actively decode in the exact same step, speculative tree verification across independent streams is disabled to avoid GPU warp divergence and dynamic KV tree collisions. While overlapping, each stream generates tokens autoregressively (~28–35 tok/s per stream).
  - However, autonomous agent turns (Hermes, Letta, Cursor, subagent teams) are inherently bursty: an agent issues a tool call (50–150 tokens), then pauses for several seconds while tool execution completes in a sandbox.
  - The instant a request is alone in flight (`going_solo`), the scheduler dynamically migrates it back to the solo speculative path, re-engaging the 4-layer MTP draft head to accelerate decode speed up to **80.0–113.8 tok/s** for the rest of its turn.

#### 3. Speculative Decoding Optimization for Agent Workloads
- Optimized MTP parameters for bursty agent turns (long prompt prefixes, short tool-call outputs):
  - `"mtp_max": 4`
  - `"stream_all_min": 64` (engages MTP drafting on prompts as short as 64 tokens)
  - `"spec_min_p": 0.05`
- Achieved **82% – 88.2% draft acceptance** on solo agent turns, yielding **2.3x–2.65x single-stream speedups**.

#### 4. Agent Reliability: Routing Prior & Reasoning Loop Guardrails
- **The `STRATA_ROUTE_PRIOR` Tradeoff:**
  - Setting `STRATA_ROUTE_PRIOR=0` disables expert cache routing bias, but drops VRAM expert hit rate from ~85% to ~55% and cuts decode speed from **~90 tok/s down to ~50 tok/s**.
  - Setting `STRATA_ROUTE_PRIOR=0.8` restores the **75% – 89.6% VRAM expert hit rate** and **80 – 113.8 tok/s decode speed**, when paired with deterministic loop recovery.
- **Universal Small-Cap Reasoning Budget Guard:**
  - On reasoning/thinking models (e.g. Qwen 3.8 Flash, DeepSeek), requests with `max_tokens <= 1024` previously exhausted their entire token allowance inside `<think>` tags without emitting code. Hydragen automatically caps the thinking budget at `max_new // 2` when `max_new < 1024`, ensuring the model exits reasoning mode and emits the requested solution.
- **Agent Loop Recovery:**
  - Automatic repetition detection wraps stuck `<think>` blocks cleanly when passage repetition (`repeat_coverage >= 0.30`) is detected, preserving the KV prefix cache and allowing the model to transition immediately to tool calls.

#### 5. Real-Time Concurrency States Web Monitor
- **Integrated Slot Grid:** Real-time visual cards for every configured slot displaying active generation state, prefill rates, decode throughput, and token progress.
- **Queue Sub-card:** Live admission status, queued request count, and head-of-line preemption indicators.
- **Universal Multi-Engine Support:** Works identically whether running with `--engine vllm` or `--engine strata`.

---

## 3. As-Measured Performance Results

### Testbed Hardware & Model Configuration
| Component | Specification |
| :--- | :--- |
| **Host Machine** | `modelserver01` (Ubuntu Server LTS bare-metal) |
| **GPU** | 1x **NVIDIA GeForce RTX 3080 Ti** (12 GB GDDR6X, 912 GB/s, PCIe Gen4 x16) |
| **CPU** | **AMD Ryzen 9 5900X** (12 Cores / 24 Threads) |
| **System RAM** | **64 GB DDR4** (~40–41 GiB locked for pinned expert arena) |
| **Model** | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (176.9B params, 48 layers, 512 experts/layer, top-10 routing, 4-layer MTP head) |
| **Context Window** | **128,000 tokens** (`max_context: 131072`) |
| **Batch Concurrency** | **2 concurrent slots** (`batch_slots: 2`) |
| **VRAM Expert Cache** | **2,200 – 2,297 experts cached** (`3.6 – 3.8 GB` in VRAM; `11.5 – 11.7 / 12 GB` total VRAM used) |

---

### A. End-to-End Inference Throughput

| Metric | Stock Single-Slot (Baseline) | Hydragen (`STRATA_ROUTE_PRIOR=0`) | **Hydragen (`STRATA_ROUTE_PRIOR=0.8` + MTP 4)** |
| :--- | :---: | :---: | :---: |
| **Single-Stream Decode Speed** | 62 – 72 tok/s | 48 – 56 tok/s | **80.0 – 113.8 tok/s** *(mean ~88–95 tok/s)* |
| **Cold Prompt Prefill Speed** | 1,600 – 2,100 tok/s | 1,800 – 2,300 tok/s | **1,860 – 2,503 tok/s** |
| **Cached Turn TTFT (`90k–105k` ctx)** | ~0.70 s | ~0.65 s | **0.55 – 0.68 s** *(99.9% token reuse)* |
| **VRAM Expert Cache Hit Rate** | 72% – 80% | 52% – 60% | **75.1% – 89.6%** *(+0.8% – 3.2% over PCIe)* |
| **MTP Draft Acceptance Rate** | ~0% on short turns | 78% – 83% | **82.0% – 88.2%** *(e.g., 379/458 drafts accepted)* |
| **2-Slot Concurrent Aggregate Decode** | N/A (serialized) | ~68 tok/s | **102 – 118 tok/s combined\*** |
| **Multimodal Vision (`strata-vision`)** | 3.1 s | 2.9 s | **2.74 s total** *(52.3 tok/s decode, 88.2% MTP)* |

*\*When multiple streams actively overlap in the exact same decode step, each stream runs autoregressively at ~28–35 tok/s (~60–70 tok/s parallel combined). The **102–118 tok/s aggregate** reflects real-world multi-agent workloads where requests interleave and leverage **Dynamic Solo MTP Promotion** (up to 113.8 tok/s) whenever the other stream completes or pauses for tool execution.*

---

### B. Multi-Token Prediction (MTP) Speculative Depth Sweep
Measured on `modelserver01` with `stream_all_min: 64` and `spec_min_p: 0.05` across coding and analytical prompts:

| MTP Draft Depth (`mtp_max`) | Draft Acceptance Rate | Decode Throughput (`tok/s`) | Speedup vs. No MTP |
| :---: | :---: | :---: | :---: |
| `mtp_max = 0` *(Disabled)* | — | 38.4 tok/s | 1.00x |
| `mtp_max = 1` | 89.1% | 61.2 tok/s | 1.59x |
| `mtp_max = 2` | 86.4% | 74.8 tok/s | 1.95x |
| `mtp_max = 3` | 84.7% | 83.5 tok/s | 2.17x |
| **`mtp_max = 4` *(Selected)*** | **83.4% – 88.2%** | **88.6 – 113.8 tok/s** | **2.31x – 2.65x** |

---

### C. Live Multi-Turn Autonomous Agent Telemetry (Hermes & Letta up to 105k Context)

| Context Size (`PROMPT`) | Cached Prefix (`REUSED`) | Cache Reuse % | Output Tokens | Prefill Speed (`t/s`) | Decode Speed (`t/s`) | VRAM Hit Rate (`+PCIe`) | Turn Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `13,313` | `13,313` | **100.0%** | 110 | 1,089 t/s | **113.8 tok/s** | **89.6%** (`+0.8% PCIe`) | 1.8 s |
| `28,561` | `41` *(cold)* | 0.1% | 1,641 | **2,144 t/s** | **77.4 tok/s** | **76.8%** (`+2.5% PCIe`) | 34.5 s |
| `43,648` | `37,028` | 84.8% | 2,188 | **2,127 t/s** | **70.5 tok/s** | **75.6%** (`+2.6% PCIe`) | 34.2 s |
| `60,744` | `60,117` | **99.0%** | 1,137 | 659 t/s | **84.2 tok/s** | **74.8%** (`+3.2% PCIe`) | 15.1 s |
| `89,263` | `89,211` | **99.9%** | 447 | 934 t/s | **106.5 tok/s** | **82.4%** (`+2.3% PCIe`) | 4.9 s |
| `104,899` | `104,845` | **99.9%** | 115 | 1,028 t/s | **91.2 tok/s** | **80.7%** (`+2.2% PCIe`) | 1.9 s |

---

## 4. Repository Structure

```text
hydragen/
├── pyproject.toml                   # PEP 621 build metadata & hydragen entrypoint
├── setup.sh / setup.ps1 / setup.bat # One-click automated setup & updater
├── run.sh / run.ps1 / run.bat       # Cross-platform server launchers
├── hydragen.json                    # Configuration template (customizable)
├── src/
│   └── hydragen/                    # Core Hydragen Python library
│       ├── __init__.py              # Package exports
│       ├── cli.py                   # hydragen CLI launcher
│       ├── engine.py                # VLLMEngine (Triton MoE integration & multi-slot scheduler)
│       ├── expert_manager.py        # PagedExpertManager (VRAM cache, LRU, async DMA)
│       ├── gguf_loader.py           # Dynamic GGUF loader & CUDA kernel bridge
│       ├── triton_moe.py            # Triton fused_moe_indexed_slot_kernel & forward passes
│       ├── vllm_plugin.py           # vLLM general plugin registration
│       ├── expert_locality_policy.py# Expert-locality queue reordering
│       ├── cpu_fallback.py          # Hybrid CPU fallback for low-frequency experts
│       └── server/                  # API server & Web Dashboard
│           ├── __init__.py          # Server exports
│           ├── server.py            # API server (OpenAI, Anthropic, live Concurrency States)
│           ├── frontend.py          # Protocol parsing & prompt template adapters
│           ├── telemetry.py         # Hardware & per-slot inference telemetry
│           ├── libstrata_iq_kernels.so # Precompiled Ampere/Ada CUDA kernels
│           └── web/                 # Web UI (Chat, Concurrency States, Monitor)
├── benchmarks/
│   └── benchmark_concurrency.py     # Multi-stream concurrency benchmark driver
├── cuda/
│   ├── setup_cuda.py                # Standalone builder for CUDA kernel libraries
│   └── hydragen_bridge.cu           # CUDA kernel bridge source
├── tests/
│   ├── test_end_to_end_triton_vllm.py # End-to-end plugin + GGUF + Triton MoE verification
│   ├── test_reasoning_loop_recovery.py# Loop recovery & small-cap budget tests
│   └── ...                          # Additional unit & integration tests
├── LICENSE                          # Apache-2.0 License
├── NOTICE                           # Attribution notices
└── README.md
```

---

## 5. Quick Installation & Usage

### Prerequisites
- **NVIDIA GPU** with 12 GB+ VRAM (RTX 3060 12GB, RTX 3080 Ti, RTX 3090/4090, RTX 4070 Ti+, etc.).
- **Host RAM:** 32 GB – 64 GB recommended for locking the full `IQ3_XXS` expert pack in system memory.
- **Python:** 3.10+ (with CUDA support).

---

### Automated One-Click Setup & Updater

> [!TIP]
> **For Autonomous AI Agents:** If an AI coding agent (Antigravity, Claude Code, Cursor, Devin) is installing or managing this service, refer to [**`AGENTS.md`**](AGENTS.md) for non-interactive preflight checks, automated polling scripts, and headless testing recipes.

Clone the repository and run the setup script for your platform:

**Linux / WSL2:**
```bash
git clone https://github.com/thirteenthd/Hydragen.git
cd Hydragen
chmod +x setup.sh && ./setup.sh
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/thirteenthd/Hydragen.git
cd Hydragen
.\setup.ps1
```

**Windows (CMD):**
```cmd
git clone https://github.com/thirteenthd/Hydragen.git
cd Hydragen
setup.bat
```

The script automatically detects your GPU, creates a `.venv` virtual environment, installs PyTorch with CUDA and Hydragen, downloads the precompiled engine binary if needed, and generates a ready-to-run `hydragen.json` configuration.

---

### Updating Hydragen

To update an existing installation to the latest release at any time:

```bash
./setup.sh --update      # Linux / WSL2
.\setup.ps1 -u           # Windows PowerShell
setup.bat -u             # Windows CMD
```

---

### Launching the Server

Start the server using the launcher script:
```bash
./run.sh      # Linux / WSL2
.\run.ps1     # Windows PowerShell
run.bat       # Windows CMD
```

Or invoke directly via the `hydragen` CLI entrypoint:
```bash
hydragen --config hydragen.json --port 8080 --api-monitor
```

### Endpoints
- **Web Chat & Live Monitor:** `http://localhost:8080/#monitor`
- **OpenAI Chat Completions:** `POST http://localhost:8080/v1/chat/completions`
- **Anthropic Messages API:** `POST http://localhost:8080/v1/messages`
- **OpenAI Responses API:** `POST http://localhost:8080/v1/responses`
- **Live Telemetry JSON:** `GET http://localhost:8080/metrics`

---

## 6. Engine Selection & Workload Tuning Guide

Hydragen provides two powerful execution backends. **Both engines fully support multi-slot concurrency (`--batch <N>`), the enhanced scheduler with interactive prompt preemption, small-cap reasoning budget guards, and the live Concurrency States dashboard:**

| Flag | Engine Name | Execution Paradigm | When to Use |
| :--- | :--- | :--- | :--- |
| **`--engine vllm`** *(Default)* | **Hydragen Hybrid Engine** | Continuous multi-slot batching with **Triton FusedMoE expert paging**, in-kernel dequantization, dynamic PCIe Gen4 DMA streaming, and Jaccard expert-affinity queue reordering. | High-concurrency agent workflows (Hermes, Letta, Cursor, multi-subagent teams) where dynamic expert paging and GPU-accelerated routing maximize aggregate throughput. |
| **`--engine strata`** | **Enhanced Strata Multi-Slot Engine** | Native C++ multi-slot batching (`BGEN`/`BT`/`BDONE`) with pinned host-RAM expert arena, dynamic solo MTP promotion, and interactive chunk yielding. Fully upgraded to Strata v0.2.9 (engine 0.1.55) with automated 9-knob parallel kernel tuning (grouped Gated DeltaNet linear attention, direct slot DMA copy, checkpoint pooling) and Tensor Core acceleration when `--batch >= 2`. | High-performance multi-slot or solo inference using the native C++ pipeline worker. Supports concurrency whenever VRAM headroom permits slot allocation. |

**Command-line examples:**
```bash
# Launch with Hydragen hybrid Triton PagedMoE engine (2 concurrent slots):
hydragen --engine vllm --batch 2 --config hydragen.json

# Launch with Enhanced Strata multi-slot engine (2 concurrent slots):
hydragen --engine strata --batch 2 --config hydragen.json
```

---

### Workload Parameter Tuning

All settings can be specified in `hydragen.json` or passed as command-line arguments to `hydragen`:

#### 1. Concurrency Slots (`--batch <N>`)
* **`--batch 1` (Solo Mode):** Single stream execution. Allows dynamic solo MTP promotion for maximum single-turn speed (up to **113.8 tok/s**).
* **`--batch 2` (Recommended Default for 12 GB GPUs):** Sustains 2 concurrent streams at **102–118 combined tok/s** without risking VRAM overflow at 128k context.
* **`--batch 3` or `4`:** Best for multi-agent workloads with higher VRAM cards (16 GB–24 GB, e.g. RTX 3090/4090/4080) or when using shorter context windows (32k–64k). Any excess requests wait cleanly in the FIFO queue displayed on the monitor.

#### 2. Context Length & KV Cache Tiering
* **`--max-context <tokens>`:** Sets the total sequence length (e.g., `32768`, `65536`, `131072`).
* **`--kv int8`:** Quantizes KV cache to 8-bit, cutting KV memory consumption by 50% so a 128k context fits on 12 GB GPUs.
* **`--kv-resident <positions>`:** Specifies how many token positions are pinned in fast GPU VRAM (e.g., `32768`), with remainder paged to host RAM.

#### 3. Speculative Decoding Depth (`--spec <N>` / MTP)
* **`--spec 4` (Recommended):** Enables 4-layer Multi-Token Prediction (MTP) speculative drafting, giving a **2.3x–2.65x speedup** on code and structured generation tasks.
* **`--spec-min-p 0.05`:** Lowers the draft acceptance probability threshold, allowing speculative drafting to engage even on shorter agent turns.
* **`--spec 0`:** Disables speculative decoding if you need to minimize memory pressure.

#### 4. VRAM Reservation Headroom (`--vram-reserve-mib <MiB>`)
* **`--vram-reserve-mib 1152`:** Sets aside dedicated VRAM headroom for Triton Fused MoE kernels, dynamic expert swap buffers, and CUDA graph capture. If you experience out-of-memory errors on 12 GB GPUs, increase this value to `1280` or `1536`.

#### 5. Dynamic PCIe Expert Streaming (`--pcie-frac <ratio>`)
* **`--pcie-frac 0.20`:** Allows the GPU to stream up to 20% of routed experts across PCIe Gen4 DMA in parallel with computation when expert slots are crowded, hiding memory latency and boosting hit rates.

#### 6. Expert Cache Routing Bias (`STRATA_ROUTE_PRIOR`)
* **`STRATA_ROUTE_PRIOR=0.8` (Environment Variable):** Biases top-10 expert routing toward currently resident VRAM experts. Keeps cache hit rate at **75%–89.6%** without sacrificing reasoning accuracy.

---

## 7. Attribution & Acknowledgments

Hydragen builds upon and brings together two foundational open-source AI projects:

* **[Strata](https://github.com/Niko1221/Strata)** by Niko Veit ([@Niko1221](https://github.com/Niko1221)): For the tiered expert memory architecture, pinned host-RAM expert arena offloading, GSQ sub-4-bit quantization (`IQ3_XXS`/`IQ2_XS`/`Q2_0_STRATA`), and speculative MTP decoding on consumer hardware.
* **[vLLM](https://github.com/vllm-project/vllm)** by the vLLM Team: For the continuous batching architecture, multi-slot scheduling paradigms, PagedAttention KV management, and high-throughput Triton FusedMoE execution.

Hydragen is released under the [Apache-2.0 License](LICENSE).
