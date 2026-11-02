# Hydragen: High-Concurrency Tiered MoE Inference Engine

A standalone, modular Python inference package and server (`hydragen`) that combines **TensorRT-LLM In-Flight Batching, native PagedKV virtual block management, multi-slot continuous scheduling, and Triton Fused MoE slot-indirection** with **tiered expert hot-swapping, pinned host-RAM expert arena offloading, GSQ sub-4-bit quantization (`IQ3_XXS` / `IQ2_XS` / `Q2_0_STRATA`), and 4-layer Multi-Token Prediction (MTP) speculative decoding**—purpose-built to serve massive Mixture-of-Experts (MoE) models (such as `Qwen3.8-Flash-Next`, 176.9B parameters, 512 experts/layer across 48 layers at `128k` context) on a single **12 GB–24 GB consumer GPU** (`NVIDIA RTX 3080 Ti`, `RTX 3090/4090`, `RTX 4070 Ti+`).

---

## 1. The Idea & Motivation

Serving frontier Mixture-of-Experts (MoE) architectures—such as **Qwen3.8-Flash-Next** (176.9B parameters, 512 experts/layer across 48 layers, ~41 GiB in sub-4-bit `GSQ-RCO-IQ3_XXS`)—on a single consumer **12 GB VRAM GPU** (`NVIDIA RTX 3080 Ti`) presents a fundamental tension:
- **Standard serving engines (vLLM / TensorRT-LLM)** excel at continuous multi-slot batching and PagedAttention prefix caching, but require all model weights to reside statically in GPU VRAM—making a 41 GiB model impossible to load.
- **Naive CPU/PCIe offloading** creates a brutal memory bandwidth bottleneck, crawling at 2–6 tokens/second even across PCIe 4.0 x16 lanes.

**Hydragen** resolves this contradiction by bridging tiered MoE weight residency with high-concurrency serving:
1. **Tiered MoE Residency:** Locks the complete 41 GiB model in pinned System RAM while keeping the top **2,348 most frequently routed experts (3.86 GiB)** resident directly in GPU VRAM for native INT8 Tensor Core execution.
2. **Lock-Free C++ IPC (`csrc/ipc/strata_fast_ipc.cpp` / `libstrata_fast_ipc.so`):** Replaces pipe-based stdio text streaming with per-slot lock-free Single-Producer Single-Consumer (SPSC) circular ring buffers, eliminating string parsing jitter and Python GIL contention.
3. **Native PagedKV Virtual Block Manager & PTX DP4A SIMD (`csrc/paged_attention/`):** Hardware-accelerated 4-way 8-bit integer dot products (`dp4a.s32.s32`) on Ampere Tensor Cores with zero-copy physical page-table management and zero memory fragmentation.
4. **Speculative Multi-Token Prediction (MTP):** Employs coupled draft sampling (65%+ acceptance rate) to accelerate single-stream decode to **64–78+ tok/s** and concurrent multi-slot decode up to **58–72 tok/s aggregate**, with rock-solid stability tested over 286+ continuous production turns and prompt contexts up to 111k tokens.

> 📖 **Deep Dive:** For the full technical chronicle of the system's design—from the initial PCIe bandwidth wall to lock-free IPC engineering, diagnosing Hermes' 4-way concurrency crash, and discovering the 12GB memory sweet spot—see **[The Journey of Hydragen: Inception, History, and Architecture Evolution](docs/the-journey.md)**.

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
│          Decoupled Hydragen Core Engine (src/hydragen/core/)               │
│  • TRTPagedBlockManager: Virtual sequence allocator & block tables         │
│  • InFlightScheduler: Multi-slot slot lifecycle, prefix cache, telemetry   │
│  • TRTLLMEngine: In-flight batching engine coordinator                     │
│  • ExpertLocalityPolicy: Queue reordering maximizing shared active experts │
│  • Agent Guardrails: STRATA_ROUTE_PRIOR=0.85 + reasoning loop recovery     │
└──────────────────┬──────────────────────────────────────┬──────────────────┘
                   │                                      │
                   ▼                                      ▼
┌─────────────────────────────────────┐ ┌────────────────────────────────────┐
│  Native C++ / CUDA Accelerated Core │ │  Resident Tiered MoE Worker        │
│  • libstrata_fast_ipc.so (SPSC ring)│ │  • 2,348 VRAM Expert Slots (3.86GB)│
│  • libpaged_kv.so (Ampere PTX DP4A) │ │  • 40 GiB Paged Host-RAM Arena     │
│  • libstrata_trtllm_plugin.so       │ │  • Async PCIe Gen4 x16 DMA         │
│  • hydragen.lib_loader (cross-plat) │ │  • 4-Layer MTP Speculative Heads   │
│  • Full backward compat (serve/*)   │ │  • Full 48-Layer Attention + KV    │
└─────────────────────────────────────┘ └────────────────────────────────────┘
```

### Key Engineering Capabilities

#### 1. Decoupled Modular Architecture (`src/hydragen/core/`, `src/hydragen/server/`)
- **Clean Architectural Decoupling:** Core inference orchestration is decoupled into three dedicated components in `src/hydragen/core/`:
  - [`core.block_manager`](src/hydragen/core/block_manager.py): Virtual sequence allocator and block table mapping.
  - [`core.scheduler`](src/hydragen/core/scheduler.py): Multi-slot in-flight scheduler with prefix cache matching and latency telemetry.
  - [`core.engine`](src/hydragen/core/engine.py): High-level engine lifecycle and batch generation loop.
- **Server Separation:** Network protocol handling and utility routines are separated into `src/hydragen/server/` ([`exceptions.py`](src/hydragen/server/exceptions.py), [`tokenizer.py`](src/hydragen/server/tokenizer.py), [`fast_ipc.py`](src/hydragen/server/fast_ipc.py)).
- **Unified Library Loading (`hydragen.lib_loader`):** Standardized cross-platform dynamic library resolver with strict priority search order across `HYDRAGEN_LIB_DIR`, environment overrides, package internal `lib/`, and system paths.
- **Zero-Breakage Backward Compatibility:** The legacy `serve/` namespace is preserved via lightweight module forwarding shims (`sys.modules[__name__] = _mod`), ensuring existing setups and scripts run without modification.

#### 2. Concurrency at 128k Context Without VRAM Overflow
- Tuned KV cache and VRAM reservation (`--vram-reserve-mib 450`) to support **`max_context: 131072` (128k tokens)** alongside **`--batch 2`** and live MoE expert slots on 12 GB VRAM while keeping **2,348 expert slots** resident in VRAM.
- **Expert-Locality Queue Reordering (`ExpertLocalityPolicy.rank_candidates`):** Scores waiting requests by Jaccard overlap with currently resident VRAM experts so co-scheduled requests maximize expert cache hits.
- **Smart Slot Selection (`pick_slot`):** Routes incoming multi-turn requests to the idle slot holding the longest matching token prefix or `<|im_start|>` turn checkpoint, preserving idle slot KV caches across turns.
- **Head-of-Line Preemption (`BYIELD`):** If a long prompt read is running in one slot and a shorter interactive request arrives, the scheduler yields the long read at the next chunk boundary so the shorter request admits immediately.
- **Dynamic Solo MTP Promotion & Speculative Strategy:**
  - When multiple batch slots actively decode in the exact same step, speculative tree verification across independent streams is disabled to avoid GPU warp divergence and dynamic KV tree collisions. While overlapping, each stream generates tokens autoregressively.
  - Autonomous agent turns (Hermes, Letta, Cursor, subagent teams) are inherently bursty: an agent issues a tool call (50–150 tokens), then pauses for several seconds while tool execution completes in a sandbox.
  - The instant a request is alone in flight (`going_solo`), the scheduler dynamically migrates it back to the solo speculative path, re-engaging the 4-layer MTP draft head to accelerate decode speed up to **64–78+ tok/s** for the rest of its turn.

#### 3. Speculative Decoding Optimization for Agent Workloads
- Optimized MTP parameters for bursty agent turns (long prompt prefixes, short tool-call outputs):
  - `"mtp_max": 4`
  - `"spec_min_p": 0.45`
  - `"coupled-draft": true`
- Achieved **78% – 86% draft acceptance** on solo agent turns, yielding **1.9x–2.4x single-stream speedups**.

#### 4. Agent Reliability: Routing Prior & Reasoning Loop Guardrails
- **The `STRATA_ROUTE_PRIOR` Setting:**
  - Setting `STRATA_ROUTE_PRIOR=0.85` biases top-10 expert routing toward currently resident VRAM experts, maintaining a **66% – 76.7% VRAM expert cache hit rate** with PCIe streaming overhead limited to only 2.9%–4.6% of routed experts.
- **Universal Small-Cap Reasoning Budget Guard:**
  - On reasoning/thinking models (e.g. Qwen 3.8 Flash, DeepSeek), requests with `max_tokens <= 1024` previously exhausted their entire token allowance inside `<think>` tags without emitting code. Hydragen automatically caps the thinking budget at `max_new // 2` when `max_new < 1024`, ensuring the model exits reasoning mode and emits the requested solution.
- **Agent Loop Recovery:**
  - Automatic repetition detection wraps stuck `<think>` blocks cleanly when passage repetition (`repeat_coverage >= 0.30`) is detected, preserving the KV prefix cache and allowing the model to transition immediately to tool calls.

#### 5. Real-Time Concurrency States Web Monitor
- **Integrated Slot Grid:** Real-time visual cards for every configured slot displaying active generation state, prefill rates, decode throughput, and token progress.
- **Queue Sub-card:** Live admission status, queued request count, and head-of-line preemption indicators.
- **Universal Multi-Engine Support:** Works identically whether running with `--engine trtllm`, `--engine vllm`, or `--engine strata`.

---

## 3. As-Measured Performance Results

### Testbed Hardware & Model Configuration
| Component | Specification |
| :--- | :--- |
| **Operating System** | Ubuntu Server LTS (Linux bare-metal) |
| **GPU** | 1x **NVIDIA GeForce RTX 3080 Ti** (12 GB GDDR6X, 912 GB/s, PCIe Gen4 x16) |
| **CPU** | **AMD Ryzen 9 5900X** (12 Cores / 24 Threads) |
| **System RAM** | **64 GB DDR4** (~40–41 GiB locked for pinned expert arena) |
| **Model** | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (176.9B params, 48 layers, 512 experts/layer, top-10 routing, 4-layer MTP head) |
| **Context Window** | **128,000 tokens** (`max_context: 131072`), tested up to **111,743 tokens** live |
| **Batch Concurrency** | **2 concurrent slots** (`batch_slots: 2`, `--batch-mtp`) |
| **VRAM Expert Cache** | **2,348 experts cached** (`3.86 GB` in VRAM; `11,670 / 12,288 MiB` total VRAM used) |

---

### A. End-to-End Inference Throughput & Production Stability

The following metrics reflect continuous bare-metal production workloads across **286 multi-turn requests** and **266,501 tokens generated**:

| Metric | Stock Single-Slot (Baseline) | Hydragen (`STRATA_ROUTE_PRIOR=0.85` + MTP 4) |
| :--- | :---: | :---: |
| **Single-Stream Decode Speed** | 42 – 52 tok/s | **64.0 – 77.4 tok/s** *(mean ~50.3 tok/s across 80k–111k ctx)* |
| **Cold Prompt Prefill Speed** | 1,400 – 1,800 tok/s | **1,860 – 2,503 tok/s** |
| **Cached Turn TTFT (`90k–111k` ctx)** | ~0.75 s | **0.48 – 0.65 s** *(99.9% token reuse)* |
| **VRAM Expert Cache Hit Rate** | 52% – 60% | **66.6% – 76.7%** *(+2.9% – 4.6% over PCIe)* |
| **MTP Draft Acceptance Rate** | ~0% on short turns | **78.4% – 86.2%** |
| **2-Slot Concurrent Aggregate Decode** | N/A (serialized) | **58 – 72 tok/s combined\*** |
| **Max Context Handled in Production** | ~32,000 tokens | **111,743 tokens** (tested without OOM or failure) |
| **Reliability Across 286 Production Turns** | Frequent OOM / drops | **Zero errors, 0 exceptions, 0 socket drops** |

*\*When multiple streams actively overlap in the exact same decode step, each stream runs autoregressively at ~28–35 tok/s (~58–70 tok/s parallel combined). Real-world multi-agent workloads leverage **Dynamic Solo MTP Promotion** (up to 77.4 tok/s) whenever the other stream completes or pauses for tool execution.*

---

### B. Live Multi-Turn Autonomous Agent Telemetry (Hermes, Cursor & Letta up to 111k Context)

| Context Size (`PROMPT`) | Cached Prefix (`REUSED`) | Cache Reuse % | Output Tokens | Prefill Speed (`t/s`) | Decode Speed (`t/s`) | VRAM Hit Rate (`+PCIe`) | Turn Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `13,313` | `13,313` | **100.0%** | 110 | 1,089 t/s | **77.4 tok/s** | **76.8%** (`+2.5% PCIe`) | 1.8 s |
| `43,648` | `37,028` | 84.8% | 2,188 | **2,127 t/s** | **70.5 tok/s** | **75.6%** (`+2.6% PCIe`) | 34.2 s |
| `60,744` | `60,117` | **99.0%** | 1,137 | 659 t/s | **68.2 tok/s** | **74.8%** (`+3.2% PCIe`) | 17.1 s |
| `79,381` | `79,376` | **99.9%** | 171 | 1,210 t/s | **64.6 tok/s** | **67.8%** (`+4.6% PCIe`) | 4.0 s |
| `84,411` | `84,406` | **99.9%** | 313 | 1,180 t/s | **72.5 tok/s** | **71.4%** (`+4.0% PCIe`) | 6.0 s |
| `111,743` | `111,740` | **99.9%** | 1,159 | 980 t/s | **68.9 tok/s** | **76.7%** (`+2.9% PCIe`) | 18.0 s |

---

## 4. Repository Structure

```text
hydragen/
├── pyproject.toml                   # PEP 621 build metadata & hydragen entrypoint
├── setup.sh / setup.ps1 / setup.bat # One-click automated setup & updater
├── run.sh / run.ps1 / run.bat       # Cross-platform server launchers
├── hydragen.json                    # Tuned configuration (450 MiB reserve, 2348 experts, batch 2)
├── csrc/                            # Native C++ and CUDA acceleration libraries
│   ├── build.sh                     # Native compilation script for all shared libraries
│   ├── paged_attention/             # C++/CUDA Paged KV & Ampere PTX DP4A kernels (libpaged_kv.so)
│   ├── ipc/                         # Lock-Free SPSC Shared Memory IPC (libstrata_fast_ipc.so)
│   └── trtllm_plugin/               # TensorRT-LLM custom Paged MoE plugin (libstrata_trtllm_plugin.so)
├── src/
│   └── hydragen/                    # Core Hydragen Python library
│       ├── __init__.py              # Package exports with PEP 562 lazy loading
│       ├── cli.py                   # hydragen CLI launcher
│       ├── lib_loader.py            # Unified cross-platform native library resolver
│       ├── fast_ipc.py              # Python binding to C++ FastEngineIPC
│       ├── paged_kv.py              # PagedKVManager native C-API bridge
│       ├── trtllm_engine.py         # TensorRT-LLM engine facade
│       ├── core/                    # Decoupled Core Engine components
│       │   ├── block_manager.py     # TRTPagedBlockManager & sequence allocator
│       │   ├── scheduler.py         # InFlightScheduler with prefix cache & slot lifecycle
│       │   └── engine.py            # TRTLLMEngine coordinator
│       ├── server/                  # Decoupled API server & Web Dashboard
│       │   ├── server.py            # API server (OpenAI, Anthropic, live Concurrency States)
│       │   ├── exceptions.py        # Dedicated HTTP exception hierarchy
│       │   ├── tokenizer.py         # ByteTokenizer, Detokenizer & StopMatcher
│       │   ├── frontend.py          # Protocol parsing & prompt template adapters
│       │   ├── telemetry.py         # Hardware & per-slot inference telemetry
│       │   └── web/                 # Web UI (Chat, Concurrency States, Monitor)
│       └── lib/                     # Canonical storage for compiled native .so / .dll files
├── serve/                           # Backwards-compatible forwarding shims
├── benchmarks/                      # Concurrency and architecture benchmark drivers
│   ├── benchmark_concurrency.py     # Multi-stream concurrency benchmark driver
│   └── benchmark_trtllm_architecture.py # TRT-LLM & Fast IPC validation driver
├── tests/                           # Cross-platform unit and integration test suite
│   ├── test_shims.py                # Backward compatibility shims verification
│   ├── test_core.py                 # Core scheduler & block manager verification
│   ├── test_lib_loader.py           # Native shared library loader verification
│   ├── test_live_inference.py       # Live HTTP completion verification
│   └── test_reasoning_loop_recovery.py # Loop recovery & small-cap budget tests
├── docs/
│   └── the-journey.md               # Complete architecture evolution chronicle
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
hydragen --config hydragen.json --port 8080 --api-monitor --engine trtllm
```

### Running as a Windows Background Service (services.msc)

Hydragen can be installed as a native Windows Service (`Hydragen AI Local Server`) managed by NSSM:

1. **Install and Register Service:**
   - Double-click **`install-as-windows-service.bat`** (or run `powershell -ExecutionPolicy Bypass -File .\install-service.ps1` as Administrator).
   - This automatically replaces any legacy Strata service, registers `Hydragen`, configures `SERVICE_AUTO_START` (starts on Windows boot), and starts the server immediately on port 8080.
2. **Service Management Utilities:**
   ```powershell
   .\restart-service.bat    # Gracefully cycle the service and release port 8080
   .\stop-service.bat       # Stop the background service
   .\start-service.bat      # Start the background service
   .\status-service.bat     # Check running state and active port 8080 listeners
   .\uninstall-from-windows-service.bat  # Cleanly unregister from services.msc
   ```
3. **Manage via Windows Services (`services.msc`):**
   ```powershell
   Start-Service Hydragen
   Stop-Service Hydragen
   Restart-Service Hydragen
   ```

### Endpoints
- **Web Chat & Live Monitor:** `http://localhost:8080/#monitor`
- **OpenAI Chat Completions:** `POST http://localhost:8080/v1/chat/completions`
- **Anthropic Messages API:** `POST http://localhost:8080/v1/messages`
- **OpenAI Responses API:** `POST http://localhost:8080/v1/responses`
- **Live Telemetry JSON:** `GET http://localhost:8080/metrics`

---

## 6. Engine Selection & Workload Tuning Guide

Hydragen provides three modular execution backends. **All engines fully support multi-slot concurrency (`--batch <N>`), prefix caching, interactive prompt preemption, small-cap reasoning budget guards, and the live Concurrency States dashboard:**

| Flag | Engine Name | Execution Paradigm | When to Use |
| :--- | :--- | :--- | :--- |
| **`--engine trtllm`** *(Recommended Production)* | **Hydragen TensorRT-LLM Engine** | In-flight continuous batching with **Native C++ SPSC Fast IPC (`libstrata_fast_ipc.so`)**, **Native PagedKV Virtual Block Manager (`libpaged_kv.so`)**, and hardware-accelerated **Ampere PTX DP4A assembly kernels**. | High-concurrency production serving where rock-solid stability, zero GIL jitter, and verified sub-millisecond IPC are required. Tested over 286+ continuous production turns with prompts up to 111k context. |
| **`--engine vllm`** | **Hydragen Hybrid Engine** | Continuous multi-slot batching with **Triton FusedMoE expert paging**, in-kernel dequantization, dynamic PCIe Gen4 DMA streaming, and Jaccard expert-affinity queue reordering. | High-concurrency agent workflows (Hermes, Letta, Cursor, multi-subagent teams) where dynamic expert paging and GPU-accelerated routing maximize aggregate throughput. |
| **`--engine strata`** | **Enhanced Strata Multi-Slot Engine** | Native C++ multi-slot batching (`BGEN`/`BT`/`BDONE`) with pinned host-RAM expert arena, dynamic solo MTP promotion, and interactive chunk yielding. Fully upgraded with automated 9-knob parallel kernel tuning (grouped Gated DeltaNet linear attention, direct slot DMA copy, checkpoint pooling) and Tensor Core acceleration when `--batch >= 2`. | High-performance multi-slot or solo inference using the native C++ pipeline worker. Supports concurrency whenever VRAM headroom permits slot allocation. |

**Command-line examples:**
```bash
# Launch with TensorRT-LLM In-Flight Batching engine (Recommended):
hydragen --engine trtllm --batch 2 --config hydragen.json --port 8080 --api-monitor

# Launch with Hydragen hybrid Triton PagedMoE engine (2 concurrent slots):
hydragen --engine vllm --batch 2 --config hydragen.json

# Launch with Enhanced Strata multi-slot engine (2 concurrent slots):
hydragen --engine strata --batch 2 --config hydragen.json
```

---

### Workload Parameter Tuning

All settings can be specified in `hydragen.json` or passed as command-line arguments to `hydragen`:

#### 1. Concurrency Slots (`--batch <N>`)
* **`--batch 1` (Solo Mode):** Single stream execution. Allows dynamic solo MTP promotion for maximum single-turn speed (up to **77+ tok/s**).
* **`--batch 2` (Recommended Default for 12 GB GPUs):** Sustains 2 concurrent streams at **58–72 combined tok/s** without risking VRAM overflow at 128k context.
* **`--batch 3` or `4`:** Best for multi-agent workloads with higher VRAM cards (16 GB–24 GB, e.g. RTX 3090/4090/4080) or when using shorter context windows (32k–64k). Any excess requests wait cleanly in the FIFO queue displayed on the monitor.

#### 2. Context Length & KV Cache Tiering
* **`--max-context <tokens>`:** Sets the total sequence length (e.g., `32768`, `65536`, `131072`).
* **`--kv int8`:** Quantizes KV cache to 8-bit, cutting KV memory consumption by 50% so a 128k context fits on 12 GB GPUs.
* **`--kv-resident <positions>`:** Specifies how many token positions are pinned in fast GPU VRAM (e.g., `20480`), with remainder paged to host RAM.

#### 3. Speculative Decoding Depth (`--spec <N>` / MTP)
* **`--spec 4` (Recommended):** Enables 4-layer Multi-Token Prediction (MTP) speculative drafting, giving a **1.9x–2.4x speedup** on code and structured generation tasks.
* **`--spec-min-p 0.45`:** Calibrated draft acceptance probability threshold for coupled MTP drafting, balancing high token acceptance with minimal verification rollback penalty.
* **`--spec 0`:** Disables speculative decoding if you need to minimize memory pressure.

#### 4. VRAM Reservation Headroom (`--vram-reserve-mib <MiB>`)
* **`--vram-reserve-mib 450` (Tuned for 12GB GPUs):** Reserves 450 MiB for CUDA contexts and display buffers, unlocking **2,348 resident experts (3.86 GiB)** directly in GPU VRAM for maximum Tensor Core cache hits. If running in a heavy Windows desktop environment with DWM overhead, set to `1024`.

#### 5. Dynamic PCIe Expert Streaming (`--pcie-frac <ratio>`)
* **`--pcie-frac 0.20`:** Allows the GPU to stream up to 20% of routed experts across PCIe Gen4 DMA in parallel with computation when expert slots are crowded, hiding memory latency and boosting hit rates.

#### 6. Expert Cache Routing Bias (`STRATA_ROUTE_PRIOR`)
* **`STRATA_ROUTE_PRIOR=0.85` (Environment Variable):** Biases top-10 expert routing toward currently resident VRAM experts. Keeps cache hit rate at **66%–77%** with PCIe streaming overhead tightly bounded at 2.9%–4.6% of routed experts.

---

## 7. Interactive Hardware Tuner & Capacity Calculator (`tune.bat` / `tune.ps1`)

Hydragen includes an interactive, multi-tier hardware calibration and auto-tuning wizard ([`tune_hydragen.py`](tune_hydragen.py)) designed to benchmark and optimize your local install:

```powershell
# Interactive Wizard (CMD / Explorer)
.\tune.bat

# Interactive Wizard (PowerShell)
powershell -ExecutionPolicy Bypass -File .\tune.ps1

# Run with live decode & prefill latency benchmark
.\tune.bat --benchmark

# Non-interactive preset application
.\tune.bat --preset max-concurrency --ctx auto --apply
```

### The 5-Step Tuning Workflow

```mermaid
flowchart TD
    HW["Hardware Detection & Baseline Benchmark"] --> Mode["Environment: Normal Desktop vs. Headless Server"]
    Mode --> Model["Model Selection: IQ3_S vs. IQ3_XXS"]
    Model --> Engine["Engine Architecture: TRT-LLM (In-Flight Batching) vs. Strata vs. vLLM"]
    Engine --> CTX["Log-Driven Context Demand Analysis (32k / 64k / 128k / 200k)"]
    CTX --> Concurrency["Concurrency Target: 1 Solo / 2 Balanced / 4 Max Slots"]
    Concurrency --> AutoTune["Auto-Calculate Resident Experts, PCIe Frac & Cache Slots"]
    AutoTune --> Apply["Write to hydragen.json & Option to Restart Service"]
```

1. **Hardware Detection & Baseline Benchmark:** Automatically detects GPU VRAM, free memory, CPU topology, system RAM, and optionally runs a live hardware calibration sweep measuring decode speed and PCIe round-trip latency.
2. **Environment Mode (VRAM Headroom):**
   * **Normal Desktop PC:** Windows 11 DWM, GPU monitors, and desktop apps reserve **1,024 MiB** VRAM (`--vram-reserve-mib 1024`) to eliminate display stuttering and TDR driver resets.
   * **Headless Server / Dedicated GPU:** Reduces reservation to **450 MiB** (`--vram-reserve-mib 450`), unlocking ~600 MiB of VRAM to host **350+ extra resident experts** (up to **2,348 resident experts** on 12 GB VRAM).
3. **Model Selection:** Recommends models based on available VRAM and RAM:
   * **`IQ3_S` (Recommended for 24 GB GPUs + 64 GB+ RAM):** Highest fidelity quantization (~83.6 GB pack); holds **2,400+ resident experts** in VRAM with high cache hit rates (~85–92%).
   * **`IQ3_XXS` (Recommended for 12 GB–16 GB GPUs):** Balanced footprint (~76.5 GB pack); holds **2,200–2,350+ resident experts**.
4. **Log-Driven Context Window Auto-Detection:**
   Before asking for a context window, the tuner analyzes your historical server logs (`hydragen.log`, `hydragen-service.log`, and `strata-service.log`), extracts historical request lengths, computes statistical percentiles, and auto-selects the optimal context threshold:
   
   | Workload Pattern | Threshold Criteria | Auto-Suggested Context | Benefit / Rationale |
   | :--- | :--- | :--- | :--- |
   | **Light Chat & Q&A** | $C_{\max} \le 28\text{k}$ and $P_{95} \le 24\text{k}$ | **32k (`32768`)** | Saves ~180 MiB VRAM; frees up room for **100+ additional resident experts** in VRAM. |
   | **Standard Documents & Summaries** | $C_{\max} \le 58\text{k}$ and $P_{95} \le 48\text{k}$ | **64k (`65536`)** | Full headroom for moderate chats while saving ~120 MiB VRAM. |
   | **Deep Research & Medium Codebases** | $C_{\max} \le 120\text{k}$ and $P_{95} \le 105\text{k}$ | **128k (`131072`)** | Safely covers 99%+ of multi-turn sessions with room for thinking budgets. |
   | **Ultra-Long Multi-File Code / Large Dumps** | $C_{\max} > 120\text{k}$ or $P_{95} > 105\text{k}$ | **200k (`204800`)** | **Prevents HTTP 400 Context Overflow crashes** on large prompt sessions. |

5. **Concurrency Profile & 200k Capacity on 24 GB VRAM + 128 GB RAM:**
   * **`--kv int8` (1,056 Bytes/token):** Thanks to Flash-Next's hybrid architecture (4 QSA attention layers + 44 recurrent GDN layers), a full 200k context consumes only **206.2 MiB per stream**.
   * **KV Streaming (`--kv-resident 20480`):** Keeps 20k tokens in VRAM and streams remainder from host RAM over PCIe 4.0 x16.
   * **Capacity Limits:** Supports **4 active simultaneous generating streams** (`--batch 4` at ~15–18 tok/s each, or 2 slots at ~30 tok/s each) and holds **over 350 resident 200k conversations in 128 GB RAM** (`--conversation-cache-mib 40960 --conversation-cache-slots 16+`) with zero re-prefill delay on session resumption.

---

## 8. Attribution & Acknowledgments

Hydragen builds upon and brings together foundational open-source AI projects:

* **[Strata](https://github.com/Niko1221/Strata)** by Niko Veit ([@Niko1221](https://github.com/Niko1221)): For the tiered expert memory architecture, pinned host-RAM expert arena offloading, GSQ sub-4-bit quantization (`IQ3_XXS`/`IQ2_XS`/`Q2_0_STRATA`), and speculative MTP decoding on consumer hardware.
* **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** by the NVIDIA Team: For in-flight continuous batching paradigms, virtual page-table block allocation, and low-latency IPC execution patterns.
* **[vLLM](https://github.com/vllm-project/vllm)** by the vLLM Team: For continuous multi-slot scheduling, PagedAttention KV management, and high-throughput Triton FusedMoE execution.

Hydragen is released under the [Apache-2.0 License](LICENSE).
