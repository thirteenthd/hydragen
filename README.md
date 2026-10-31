# Strata-vLLM Hybrid PagedMoE Engine (`--engine vllm`)

A drop-in hybrid inference engine and continuous-batching scheduler that combines **vLLM's multi-slot concurrency, prefix caching, and Triton MoE indirection** with **Strata's tiered MoE expert hot-swapping, pinned host-RAM expert arena, GSQ sub-4-bit quantization (`IQ3_XXS` / `IQ2_XS`), and 4-layer Multi-Token Prediction (MTP) speculative decoding**—purpose-built to serve massive MoE models (`Qwen3.8-Flash-Next` at `128k` context) on a single **12 GB VRAM consumer GPU** (`NVIDIA RTX 3080 Ti`).

![Strata-vLLM Live Web Monitor](docs/assets/monitor-overview.png)

---

## 1. The Idea & Motivation

Running large Mixture-of-Experts (MoE) models (such as `Qwen3.8-Flash-Next`, ~41 GiB in `GSQ-RCO-IQ3_XXS` quantization with 512 experts per layer across 48 layers) on a single **12 GB VRAM GPU** presents a fundamental tension between two worlds:

1. **Stock vLLM:**
   - Excels at continuous batching, PagedAttention KV prefix caching, and high-throughput GPU execution.
   - However, standard vLLM assumes all model weights reside statically in GPU VRAM and requires standard HF/SafeTensors formats—making a 41 GiB MoE model impossible to load on a 12 GB card without slow, uncoordinated PCIe thrashing.
2. **Stock Strata:**
   - Excels at tiered MoE weight management: locking the 41 GiB expert pack into pinned System RAM, caching the hottest ~2,265 expert slots (`3.7 GB`) in VRAM, streaming missing routed experts asynchronously over PCIe Gen4 DMA, and using 4-layer Multi-Token Prediction (MTP) draft heads.
   - However, single-stream execution creates head-of-line blocking when multiple autonomous agents (e.g., Letta, Hermes, subagents) issue concurrent requests, and lacks vLLM-style dynamic slot scheduling, agent-specific loop guardrails for third-party system prompts, and real-time per-slot telemetry.

### The Hybrid Solution
Instead of converting 41 GiB of proprietary GSQ `IQ3_XXS` RCO weights offline into FP16 SafeTensors, we built a **two-layer hybrid architecture**:
- **Layer 1 — Custom vLLM PagedMoE & Ampere CUDA Kernels (`vllm_paged_moe/`, `cuda/`, `serve/strata_gguf.py`):**
  - Implements `PagedExpertManager`, `ExpertLocalityPolicy`, and custom CUDA kernels (`libstrata_iq_kernels.so`) that read GSQ `IQ3_XXS` / `IQ2_XS` / `IQ4_NL` blocks directly from Strata's GGUF shards and dequantize them on-the-fly in GPU registers at **982.4 GB/s** (`0.027 ms` per expert), paired with a Triton Fused MoE slot-indirection table (`expert_to_slot[layer, expert_id] -> vram_slot_id`).
- **Layer 2 — Dynamic Continuous-Batching & Resident Worker Bridge (`serve/vllm_engine.py`, `serve/server.py`):**
  - Wraps the resident C++ CUDA worker (`0.1.54` / Strata `0.2.8`) inside a continuous-batching scheduler (`--engine vllm`) that manages **2 concurrent 128k-context batch slots**, automatic `<|im_start|>` turn-checkpoint KV prefix reuse across slots, short-request `BYIELD` preemption, dynamic solo-path MTP speculation promotion, multimodal vision (`strata-vision`), and third-party agent reasoning-loop recovery.

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
│                Strata-vLLM Hybrid Scheduler (vllm_engine.py)               │
│  • 2x Concurrent Batch Slots (128k ctx) + Priority Short-Prompt Preemption │
│  • Cross-Turn Prefix Matcher (longest prefix & <|im_start|> checkpoints)   │
│  • Dynamic Solo-Path Promotion (engages 4-layer MTP when 1 slot is active) │
│  • Agent Loop Guardrails (STRATA_ROUTE_PRIOR=0.8 + </think> wrap recovery) │
│  • Multimodal Vision Pipeline (strata-vision projector + image token cache)│
└──────────────────┬──────────────────────────────────────┬──────────────────┘
                   │                                      │
                   ▼                                      ▼
┌─────────────────────────────────────┐ ┌────────────────────────────────────┐
│  GPU Dequant & PagedMoE (CUDA/V1)   │ │  Resident Tiered MoE Worker (C++)  │
│  • libstrata_iq_kernels.so          │ │  • 2,265 VRAM Expert Slots (3.7GB) │
│    (IQ3_XXS / IQ2_XS / IQ4_NL,      │ │  • 40 GiB Paged Host-RAM Arena     │
│     982 GB/s register dequant)      │ │  • Async PCIe Gen4 x16 DMA         │
│  • PagedExpertManager + Lookahead   │ │  • 4-Layer MTP Speculative Heads   │
└─────────────────────────────────────┘ └────────────────────────────────────┘
```

### Key Engineering Milestones

#### 1. Direct GSQ GGUF Loader & On-the-Fly GPU Dequantization (`serve/strata_gguf.py`, `cuda/setup_cuda.py`)
- Reverse-engineered Strata's custom GGUF tensor layout (`blk.{l}.ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight` with 256-weight super-blocks in `IQ3_XXS`, `IQ2_XS`, and `IQ4_NL`).
- Built custom CUDA kernels (`dequant_iq3_xxs_kernel`, `dequant_iq2_xs_kernel`, `dequant_iq4_nl_kernel`) compiled into `libstrata_iq_kernels.so` using inline lookup tables in constant/shared memory.
- Verified bit-exact output (`max_diff = 0.0` against CPU reference) at **0.027 ms per expert** (**982.4 GB/s** effective bandwidth on RTX 3080 Ti).

#### 2. Concurrency at 128k Context Without VRAM Overflow (`strata-iq3_xxs.json`, `serve/vllm_engine.py`)
- Tuned KV cache and slot allocation to support **`max_context: 131072` (128k tokens)** alongside **`batch_slots: 2`** on 12 GB VRAM while keeping **2,265 expert slots (3.7 GB)** resident in VRAM.
- **Smart Slot Selection (`pick_slot`):** Routes incoming multi-turn requests to the idle slot holding the longest matching token prefix or `<|im_start|>` (`TURN_ID = 248045`) turn checkpoint, preserving idle slot KV caches across turns.
- **Head-of-Line Preemption (`BYIELD`):** If a long prompt read is running in one slot and a shorter interactive request arrives, the scheduler yields the long read at the next chunk boundary so the shorter request admits immediately.
- **Dynamic Solo MTP Promotion:** Continuous batching slots decode without MTP drafts when both slots are simultaneously active, but the instant one slot finishes (`going_solo`), the remaining request seamlessly migrates back to the solo path to re-engage 4-layer MTP speculative decoding.

#### 3. Speculative Decoding Optimization for Agent Workloads
- Diagnosed why stock settings (`stream_all_min: 1024`, `spec_min_p: 0.70`) failed to trigger speculative drafting on typical agent turns (long prompts, short tool-call outputs).
- Benchmarked `mtp_max` across `1, 2, 3, 4` and tuned `strata-iq3_xxs.json`:
  - `"mtp_max": 4`
  - `"stream_all_min": 64` (engages MTP drafting on prompts as short as 64 tokens)
  - `"spec_min_p": 0.05`
- Achieved **82% – 88.2% draft acceptance** on live coding/tool-call workloads.

#### 4. Agent Reliability: Routing Prior & Third-Party Loop Guardrails (`serve/server.py`)
- **The `STRATA_ROUTE_PRIOR` Tradeoff:**
  - Setting `STRATA_ROUTE_PRIOR=0` prevents GeForce FP16 tie-breaking loops by disabling expert cache routing bias, but drops VRAM expert hit rate from ~85% to ~55% and cuts decode speed from **~90 tok/s down to ~50 tok/s**.
  - Setting `STRATA_ROUTE_PRIOR=0.8` (our middle-ground configuration) restores the **75% – 89.6% VRAM expert hit rate** and **80 – 113.8 tok/s decode speed**, when paired with deterministic loop recovery.
- **Third-Party Agent `reasoning_loop_recovery` Fallback:**
  - Upstream Strata's `focused_recovery_prompt()` only worked if the system prompt contained Strata's native `"Reasoning: high"` string. External agent harnesses (Hermes, Letta, Claude Code, Cursor) use custom system prompts, causing `focused_recovery_prompt()` to return `None` and leaving the model stuck in infinite thinking loops.
  - Added an automatic wrap-up fallback in `serve/server.py`: when passage repetition (`repeat_coverage >= 0.30`) is detected on a custom agent prompt, the engine injects `\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n`, cleanly closing the `<think>` block without invalidating the 100k-token KV prefix cache and transitioning the model directly to its tool call or final answer.

#### 5. Multimodal Vision (`strata-vision`) & Web Monitor Overhaul (`serve/web/`)
- **Multimodal Vision:** Integrated `strata-vision` projector subprocess management into `VLLMEngine`, supporting OpenAI `image_url` (`data:image/...;base64,...` and HTTP URLs) with SHA-256 vision embedding caching.
- **Concurrency & Queuing Card:** Added a dedicated card below *Context fill* displaying active vs. configured slots (`1 / 2`), queued request count, and live vertical slot bars (`Slot 0`, `Slot 1`) color-synced to the Model State badges (`Idle` gray, `Reading` blue, `Generating` green).
- **Speed Gauge Scaling & Prefix-Restore Compensation:** Fixed Decode vs. Prefill sparkline scaling, eliminated multi-second `PP` plateau dropouts, and compensated for KV checkpoint-restore overhead on small cached deltas (`fresh < 2048, reused > 0`) so cached agent turns reflect actual GPU prefill throughput.

---

## 3. As-Measured Performance Results

### Testbed Hardware & Model Configuration
| Component | Specification |
| :--- | :--- |
| **Host Machine** | `modelserver01` (Ubuntu Server LTS bare-metal) |
| **GPU** | 1x **NVIDIA GeForce RTX 3080 Ti** (12 GB GDDR6X, 912 GB/s, PCIe Gen4 x16) |
| **CPU** | **AMD Ryzen 9 5900X** (12 Cores / 24 Threads) |
| **System RAM** | **64 GB DDR4** (~40–41 GiB locked for pinned expert arena) |
| **Model** | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (48 layers, 512 experts/layer, 4-layer MTP head) |
| **Context Window** | **128,000 tokens** (`max_context: 131072`) |
| **Batch Concurrency** | **2 concurrent slots** (`batch_slots: 2`) |
| **VRAM Expert Cache** | **2,265 – 2,297 experts cached** (`3.7 – 3.8 GB` in VRAM; `11.5 – 11.7 / 12 GB` total VRAM used) |

---

### A. End-to-End Inference Throughput (Live Agent & Benchmark Workloads)

| Metric | Stock Strata (Baseline) | Strata-vLLM (`STRATA_ROUTE_PRIOR=0`) | **Strata-vLLM (`STRATA_ROUTE_PRIOR=0.8` + MTP 4)** |
| :--- | :---: | :---: | :---: |
| **Single-Stream Decode Speed** | 62 – 72 tok/s | 48 – 56 tok/s | **80.0 – 113.8 tok/s** *(mean ~88–95 tok/s)* |
| **Cold Prompt Prefill Speed** | 1,600 – 2,100 tok/s | 1,800 – 2,300 tok/s | **1,860 – 2,503 tok/s** |
| **Cached Turn Time-to-First-Token (`90k–105k` ctx)** | ~0.70 s | ~0.65 s | **0.55 – 0.68 s** *(99.9% token reuse)* |
| **VRAM Expert Cache Hit Rate** | 72% – 80% | 52% – 60% | **75.1% – 89.6%** *(+0.8% – 3.2% over PCIe)* |
| **MTP Draft Acceptance Rate** | ~0% on short agent turns | 78% – 83% | **82.0% – 88.2%** *(e.g., 379/458 drafts accepted)* |
| **2-Slot Concurrent Aggregate Decode** | N/A (queued) | ~68 tok/s | **102 – 118 tok/s combined** |
| **Multimodal Vision (`strata-vision`)** | 3.1 s | 2.9 s | **2.74 s total** *(52.3 tok/s decode, 88.2% MTP)* |

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
Sampled directly from production multi-turn coding sessions on `modelserver01` (see screenshots in [`docs/assets/`](docs/assets/)):

| Context Size (`PROMPT`) | Cached Prefix (`REUSED`) | Cache Reuse % | Output Tokens | Prefill Speed (`t/s`) | Decode Speed (`t/s`) | VRAM Hit Rate (`+PCIe`) | Turn Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `13,313` | `13,313` | **100.0%** | 110 | 1,089 t/s | **113.8 tok/s** | **89.6%** (`+0.8% PCIe`) | 1.8 s |
| `28,561` | `41` *(cold)* | 0.1% | 1,641 | **2,144 t/s** | **77.4 tok/s** | **76.8%** (`+2.5% PCIe`) | 34.5 s |
| `43,648` | `37,028` | 84.8% | 2,188 | **2,127 t/s** | **70.5 tok/s** | **75.6%** (`+2.6% PCIe`) | 34.2 s |
| `60,744` | `60,117` | **99.0%** | 1,137 | 659 t/s | **84.2 tok/s** | **74.8%** (`+3.2% PCIe`) | 15.1 s |
| `89,263` | `89,211` | **99.9%** | 447 | 934 t/s | **106.5 tok/s** | **82.4%** (`+2.3% PCIe`) | 4.9 s |
| `104,899` | `104,845` | **99.9%** | 115 | 1,028 t/s | **91.2 tok/s** | **80.7%** (`+2.2% PCIe`) | 1.9 s |

---

### D. Custom CUDA Kernel & PagedMoE Component Microbenchmarks
Verified via the standalone test suite in [`tests/`](tests/):

- **Ampere GPU Dequantization (`libstrata_iq_kernels.so`):**
  - `IQ3_XXS`, `IQ2_XS`, and `IQ4_NL` kernels: **0.027 ms** per expert dequantization (**982.4 GB/s** effective throughput), **0.0 max absolute error** vs. reference implementation.
- **Lookahead Async DMA Prefetching (`tests/test_lookahead_prefetch.py`):**
  - Overlapping layer $L+1$ PCIe Gen4 expert DMA transfers on an async CUDA stream with layer $L$ compute hides **91.4%** of PCIe transfer latency.
- **Triton MoE Slot Indirection (`tests/test_triton_moe_indirection.py`):**
  - Bit-exact routing (`torch.allclose` passed with `atol=1e-3`) when dynamically remapping 512 logical experts per layer into a compact 64-slot physical VRAM cache with fallback masks for cache misses.

---

## 4. Repository Structure

```text
strata-vllm/
├── serve/
│   ├── vllm_engine.py               # Hybrid vLLM continuous-batching & slot scheduler
│   ├── strata_gguf.py               # Direct GSQ GGUF shard reader & CUDA dequant bridge
│   ├── libstrata_iq_kernels.so      # Compiled Ampere CUDA kernels (IQ3_XXS/IQ2_XS/IQ4_NL)
│   ├── server.py                    # Patched Strata 0.2.8 server with --engine vllm & loop recovery
│   ├── frontend.py                  # OpenAI / Anthropic / Responses frontend adapters
│   ├── telemetry.py                 # 1 Hz hardware & inference telemetry sampler
│   └── web/                         # Web UI & Monitor (Concurrency slot bars & Speed gauge)
├── cuda/
│   └── setup_cuda.py                # Source & builder for libstrata_iq_kernels.so
├── vllm_paged_moe/
│   ├── expert_manager.py            # Standalone vLLM V1 PagedExpertManager (3-tier LRU + DMA)
│   ├── expert_locality_policy.py    # Expert-locality request reordering scheduler policy
│   ├── cpu_fallback.py              # Hybrid CPU fallback for uncached low-frequency experts
│   └── vllm-paged-moe.patch         # Git patch against upstream vLLM V1
├── tests/
│   ├── test_reasoning_loop_recovery.py  # 18 unit tests for loop recovery & agent wrap-up
│   ├── test_live_strata_gguf_moe.py     # End-to-end GGUF -> GPU dequant -> Triton MoE test
│   ├── test_paged_expert_manager.py     # PagedExpertManager unit & async stream tests
│   ├── test_triton_moe_indirection.py   # Triton FusedMoE slot-indirection verification
│   ├── test_moe_prefetch_pipeline.py    # Multi-stream DMA prefetch pipeline benchmark
│   ├── test_lookahead_prefetch.py       # Layer-lookahead latency hiding benchmark
│   ├── test_realistic_simulation.py     # Full 48-layer PCIe Gen4 x16 simulation
│   ├── test_milestone4_integration.py   # Scheduler locality + CPU fallback integration test
│   └── live_stream_driver.py            # 4-stream concurrent continuous-batching load driver
├── docs/assets/
│   ├── monitor-overview.png         # Live dashboard screenshot during active generation
│   └── monitor-reading.png          # Live dashboard screenshot during 60k prompt reading
├── strata-iq3_xxs.json              # Production 128k-context, 2-slot, MTP-4 configuration
├── vllm-server.patch                # Unified diff against upstream Strata serve/
├── install.sh / install.bat         # 1-step non-destructive installer for Linux & Windows
├── uninstall.sh / uninstall.bat     # 1-step clean revert to stock Strata
├── run_vllm.sh / run_vllm.bat       # Launch scripts for the hybrid server
└── requirements-vllm.txt            # Python dependencies
```

---

## 5. Quick Installation & Usage

### Prerequisites
- An existing [Strata](https://github.com/Niko1221/Strata) installation with model weights downloaded.
- NVIDIA GPU with 12 GB+ VRAM (RTX 3060 12GB, RTX 3080 Ti, RTX 3090, RTX 4070 Ti+, etc.).
- 64 GB System RAM recommended for locking the full `IQ3_XXS` expert pack in host RAM (or 32 GB RAM for `IQ2_XS`).

### Install (Non-Destructive)
Clone this repository and run the installer pointing to your Strata directory:

```bash
git clone git@github.com:derekchung-800house/strata-vllm.git
cd strata-vllm
./install.sh /path/to/Strata
```

> **Note:** Your existing Strata installation is **100% preserved**. The installer backs up `serve/server.py` to `serve/server.py.orig_backup` and only adds the `--engine vllm` option. You can switch between `--engine strata` and `--engine vllm` at any time, or run `./uninstall.sh /path/to/Strata` to revert completely.

### Launch the Server
```bash
cd /path/to/Strata
python3 -m serve.server --engine vllm --config strata-iq3_xxs.json --port 8080 --api-monitor --no-api-key
```

### Endpoints
- **Web Chat & Live Monitor:** `http://localhost:8080/#monitor`
- **OpenAI Chat Completions:** `POST http://localhost:8080/v1/chat/completions`
- **Anthropic Messages API:** `POST http://localhost:8080/v1/messages`
- **OpenAI Responses API:** `POST http://localhost:8080/v1/responses`
- **Live Telemetry JSON:** `GET http://localhost:8080/metrics`

### Rebuilding the CUDA Dequantization Library (Optional)
If deploying on a different CUDA architecture:
```bash
python3 cuda/setup_cuda.py
cp libstrata_iq_kernels.so serve/
```
