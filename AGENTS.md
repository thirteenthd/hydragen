# Hydragen: Autonomous Agent Installation & Operation Guide (AGENTS.md)

This document provides deterministic, non-interactive instructions for AI coding agents (such as Antigravity, Claude Code, Cursor, Devin, Codex) to install, configure, verify, and operate the **Hydragen** inference engine on Linux, WSL2, or Windows.

---

## 1. System Requirements & Agent Preflight Checks

Before running the installer, execute the following preflight checks:

### Preflight Script (Linux / WSL2)
```bash
python3 -c "
import sys, shutil, subprocess

# 1. Check Python version (Must be >= 3.10 and <= 3.13; PyTorch/Numba does not yet support 3.14)
major, minor = sys.version_info.major, sys.version_info.minor
if not (major == 3 and 10 <= minor <= 13):
    print(f'[!] Current Python {major}.{minor} is incompatible. Python 3.10-3.13 is required.')
    sys.exit(1)
print(f'[✓] Python {major}.{minor} OK')

# 2. Check NVIDIA GPU
smi = shutil.which('nvidia-smi')
if not smi:
    print('[!] nvidia-smi not found. CUDA GPU required.')
    sys.exit(1)
gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']).decode().strip()
print(f'[✓] GPU: {gpu_info}')

# 3. Check Free System RAM (Need >= 48 GB free for full 176B Qwen MoE arena)
import os
mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
mem_gib = mem_bytes / (1024**3)
print(f'[✓] System RAM: {mem_gib:.1f} GiB')
if mem_gib < 32:
    print('[!] Warning: System RAM is less than 32 GiB.')
"
```

> [!IMPORTANT]
> If the host's default `python3` is **Python 3.14+**, `setup.sh` will automatically search for installed Python 3.12 runtimes under `~/.local/share/uv/python/*/bin/python3` or `python3.12`. If none is found, install Python 3.12 before proceeding (`uv python install 3.12` or `sudo apt install python3.12 python3.12-venv`).

---

## 2. Non-Interactive One-Command Installation

Execute the installation non-interactively using the automated setup script.

### Option A: If Model Weights Already Exist on Host
If model weights are already stored at `/path/to/models_directory` (e.g. from an earlier Strata download):

```bash
git clone https://github.com/thirteenthd/hydragen.git
cd hydragen
chmod +x setup.sh run.sh
./setup.sh --model-dir=/path/to/models_directory
```

### Option B: Fresh Install (Models to be Linked/Added Later)
```bash
git clone https://github.com/thirteenthd/hydragen.git
cd hydragen
chmod +x setup.sh run.sh
./setup.sh
```

### Windows PowerShell:
```powershell
git clone https://github.com/thirteenthd/hydragen.git
cd hydragen
.\setup.ps1 -ModelDir "C:\path\to\models_directory"
```

### What `setup.sh` / `setup.ps1` Automatically Does:
1. Detects compatible Python (3.10–3.13) and NVIDIA GPU.
2. Creates `.venv` and installs `hydragen` in editable mode (`pip install -e .`) with PyTorch, Triton, GGUF, Pillow, and regex.
3. Downloads the official precompiled `strata` and `strata-vision` worker binaries (v0.2.9 / engine 0.1.55) into `bin/`.
4. Downloads control modules, tools, and `data/expert-profile.bin`.
5. Links the model directory into `models/` (if `--model-dir` is provided).
6. Generates the default `hydragen.json` configuration with relative paths.

---

## 3. Model Directory Structure & Verification

Hydragen configs (`hydragen.json`) use self-contained relative paths resolved from the repository root. Ensure the `models` directory contains the expected subpaths:

```
hydragen/
├── bin/
│   ├── strata                  # Engine executable (v0.2.9 / 0.1.55)
│   └── strata-vision           # Vision encoder executable
├── data/
│   └── expert-profile.bin      # Precalculated expert activation profiles
├── hydragen.json               # Primary configuration file
└── models/                     # Symlinked or physical directory
    ├── packs/
    │   └── iq3_xxs/
    │       ├── conversions.json
    │       ├── dense.bin       # Dense projection weights (~1.5 GB)
    │       ├── index.txt
    │       ├── native_experts.txt
    │       └── tokenizer/      # Vocabulary, merges, chat template
    ├── models/
    │   ├── IQ3_XXS/
    │   │   ├── Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf (44 GB)
    │   │   └── Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf (27 GB)
    │   └── mmproj-Qwen3.8-Flash-Next-BF16.gguf (Vision encoder, 866 MB)
    └── mtp/
        └── rt/
            ├── dense.bin
            ├── draft_vocab.bin
            └── experts.bin     # 4-layer MTP draft weights (~5 GB)
```

### Agent Path Pre-Flight Verification:
Run this single command to verify all configuration paths before launching:
```bash
test -f bin/strata && \
test -f bin/strata-vision && \
test -f data/expert-profile.bin && \
test -d models/packs/iq3_xxs && \
test -f models/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf && \
test -f models/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf && \
test -d models/mtp/rt && \
test -d models/packs/iq3_xxs/tokenizer && \
echo "ALL CONFIG PATHS VERIFIED VALID"
```

---

## 4. Headless Server Startup & Readiness Polling

Agents running without an interactive terminal should start the server in the background and poll the health endpoint until ready:

### Start Daemon in Background (Linux / WSL2):
```bash
nohup ./run.sh --no-api-key > hydragen_run.log 2>&1 < /dev/null &
```

### Windows Service Management:
On Windows, Hydragen can be registered and run as a permanent background service:
```powershell
# Register and start as a Windows Service (Administrative prompt required):
& "C:\Hydragen\install-service.ps1"

# Safe restart & release port 8080:
& "C:\Hydragen\restart-service.bat"

# Quick status check:
& "C:\Hydragen\status-service.bat"
```

### Non-Interactive Hardware Tuning & Capacity Optimization:
Agents can calibrate and apply optimal settings without manual prompt interaction:
```bash
# Auto-tune with log-derived context recommendation and balanced concurrency:
python tune_hydragen.py --preset balanced --ctx auto --mode desktop --apply

# Auto-tune for high concurrency on 24 GB GPUs with 200k context:
python tune_hydragen.py --preset max-concurrency --ctx 200k --mode desktop --apply
```

### Wait / Poll for Readiness (Python Script):
```python
import time, urllib.request, json, sys

url = "http://127.0.0.1:8080/health"
print("[+] Polling Hydragen startup (loading 40-47 GB into RAM arena takes ~25-45s)...")

start_time = time.time()
while time.time() - start_time < 180:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "ok":
                print(f"[✓] Hydragen is READY in {time.time() - start_time:.1f}s!")
                print(f"    Model: {data.get('model')}, Context: {data.get('max_context')}, Slots: 2")
                sys.exit(0)
    except Exception:
        pass
    time.sleep(3)

print("[-] Timeout waiting for Hydragen to start. Check hydragen_run.log and hydragen.log.")
sys.exit(1)
```

---

## 5. Automated Verification & Smoke Tests

Once ready, execute the following tests to verify correct operation:

### 1. Check Slots & Concurrency (`GET /slots`):
```bash
curl -s http://127.0.0.1:8080/slots
# Expected output: 2 slots with n_ctx=131072, is_processing=false
```

### 2. Basic Chat Completion (`POST /v1/chat/completions`):
```python
import urllib.request, json

req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps({
        "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        "max_tokens": 10,
        "temperature": 0.0
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
res = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
print("Response:", res["choices"][0]["message"]["content"].strip())
```

### 3. Agent Tool Calling Test:
```python
import urllib.request, json

tools = [{
    "type": "function",
    "function": {
        "name": "lookup_stock",
        "description": "Get current stock price",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"]
        }
    }
}]

req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps({
        "messages": [{"role": "user", "content": "What is NVDA stock price?"}],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 100
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
res = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
tool_calls = res["choices"][0]["message"].get("tool_calls", [])
print(f"[✓] Tool calls emitted: {len(tool_calls)}")
if tool_calls:
    print(f"    Function: {tool_calls[0]['function']['name']}, Args: {tool_calls[0]['function']['arguments']}")
```

---

## 6. Troubleshooting & Common Agent Pitfalls

| Issue | Symptom | Agent Remediation |
| :--- | :--- | :--- |
| **Python 3.14 Incompatibility** | `RuntimeError: Cannot install on Python version 3.14` during numba/pytorch build | Run `setup.sh` with Python 3.12 (`~/.local/share/uv/python/*/bin/python3` or `python3.12 -m venv .venv`). `setup.sh` automatically detects and recreates `.venv` if Python is > 3.13. |
| **Port 8080 in use** | `OSError: [Errno 98] Address already in use` | Find and kill the stale process: `fuser -k 8080/tcp` or `pkill -f hydragen`. Alternatively, pass `--port 8081` in `run.sh`. |
| **GPU Out of Memory (OOM)** | CUDA error `out of memory` during start | In `hydragen.json`, ensure `--vram-reserve-mib 1152` and `--batch 2`. On 12GB cards, avoid increasing `--batch` beyond 2. |
| **Missing Model File** | `FileNotFoundError` or engine immediate exit | Check `hydragen.log`. Verify that `models -> /path/to/models` is a valid symlink with `ls -la models`. |
| **Stale Cache / Shims** | `ImportError: cannot import name ...` | Ensure `/home/<user>/Strata` or old drop-in packages are not in `PYTHONPATH`. Hydragen is completely self-contained in its repository root. |
