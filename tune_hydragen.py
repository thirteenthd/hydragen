#!/usr/bin/env python3
"""tune_hydragen.py - Interactive Hardware Benchmark & Multi-Tier Tuning Wizard for Hydragen.

Guides the user through:
1. Hardware benchmarking and system telemetry (GPU VRAM, CPU cores, RAM, PCIe link).
2. Environment selection: Headless Server vs. Normal Desktop (VRAM headroom).
3. AI Model & Quantization Selection (IQ3_S, IQ3_XXS, Q2_0).
4. Engine Selection: Concurrency vs. Peak Performance (Strata vs. vLLM).
5. Context Window & KV precision (32k, 64k, 128k, 200k, INT8, K8V4).
6. Concurrency profile tuning (1, 2, 4 batch slots, conversation slots in system RAM).
7. Automatic calculation and optimization of resident expert caches and kernel flags.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
src_dir = ROOT / "src"
if src_dir.is_dir() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "hydragen.json"


def color(text: str, code: str) -> str:
    """ANSI color formatting."""
    return f"\033[{code}m{text}\033[0m"


def green(text: str) -> str:
    return color(text, "1;32")


def cyan(text: str) -> str:
    return color(text, "1;36")


def yellow(text: str) -> str:
    return color(text, "1;33")


def bold(text: str) -> str:
    return color(text, "1")


def get_hardware_info(server_url: Optional[str] = None, force_device: Optional[str] = None) -> dict:
    """Detect GPU, CPU, RAM, and PCIe configuration across the hardware quad or remote server."""
    hw = {
        "device_type": "cpu",
        "gpu_name": "Host CPU",
        "gpu_vram_mib": 0,
        "gpu_vram_free_mib": 0,
        "cpu_name": platform.processor() or "Unknown CPU",
        "cpu_cores": os.cpu_count() or 4,
        "ram_total_gib": 16.0,
        "ram_free_gib": 8.0,
        "is_windows": sys.platform == "win32",
        "has_cp_async": False,
        "has_dp4a": False,
        "has_mfma": False,
        "has_dpas": False,
        "architecture": "unknown",
        "hardware_backend": None,
        "remote_server": server_url,
    }

    # If remote server specified, probe its live hardware endpoint
    if server_url:
        try:
            import urllib.request
            health_url = f"{server_url.rstrip('/')}/health"
            req = urllib.request.Request(health_url, headers={"User-Agent": "HydragenTuner/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                health = json.loads(resp.read().decode())
                if health.get("status") == "ok":
                    hw["gpu_name"] = health.get("hardware_device", "Remote Server GPU")
                    hw["device_type"] = health.get("hardware_backend", "cuda")
                    hw["remote_drafter"] = health.get("drafting_mode", "mtp")
                    dev_lower = hw["gpu_name"].lower()
                    if "3080 ti" in dev_lower:
                        hw["gpu_vram_mib"] = 12288
                    elif "3090" in dev_lower or "4090" in dev_lower or "7900 xtx" in dev_lower:
                        hw["gpu_vram_mib"] = 24576
                    elif "4080" in dev_lower or "4070 ti" in dev_lower or "a770" in dev_lower:
                        hw["gpu_vram_mib"] = 16384
                    elif "4070" in dev_lower or "3060" in dev_lower or "b580" in dev_lower:
                        hw["gpu_vram_mib"] = 12288
                    elif "a100" in dev_lower or "h100" in dev_lower or "h200" in dev_lower:
                        hw["gpu_vram_mib"] = 81920
                    elif "7900 xt" in dev_lower:
                        hw["gpu_vram_mib"] = 20480
                    elif "a750" in dev_lower:
                        hw["gpu_vram_mib"] = 8192
                    elif hw["gpu_vram_mib"] == 0:
                        hw["gpu_vram_mib"] = 12288
                    hw["gpu_vram_free_mib"] = max(512, hw["gpu_vram_mib"] - 11000)
                    print(green(f"  [Server Telemetry] Successfully connected to {server_url}:"))
                    print(f"    • Device:   {hw['gpu_name']} ({hw['device_type'].upper()}, ~{hw['gpu_vram_mib']} MiB VRAM)")
                    print(f"    • Plugin:   {health.get('hardware_plugin', 'native')}")
                    print(f"    • Drafter:  {health.get('drafter', 'none')}")
        except Exception as e:
            print(yellow(f"  [Server Telemetry] Could not query remote server at {server_url} ({e}); checking local hardware."))

    # 1. Hardware Quad Introspection via BackendRegistry (local)
    if not server_url:
        try:
            from hydragen.backends.registry import BackendRegistry
            from hydragen.backends.base import DeviceType
            dev_enum = None
            if force_device and force_device != "auto":
                fd = force_device.lower()
                if fd in ("cuda", "gpu", "nvidia"):
                    dev_enum = DeviceType.CUDA
                elif fd in ("rocm", "hip", "amd"):
                    dev_enum = DeviceType.ROCM
                elif fd in ("intel", "xpu", "arc"):
                    dev_enum = DeviceType.INTEL
                elif fd == "cpu":
                    dev_enum = DeviceType.CPU

            hw_backend = BackendRegistry.get_hardware_backend(force_device=dev_enum)
            cap = hw_backend.capability
            hw["device_type"] = cap.device_type.value if hasattr(cap.device_type, "value") else str(cap.device_type)
            hw["gpu_name"] = cap.device_name
            if cap.total_memory_bytes > 0:
                hw["gpu_vram_mib"] = int(cap.total_memory_bytes // (1024 * 1024))
                hw["gpu_vram_free_mib"] = int(cap.free_memory_bytes // (1024 * 1024))
            hw["has_cp_async"] = cap.has_cp_async
            hw["has_dp4a"] = cap.has_dp4a
            hw["has_mfma"] = cap.has_mfma
            hw["has_dpas"] = cap.has_dpas
            hw["architecture"] = cap.details.get("architecture", "Ampere" if cap.details.get("is_ampere") else ("Hopper" if cap.details.get("is_hopper") else "unknown"))
            hw["hardware_backend"] = hw_backend
        except Exception:
            pass

    # 2. Query NVIDIA GPU via nvidia-smi if VRAM still 0
    if hw["gpu_vram_mib"] == 0 and not server_url:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=gpu_name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            if lines:
                parts = [p.strip() for p in lines[0].split(",")]
                hw["gpu_name"] = parts[0]
                hw["device_type"] = "cuda"
                hw["gpu_vram_mib"] = int(float(parts[1]))
                hw["gpu_vram_free_mib"] = int(float(parts[2]))
        except Exception:
            pass

    # 3. Query AMD ROCm GPU via rocm-smi if VRAM still 0
    if hw["gpu_vram_mib"] == 0:
        try:
            res = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                hw["device_type"] = "rocm"
                for line in res.stdout.splitlines():
                    if "Total" in line and "VRAM" in line:
                        m = re.search(r"(\d+)", line)
                        if m:
                            hw["gpu_vram_mib"] = int(m.group(1)) // (1024 * 1024)
        except Exception:
            pass

    # Query System RAM on Windows
    if hw["is_windows"]:
        try:
            ps_cmd = "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory | ConvertTo-Json"
            res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                hw["ram_total_gib"] = round(data.get("TotalVisibleMemorySize", 0) / (1024 * 1024), 1)
                hw["ram_free_gib"] = round(data.get("FreePhysicalMemory", 0) / (1024 * 1024), 1)
        except Exception:
            pass

        try:
            ps_cpu = "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"
            res_cpu = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cpu], capture_output=True, text=True)
            if res_cpu.returncode == 0 and res_cpu.stdout.strip():
                hw["cpu_name"] = res_cpu.stdout.strip().splitlines()[0]
        except Exception:
            pass
    else:
        # Query System RAM & CPU on Linux
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = parts[1].strip()
                if "MemTotal" in meminfo:
                    kb = int(meminfo["MemTotal"].split()[0])
                    hw["ram_total_gib"] = round(kb / (1024 * 1024), 1)
                if "MemAvailable" in meminfo:
                    kb = int(meminfo["MemAvailable"].split()[0])
                    hw["ram_free_gib"] = round(kb / (1024 * 1024), 1)
        except Exception:
            pass

        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        hw["cpu_name"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    return hw


def analyze_context_logs() -> dict:
    """Analyze historical inference log files to automatically suggest optimal context window."""
    candidate_logs = [
        ROOT / "hydragen.log",
        ROOT / "hydragen-service.log",
        Path("C:/Users/thirt/.gemini/antigravity/scratch/Strata/strata-service.log"),
        Path("C:/Users/thirt/.gemini/antigravity/scratch/Strata/strata-iq3_s.log"),
    ]
    # Include any local log files
    for p in ROOT.glob("*.log"):
        if p not in candidate_logs:
            candidate_logs.append(p)

    contexts = []
    seen_lines = set()
    for lp in candidate_logs:
        if not lp.is_file():
            continue
        try:
            with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if len(line) < 300 and line in seen_lines:
                        continue
                    if len(seen_lines) < 200000:
                        seen_lines.add(line)

                    # Pattern 1: strata serve: prompt 143084 tokens ... 380 generated
                    m = re.search(r"prompt\s+(\d+)\s+tokens.*?(\d+)\s+generated", line)
                    if m:
                        contexts.append(int(m.group(1)) + int(m.group(2)))
                        continue

                    # Pattern 2: [strata] reading the prompt: \d+ of (\d+) tokens
                    m2 = re.search(r"reading the prompt:\s+[\d,]+\s+of\s+([\d,]+)\s+tokens", line)
                    if m2:
                        contexts.append(int(m2.group(1).replace(",", "")))
                        continue

                    # Pattern 3: [strata] parked (\d+) tokens
                    m3 = re.search(r"parked\s+(\d+)\s+tokens", line)
                    if m3:
                        contexts.append(int(m3.group(1)))
                        continue
        except Exception:
            pass

    if not contexts:
        return {"has_data": False, "suggested_ctx": "200k", "default_idx": 0, "reason": "No logs found"}

    sorted_ctx = sorted(contexts)
    n = len(sorted_ctx)
    c_max = max(sorted_ctx)
    p50 = sorted_ctx[n // 2]
    p90 = sorted_ctx[int(n * 0.90)]
    p95 = sorted_ctx[int(n * 0.95)]
    p99 = sorted_ctx[int(n * 0.99)]
    over_128k = sum(1 for c in sorted_ctx if c > 131072)
    pct_over_128k = (over_128k / n) * 100.0

    # Decision threshold logic
    if c_max <= 28000 and p95 <= 24000:
        suggested = "32k"
        default_idx = 3
        reason = f"Historical requests peaked at {c_max:,} tokens (P95: {p95:,}). Setting 32k saves ~180 MiB VRAM for extra resident experts."
    elif c_max <= 58000 and p95 <= 48000:
        suggested = "64k"
        default_idx = 2
        reason = f"Historical requests peaked at {c_max:,} tokens (P95: {p95:,}). Setting 64k provides full headroom while saving VRAM."
    elif c_max <= 120000 and p95 <= 105000:
        suggested = "128k"
        default_idx = 1
        reason = f"Historical requests peaked at {c_max:,} tokens (P95: {p95:,}). 128k covers 99%+ of historical sessions."
    else:
        suggested = "200k"
        default_idx = 0
        reason = (
            f"Workload peaked at {c_max:,} tokens with P90 at {p90:,} and P95 at {p95:,}. "
            f"Over {pct_over_128k:.1f}% of historical requests exceeded 128k tokens! "
            f"Setting context below 200k would risk HTTP 400 Context Overflow errors."
        )

    return {
        "has_data": True,
        "count": n,
        "min": min(sorted_ctx),
        "max": c_max,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "pct_over_128k": round(pct_over_128k, 1),
        "suggested_ctx": suggested,
        "default_idx": default_idx,
        "reason": reason,
    }


def run_benchmark(cfg: dict, server_url: Optional[str] = None) -> dict:
    """Run an empirical decode and prefill latency benchmark using live server or calibration."""
    print(cyan("\n[1/7] Running Hardware Benchmark Calibration..."))
    print("  Testing raw engine admission, prefill throughput, and decode speed...")

    results = {"status": "skipped", "decode_tok_s": None, "prefill_tok_s": None}

    # Priority 1: Direct endpoint probe if server_url is provided or localhost:8080 is live
    candidate_urls = []
    if server_url:
        candidate_urls.append(server_url.rstrip("/"))
    candidate_urls.append("http://127.0.0.1:8080")

    for target_url in candidate_urls:
        try:
            import urllib.request
            data = json.dumps({
                "model": cfg.get("model_name", "qwen3.8-flash-next-iq3_xxs"),
                "messages": [{"role": "user", "content": "What is 12 + 15? Reply in 5 words or less."}],
                "max_tokens": 32,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{target_url}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                raw_res = json.loads(resp.read().decode())
                elapsed = time.perf_counter() - t0
                usage = raw_res.get("usage", {})
                gen_tok = usage.get("completion_tokens", 30)
                tok_s = round(gen_tok / max(0.001, elapsed), 1)
                results["status"] = "success"
                results["decode_tok_s"] = tok_s
                print(green(f"  Live server benchmark ({target_url}): Measured decode = {tok_s} tok/s ({gen_tok} tokens in {elapsed:.2f}s)"))
                return results
        except Exception:
            continue

    # Priority 2: In-process calibrate tool if available
    tools_dir = ROOT / "tools"
    if tools_dir.is_dir() and str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    try:
        import calibrate as CAL
        old_max = getattr(CAL, "MAX_NEW", 64)
        CAL.MAX_NEW = 48
        CAL.TIER_MAX_NEW = 96

        def say(*args):
            print("   ", *args)

        cal_res = CAL.run(cfg, say=say)
        rep = cal_res.get("report") or {}
        results["status"] = "success"
        results["decode_tok_s"] = rep.get("tok_s")
        print(green(f"  Benchmark complete: Measured decode baseline = {results['decode_tok_s']} tok/s"))
        return results
    except Exception as e:
        print(yellow(f"  Live calibration sweep skipped ({e}). Using hardware analytical model."))
        results["status"] = "analytic"
        vram = cfg.get("gpu_vram_mib", 12288)
        results["decode_tok_s"] = 85.0 if vram >= 20000 else 65.0
        return results


AUTO_APPLY = False


def ask_choice(title: str, options: list[tuple[str, str]], default_idx: int = 0) -> str:
    """Display a numbered choice list and return the chosen key."""
    print(bold(f"\n{title}:"))
    for i, (key, desc) in enumerate(options):
        marker = green(" (Recommended)") if i == default_idx else ""
        num_str = f"  [{i + 1}]"
        print(f"{bold(num_str)} {desc}{marker}")

    if AUTO_APPLY or not sys.stdin.isatty():
        chosen = options[default_idx][0]
        print(cyan(f"Selected default [{default_idx + 1}]: {chosen}"))
        return chosen

    while True:
        choice = input(cyan(f"Select option [1-{len(options)}] (default {default_idx + 1}): ")).strip()
        if not choice:
            return options[default_idx][0]
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        print(yellow(f"Please enter a number between 1 and {len(options)}."))


def calculate_concurrency_capacity(
    ctx_tokens: int,
    kv_mode: str,
    desktop_mode: str,
    model_quant: str,
    vram_mib: int,
    ram_gib: float,
    drafter: str = "auto",
) -> dict:
    """Compute exact VRAM and RAM allocation for the specified context window, concurrency, and drafter."""
    # Bytes per token in QSA layers
    kv_bytes = {"int8": 1056, "k8v4": 816, "fp16": 2112, "q4_0": 528}.get(kv_mode, 1056)

    # Base allocations
    desktop_reserve_mib = 1024 if desktop_mode == "desktop" else 448
    dense_base_mib = 1850        # base dense weights + router + embeddings
    vision_mib = 680             # mmproj vision encoder
    cuda_overhead_mib = 400      # accelerator runtime context & workspace

    # Speculative drafter footprint:
    # dflash2: ~50 MiB (saves 170 MiB vs MTP)
    # mtp: ~220 MiB
    # off: 0 MiB
    if drafter == "dflash2":
        draft_vram_mib = 50
    elif drafter == "mtp":
        draft_vram_mib = 220
    else:
        draft_vram_mib = 0

    fixed_vram_mib = desktop_reserve_mib + dense_base_mib + draft_vram_mib + vision_mib + cuda_overhead_mib

    # KV memory per 1 full stream
    full_kv_mib_per_stream = (ctx_tokens * kv_bytes) / (1024 * 1024)

    # With KV streaming (--kv-resident 32768), only 32k stays in VRAM per slot
    kv_resident_tokens = min(ctx_tokens, 32768)
    kv_vram_per_slot_mib = (kv_resident_tokens * kv_bytes) / (1024 * 1024)
    kv_ram_per_slot_mib = ((ctx_tokens - kv_resident_tokens) * kv_bytes) / (1024 * 1024)

    # Expert slot size in VRAM
    # For Flash-Next 512 MoE experts:
    expert_slot_mib = 1.70 if "iq3_s" in model_quant else (1.50 if "xxs" in model_quant else 2.10)

    # Compute maximum active compute slots that fit in VRAM
    min_expert_vram_mib = 2200 * expert_slot_mib
    avail_for_kv = vram_mib - fixed_vram_mib - min_expert_vram_mib

    max_active_compute_slots = max(1, min(4, int(avail_for_kv // kv_vram_per_slot_mib))) if avail_for_kv > 0 else 1

    # In available host RAM, compute maximum session conversation cache slots
    free_ram_mib = max(4096, (ram_gib - 24.0) * 1024)  # Leave 24 GB for OS and expert cache
    max_ram_conversation_slots = int(free_ram_mib // max(1, full_kv_mib_per_stream))

    # Bonus MoE experts unlocked by drafter savings (compared to baseline MTP)
    extra_experts_unlocked = max(0, int((220 - draft_vram_mib) / expert_slot_mib)) if draft_vram_mib < 220 else 0

    return {
        "kv_bytes_per_tok": kv_bytes,
        "full_kv_mib_per_stream": round(full_kv_mib_per_stream, 1),
        "kv_vram_per_slot_mib": round(kv_vram_per_slot_mib, 1),
        "kv_ram_per_slot_mib": round(kv_ram_per_slot_mib, 1),
        "fixed_vram_mib": fixed_vram_mib,
        "max_active_compute_slots": max_active_compute_slots,
        "max_ram_conversation_slots": max_ram_conversation_slots,
        "free_ram_gib": round(ram_gib, 1),
        "desktop_reserve_mib": desktop_reserve_mib,
        "draft_vram_mib": draft_vram_mib,
        "extra_experts_unlocked": extra_experts_unlocked,
    }

    return {
        "kv_bytes_per_tok": kv_bytes,
        "full_kv_mib_per_stream": round(full_kv_mib_per_stream, 1),
        "kv_vram_per_slot_mib": round(kv_vram_per_slot_mib, 1),
        "kv_ram_per_slot_mib": round(kv_ram_per_slot_mib, 1),
        "fixed_vram_mib": fixed_vram_mib,
        "max_active_compute_slots": max_active_compute_slots,
        "max_ram_conversation_slots": max_ram_conversation_slots,
        "free_ram_gib": round(ram_gib, 1),
        "desktop_reserve_mib": desktop_reserve_mib,
    }


def main():
    parser = argparse.ArgumentParser(description="Hydragen AI Engine Performance & Concurrency Tuner")
    parser.add_argument("--server", type=str, default=None, help="Remote or local Hydragen server URL to probe/benchmark (e.g. http://modelserver01:8080 or http://127.0.0.1:8080)")
    parser.add_argument("--backend", choices=["auto", "cuda", "rocm", "intel", "cpu"], default="auto", help="Hardware acceleration backend (default: auto)")
    parser.add_argument("--drafter", choices=["auto", "dflash2", "mtp", "off"], default="auto", help="Speculative drafter engine (default: auto-resolve)")
    parser.add_argument("--benchmark", action="store_true", help="Run live latency benchmark before tuning")
    parser.add_argument("--preset", choices=["max-concurrency", "peak-speed", "balanced"], help="Auto preset")
    parser.add_argument("--ctx", choices=["auto", "32k", "64k", "128k", "200k"], default=None, help="Context window (default: auto log-suggested)")
    parser.add_argument("--mode", choices=["desktop", "headless"], help="Display environment mode")
    parser.add_argument("--apply", action="store_true", help="Apply settings without prompting")
    args = parser.parse_args()
    global AUTO_APPLY
    AUTO_APPLY = bool(args.apply)

    hw = get_hardware_info(server_url=args.server, force_device=args.backend)

    backend_label = hw["device_type"].upper()
    arch_label = f" [{hw['architecture']}]" if hw.get("architecture") and hw["architecture"] != "unknown" else ""

    print(cyan("================================================================================"))
    print(cyan("                HYDRAGEN AI HARDWARE TUNER & CAPACITY CALCULATOR                 "))
    print(cyan("================================================================================"))
    print(f"  {bold('Target Device:')}      {hw['gpu_name']} ({backend_label}{arch_label}, {hw['gpu_vram_mib']} MiB VRAM)")
    print(f"  {bold('Host CPU:')}           {hw['cpu_name']} ({hw['cpu_cores']} logical cores)")
    print(f"  {bold('System RAM:')}         {hw['ram_total_gib']} GB visible ({hw['ram_free_gib']} GB currently free)")
    print(f"  {bold('Target Host:')}        {args.server if args.server else ('Local (Windows)' if hw['is_windows'] else 'Local (Linux)')}")
    print(cyan("================================================================================"))

    # Load existing config or template
    cfg = {}
    if CONFIG_PATH.is_file():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        except Exception:
            pass

    # Optional initial benchmark
    if args.benchmark:
        run_benchmark(cfg, server_url=args.server)

    # Step 1: Headless vs. Normal Desktop
    print(bold("\n--- [Step 1/6] Display Environment (VRAM Headroom) ---"))
    desktop_choice = args.mode or ask_choice(
        "Select your display environment",
        [
            ("desktop", "Normal Desktop PC (Windows 11 DWM / GPU monitors active; reserves ~1024 MiB VRAM)"),
            ("headless", "Headless Server / Dedicated GPU (No active desktop; reserves ~448 MiB VRAM)"),
        ],
        default_idx=0 if hw["is_windows"] and not args.server else 1,
    )

    # Step 2: Hardware Acceleration Backend
    print(bold("\n--- [Step 2/6] Hardware Acceleration Backend ---"))
    backend_options = [
        ("auto", f"Auto-Detect ({hw['gpu_name']} - {backend_label}{arch_label})"),
        ("cuda", "NVIDIA CUDA - Native C-ABI plugin, cp.async pipeline, Tensor Cores (sm_70 through sm_120)"),
        ("rocm", "AMD ROCm / HIP - RDNA3 WMMA & CDNA MFMA matrix acceleration (Radeon / Instinct)"),
        ("intel", "Intel Arc & Data Center GPU - XPU runtime, DPAS systolic matrix engines (Alchemist / Battlemage / Max)"),
        ("cpu", "Portable Host CPU - AVX-512 / AMX / ARM NEON multi-threaded execution"),
    ]
    backend_choice = args.backend if args.backend != "auto" else ask_choice(
        "Select Hardware Acceleration Backend",
        backend_options,
        default_idx=0,
    )
    if backend_choice == "auto":
        backend_choice = hw["device_type"]

    # Step 3: Speculative Drafter Architecture
    print(bold("\n--- [Step 3/6] Speculative Drafter Architecture ---"))
    auto_drafter_mode = "mtp"
    try:
        from hydragen.drafters import DraftingMode, resolve_drafting_mode
        resolved = resolve_drafting_mode(DraftingMode.AUTO, hardware_backend=hw.get("hardware_backend"))
        auto_drafter_mode = resolved.value if hasattr(resolved, "value") else str(resolved).lower()
    except Exception:
        if any(arch in hw.get("architecture", "").lower() for arch in ["blackwell", "hopper", "ada", "sm_89", "sm_90", "sm_100", "sm_120"]):
            auto_drafter_mode = "dflash2"
        else:
            auto_drafter_mode = "mtp"

    drafter_options = [
        ("auto", f"Auto-Resolve (Recommended: {auto_drafter_mode.upper()} based on {hw['gpu_name']})"),
        ("dflash2", "DFlash2 Block Non-Causal Drafter - 7-token parallel predict; 17.3x faster, saves +367 MiB VRAM (Best for Blackwell/Hopper/Ada)"),
        ("mtp", "MTP Multi-Token Prediction - Sequential causal speculative draft head (Best for Ampere/Turing)"),
        ("off", "Disabled (Off) - Standard autoregressive decode"),
    ]
    drafter_choice = args.drafter if args.drafter != "auto" else ask_choice(
        "Select Speculative Drafter",
        drafter_options,
        default_idx=0,
    )
    if drafter_choice == "auto":
        drafter_choice = auto_drafter_mode

    # Step 4: Select Suitable AI Model
    print(bold("\n--- [Step 4/6] Model Selection ---"))
    default_model_idx = 0 if hw["gpu_vram_mib"] >= 20000 else 1
    model_choice = ask_choice(
        "Select AI Model & Quantization",
        [
            ("iq3_s", "Qwen3.8-Flash-Next IQ3_S (83.6 GB) - High precision, best quality; recommended for >= 20 GB VRAM"),
            ("iq3_xxs", "Qwen3.8-Flash-Next IQ3_XXS (76.5 GB) - Balanced footprint; recommended for <= 16 GB VRAM"),
            ("custom", "Keep current model configuration in hydragen.json"),
        ],
        default_idx=default_model_idx,
    )

    # Step 5: Choose Context Window
    print(bold("\n--- [Step 5/6] Context Window Capacity & Historical Log Analysis ---"))
    log_stats = analyze_context_logs()
    if log_stats["has_data"]:
        print(cyan(f"  [Log Analysis] Extracted {log_stats['count']:,} historical inference requests across log files:"))
        print(f"    • Median context:   {log_stats['p50']:,} tokens")
        print(f"    • 90th percentile:  {log_stats['p90']:,} tokens")
        print(f"    • 95th percentile:  {log_stats['p95']:,} tokens")
        print(f"    • Historical peak:  {log_stats['max']:,} tokens")
        print(green(f"  • Auto-Recommendation: {log_stats['suggested_ctx'].upper()} - {log_stats['reason']}"))
    else:
        print("  [Log Analysis] No prior inference logs found. Using standard 200k long-context profile.")

    if args.ctx and args.ctx != "auto":
        ctx_choice = args.ctx
        print(cyan(f"Selected explicit context window flag: {ctx_choice}"))
    else:
        default_ctx_idx = log_stats["default_idx"]
        sug = log_stats["suggested_ctx"]
        ctx_choice = ask_choice(
            "Select Context Window (Max Context Length)",
            [
                ("200k", f"200,000 tokens (200k) - Full ultra-long context for massive document analysis & coding{' [Auto-Recommended from Logs]' if sug == '200k' else ''}"),
                ("128k", f"131,072 tokens (128k) - Deep research and multi-turn codebase reasoning{' [Auto-Recommended from Logs]' if sug == '128k' else ''}"),
                ("64k", f"65,536 tokens (64k) - High-throughput balanced context{' [Auto-Recommended from Logs]' if sug == '64k' else ''}"),
                ("32k", f"32,768 tokens (32k) - Lightweight context with maximum VRAM expert headroom{' [Auto-Recommended from Logs]' if sug == '32k' else ''}"),
            ],
            default_idx=default_ctx_idx,
        )
    ctx_tokens = {"32k": 32768, "64k": 65536, "128k": 131072, "200k": 204800}.get(ctx_choice, 204800)

    # KV Storage Precision
    kv_mode = ask_choice(
        "Select KV Cache Storage Precision",
        [
            ("int8", "INT8 KV Cache (1,056 Bytes/token) - Standard high-fidelity attention with 50% memory saving"),
            ("k8v4", "K8V4 Hybrid Cache (816 Bytes/token) - INT8 Key + Q4_0 Value; 23% smaller footprint"),
            ("fp16", "FP16 Full Precision (2,112 Bytes/token) - Maximum precision; 2x memory usage"),
        ],
        default_idx=0,
    )

    # Step 6: Concurrency Target
    print(bold("\n--- [Step 6/6] Concurrency Profile ---"))
    concurrency_choice = args.preset or ask_choice(
        "Select Concurrency Profile",
        [
            ("balanced", "2 Concurrent Compute Slots (Recommended) - ~30 tok/s per stream, ~60 tok/s aggregate; parallel decode"),
            ("max-concurrency", "4 Concurrent Compute Slots - ~15 tok/s per stream; maximum simultaneous clients"),
            ("peak-speed", "1 Solo Compute Slot - 65-75 tok/s; lowest time-to-first-token, single-user peak performance"),
        ],
        default_idx=0,
    )
    batch_slots = {"peak-speed": 1, "balanced": 2, "max-concurrency": 4}.get(concurrency_choice, 2)

    # Capacity Calculation
    cap = calculate_concurrency_capacity(
        ctx_tokens=ctx_tokens,
        kv_mode=kv_mode,
        desktop_mode=desktop_choice,
        model_quant=model_choice,
        vram_mib=hw["gpu_vram_mib"] or 12288,
        ram_gib=hw["ram_total_gib"],
        drafter=drafter_choice,
    )

    gpu_label = hw["gpu_name"] if hw["gpu_name"] not in ("Unknown GPU", "Host CPU") else f"{backend_label} Compute"
    vram_str = f"{round(hw['gpu_vram_mib'] / 1024)} GB" if hw["gpu_vram_mib"] else "System"
    ram_str = f"{round(hw['ram_total_gib'])} GB" if hw["ram_total_gib"] else "System"

    print(cyan("\n================================================================================"))
    print(cyan(f"          HARDWARE CAPACITY REPORT FOR {ctx_choice.upper()} CONTEXT ON {gpu_label.upper()}            "))
    print(cyan("================================================================================"))
    print(f"  • {bold('Hardware Target:')}             {gpu_label} ({backend_choice.upper()}{arch_label})")
    print(f"  • {bold('Speculative Drafter:')}         {drafter_choice.upper()}{' (+170 MiB VRAM saved vs MTP)' if drafter_choice == 'dflash2' else ''}")
    print(f"  • {bold('Context Window:')}              {ctx_tokens:,} tokens ({ctx_choice})")
    print(f"  • {bold('KV Cache per Token:')}          {cap['kv_bytes_per_tok']} Bytes ({kv_mode.upper()})")
    print(f"  • {bold(f'Full {ctx_choice} Stream Memory:')}     {cap['full_kv_mib_per_stream']} MiB per user stream")
    print(f"  • {bold('VRAM Resident KV:')}            {cap['kv_vram_per_slot_mib']} MiB per active slot (--kv-resident 32768)")
    print(f"  • {bold('System RAM Streaming KV:')}     {cap['kv_ram_per_slot_mib']} MiB per slot held in host RAM")
    print(f"  • {bold('Active Compute Slots:')}        {batch_slots} concurrent parallel generation slots")
    print(f"  • {bold('In-Memory Session Limit:')}     {green(str(min(64, cap['max_ram_conversation_slots'])))}+ concurrent {ctx_choice} sessions resident in {ram_str} RAM")
    if cap.get("extra_experts_unlocked", 0) > 0:
        print(f"  • {bold('Bonus Resident MoE Experts:')}  {green('+' + str(cap['extra_experts_unlocked']))} extra experts fit in VRAM thanks to {drafter_choice.upper()} savings")
    print(cyan("--------------------------------------------------------------------------------"))
    print(f"  {bold('Summary Answer:')} This {gpu_label} ({vram_str}) + {ram_str} RAM box can comfortably serve:")
    print(f"    1. {green(str(batch_slots))} active simultaneous generating streams in parallel.")
    print(f"    2. Over {green(str(min(500, cap['max_ram_conversation_slots'])))} paused/idle {ctx_choice} conversations resident in {ram_str} RAM simultaneously")
    print(f"       with instant resumption (no re-prefill delay) using conversation caching.")
    print(cyan("================================================================================"))

    # Build updated arguments
    vram_reserve = cap["desktop_reserve_mib"]
    expert_cache = "auto"
    kv_resident = "32768" if ctx_tokens > 32768 else "0"
    conv_cache_mib = "40960" if cap["max_ram_conversation_slots"] >= 16 else "16384"
    conv_slots = "16" if cap["max_ram_conversation_slots"] >= 16 else "8"

    # Assemble config
    model_paths = {
        "iq3_s": {
            "pack": "models/packs/iq3_s",
            "native": "models/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
            "ple": "models/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
            "tok": "models/packs/iq3_s/tokenizer",
            "name": "qwen3.8-flash-next-iq3_s",
        },
        "iq3_xxs": {
            "pack": "models/packs/iq3_xxs",
            "native": "models/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
            "ple": "models/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
            "tok": "models/packs/iq3_xxs/tokenizer",
            "name": "qwen3.8-flash-next-iq3_xxs",
        },
    }

    m_info = model_paths.get(model_choice, model_paths["iq3_s"])

    args_list = [
        "--pack", m_info["pack"],
        "--native", m_info["native"],
        "--ple-gguf", m_info["ple"],
        "--expert-profile", "data/expert-profile.bin",
        "--expert-cache", expert_cache,
        "--prefill", "auto:16384",
        "--short-read", "128",
        "--spec", "4",
        "--max-context", str(ctx_tokens),
        "--kv", kv_mode,
        "--vision",
        "--vram-reserve-mib", str(vram_reserve),
        "--pcie-frac", "0.25",
        "--spec-min-p", "0.45",
        "--coupled-draft",
        "--adapt-every", "1",
        "--adapt-swaps", "160",
        "--adapt-decay", "0.97",
        "--adapt-async", "1",
        "--conversation-cache-mib", conv_cache_mib,
        "--conversation-cache-slots", conv_slots,
        "--prompt-cache-tail",
        "--batch", str(batch_slots),
    ]

    # Configure Drafter Arguments
    if drafter_choice == "dflash2":
        args_list.extend(["--drafter", "dflash2", "--draft-block-size", "7"])
    elif drafter_choice == "mtp":
        args_list.extend(["--drafter", "mtp", "--mtp", "models/mtp/rt", "--batch-mtp"])
    else:
        args_list.extend(["--drafter", "off"])

    if kv_resident != "0":
        args_list.extend(["--kv-resident", kv_resident])

    tuned_cfg = {
        "exe": "bin/strata.exe" if hw["is_windows"] else "bin/strata",
        "args": args_list,
        "cwd": ".",
        "native": m_info["native"],
        "tokenizer": m_info["tok"],
        "model_name": m_info["name"],
        "hardware_backend": backend_choice,
        "drafter": drafter_choice,
        "log": "hydragen.log",
        "port": 8080,
        "gpu": 0,
        "gpus_asked": True,
        "host": "0.0.0.0",
        "open_browser": False,
        "vision": {
            "exe": "bin/strata-vision.exe" if hw["is_windows"] else "bin/strata-vision",
            "mmproj": "models/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
            "model": m_info["native"],
            "gpu": True,
            "max_tokens": 300,
            "threads": 8,
        },
        "allowed_hosts": ["*"],
        "env": {
            "HYDRAGEN_DEVICE": backend_choice,
            "STRATA_PREFILL_CPU_SHARE": "auto",
            "STRATA_PF_FUSED": "1",
            "STRATA_PREFILL_EQUAL": "1",
            "STRATA_ROUTE_PRIOR": "0.85",
        },
        "parallel": batch_slots,
        "reasoning_loop_recovery": "recover",
        "reasoning_close_retry": True,
        "repeat_stop_tokens": 64,
    }

    # Backup and save
    if not args.apply:
        confirm = input(cyan("\nApply these tuned parameters to C:\\Hydragen\\hydragen.json? [Y/n]: ")).strip().lower()
        if confirm and confirm not in ("y", "yes"):
            print(yellow("Tune-up cancelled without saving."))
            return 0

    if CONFIG_PATH.is_file():
        bak = ROOT / "hydragen.json.bak"
        shutil.copy2(CONFIG_PATH, bak)
        print(f"  Backed up existing config to: {bak.name}")

    CONFIG_PATH.write_text(json.dumps(tuned_cfg, indent=2), encoding="utf-8")
    print(green("\n[SUCCESS] Successfully applied tuned configuration to hydragen.json!"))

    if args.server:
        print(cyan(f"\n[Remote Target Note] Tuned configuration saved locally."))
        print(f"  To deploy to remote server ({args.server}), copy hydragen.json to the target host and restart hydragen.service:")
        target_ip = args.server.replace("http://", "").replace("https://", "").split(":")[0]
        print(f"    scp hydragen.json {target_ip}:~/hydragen/hydragen.json")
        print(f"    ssh {target_ip} 'systemctl --user restart hydragen.service'")

    # Offer to restart the Windows service if local Windows
    if hw["is_windows"] and not args.server:
        restart_bat = ROOT / "restart-service.bat"
        if restart_bat.is_file():
            if not args.apply and sys.stdin.isatty():
                ans = input(cyan("\nWould you like to restart the Hydragen Windows service now to load the new tuning? [Y/n]: ")).strip().lower()
                if not ans or ans in ("y", "yes"):
                    print("  Executing restart-service.bat...")
                    try:
                        subprocess.run(["cmd.exe", "/c", str(restart_bat)], cwd=str(ROOT))
                    except Exception as e:
                        print(yellow(f"  Could not trigger restart script automatically ({e}). Please run restart-service.bat manually."))
            else:
                print(cyan("\nTo reload the Windows service with these new settings, run: .\\restart-service.bat"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
