#!/usr/bin/env python3
"""Hydragen Automated Performance Benchmark Suite.

Executes standardized benchmark sweeps across:
1. Prefill Scaling Sweep (1k, 4k, 8k, 16k, 32k prompt tokens):
   - Measures TTFT (Time-to-First-Token) and Prefill tok/s.
2. Decode Profile Sweep:
   - Cold Start (First 50 tokens on fresh domain prompt): measures initial VRAM hit-rate and cold tok/s.
   - Sustained Generation (250 tokens): measures peak decode tok/s and MTP speculative draft acceptance.
3. Concurrency Stress (C=1 vs C=2 slots):
   - Measures parallel throughput and slot contention.

Pulls ground-truth telemetry directly from Hydragen's `/metrics` and `/health` endpoints.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkRunRecord:
    timestamp: str
    suite_name: str
    target_url: str
    hardware_gpu: str
    hardware_cpu: str
    bar1_rebar_mb: Optional[int]
    vram_expert_slots: int
    test_type: str
    prompt_tokens: int
    output_tokens: int
    concurrency: int
    ttft_client_s: float
    decode_client_tok_s: float
    server_prefill_tok_s: float
    server_decode_tok_s: float
    vram_hit_rate: float
    pcie_share: float
    mtp_acceptance_rate: float


def get_json(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "HydragenBench/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: Dict[str, Any], timeout: float = 180.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "HydragenBench/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate_synthetic_prompt(token_count: int, salt: str = "") -> str:
    """Generate a realistic synthetic prompt to hit exact token targets."""
    # Each code chunk is ~60 tokens
    code_unit = (
        "def transform_tensor_block(tensor_id: int, matrix_a: list, matrix_b: list) -> dict:\n"
        "    \"\"\"Compute blockwise dot-product and apply fused activation scaling.\"\"\"\n"
        "    filtered = [x for x in matrix_a if x > 0.0]\n"
        "    scale_factor = sum(filtered) / (len(matrix_b) + 1e-6)\n"
        "    out_vector = [val * scale_factor for val in matrix_b]\n"
        "    return {'tensor_id': tensor_id, 'norm': scale_factor, 'data': out_vector[:16]}\n\n"
    )
    units_needed = max(1, token_count // 60)
    body = code_unit * units_needed
    unique_tag = f"// Benchmark Salt: {salt or time.time_ns()}\n"
    return (
        f"{unique_tag}"
        f"Below is a high-performance numerical kernel module containing {units_needed} execution routines.\n"
        f"Review the algorithmic logic and cache implications.\n\n"
        f"{body}\n\n"
        f"Provide a concise technical assessment of the memory access pattern."
    )


def execute_completion_request(
    base_url: str,
    prompt_tokens: int,
    max_tokens: int,
    temperature: float = 0.6,
    salt: str = "",
) -> Dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    prompt = generate_synthetic_prompt(prompt_tokens, salt=salt)
    payload = {
        "model": "qwen3.8-flash-next-iq3_xxs",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    t0 = time.perf_counter()
    t_first = None
    output_tokens = 0

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            msg = line[6:]
            if msg == "[DONE]":
                break
            try:
                chunk = json.loads(msg)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    if t_first is None:
                        t_first = time.perf_counter()
                    output_tokens += 1
            except Exception:
                continue

    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end

    ttft = t_first - t0
    decode_duration = max(1e-4, t_end - t_first)
    decode_tok_s = output_tokens / decode_duration if output_tokens > 0 else 0.0

    return {
        "ttft_s": ttft,
        "decode_duration_s": decode_duration,
        "decode_tok_s": decode_tok_s,
        "output_tokens": output_tokens,
    }


def fetch_latest_server_request_metrics(base_url: str) -> Dict[str, Any]:
    """Fetch the latest request metrics from /metrics."""
    try:
        metrics = get_json(f"{base_url.rstrip('/')}/metrics")
        reqs = metrics.get("requests", [])
        if reqs:
            return reqs[0]
    except Exception:
        pass
    return {}


def run_benchmark_suite(
    base_url: str,
    output_dir: Path,
    suite_tag: str = "baseline",
    run_prefill_sweep: bool = True,
    run_decode_sweep: bool = True,
    run_concurrency: bool = True,
) -> Dict[str, Any]:
    print(f"\n========================================================================")
    print(f" HYDRAGEN AUTOMATED PERFORMANCE BENCHMARK SUITE")
    print(f" Target: {base_url} | Suite Tag: {suite_tag}")
    print(f"========================================================================")

    # 1. Health & Environment Validation
    health = get_json(f"{base_url.rstrip('/')}/health")
    metrics = get_json(f"{base_url.rstrip('/')}/metrics")

    hw_static = metrics.get("hardware_static", {})
    gpu_name = hw_static.get("gpu_name", "Unknown GPU")
    cpu_name = hw_static.get("cpu_name", "Unknown CPU")
    engine_cfg = metrics.get("engine", {})
    expert_slots = engine_cfg.get("expert_slots", 0)

    print(f"GPU Detected   : {gpu_name}")
    print(f"CPU Detected   : {cpu_name}")
    print(f"Engine Model   : {health.get('model')}")
    print(f"VRAM MoE Slots : {expert_slots} resident experts ({engine_cfg.get('expert_cache_mib', 0)} MiB)")
    print(f"Max Context    : {health.get('max_context')} tokens")
    print("------------------------------------------------------------------------\n")

    timestamp_iso = datetime.now(timezone.utc).isoformat()
    results: List[BenchmarkRunRecord] = []

    # 2. Prefill Scaling Sweep
    if run_prefill_sweep:
        print("[1/3] Running Prefill Scaling Sweep (1k, 4k, 8k, 16k, 32k)...")
        prefill_sizes = [1024, 4096, 8192, 16384, 32768]
        for p_size in prefill_sizes:
            print(f"  -> Testing Prefill Size ~{p_size:,} tokens (Cold Prompt)...", end="", flush=True)
            res = execute_completion_request(
                base_url=base_url,
                prompt_tokens=p_size,
                max_tokens=8,
                salt=f"prefill_sweep_{p_size}_{time.time_ns()}",
            )
            time.sleep(1.0)
            srv = fetch_latest_server_request_metrics(base_url)

            rec = BenchmarkRunRecord(
                timestamp=timestamp_iso,
                suite_name=suite_tag,
                target_url=base_url,
                hardware_gpu=gpu_name,
                hardware_cpu=cpu_name,
                bar1_rebar_mb=None,
                vram_expert_slots=expert_slots,
                test_type="prefill_sweep",
                prompt_tokens=srv.get("prompt_tokens", p_size),
                output_tokens=res["output_tokens"],
                concurrency=1,
                ttft_client_s=round(res["ttft_s"], 3),
                decode_client_tok_s=round(res["decode_tok_s"], 1),
                server_prefill_tok_s=round(srv.get("prefill_tok_s", 0.0), 1),
                server_decode_tok_s=round(srv.get("decode_tok_s", 0.0), 1),
                vram_hit_rate=round(srv.get("hit_rate", 0.0), 3),
                pcie_share=round(srv.get("pcie_share", 0.0), 3),
                mtp_acceptance_rate=round(
                    (srv.get("drafts_accepted", 0) / max(1, srv.get("drafts_offered", 1))), 3
                ),
            )
            results.append(rec)
            print(f" Done! TTFT: {rec.ttft_client_s}s | Prefill: {rec.server_prefill_tok_s} tok/s")

    # 3. Decode Profile Sweep
    if run_decode_sweep:
        print("\n[2/3] Running Decode Profile Sweep (Cold vs Sustained)...")

        # Test A: Cold Start first 50 tokens
        print("  -> Testing Cold-Start First 50 Tokens (Cold Domain Switch)...", end="", flush=True)
        res_cold = execute_completion_request(
            base_url=base_url,
            prompt_tokens=2048,
            max_tokens=50,
            salt=f"cold_decode_{time.time_ns()}",
        )
        time.sleep(1.0)
        srv_cold = fetch_latest_server_request_metrics(base_url)
        rec_cold = BenchmarkRunRecord(
            timestamp=timestamp_iso,
            suite_name=suite_tag,
            target_url=base_url,
            hardware_gpu=gpu_name,
            hardware_cpu=cpu_name,
            bar1_rebar_mb=None,
            vram_expert_slots=expert_slots,
            test_type="decode_cold_50",
            prompt_tokens=srv_cold.get("prompt_tokens", 2048),
            output_tokens=res_cold["output_tokens"],
            concurrency=1,
            ttft_client_s=round(res_cold["ttft_s"], 3),
            decode_client_tok_s=round(res_cold["decode_tok_s"], 1),
            server_prefill_tok_s=round(srv_cold.get("prefill_tok_s", 0.0), 1),
            server_decode_tok_s=round(srv_cold.get("decode_tok_s", 0.0), 1),
            vram_hit_rate=round(srv_cold.get("hit_rate", 0.0), 3),
            pcie_share=round(srv_cold.get("pcie_share", 0.0), 3),
            mtp_acceptance_rate=round(
                (srv_cold.get("drafts_accepted", 0) / max(1, srv_cold.get("drafts_offered", 1))), 3
            ),
        )
        results.append(rec_cold)
        print(f" Done! Decode: {rec_cold.server_decode_tok_s} tok/s | Hit Rate: {rec_cold.vram_hit_rate * 100:.1f}%")

        # Test B: Sustained 250 tokens
        print("  -> Testing Sustained 250 Tokens (Full MTP Pipelining)...", end="", flush=True)
        res_sust = execute_completion_request(
            base_url=base_url,
            prompt_tokens=2048,
            max_tokens=250,
            salt=f"sustained_decode_{time.time_ns()}",
        )
        time.sleep(1.0)
        srv_sust = fetch_latest_server_request_metrics(base_url)
        rec_sust = BenchmarkRunRecord(
            timestamp=timestamp_iso,
            suite_name=suite_tag,
            target_url=base_url,
            hardware_gpu=gpu_name,
            hardware_cpu=cpu_name,
            bar1_rebar_mb=None,
            vram_expert_slots=expert_slots,
            test_type="decode_sustained_250",
            prompt_tokens=srv_sust.get("prompt_tokens", 2048),
            output_tokens=res_sust["output_tokens"],
            concurrency=1,
            ttft_client_s=round(res_sust["ttft_s"], 3),
            decode_client_tok_s=round(res_sust["decode_tok_s"], 1),
            server_prefill_tok_s=round(srv_sust.get("prefill_tok_s", 0.0), 1),
            server_decode_tok_s=round(srv_sust.get("decode_tok_s", 0.0), 1),
            vram_hit_rate=round(srv_sust.get("hit_rate", 0.0), 3),
            pcie_share=round(srv_sust.get("pcie_share", 0.0), 3),
            mtp_acceptance_rate=round(
                (srv_sust.get("drafts_accepted", 0) / max(1, srv_sust.get("drafts_offered", 1))), 3
            ),
        )
        results.append(rec_sust)
        print(
            f" Done! Decode: {rec_sust.server_decode_tok_s} tok/s | "
            f"Hit Rate: {rec_sust.vram_hit_rate * 100:.1f}% | "
            f"MTP Accept: {rec_sust.mtp_acceptance_rate * 100:.1f}%"
        )

    # 4. Concurrency Stress (C=2 slots)
    if run_concurrency:
        print("\n[3/3] Running Concurrent Dual-Stream Scaling (C=2 Slots)...")
        print("  -> Dispatching 2 concurrent streams...", end="", flush=True)
        t_wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(
                execute_completion_request, base_url, 2048, 100, 0.6, f"c2_stream1_{time.time_ns()}"
            )
            f2 = executor.submit(
                execute_completion_request, base_url, 2048, 100, 0.6, f"c2_stream2_{time.time_ns()}"
            )
            r1 = f1.result()
            r2 = f2.result()
        t_wall_end = time.perf_counter()
        total_tokens = r1["output_tokens"] + r2["output_tokens"]
        aggregate_tok_s = round(total_tokens / max(1e-4, (t_wall_end - t_wall_start)), 1)

        rec_c2 = BenchmarkRunRecord(
            timestamp=timestamp_iso,
            suite_name=suite_tag,
            target_url=base_url,
            hardware_gpu=gpu_name,
            hardware_cpu=cpu_name,
            bar1_rebar_mb=None,
            vram_expert_slots=expert_slots,
            test_type="concurrency_c2",
            prompt_tokens=2048,
            output_tokens=total_tokens,
            concurrency=2,
            ttft_client_s=round(max(r1["ttft_s"], r2["ttft_s"]), 3),
            decode_client_tok_s=aggregate_tok_s,
            server_prefill_tok_s=0.0,
            server_decode_tok_s=aggregate_tok_s,
            vram_hit_rate=0.0,
            pcie_share=0.0,
            mtp_acceptance_rate=0.0,
        )
        results.append(rec_c2)
        print(f" Done! Aggregate 2-Slot Throughput: {aggregate_tok_s} tok/s")

    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"benchmark_{suite_tag}_{int(time.time())}.json"
    data_out = {
        "metadata": {
            "timestamp": timestamp_iso,
            "suite_tag": suite_tag,
            "gpu": gpu_name,
            "cpu": cpu_name,
            "base_url": base_url,
            "expert_slots": expert_slots,
        },
        "records": [asdict(r) for r in results],
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data_out, f, indent=2)

    print("\n========================================================================")
    print(f" BENCHMARK COMPLETE — Results saved to {out_file}")
    print("========================================================================")
    print_markdown_summary(results)
    return data_out


def print_markdown_summary(results: List[BenchmarkRunRecord]) -> None:
    print("\n### Benchmark Summary Table\n")
    print("| Test Type | Prompt (tok) | Out (tok) | Conc | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | VRAM Hit % | MTP Accept % |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        vram_str = f"{r.vram_hit_rate * 100:.1f}%" if r.vram_hit_rate > 0 else "-"
        mtp_str = f"{r.mtp_acceptance_rate * 100:.1f}%" if r.mtp_acceptance_rate > 0 else "-"
        prefill_str = f"{r.server_prefill_tok_s:.1f}" if r.server_prefill_tok_s > 0 else "-"
        print(
            f"| {r.test_type} | {r.prompt_tokens:,} | {r.output_tokens} | {r.concurrency} | "
            f"{r.ttft_client_s}s | {prefill_str} | {r.server_decode_tok_s:.1f} | {vram_str} | {mtp_str} |"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hydragen Automated Performance Benchmark Suite")
    parser.add_argument(
        "--url",
        default=os.environ.get("HYDRAGEN_URL", "http://192.168.7.152:8080"),
        help="Base URL of Hydragen server (default: http://192.168.7.152:8080)",
    )
    parser.add_argument(
        "--tag",
        default="baseline",
        help="Identifier tag for this benchmark run (e.g. baseline, lpt_attention, post_prefill_adapt)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Directory to save benchmark JSON artifacts",
    )
    parser.add_argument("--skip-prefill", action="store_true", help="Skip prefill scaling sweep")
    parser.add_argument("--skip-decode", action="store_true", help="Skip decode profile sweep")
    parser.add_argument("--skip-concurrency", action="store_true", help="Skip concurrency test")

    args = parser.parse_args()

    try:
        run_benchmark_suite(
            base_url=args.url,
            output_dir=Path(args.output_dir),
            suite_tag=args.tag,
            run_prefill_sweep=not args.skip_prefill,
            run_decode_sweep=not args.skip_decode,
            run_concurrency=not args.skip_concurrency,
        )
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Connection failed to {args.url}: {e}")
        print("Ensure the Hydragen server is active and reachable before benchmarking.")
        sys.exit(1)


if __name__ == "__main__":
    main()
