#!/usr/bin/env python3
"""Hydragen Multi-Stream Concurrency Scaling Benchmark.

Tests throughput and latency scaling across concurrency levels C=1, C=2, C=4
using distinct domain prompts to stress MoE expert slot contention.
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
from pathlib import Path
from typing import Any, Dict, List, Tuple


DOMAIN_PROMPTS = [
    # Domain 1: Low-level GPU kernels & memory access
    (
        "Below is a technical review of GPU memory bandwidth and warp divergence in SIMD prefill kernels. "
        "Analyze the cache line transactions, L2 cache hit probability, and register spilling effects in detail: "
        + " void compute_block(float* d_out, const float* d_in, int n) { int tid = blockIdx.x * blockDim.x + threadIdx.x; } " * 80
    ),
    # Domain 2: Medieval European economics & trade routes
    (
        "Discuss the fiscal mechanics and currency debasement of the Hanseatic League and Venetian merchant banking during the 14th century. "
        "Detail the impact of double-entry bookkeeping, maritime insurance contracts, and trade monopolies across the Baltic and Mediterranean seas: "
        + " The commercial revolution transformed merchant capital into financial credit instruments across northern European ports. " * 70
    ),
    # Domain 3: Quantum electrodynamics & field theory
    (
        "Explain the renormalization group equations, Feynman path integrals, and gauge invariance in quantum electrodynamics. "
        "Derive the Ward-Takahashi identities and discuss radiative corrections to the electron magnetic moment: "
        + " Perturbative expansion of the S-matrix involves evaluating one-loop vertex corrections and vacuum polarization diagrams. " * 65
    ),
    # Domain 4: High-frequency market microstructure & order book dynamics
    (
        "Analyze the stochastic modeling of limit order books using Hawkes processes and Fokker-Planck equations. "
        "Explain adverse selection, queue position priority, and market making inventory risk under tick size constraints: "
        + " Matching engines process cancel and limit orders sequentially through price-time priority data structures. " * 70
    ),
    # Domain 5: CRISPR gene editing & synthetic biology
    (
        "Examine the biochemical kinetics of Cas9 endonuclease target recognition, PAM sequence binding, and R-loop propagation. "
        "Compare base editors, prime editors, and HDR-mediated donor template integration for precision genomic modifications: "
        + " Guide RNA hybridization triggers conformational activation of the HNH and RuvC catalytic domains. " * 70
    ),
    # Domain 6: Compiler AST optimizations & LLVM register allocation
    (
        "Detail the graph coloring heuristics and iterated register coalescing algorithms used in modern SSA compiler backends. "
        "Analyze static single assignment phi-node elimination, loop invariant code motion, and polyhedral loop transformations: "
        + " Interference graphs represent live range conflicts across basic block control flow edges. " * 70
    ),
    # Domain 7: Atmospheric thermodynamics & climate dynamics
    (
        "Explain Navier-Stokes primitive equations under hydrostatic approximation and Coriolis force in global circulation models. "
        "Detail baroclinic instability, Rossby wave propagation, and latent heat convection in tropospheric energy budgets: "
        + " Geostrophic balance couples horizontal pressure gradients to planetary vorticity advection. " * 70
    ),
    # Domain 8: Cryptographic lattice problems & post-quantum signatures
    (
        "Discuss the hardness of Learning With Errors (LWE) and Shortest Vector Problem (SVP) in Euclidean lattices. "
        "Analyze the security reductions, modulus switching, and rejection sampling in Dilithium and Kyber cryptographic primitives: "
        + " Module lattices provide algebraic compactness while maintaining worst-case to average-case lattice hardness. " * 70
    ),
]


def execute_stream(
    base_url: str,
    prompt: str,
    max_tokens: int,
    stream_id: int,
) -> Dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "qwen3.8-flash-next-iq3_xxs",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
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
    ttft_s = (t_first - t0) if t_first else (t_end - t0)
    gen_time_s = max(1e-4, t_end - (t_first or t0))
    tok_s = round(output_tokens / gen_time_s, 1)

    return {
        "stream_id": stream_id,
        "output_tokens": output_tokens,
        "ttft_s": round(ttft_s, 3),
        "wall_time_s": round(t_end - t0, 3),
        "decode_tok_s": tok_s,
    }


def run_concurrency_test(
    base_url: str,
    concurrency_levels: List[int],
    max_tokens_per_stream: int = 100,
    tag: str = "concurrency",
) -> Dict[str, Any]:
    print("=" * 86)
    print(f" HYDRAGEN MULTI-STREAM CONCURRENCY SCALING BENCHMARK")
    print(f" Target: {base_url} | Tag: {tag}")
    print(f" Levels to test: {concurrency_levels} streams | Max tokens/stream: {max_tokens_per_stream}")
    print("=" * 86)

    records: List[Dict[str, Any]] = []

    for c in concurrency_levels:
        print(f"\n[+] Testing Concurrency C={c}...")
        prompts = [DOMAIN_PROMPTS[i % len(DOMAIN_PROMPTS)] for i in range(c)]

        # Query metrics before run
        try:
            req_m = urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=5)
            metrics_before = json.loads(req_m.read().decode("utf-8"))
        except Exception:
            metrics_before = {}

        t_wall_start = time.perf_counter()
        stream_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
            futures = [
                executor.submit(execute_stream, base_url, prompts[i], max_tokens_per_stream, i)
                for i in range(c)
            ]
            for fut in concurrent.futures.as_completed(futures):
                stream_results.append(fut.result())

        t_wall_end = time.perf_counter()
        wall_time_s = max(1e-4, t_wall_end - t_wall_start)

        # Query metrics after run
        try:
            req_m = urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=5)
            metrics_after = json.loads(req_m.read().decode("utf-8"))
        except Exception:
            metrics_after = {}

        total_tokens = sum(r["output_tokens"] for r in stream_results)
        aggregate_tok_s = round(total_tokens / wall_time_s, 1)
        mean_ttft = round(sum(r["ttft_s"] for r in stream_results) / max(1, len(stream_results)), 3)
        max_ttft = round(max(r["ttft_s"] for r in stream_results), 3)

        hit_rate = metrics_after.get("engine", {}).get("vram_hit_rate") or metrics_after.get("expert_cache_hit_rate") or 0.0

        rec = {
            "concurrency": c,
            "wall_time_s": round(wall_time_s, 2),
            "total_output_tokens": total_tokens,
            "aggregate_tok_s": aggregate_tok_s,
            "mean_ttft_s": mean_ttft,
            "max_ttft_s": max_ttft,
            "vram_hit_rate": hit_rate,
            "per_stream": stream_results,
        }
        records.append(rec)

        print(f"  -> Concurrency C={c} Complete:")
        print(f"     Wall Time       : {wall_time_s:.2f} s")
        print(f"     Total Output    : {total_tokens} tokens")
        print(f"     Aggregate Speed : {aggregate_tok_s} tok/s")
        print(f"     Mean TTFT       : {mean_ttft} s (Max: {max_ttft} s)")
        for sr in sorted(stream_results, key=lambda x: x["stream_id"]):
            print(f"       Stream #{sr['stream_id']}: {sr['output_tokens']} tok in {sr['wall_time_s']}s (TTFT: {sr['ttft_s']}s, {sr['decode_tok_s']} tok/s)")

    # Print summary table
    print("\n" + "=" * 86)
    print(" CONCURRENCY SCALING SUMMARY TABLE")
    print("=" * 86)
    print(f"{'Concurrency':<12} | {'Tokens':<8} | {'Wall (s)':<10} | {'Aggregate tok/s':<18} | {'Mean TTFT (s)':<15} | {'Max TTFT (s)':<12}")
    print("-" * 86)
    for r in records:
        print(f"C={r['concurrency']:<10} | {r['total_output_tokens']:<8} | {r['wall_time_s']:<10.2f} | {r['aggregate_tok_s']:<18.1f} | {r['mean_ttft_s']:<15.3f} | {r['max_ttft_s']:<12.3f}")
    print("=" * 86)

    # Save to file
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"concurrency_scaling_{tag}_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump({"tag": tag, "records": records}, f, indent=2)
    print(f"\n[+] Results saved to {out_path}\n")
    return {"tag": tag, "records": records}


def main():
    parser = argparse.ArgumentParser(description="Multi-stream concurrency benchmark")
    parser.add_argument("--url", default="http://192.168.7.152:8080", help="Base URL of server")
    parser.add_argument("--concurrency", default="1,2", help="Comma-separated concurrency levels (e.g. 1,2)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per stream")
    parser.add_argument("--tag", default="test", help="Tag for this benchmark run")
    args = parser.parse_args()

    c_levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    run_concurrency_test(
        base_url=args.url,
        concurrency_levels=c_levels,
        max_tokens_per_stream=args.max_tokens,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
