#!/usr/bin/env python3
"""Concurrency Benchmark Runner for Strata vs Strata-vLLM Hybrid Engine.

Measures Time to First Token (TTFT), Decode Speed, Queue Latency, and Aggregate
Throughput at Concurrency = 2 and Concurrency = 3.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request
from typing import Any, Dict, List


PROMPTS = [
    (
        "Request A (Python Systems)",
        "Write a robust Python worker pool class using asyncio.Queue with exponential backoff, "
        "dead-letter queue handling, and clean shutdown signals. Provide the implementation and explain the concurrency mechanics."
    ),
    (
        "Request B (Database Schema)",
        "Design a PostgreSQL schema for an e-commerce order fulfillment system with multi-warehouse inventory, "
        "partitioning by creation date, and idempotency keys. Output the SQL CREATE TABLE statements and indexes."
    ),
    (
        "Request C (TypeScript / Frontend)",
        "Implement a custom React hook in TypeScript `useReconnectingWebSocket` that handles heartbeats, "
        "exponential backoff reconnection, and message batching with clean cleanup on unmount."
    ),
]


def run_single_stream(
    idx: int,
    name: str,
    prompt: str,
    url: str,
    max_tokens: int,
) -> Dict[str, Any]:
    req_body = {
        "model": "qwen3.8-flash-next-iq3_xxs",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "stream": True,
    }
    data = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    t_first = None
    tokens = 0
    full_text = []

    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str or not line_str.startswith("data: "):
                continue
            payload = line_str[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    if t_first is None:
                        t_first = time.perf_counter()
                    tokens += 1
                    full_text.append(content)
            except Exception:
                continue

    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end

    ttft_s = t_first - t_start
    decode_s = max(1e-4, t_end - t_first)
    total_s = t_end - t_start
    decode_tok_s = tokens / decode_s if tokens > 0 else 0.0

    return {
        "idx": idx,
        "name": name,
        "tokens": tokens,
        "ttft_ms": round(ttft_s * 1000, 1),
        "decode_s": round(decode_s, 2),
        "total_s": round(total_s, 2),
        "decode_tok_s": round(decode_tok_s, 1),
        "t_start": t_start,
        "t_first": t_first,
        "t_end": t_end,
    }


def run_benchmark(
    concurrency: int,
    url: str = "http://127.0.0.1:8080/v1/chat/completions",
    max_tokens: int = 128,
    warmup: bool = True,
) -> Dict[str, Any]:
    print(f"\n========================================================")
    print(f" Running Benchmark: Concurrency = {concurrency} ({concurrency} simultaneous streams)")
    print(f" Target URL: {url} | Max Tokens: {max_tokens}")
    print(f"========================================================")

    if warmup:
        print("[Warmup] Sending single warmup request ...", end="", flush=True)
        _ = run_single_stream(0, "Warmup", "Say 'Ready' in one word.", url, max_tokens=16)
        print(" Done.\n")
        time.sleep(1.0)

    selected_prompts = PROMPTS[:concurrency]
    t_global_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_single_stream, i, p[0], p[1], url, max_tokens)
            for i, p in enumerate(selected_prompts)
        ]
        results = [f.result() for f in futures]

    t_global_end = time.perf_counter()
    wall_clock_s = round(t_global_end - t_global_start, 2)
    total_tokens = sum(r["tokens"] for r in results)
    agg_tok_s = round(total_tokens / max(1e-4, wall_clock_s), 1)

    # Sort results by index
    results.sort(key=lambda x: x["idx"])

    print(f"\n--- Results for Concurrency = {concurrency} ---")
    print(f"Wall-Clock Duration : {wall_clock_s:.2f} s")
    print(f"Total Tokens Output : {total_tokens} tokens")
    print(f"Aggregate Throughput: {agg_tok_s:.1f} tok/s\n")

    print(f"{'Stream':<32} | {'Tokens':<7} | {'TTFT (ms)':<10} | {'Decode (s)':<10} | {'Speed (tok/s)':<14} | {'Total (s)':<10}")
    print("-" * 92)
    for r in results:
        print(f"{r['name']:<32} | {r['tokens']:<7} | {r['ttft_ms']:<10} | {r['decode_s']:<10} | {r['decode_tok_s']:<14} | {r['total_s']:<10}")

    return {
        "concurrency": concurrency,
        "wall_clock_s": wall_clock_s,
        "total_tokens": total_tokens,
        "aggregate_tok_s": agg_tok_s,
        "streams": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    res = run_benchmark(
        concurrency=args.concurrency,
        url=args.url,
        max_tokens=args.max_tokens,
        warmup=not args.no_warmup,
    )
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
