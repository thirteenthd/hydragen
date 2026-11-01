#!/usr/bin/env python3
"""Context Tiers Benchmark Runner for Strata / Hydragen.

Tests Short (~100 tokens), Medium (~4,000 tokens), and Long (~32,000 tokens)
measuring:
- Prefill Rate (tokens/sec)
- Time to First Token (TTFT)
- Sustained Decode Rate (tokens/sec)
- KV Cache spill behavior (resident VRAM vs host RAM)
- Single-stream (C=1) vs Dual-stream (C=2) concurrency under context load
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request
from typing import Any, Dict, List


def generate_prompt_for_tokens(target_tokens: int, unique_id: str = "") -> str:
    """Generate a realistic synthetic code/analysis prompt with approximately target_tokens."""
    # A realistic Python function block with docstrings and logic is ~50 tokens (~200 chars)
    chunk = (
        "def audit_transaction_pipeline(record_id: int, payload: dict, metadata: list) -> dict:\n"
        "    \"\"\"Validate transaction signatures, cross-reference ledger balances, and check replay protection.\"\"\"\n"
        "    sanitized = {k: v for k, v in payload.items() if not k.startswith('_')}\n"
        "    signature_hash = hash(record_id) ^ hash(str(payload))\n"
        "    validated_entries = [m for m in metadata if m.get('active', True) and m.get('tier', 0) > 1]\n"
        "    return {'status': 'verified', 'record_id': record_id, 'sig': signature_hash, 'count': len(validated_entries)}\n\n"
    )
    needed_chunks = max(1, target_tokens // 60)
    body = chunk * needed_chunks
    salt_header = f"// Session Run ID: {unique_id or time.time_ns()}\n"
    return (
        f"{salt_header}"
        f"Below is a production pipeline module containing {needed_chunks} service handler routines.\n"
        f"Examine the architecture, logic, and structure.\n\n"
        f"{body}\n\n"
        "Task: Provide a concise architectural review highlighting 3 security considerations and 2 performance bottlenecks."
    )


def run_stream_request(
    stream_idx: int,
    tier_name: str,
    target_tokens: int,
    url: str,
    max_tokens: int,
    force_cold: bool = True,
) -> Dict[str, Any]:
    uid = f"cold_{time.time_ns()}_{stream_idx}" if force_cold else "cached_prefix"
    prompt = generate_prompt_for_tokens(target_tokens, unique_id=uid)
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

    with urllib.request.urlopen(req, timeout=300) as resp:
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
        "stream_idx": stream_idx,
        "tier_name": tier_name,
        "prompt_chars": len(prompt),
        "output_tokens": tokens,
        "ttft_s": round(ttft_s, 3),
        "decode_s": round(decode_s, 2),
        "total_s": round(total_s, 2),
        "decode_tok_s": round(decode_tok_s, 1),
    }


def run_tier_test(
    tier_name: str,
    target_prompt_tokens: int,
    concurrency: int = 1,
    url: str = "http://192.168.7.138:8080/v1/chat/completions",
    max_tokens: int = 64,
    force_cold: bool = True,
) -> Dict[str, Any]:
    print(f"\n========================================================")
    print(f" Tier: {tier_name} (~{target_prompt_tokens:,} tokens) | Concurrency = {concurrency} | {'Cold' if force_cold else 'Warm'}")
    print(f" Target URL: {url} | Max Gen Tokens: {max_tokens}")
    print(f"========================================================")

    t_wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_stream_request, i, tier_name, target_prompt_tokens, url, max_tokens, force_cold)
            for i in range(concurrency)
        ]
        results = [f.result() for f in futures]
    t_wall_end = time.perf_counter()

    wall_s = t_wall_end - t_wall_start
    total_out_toks = sum(r["output_tokens"] for r in results)
    agg_tok_s = round(total_out_toks / max(1e-4, wall_s), 1)

    print(f"\n--- Results for {tier_name} (Concurrency = {concurrency}) ---")
    print(f"Wall Clock Duration  : {wall_s:.2f} s")
    print(f"Total Output Tokens  : {total_out_toks}")
    print(f"Aggregate Generation : {agg_tok_s:.1f} tok/s\n")

    print(f"{'Stream':<12} | {'Prompt Chars':<14} | {'TTFT (s)':<10} | {'Decode (s)':<10} | {'Decode Speed':<14} | {'Total (s)':<10}")
    print("-" * 80)
    for r in results:
        print(f"Stream #{r['stream_idx']:<4} | {r['prompt_chars']:<14,} | {r['ttft_s']:<10.3f} | {r['decode_s']:<10.2f} | {r['decode_tok_s']:<14.1f} | {r['total_s']:<10.2f}")

    return {
        "tier": tier_name,
        "concurrency": concurrency,
        "wall_clock_s": round(wall_s, 2),
        "total_output_tokens": total_out_toks,
        "aggregate_tok_s": agg_tok_s,
        "streams": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["medium", "long", "all"], default="medium")
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--url", type=str, default="http://192.168.7.138:8080/v1/chat/completions")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    tiers_to_run = []
    if args.tier in ["medium", "all"]:
        tiers_to_run.append(("Medium (4k)", 4000))
    if args.tier in ["long", "all"]:
        tiers_to_run.append(("Long (32k)", 32000))

    for name, tokens in tiers_to_run:
        run_tier_test(name, tokens, concurrency=args.concurrency, url=args.url, max_tokens=args.max_tokens)
