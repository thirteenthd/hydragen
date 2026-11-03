# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark measuring Reverse Query-Tile Attention Scheduling (LPT) vs Forward Scheduling on GPU."""

import argparse
import json
import math
import os
import time
import torch

from hydragen.paged_kv import PagedKVManager


def run_bench(device="cuda", dtype=torch.float32, is_int8_kv=False):
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, running on CPU")
        device = "cpu"

    num_heads = 16
    num_kv_heads = 4
    head_dim = 128
    block_size = 16
    scale = 1.0 / math.sqrt(head_dim)

    # Test context lengths across prefill spectrum
    test_lengths = [2048, 8192, 16384, 32768]
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Running on GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB VRAM)")
        if gpu_mem >= 10:
            test_lengths.append(64000)

    results = []

    for query_len in test_lengths:
        needed_blocks = (query_len + block_size - 1) // block_size + 16
        mgr = PagedKVManager(
            block_size=block_size,
            num_blocks=needed_blocks + 32,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        seq_id = 1
        mgr.allocate_sequence(seq_id, query_len)
        block_list = mgr.get_block_table(seq_id)

        block_tables = torch.tensor([block_list], dtype=torch.int32, device=device)
        context_lens = torch.tensor([0], dtype=torch.int32, device=device)

        q = torch.randn(1, query_len, num_heads, head_dim, dtype=dtype, device=device)
        k_cache = torch.randn(needed_blocks + 32, num_kv_heads, block_size, head_dim, dtype=dtype, device=device)
        v_cache = torch.randn(needed_blocks + 32, num_kv_heads, block_size, head_dim, dtype=dtype, device=device)

        out_fwd = torch.zeros_like(q)
        out_rev = torch.zeros_like(q)

        # Warmup
        for _ in range(3):
            mgr.paged_attention_prefill(
                out_fwd, q, k_cache, v_cache, block_tables, context_lens,
                scale=scale, is_int8_kv=is_int8_kv, reverse_lpt=False
            )
            mgr.paged_attention_prefill(
                out_rev, q, k_cache, v_cache, block_tables, context_lens,
                scale=scale, is_int8_kv=is_int8_kv, reverse_lpt=True
            )
        if device == "cuda":
            torch.cuda.synchronize()

        # Measure Forward Scheduling (Standard)
        iters = 10 if query_len <= 16384 else 5
        fwd_times = []
        for _ in range(iters):
            if device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                mgr.paged_attention_prefill(
                    out_fwd, q, k_cache, v_cache, block_tables, context_lens,
                    scale=scale, is_int8_kv=is_int8_kv, reverse_lpt=False
                )
                end_event.record()
                torch.cuda.synchronize()
                fwd_times.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                mgr.paged_attention_prefill(
                    out_fwd, q, k_cache, v_cache, block_tables, context_lens,
                    scale=scale, is_int8_kv=is_int8_kv, reverse_lpt=False
                )
                fwd_times.append((time.perf_counter() - t0) * 1000.0)

        # Measure Reverse Query-Tile Scheduling (LPT)
        rev_times = []
        for _ in range(iters):
            if device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                mgr.paged_attention_prefill(
                    out_rev, q, k_cache, v_cache, block_tables, context_lens,
                    scale=scale, is_int8_kv=is_int8_kv, reverse_lpt=True
                )
                end_event.record()
                torch.cuda.synchronize()
                rev_times.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                mgr.paged_attention_prefill(
                    out_rev, q, k_cache, v_cache, block_tables, context_lens,
                    scale=scale, is_int8_kv=is_int8_kv, reverse_lpt=True
                )
                rev_times.append((time.perf_counter() - t0) * 1000.0)

        # Check numerical equality
        max_diff = torch.max(torch.abs(out_fwd - out_rev)).item()

        fwd_median = sorted(fwd_times)[len(fwd_times) // 2]
        rev_median = sorted(rev_times)[len(rev_times) // 2]
        speedup = ((fwd_median - rev_median) / fwd_median) * 100.0 if fwd_median > 0 else 0.0

        res = {
            "query_len": query_len,
            "forward_ms": round(fwd_median, 3),
            "reverse_lpt_ms": round(rev_median, 3),
            "speedup_pct": round(speedup, 2),
            "max_abs_diff": float(max_diff),
        }
        results.append(res)
        print(f"Context {query_len:6d} tok: Forward = {fwd_median:7.2f} ms | Reverse LPT = {rev_median:7.2f} ms | Speedup: {speedup:+5.2f}% (max diff: {max_diff:.1e})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=str, default="benchmarks/results/bench_lpt_micro.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    results = run_bench(device=args.device)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save}")
