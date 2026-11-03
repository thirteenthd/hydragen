#!/usr/bin/env python3
"""Direct Microbenchmark: MoE Concurrency Scaling Under Expert Contention.

Compares:
  Mode A: GPU-Only PagedMoE (forced PCIe DMA eviction on cache misses)
  Mode B: Concurrent CPU+GPU Hybrid Expert Mailbox (host RAM compute for missed slices)
across concurrent streams C=1, C=2, C=4 under high cross-stream expert conflict.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch

# Ensure local hydragen package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hydragen.expert_mailbox import HybridExpertMailbox
from hydragen.expert_manager import PagedExpertManager
from hydragen.triton_moe import triton_paged_moe_forward


def run_moe_concurrency_microbench(
    device: torch.device,
    total_experts: int = 64,
    num_slots: int = 16,
    k_dim: int = 64,
    n_ff: int = 128,
    tokens_per_stream: int = 16,
    num_steps: int = 10,
) -> Dict[str, Any]:
    print("=" * 80)
    print(" DIRECT MoE CONCURRENCY MICROBENCHMARK (EXPERT CONTENTION)")
    print(f" Device: {device} | Total Experts: {total_experts} | VRAM Slots: {num_slots}")
    print(f" Dimensions: K={k_dim}, N_ff={n_ff} | Steps: {num_steps}")
    print("=" * 80)

    results: Dict[str, Any] = {}

    for c in [1, 2, 4, 8]:
        print(f"\n--- Testing Concurrency Level C={c} Concurrent Streams ---")

        # Distinct expert interest sets per stream to simulate high cross-stream contention
        stream_expert_pools = [
            list(range((i * 16) % total_experts, min(total_experts, (i * 16) % total_experts + 16)))
            for i in range(c)
        ]

        # Generate inputs for all streams
        batch_tokens = tokens_per_stream * c
        x_all = torch.randn((batch_tokens, k_dim), dtype=torch.float16, device=device)

        # Route top-2 experts per token from each stream's conflicting pool
        topk_ids_list = []
        for i in range(c):
            pool = stream_expert_pools[i]
            # Pick 2 distinct experts per token in stream i
            t_ids = torch.tensor(
                [[pool[j % len(pool)], pool[(j + 3) % len(pool)]] for j in range(tokens_per_stream)],
                dtype=torch.int32,
                device=device,
            )
            topk_ids_list.append(t_ids)
        topk_ids = torch.cat(topk_ids_list, dim=0)  # [batch_tokens, 2]
        topk_weights = torch.full((batch_tokens, 2), 0.5, dtype=torch.float32, device=device)

        # -------------------------------------------------------------
        # Mode A: Standard GPU-Only PagedMoE (Mailbox Disabled)
        # -------------------------------------------------------------
        mgr_gpu_only = PagedExpertManager(
            num_slots=num_slots,
            expert_shape=(3, n_ff, k_dim),
            total_experts=total_experts,
            dtype=torch.float16,
            device=device,
            compressed_cache=False,
        )
        mgr_gpu_only.mailbox = None  # Ensure mailbox disabled
        torch.manual_seed(1234)
        mgr_gpu_only.host_pool.copy_(torch.randn_like(mgr_gpu_only.host_pool))
        # Warmup slots with stream 0 experts
        mgr_gpu_only.request_experts(stream_expert_pools[0][:num_slots])

        # Benchmark Mode A
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for step in range(num_steps):
            out_gpu, m_gpu = triton_paged_moe_forward(
                hidden_states=x_all,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                expert_manager=mgr_gpu_only,
                layer_idx=0,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
        t_gpu_only = (time.perf_counter() - t0) * 1000.0  # ms
        tok_s_gpu_only = round((batch_tokens * num_steps) / max(1e-4, t_gpu_only / 1000.0), 1)

        # -------------------------------------------------------------
        # Mode B: Concurrent CPU+GPU Hybrid Expert Mailbox
        # -------------------------------------------------------------
        mgr_mailbox = PagedExpertManager(
            num_slots=num_slots,
            expert_shape=(3, n_ff, k_dim),
            total_experts=total_experts,
            dtype=torch.float16,
            device=device,
            compressed_cache=False,
        )
        mgr_mailbox.mailbox = HybridExpertMailbox(max_workers=min(8, os.cpu_count() or 4), device=device)
        mgr_mailbox.host_pool.copy_(mgr_gpu_only.host_pool)
        # Warmup slots with stream 0 experts
        mgr_mailbox.request_experts(stream_expert_pools[0][:num_slots])

        # Benchmark Mode B
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for step in range(num_steps):
            out_box, m_box = triton_paged_moe_forward(
                hidden_states=x_all,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                expert_manager=mgr_mailbox,
                layer_idx=0,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
        t_mailbox = (time.perf_counter() - t0) * 1000.0  # ms
        tok_s_mailbox = round((batch_tokens * num_steps) / max(1e-4, t_mailbox / 1000.0), 1)
        mgr_mailbox.mailbox.close()

        speedup = ((tok_s_mailbox - tok_s_gpu_only) / tok_s_gpu_only) * 100.0

        print(f"  Mode A (GPU-Only PagedMoE)      : {t_gpu_only:6.1f} ms | {tok_s_gpu_only:7.1f} tok/s")
        print(f"  Mode B (Hybrid Mailbox MoE)     : {t_mailbox:6.1f} ms | {tok_s_mailbox:7.1f} tok/s | [{speedup:+6.1f}%]")

        results[f"C={c}"] = {
            "gpu_only_ms": round(t_gpu_only, 2),
            "gpu_only_tok_s": tok_s_gpu_only,
            "mailbox_ms": round(t_mailbox, 2),
            "mailbox_tok_s": tok_s_mailbox,
            "speedup_pct": round(speedup, 2),
        }

    return results


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    run_moe_concurrency_microbench(device=device)


if __name__ == "__main__":
    main()
