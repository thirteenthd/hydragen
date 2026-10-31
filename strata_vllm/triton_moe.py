# SPDX-License-Identifier: Apache-2.0
"""Triton Fused MoE Execution Engine with Slot Indirection for Strata Paged MoE.

Executes Mixture-of-Experts forward passes using Triton Tensor-Core kernels
(`tl.dot`) directly over PagedExpertManager's GPU slot arena and GPU indirection
table (`gpu_slot_table`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_moe_slot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    slot_mapping_ptr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Triton grouped MoE GEMM with GPU slot indirection (`slot_mapping_ptr`).

    For each logical expert `off_experts` in the grid, looks up its physical
    VRAM slot `slot_idx = slot_mapping_ptr[off_experts]`. If `slot_idx < 0`
    (expert not active/resident), exits immediately in <1 cycle. Otherwise
    computes `C[off_experts] = A @ B[slot_idx]` on Ampere FP16 Tensor Cores.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_experts = tl.program_id(2)

    slot_idx = tl.load(slot_mapping_ptr + off_experts)
    if slot_idx < 0:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (
        b_ptr
        + slot_idx * stride_be
        + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    c_ptrs = (
        c_ptr
        + off_experts * (M * N)
        + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    )
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def fused_moe_indexed_slot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    active_expert_ids_ptr,
    slot_mapping_ptr,
    stride_az,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Compact Triton grouped MoE GEMM over only the unique active experts in the batch.

    Grid Z is `num_active_experts` (instead of all 512 experts), looking up
    `expert_id = active_expert_ids_ptr[pid_z]` and then `slot_idx = slot_mapping_ptr[expert_id]`.
    `stride_az` is 0 when all experts share input `A` (Gate/Up projection) and
    `M * K` when each active expert `pid_z` has its own input slice `A[pid_z]` (Down projection).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_z = tl.program_id(2)

    expert_id = tl.load(active_expert_ids_ptr + pid_z)
    slot_idx = tl.load(slot_mapping_ptr + expert_id)
    if slot_idx < 0:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = (
        a_ptr
        + pid_z * stride_az
        + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    )
    b_ptrs = (
        b_ptr
        + slot_idx * stride_be
        + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    c_ptrs = (
        c_ptr
        + pid_z * (M * N)
        + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    )
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_paged_moe_forward(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_manager: "Any",
    layer_idx: int = 0,
    block_size_m: int = 16,
    block_size_n: int = 64,
    block_size_k: int = 64,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Execute a complete MoE layer forward pass using PagedExpertManager + Triton.

    1. Extracts unique active experts from `topk_ids [M, top_k]`.
    2. Requests `expert_manager.request_experts(active_ids, layer_idx=layer_idx)`
       which triggers async PCIe DMA + on-GPU `dequant_f16` for any cache misses.
    3. Synchronizes the compute stream with `expert_manager.dma_stream`.
    4. Runs Triton `fused_moe_indexed_slot_kernel` for Gate/Up and Down projections
       with SwiGLU activation and `topk_weights` aggregation.

    Returns:
        (output_hidden_states [M, K], step_metrics_dict)
    """
    orig_shape = hidden_states.shape
    K = orig_shape[-1]
    x = hidden_states.reshape(-1, K).contiguous()
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    M = x.shape[0]
    top_k = topk_ids.shape[-1]

    # 1. Identify unique active experts in this batch step
    unique_experts_t = torch.unique(topk_ids.reshape(-1)).to(torch.int32)
    unique_expert_list: List[int] = unique_experts_t.detach().cpu().tolist()
    num_active = len(unique_expert_list)

    # 2. Request experts from PagedExpertManager (async DMA + GPU dequant on misses)
    # and execute Triton Gate/Up/Down projections in tiles of `expert_manager.num_slots`
    hits_before = getattr(expert_manager, "cache_hits", 0)
    misses_before = getattr(expert_manager, "cache_misses", 0)
    pcie_before = getattr(expert_manager, "pcie_bytes_transferred", 0)

    # 3. Prepare padded M>=16 input for Triton tl.dot Tensor Cores
    M_pad = max(16, triton.cdiv(M, block_size_m) * block_size_m)
    if M_pad > M:
        x_pad = torch.zeros((M_pad, K), dtype=torch.float16, device=x.device)
        x_pad[:M] = x
    else:
        x_pad = x

    arena = expert_manager.vram_arena
    has_up = arena.shape[1] >= 3
    if arena.shape[-1] == K:
        N_ff = arena.shape[-2]
        w_gate = arena[:, 0].transpose(-1, -2)  # [num_slots, K, N_ff]
        w_up = arena[:, 1].transpose(-1, -2) if has_up else None
        w_down_idx = 2 if has_up else 1
        w_down = arena[:, w_down_idx].reshape(arena.shape[0], K, N_ff).transpose(-1, -2)
    else:
        N_ff = arena.shape[-1]
        w_gate = arena[:, 0]  # [num_slots, K, N_ff]
        w_up = arena[:, 1] if has_up else None
        w_down_idx = 2 if has_up else 1
        w_down = arena[:, w_down_idx].transpose(-1, -2)  # [num_slots, N_ff, K]

    down_out = torch.empty((num_active, M, K), dtype=torch.float16, device=x.device)
    tile_slots = max(1, int(getattr(expert_manager, "num_slots", num_active)))

    for start in range(0, num_active, tile_slots):
        end = min(num_active, start + tile_slots)
        chunk_len = end - start
        chunk_list = unique_expert_list[start:end]
        chunk_experts_t = unique_experts_t[start:end].contiguous()

        if hasattr(expert_manager, "request_layer_experts"):
            expert_manager.request_layer_experts(layer_idx, chunk_list)
            slot_table = expert_manager.get_gpu_slot_table(layer_idx)
        else:
            expert_manager.request_experts(chunk_list)
            slot_table = expert_manager.gpu_slot_table

        expert_manager.synchronize()

        gate_out = torch.empty((chunk_len, M_pad, N_ff), dtype=torch.float16, device=x.device)
        grid_gu = (
            triton.cdiv(M_pad, block_size_m),
            triton.cdiv(N_ff, block_size_n),
            chunk_len,
        )

        # 4. Launch Triton Gate Projection
        fused_moe_indexed_slot_kernel[grid_gu](
            x_pad,
            w_gate,
            gate_out,
            chunk_experts_t,
            slot_table,
            0,
            x_pad.stride(0),
            x_pad.stride(1),
            w_gate.stride(0),
            w_gate.stride(1),
            w_gate.stride(2),
            gate_out.stride(1),
            gate_out.stride(2),
            M_pad,
            N_ff,
            K,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
        )

        if w_up is not None:
            up_out = torch.empty((chunk_len, M_pad, N_ff), dtype=torch.float16, device=x.device)
            fused_moe_indexed_slot_kernel[grid_gu](
                x_pad,
                w_up,
                up_out,
                chunk_experts_t,
                slot_table,
                0,
                x_pad.stride(0),
                x_pad.stride(1),
                w_up.stride(0),
                w_up.stride(1),
                w_up.stride(2),
                up_out.stride(1),
                up_out.stride(2),
                M_pad,
                N_ff,
                K,
                BLOCK_SIZE_M=block_size_m,
                BLOCK_SIZE_N=block_size_n,
                BLOCK_SIZE_K=block_size_k,
            )
            inter_pad = F.silu(gate_out) * up_out
        else:
            inter_pad = F.silu(gate_out)

        # 5. Launch Triton Down Projection over active experts in this tile
        down_pad = torch.empty((chunk_len, M_pad, K), dtype=torch.float16, device=x.device)
        grid_down = (
            triton.cdiv(M_pad, block_size_m),
            triton.cdiv(K, block_size_n),
            chunk_len,
        )
        fused_moe_indexed_slot_kernel[grid_down](
            inter_pad,
            w_down,
            down_pad,
            chunk_experts_t,
            slot_table,
            inter_pad.stride(0),
            inter_pad.stride(1),
            inter_pad.stride(2),
            w_down.stride(0),
            w_down.stride(1),
            w_down.stride(2),
            down_pad.stride(1),
            down_pad.stride(2),
            M_pad,
            K,
            N_ff,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
        )
        down_out[start:end] = down_pad[:, :M, :]

    # 6. Weighted reduction over top_k selected experts per token
    # Map each expert_id in topk_ids [M, top_k] to its compact index in [0..num_active-1]
    expert_to_compact = torch.full(
        (slot_table.shape[0],), 0, dtype=torch.long, device=x.device
    )
    expert_to_compact[unique_experts_t.long()] = torch.arange(
        num_active, dtype=torch.long, device=x.device
    )
    compact_ids = expert_to_compact[topk_ids.reshape(M, top_k).long()]  # [M, top_k]
    token_indices = torch.arange(M, device=x.device).unsqueeze(1).expand(M, top_k)
    selected_out = down_out[compact_ids, token_indices]  # [M, top_k, K]
    final_out = (selected_out.float() * topk_weights.reshape(M, top_k, 1).float()).sum(dim=1).to(hidden_states.dtype)

    hits_delta = getattr(expert_manager, "cache_hits", 0) - hits_before
    misses_delta = getattr(expert_manager, "cache_misses", 0) - misses_before
    pcie_delta = getattr(expert_manager, "pcie_bytes_transferred", 0) - pcie_before
    total_reqs = max(1, hits_delta + misses_delta)

    metrics = {
        "active_experts": float(num_active),
        "cache_hits": float(hits_delta),
        "cache_misses": float(misses_delta),
        "hit_rate": float(hits_delta) / float(total_reqs),
        "pcie_bytes": float(pcie_delta),
    }
    return final_out.reshape(orig_shape), metrics
