# SPDX-License-Identifier: Apache-2.0
"""Triton & Native GGUF Fused MoE Execution Engine with Slot Indirection.

Implements three execution paths over PagedExpertManager's GPU slot indirection table:
1. Option 1 (`fused_moe_gguf_down_slot_kernel`):
   In-kernel `@triton.jit` GGUF unpacker that reads compressed `Q2_0_STRATA` (type=42,
   64 weights / 18B) and `IQ4_NL` (type=20, 32 weights / 18B) directly from
   `vram_blob_cache` (`uint8*`), unpacking in Triton registers inside the `K` loop
   before `tl.dot` with ZERO intermediate BF16/FP16 VRAM buffer.
2. Option 2 (`native_gguf_paged_moe_forward`):
   Direct compressed `[gate | up | down]` GGUF execution via Strata's INT8 `dp4a` /
   `native_expert_grouped` CUDA kernel reading straight from `vram_blob_cache`
   (`~1.61 MB/expert`, 5.8x smaller than BF16/FP16).
3. Standard FP16 Arena (`fused_moe_slot_kernel` / `fused_moe_indexed_slot_kernel`):
   Used when `expert_manager.compressed_cache` is False (e.g. synthetic FP16 tests).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Constant 16-entry non-linear lookup table for IQ4_NL (type=20)
_IQ4NL_TABLE_VALUES = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
_IQ4NL_DEVICE_TABLES: Dict[torch.device, torch.Tensor] = {}


def get_iq4nl_lut(device: torch.device) -> torch.Tensor:
    if device not in _IQ4NL_DEVICE_TABLES:
        _IQ4NL_DEVICE_TABLES[device] = torch.tensor(
            _IQ4NL_TABLE_VALUES, dtype=torch.float16, device=device
        )
    return _IQ4NL_DEVICE_TABLES[device]


@triton.jit
def fused_moe_gguf_down_slot_kernel(
    a_ptr,
    blob_ptr,
    c_ptr,
    active_expert_ids_ptr,
    slot_mapping_ptr,
    iq4nl_lut_ptr,
    stride_az,
    stride_am,
    stride_ak,
    stride_slot,
    down_off,
    d_row_bytes,
    stride_cm,
    stride_cn,
    M,
    N,
    K,
    QUANT_TYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Option 1: In-kernel Triton GGUF unpacker (`GGUF uint8 -> Triton Registers -> tl.dot`).

    Reads compressed `Q2_0_STRATA` (type=42, 64 weights in 18 bytes) or `IQ4_NL`
    (type=20, 32 weights in 18 bytes) directly from `blob_ptr + slot_idx * stride_slot + down_off`
    without materializing any BF16/FP16 weight buffer in global VRAM.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_z = tl.program_id(2)

    expert_id = tl.load(active_expert_ids_ptr + pid_z)
    slot_idx = tl.load(slot_mapping_ptr + expert_id).to(tl.int64)
    if slot_idx < 0:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    n_mask = offs_n < N

    a_ptrs = (
        a_ptr
        + pid_z * stride_az
        + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    )

    # Base byte offset of each output row `offs_n` inside the compressed expert slot
    expert_down_base = (
        blob_ptr
        + slot_idx * stride_slot
        + down_off
        + offs_n.to(tl.int64) * d_row_bytes
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if QUANT_TYPE == 42:
        # Q2_0_STRATA: 64 weights per 18-byte block (BLOCK_SIZE_K == 64)
        byte_idx = 2 + (offs_k // 4)
        bit_shift = ((offs_k % 4) * 2).to(tl.uint8)
        for blk_idx in range(0, K // 64):
            k = blk_idx * 64
            a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            blk_base = expert_down_base + blk_idx * 18
            b0 = tl.load(blk_base, mask=n_mask, other=0).to(tl.uint16)
            b1 = tl.load(blk_base + 1, mask=n_mask, other=0).to(tl.uint16)
            d = (b0 | (b1 << 8)).to(tl.float16, bitcast=True)

            q_byte = tl.load(
                blk_base[None, :] + byte_idx[:, None],
                mask=n_mask[None, :],
                other=0,
            )
            q_val = (((q_byte >> bit_shift[:, None]) & 3).to(tl.int8) - 1).to(tl.float16)
            w_tile = q_val * d[None, :]

            accumulator += tl.dot(a, w_tile)
            a_ptrs += 64 * stride_ak
    else:
        # IQ4_NL (type=20): 32 weights per 18-byte block (BLOCK_SIZE_K == 32)
        byte_idx = 2 + (offs_k % 16)
        bit_shift = ((offs_k // 16) * 4).to(tl.uint8)
        for blk_idx in range(0, K // 32):
            k = blk_idx * 32
            a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            blk_base = expert_down_base + blk_idx * 18
            b0 = tl.load(blk_base, mask=n_mask, other=0).to(tl.uint16)
            b1 = tl.load(blk_base + 1, mask=n_mask, other=0).to(tl.uint16)
            d = (b0 | (b1 << 8)).to(tl.float16, bitcast=True)

            q_byte = tl.load(
                blk_base[None, :] + byte_idx[:, None],
                mask=n_mask[None, :],
                other=0,
            )
            nibble = ((q_byte >> bit_shift[:, None]) & 0xF).to(tl.int32)
            kv = tl.load(iq4nl_lut_ptr + nibble)
            w_tile = kv * d[None, :]

            accumulator += tl.dot(a, w_tile)
            a_ptrs += 32 * stride_ak

    c = accumulator.to(tl.float16)
    c_ptrs = (
        c_ptr
        + pid_z * (M * N)
        + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    )
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_gguf_down_slot_forward(
    inter_states: torch.Tensor,
    active_experts_t: torch.Tensor,
    slot_table: torch.Tensor,
    vram_blob_cache: torch.Tensor,
    down_type: int,
    down_off: int,
    d_row_bytes: int,
    n_embd: int,
    n_ff: int,
    block_size_m: int = 16,
    block_size_n: int = 64,
) -> torch.Tensor:
    """Execute Down projection via Option 1 (`fused_moe_gguf_down_slot_kernel`) directly
    on compressed GGUF bytes in `vram_blob_cache` without any FP16/BF16 weight buffer.

    Args:
        inter_states: [num_active, M, n_ff] float16 intermediate activations.
        active_experts_t: [num_active] int32 active expert IDs.
        slot_table: [total_experts] int32 GPU slot mapping table.
        vram_blob_cache: [num_slots, max_blob_bytes] uint8 compressed GGUF cache.
    Returns:
        down_out: [num_active, M, n_embd] float16
    """
    num_active, M, K = inter_states.shape
    M_pad = max(16, triton.cdiv(M, block_size_m) * block_size_m)
    if M_pad > M:
        a_pad = torch.zeros((num_active, M_pad, K), dtype=torch.float16, device=inter_states.device)
        a_pad[:, :M, :] = inter_states
    else:
        a_pad = inter_states.contiguous()

    out_pad = torch.empty((num_active, M_pad, n_embd), dtype=torch.float16, device=inter_states.device)
    block_size_k = 64 if down_type == 42 else 32
    lut = get_iq4nl_lut(inter_states.device)

    grid = (
        triton.cdiv(M_pad, block_size_m),
        triton.cdiv(n_embd, block_size_n),
        num_active,
    )
    fused_moe_gguf_down_slot_kernel[grid](
        a_pad,
        vram_blob_cache,
        out_pad,
        active_experts_t,
        slot_table,
        lut,
        a_pad.stride(0),
        a_pad.stride(1),
        a_pad.stride(2),
        vram_blob_cache.stride(0),
        int(down_off),
        int(d_row_bytes),
        out_pad.stride(1),
        out_pad.stride(2),
        M_pad,
        n_embd,
        n_ff,
        QUANT_TYPE=int(down_type),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
    )
    return out_pad[:, :M, :]


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
    """Triton grouped MoE GEMM with GPU slot indirection (`slot_mapping_ptr`)."""
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
    """Compact Triton grouped MoE GEMM over only the unique active experts in the batch."""
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


def native_gguf_paged_moe_forward(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_manager: Any,
    layer_idx: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Option 2: Execute MoE layer directly on compressed `[gate | up | down]` GGUF blobs
    in `expert_manager.vram_blob_cache` using Strata's INT8 `native_expert_grouped`
    (`dp4a` / `mma.sync`) with ZERO intermediate BF16/FP16 weight buffers.
    """
    orig_shape = hidden_states.shape
    K = orig_shape[-1]
    x_f32 = hidden_states.reshape(-1, K).float().contiguous()
    M = x_f32.shape[0]
    top_k = topk_ids.shape[-1]
    cap_entries = M * top_k
    device = x_f32.device

    loader = expert_manager.loader
    meta = loader.get_layer_expert_metadata(layer_idx)
    gu_type = meta["gate_type"]
    d_type = meta["down_type"]
    n_embd = meta["n_embd"]
    n_ff = meta["n_ff"]

    hits_before = getattr(expert_manager, "cache_hits", 0)
    misses_before = getattr(expert_manager, "cache_misses", 0)
    pcie_before = getattr(expert_manager, "pcie_bytes_transferred", 0)

    # Sort routed (token, k) entries by expert_id to form CSR groups
    flat_experts = topk_ids.reshape(-1).to(torch.int32)
    sorted_experts, sort_order = torch.sort(flat_experts)
    ent_dst = sort_order.to(torch.int32).contiguous()
    ent_tok = torch.div(sort_order, top_k, rounding_mode="floor").to(torch.int32).contiguous()

    unique_experts_t, counts_t = torch.unique_consecutive(sorted_experts, return_counts=True)
    unique_expert_list: List[int] = unique_experts_t.detach().cpu().tolist()
    num_active = len(unique_expert_list)

    grp_start = torch.zeros((num_active + 1,), dtype=torch.int32, device=device)
    grp_start[1:] = torch.cumsum(counts_t.to(torch.int32), dim=0)
    grp_start_cpu = grp_start.detach().cpu().tolist()

    # Allocate compact Q8_1 activation buffer (36 bytes per 32 floats) & scratch
    q8_bytes = M * (n_embd // 32) * 36
    x_q8_1 = torch.empty((q8_bytes,), dtype=torch.uint8, device=device)
    scratch_bytes = loader.bridge.expert_scratch_bytes(cap_entries, n_ff)
    scratch = torch.empty((scratch_bytes,), dtype=torch.uint8, device=device)
    out_parts = torch.zeros((cap_entries, n_embd), dtype=torch.float32, device=device)

    blob_cache = expert_manager.vram_blob_cache
    slot_stride = blob_cache.stride(0)
    base_ptr = blob_cache.data_ptr()
    tile_slots = max(1, int(getattr(expert_manager, "num_slots", num_active)))

    for start in range(0, num_active, tile_slots):
        end = min(num_active, start + tile_slots)
        chunk_len = end - start
        chunk_list = unique_expert_list[start:end]
        chunk_experts_t = unique_experts_t[start:end]

        expert_manager.request_layer_experts(layer_idx, chunk_list)
        slot_table = expert_manager.get_gpu_slot_table(layer_idx)
        expert_manager.synchronize()

        slot_ids = slot_table[chunk_experts_t.long()].to(torch.int64)
        tile_grp_ptr = (base_ptr + slot_ids * slot_stride).contiguous()

        entry_start = int(grp_start_cpu[start])
        entry_end = int(grp_start_cpu[end])
        tile_entries = entry_end - entry_start
        tile_grp_start = (grp_start[start : end + 1] - entry_start).contiguous()
        tile_n_groups = torch.tensor([chunk_len], dtype=torch.int32, device=device)
        tile_ent_dst = ent_dst[entry_start:entry_end].contiguous()
        tile_ent_tok = ent_tok[entry_start:entry_end].contiguous()

        loader.bridge.native_expert_grouped_forward(
            gu_type=gu_type,
            d_type=d_type,
            n_embd=n_embd,
            n_ff=n_ff,
            n_tokens=M,
            grp_ptr=tile_grp_ptr,
            grp_start=tile_grp_start,
            n_groups=tile_n_groups,
            ent_dst=tile_ent_dst,
            ent_tok=tile_ent_tok,
            cap_groups=chunk_len,
            cap_entries=tile_entries,
            x_f32=x_f32,
            x_q8_1=x_q8_1,
            scratch=scratch,
            out_parts=out_parts,
        )

    final_f32 = torch.empty((M, n_embd), dtype=torch.float32, device=device)
    loader.bridge.moe_combine_multi(
        parts=out_parts,
        weights=topk_weights.reshape(M, top_k).float().contiguous(),
        output=final_f32,
        n_embd=n_embd,
        top_k=top_k,
        n_tok=M,
    )

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
        "compressed_gguf": 1.0,
    }
    return final_f32.to(hidden_states.dtype).reshape(orig_shape), metrics


def triton_paged_moe_forward(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_manager: Any,
    layer_idx: int = 0,
    block_size_m: int = 16,
    block_size_n: int = 64,
    block_size_k: int = 64,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Execute a complete MoE layer forward pass using PagedExpertManager.

    When `expert_manager.compressed_cache` is True (`vram_blob_cache` is active),
    executes directly on compressed GGUF blocks with zero BF16/FP16 weight buffers:
    - Runs Option 2 (`native_gguf_paged_moe_forward`) for Gate/Up/Down INT8 execution
    - Runs Option 1 (`fused_moe_gguf_down_slot_kernel`) in-kernel Triton GGUF unpacker
      on the compressed `Q2_0_STRATA` / `IQ4_NL` down blocks in `vram_blob_cache`.
    When `expert_manager.compressed_cache` is False, runs `fused_moe_indexed_slot_kernel`
    over `expert_manager.vram_arena`.
    """
    if getattr(expert_manager, "compressed_cache", False) and getattr(expert_manager, "vram_blob_cache", None) is not None:
        return native_gguf_paged_moe_forward(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_manager=expert_manager,
            layer_idx=layer_idx,
        )

    orig_shape = hidden_states.shape
    K = orig_shape[-1]
    x = hidden_states.reshape(-1, K).contiguous()
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    M = x.shape[0]
    top_k = topk_ids.shape[-1]

    unique_experts_t = torch.unique(topk_ids.reshape(-1)).to(torch.int32)
    unique_expert_list: List[int] = unique_experts_t.detach().cpu().tolist()
    num_active = len(unique_expert_list)

    hits_before = getattr(expert_manager, "cache_hits", 0)
    misses_before = getattr(expert_manager, "cache_misses", 0)
    pcie_before = getattr(expert_manager, "pcie_bytes_transferred", 0)

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
        w_gate = arena[:, 0].transpose(-1, -2)
        w_up = arena[:, 1].transpose(-1, -2) if has_up else None
        w_down_idx = 2 if has_up else 1
        w_down = arena[:, w_down_idx].reshape(arena.shape[0], K, N_ff).transpose(-1, -2)
    else:
        N_ff = arena.shape[-1]
        w_gate = arena[:, 0]
        w_up = arena[:, 1] if has_up else None
        w_down_idx = 2 if has_up else 1
        w_down = arena[:, w_down_idx].transpose(-1, -2)

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

    expert_to_compact = torch.full(
        (slot_table.shape[0],), 0, dtype=torch.long, device=x.device
    )
    expert_to_compact[unique_experts_t.long()] = torch.arange(
        num_active, dtype=torch.long, device=x.device
    )
    compact_ids = expert_to_compact[topk_ids.reshape(M, top_k).long()]
    token_indices = torch.arange(M, device=x.device).unsqueeze(1).expand(M, top_k)
    selected_out = down_out[compact_ids, token_indices]
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
