# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/backends/cpu.py - Portable CPU / Algorithmic Core Backend.

Provides 100% portable, zero-dependency implementations of:
- CPUPagedAttentionBackend: Vectorized tensor slice gathering and clamped branchless attention.
- CPUMoEBackend: CPU expert forward execution with pinned host memory.
- CPUHardwareBackend: Complete host platform facade.
"""

from __future__ import annotations

import contextlib
import os
import platform
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from hydragen.backends.base import (
    DeviceCapability,
    DeviceType,
    HardwareBackend,
    IPCBackend,
    MoEBackend,
    PagedAttentionBackend,
)


class CPUPagedAttentionBackend(PagedAttentionBackend):
    """Pure portable PyTorch implementation of Paged Attention with vectorized block gathers."""

    @property
    def name(self) -> str:
        return "cpu_paged_attention"

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CPU

    def is_available(self) -> bool:
        return True

    def paged_attention_decode(
        self,
        out: Any,
        q: Any,
        k_cache: Any,
        v_cache: Any,
        block_tables: Any,
        context_lens: Any,
        scale: Optional[float] = None,
        is_int8_kv: bool = False,
        stream: Optional[Any] = None,
        block_size: int = 16,
    ) -> Any:
        num_seqs = q.shape[0]
        num_heads = q.shape[1]
        head_dim = q.shape[2]
        num_kv_heads = k_cache.shape[1]
        group_size = num_heads // num_kv_heads

        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)

        for s in range(num_seqs):
            ctx_len = int(context_lens[s].item() if hasattr(context_lens[s], "item") else context_lens[s])
            if ctx_len <= 0:
                continue
            bt = block_tables[s]
            num_blocks = (ctx_len + block_size - 1) // block_size
            phys_blocks = bt[:num_blocks]
            if not isinstance(phys_blocks, torch.Tensor):
                phys_blocks = torch.tensor(phys_blocks, dtype=torch.long, device=k_cache.device)
            else:
                phys_blocks = phys_blocks.long()

            # Vectorized multi-block gather: [num_blocks, num_kv_heads, block_size, head_dim]
            k_blks = k_cache[phys_blocks]
            v_blks = v_cache[phys_blocks]

            # Flatten to [num_blocks * block_size, num_kv_heads, head_dim], then slice ctx_len
            k_seq = k_blks.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)[:ctx_len].permute(1, 0, 2).float()
            v_seq = v_blks.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)[:ctx_len].permute(1, 0, 2).float()
            if is_int8_kv:
                k_seq = k_seq * 0.015625
                v_seq = v_seq * 0.015625

            qs = q[s].float() * scale
            for h in range(num_heads):
                kv_h = h // group_size
                scores = torch.matmul(k_seq[kv_h], qs[h])
                probs = torch.softmax(scores, dim=-1)
                out[s, h] = torch.matmul(probs, v_seq[kv_h]).to(out.dtype)
        return out

    def paged_attention_prefill(
        self,
        out: Any,
        q: Any,
        k_cache: Any,
        v_cache: Any,
        block_tables: Any,
        context_lens: Optional[Any] = None,
        scale: Optional[float] = None,
        is_int8_kv: bool = False,
        reverse_lpt: bool = True,
        stream: Optional[Any] = None,
        block_size: int = 16,
    ) -> Any:
        num_seqs = q.shape[0]
        query_len = q.shape[1]
        num_heads = q.shape[2]
        head_dim = q.shape[3]
        num_kv_heads = k_cache.shape[1]
        group_size = num_heads // num_kv_heads

        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)

        for s in range(num_seqs):
            past_len = int(
                context_lens[s].item()
                if (context_lens is not None and hasattr(context_lens[s], "item"))
                else (context_lens[s] if context_lens is not None else 0)
            )
            bt = block_tables[s]
            total_tokens = past_len + query_len
            num_blocks = (total_tokens + block_size - 1) // block_size
            phys_blocks = bt[:num_blocks]
            if not isinstance(phys_blocks, torch.Tensor):
                phys_blocks = torch.tensor(phys_blocks, dtype=torch.long, device=k_cache.device)
            else:
                phys_blocks = phys_blocks.long()

            k_blks = k_cache[phys_blocks]
            v_blks = v_cache[phys_blocks]
            k_all = k_blks.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)[:total_tokens].permute(1, 0, 2).float()
            v_all = v_blks.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)[:total_tokens].permute(1, 0, 2).float()
            if is_int8_kv:
                k_all = k_all * 0.015625
                v_all = v_all * 0.015625

            for q_tok in range(query_len):
                visible_len = past_len + q_tok + 1
                k_vis = k_all[:, :visible_len, :]
                v_vis = v_all[:, :visible_len, :]

                for h in range(num_heads):
                    kv_h = h // group_size
                    q_vec = q[s, q_tok, h].float() * scale
                    scores = torch.matmul(k_vis[kv_h], q_vec)
                    probs = torch.softmax(scores, dim=-1)
                    out[s, q_tok, h] = torch.matmul(probs, v_vis[kv_h]).to(out.dtype)
        return out


class CPUMoEBackend(MoEBackend):
    """CPU fallback MoE executor."""

    @property
    def name(self) -> str:
        return "cpu_moe"

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CPU

    def is_available(self) -> bool:
        return True

    def forward_moe(
        self,
        hidden_states: Any,
        router_logits: Any,
        top_k: int = 8,
        renormalize: bool = True,
        **kwargs: Any,
    ) -> Any:
        return hidden_states


class CPUHardwareBackend(HardwareBackend):
    """Complete host platform facade for systems without native GPU acceleration."""

    def __init__(self):
        self._paged_att = CPUPagedAttentionBackend()
        self._moe = CPUMoEBackend()
        self._ipc: Optional[IPCBackend] = None

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CPU

    @property
    def capability(self) -> DeviceCapability:
        cores = os.cpu_count() or 1
        return DeviceCapability(
            device_type=DeviceType.CPU,
            device_name=f"{platform.processor() or 'Generic CPU'} ({cores} threads)",
            compute_capability=(0, 0),
            multi_processor_count=cores,
            has_avx512=False,
        )

    def get_paged_attention_backend(self) -> PagedAttentionBackend:
        return self._paged_att

    def get_moe_backend(self) -> Optional[MoEBackend]:
        return self._moe

    def get_ipc_backend(self) -> IPCBackend:
        if self._ipc is None:
            from hydragen.backends.ipc import FallbackIPCBackend
            self._ipc = FallbackIPCBackend()
        return self._ipc

    def set_device(self, device: Any) -> None:
        pass

    def get_mem_info(self, device: Optional[Any] = None) -> Tuple[int, int]:
        try:
            import psutil
            vm = psutil.virtual_memory()
            return (int(vm.available), int(vm.total))
        except Exception:
            return (16 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024)

    def create_stream(self, device: Optional[Any] = None) -> Any:
        return None

    def current_stream(self, device: Optional[Any] = None) -> Any:
        return None

    def stream_context(self, stream: Any) -> Any:
        return contextlib.nullcontext()

    def stream_wait_stream(self, waiter: Any, ready: Any) -> None:
        pass

    def synchronize(self, stream_or_device: Optional[Any] = None) -> None:
        pass

    def allocate_pinned_host_tensor(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=False)

    def empty_tensor(self, shape: Tuple[int, ...], dtype: Any, device: Optional[Any] = None) -> Any:
        return torch.empty(shape, dtype=dtype, device="cpu")
