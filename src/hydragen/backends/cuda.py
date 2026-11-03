# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/backends/cuda.py - CUDA SM80+ Hardware Plugin Backend.

Encapsulates all CUDA-specific execution components:
- CUDAPagedAttentionBackend: Hardware-accelerated PTX DP4A SIMD, 128-bit vector loads, and LPT prefill.
- CUDAMoEBackend: Triton MoE indirection and IQ3_XXS dequantization kernels.
- CUDAHardwareBackend: CUDA device introspection, SM_86 capability flags, and memory management.
"""

from __future__ import annotations

import ctypes
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


class CUDAPagedAttentionBackend(PagedAttentionBackend):
    """Native CUDA Paged Attention kernel backend using compiled libpaged_kv.so."""

    def __init__(self, native_lib: Optional[ctypes.CDLL] = None):
        self._native_lib = native_lib
        if self._native_lib is not None:
            self._setup_c_api(self._native_lib)

    @property
    def name(self) -> str:
        return "cuda_paged_attention"

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CUDA

    def is_available(self) -> bool:
        return (
            torch.cuda.is_available()
            and self._native_lib is not None
            and hasattr(self._native_lib, "paged_attention_decode")
            and hasattr(self._native_lib, "paged_attention_prefill")
        )

    def _setup_c_api(self, lib: ctypes.CDLL) -> None:
        if hasattr(lib, "paged_attention_decode"):
            lib.paged_attention_decode.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_int32, ctypes.c_int32,
                ctypes.c_float, ctypes.c_int32, ctypes.c_void_p,
            ]
        if hasattr(lib, "paged_attention_prefill"):
            lib.paged_attention_prefill.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
            ]

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
        if not self.is_available() or not (hasattr(q, "is_cuda") and q.is_cuda):
            # Fall back to CPU implementation if invoked on non-CUDA tensor
            from hydragen.backends.cpu import CPUPagedAttentionBackend
            return CPUPagedAttentionBackend().paged_attention_decode(
                out=out, q=q, k_cache=k_cache, v_cache=v_cache,
                block_tables=block_tables, context_lens=context_lens,
                scale=scale, is_int8_kv=is_int8_kv, stream=stream,
                block_size=block_size,
            )

        num_seqs = q.shape[0]
        num_heads = q.shape[1]
        head_dim = q.shape[2]
        num_kv_heads = k_cache.shape[1]
        max_blocks = block_tables.shape[1]
        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)

        stream_ptr = stream.cuda_stream if hasattr(stream, "cuda_stream") else (stream or 0)
        self._native_lib.paged_attention_decode(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(q.data_ptr()),
            ctypes.c_void_p(k_cache.data_ptr()),
            ctypes.c_void_p(v_cache.data_ptr()),
            ctypes.c_void_p(block_tables.data_ptr()),
            ctypes.c_void_p(context_lens.data_ptr()),
            int(num_seqs), int(num_heads), int(num_kv_heads), int(head_dim),
            int(block_size), int(max_blocks),
            float(scale), int(1 if is_int8_kv else 0),
            ctypes.c_void_p(stream_ptr)
        )
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
        if not self.is_available() or not (hasattr(q, "is_cuda") and q.is_cuda):
            from hydragen.backends.cpu import CPUPagedAttentionBackend
            return CPUPagedAttentionBackend().paged_attention_prefill(
                out=out, q=q, k_cache=k_cache, v_cache=v_cache,
                block_tables=block_tables, context_lens=context_lens,
                scale=scale, is_int8_kv=is_int8_kv, reverse_lpt=reverse_lpt,
                stream=stream, block_size=block_size,
            )

        num_seqs = q.shape[0]
        query_len = q.shape[1]
        num_heads = q.shape[2]
        head_dim = q.shape[3]
        num_kv_heads = k_cache.shape[1]
        max_blocks = block_tables.shape[1]
        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)

        stream_ptr = stream.cuda_stream if hasattr(stream, "cuda_stream") else (stream or 0)
        ctx_ptr = context_lens.data_ptr() if context_lens is not None else 0
        self._native_lib.paged_attention_prefill(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(q.data_ptr()),
            ctypes.c_void_p(k_cache.data_ptr()),
            ctypes.c_void_p(v_cache.data_ptr()),
            ctypes.c_void_p(block_tables.data_ptr()),
            ctypes.c_void_p(ctx_ptr),
            int(num_seqs), int(query_len), int(num_heads), int(num_kv_heads),
            int(head_dim), int(block_size), int(max_blocks),
            float(scale), int(1 if is_int8_kv else 0), int(1 if reverse_lpt else 0),
            ctypes.c_void_p(stream_ptr)
        )
        return out


class CUDAMoEBackend(MoEBackend):
    """CUDA MoE execution backend using Triton fused kernels and IQ dequantization."""

    @property
    def name(self) -> str:
        return "cuda_triton_moe"

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CUDA

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def forward_moe(
        self,
        hidden_states: Any,
        router_logits: Any,
        top_k: int = 8,
        renormalize: bool = True,
        **kwargs: Any,
    ) -> Any:
        from hydragen.triton_moe import triton_paged_moe_forward
        return triton_paged_moe_forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            **kwargs,
        )


class CUDAHardwareBackend(HardwareBackend):
    """Hardware backend for NVIDIA Ampere, Ada Lovelace, and Hopper architectures."""

    def __init__(self, paged_kv_lib: Optional[ctypes.CDLL] = None):
        self._paged_att = CUDAPagedAttentionBackend(paged_kv_lib)
        self._moe = CUDAMoEBackend()
        self._ipc: Optional[IPCBackend] = None

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CUDA

    @property
    def capability(self) -> DeviceCapability:
        if not torch.cuda.is_available():
            return DeviceCapability(device_type=DeviceType.CUDA, device_name="CUDA Unavailable")

        props = torch.cuda.get_device_properties(0)
        major, minor = torch.cuda.get_device_capability(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()

        # Ampere (SM 80+) has hardware DP4A SIMD and cp.async
        has_dp4a = major >= 6 or (major == 8 and minor >= 6)
        has_cp_async = major >= 8

        return DeviceCapability(
            device_type=DeviceType.CUDA,
            device_name=props.name,
            compute_capability=(major, minor),
            total_memory_bytes=total_bytes,
            free_memory_bytes=free_bytes,
            multi_processor_count=props.multi_processor_count,
            has_dp4a=has_dp4a,
            has_cp_async=has_cp_async,
            details={
                "cuda_version": torch.version.cuda,
                "is_ampere": major == 8,
                "is_hopper": major == 9,
            },
        )

    def get_paged_attention_backend(self) -> PagedAttentionBackend:
        return self._paged_att

    def get_moe_backend(self) -> Optional[MoEBackend]:
        return self._moe

    def get_ipc_backend(self) -> IPCBackend:
        if self._ipc is None:
            from hydragen.backends.ipc import NativeIPCBackend
            self._ipc = NativeIPCBackend()
        return self._ipc
