# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/backends/base.py - Abstract Interfaces for Dynamic Hardware Plugins.

Defines the contract between the portable algorithmic core of Hydragen and
hardware-specific native execution libraries (.so/.dll):
- DeviceType & DeviceCapability: Hardware introspection and capability queries.
- PagedAttentionBackend: Virtual page allocation, 128-bit KV fetch, and decode/prefill attention.
- MoEBackend: GGUF MoE dequantization, GEMM/GEMV kernel dispatch, and expert cache management.
- IPCBackend: Lock-free ring buffer and inter-process token streaming.
- HardwareBackend: Unified facade binding a target device family (CUDA, ROCm, CPU, Hybrid).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class DeviceType(str, Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    INTEL = "intel"
    XPU = "xpu"
    CPU = "cpu"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


@dataclass
class DeviceCapability:
    device_type: DeviceType
    device_name: str
    compute_capability: Tuple[int, int] = (0, 0)
    total_memory_bytes: int = 0
    free_memory_bytes: int = 0
    multi_processor_count: int = 0
    has_dp4a: bool = False
    has_cp_async: bool = False
    has_wmma: bool = False
    has_avx512: bool = False
    has_dpas: bool = False
    has_amx: bool = False
    has_vnni: bool = False
    has_mfma: bool = False
    details: Optional[Dict[str, Any]] = None


class PagedAttentionBackend(ABC):
    """Abstract interface for Paged Attention memory operations and CUDA/CPU kernels."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of this backend implementation."""
        pass

    @property
    @abstractmethod
    def device_type(self) -> DeviceType:
        """Target device family."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if native library / requirements are met and ready to execute."""
        pass

    @abstractmethod
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
        """Executes single-token decode attention across paged virtual memory blocks."""
        pass

    @abstractmethod
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
        """Executes causal prefill attention with Reverse Query-Tile Attention Scheduling (LPT)."""
        pass


class MoEBackend(ABC):
    """Abstract interface for MoE expert weight dequantization and execution."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def device_type(self) -> DeviceType:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def forward_moe(
        self,
        hidden_states: Any,
        router_logits: Any,
        top_k: int = 8,
        renormalize: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Executes MoE routing and expert kernel execution."""
        pass


class IPCBackend(ABC):
    """Abstract interface for Lock-Free Inter-Process Communication."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def create(self, stdout_fd: int, num_slots: int = 4) -> Any:
        pass

    @abstractmethod
    def set_control_callback(self, handle: Any, cb: Callable[[bytes], None]) -> None:
        pass

    @abstractmethod
    def read_token(self, handle: Any, slot: int, timeout_ms: int = 100) -> Optional[int]:
        pass

    @abstractmethod
    def clear_slot(self, handle: Any, slot: int) -> None:
        pass

    @abstractmethod
    def stop(self, handle: Any) -> None:
        pass

    @abstractmethod
    def destroy(self, handle: Any) -> None:
        pass


class HardwareBackend(ABC):
    """Unified hardware backend facade uniting memory, compute, and IPC subsystems."""

    @property
    @abstractmethod
    def device_type(self) -> DeviceType:
        pass

    @property
    @abstractmethod
    def capability(self) -> DeviceCapability:
        pass

    @abstractmethod
    def get_paged_attention_backend(self) -> PagedAttentionBackend:
        pass

    @abstractmethod
    def get_moe_backend(self) -> Optional[MoEBackend]:
        pass

    @abstractmethod
    def get_ipc_backend(self) -> IPCBackend:
        pass

    def set_device(self, device: Any) -> None:
        """Sets active execution device context."""
        pass

    def get_mem_info(self, device: Optional[Any] = None) -> Tuple[int, int]:
        """Returns (free_bytes, total_bytes) for the specified device."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            return (int(vm.available), int(vm.total))
        except Exception:
            return (16 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024)

    def create_stream(self, device: Optional[Any] = None) -> Any:
        """Creates an asynchronous hardware execution stream (or None on CPU)."""
        return None

    def current_stream(self, device: Optional[Any] = None) -> Any:
        """Returns the current active hardware execution stream."""
        return None

    def stream_context(self, stream: Any) -> Any:
        """Context manager executing operations inside the given hardware stream."""
        import contextlib
        return contextlib.nullcontext()

    def stream_wait_stream(self, waiter: Any, ready: Any) -> None:
        """Synchronizes two execution streams so waiter blocks until ready completes."""
        pass

    def synchronize(self, stream_or_device: Optional[Any] = None) -> None:
        """Synchronizes device execution."""
        pass

    def allocate_pinned_host_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: Any,
    ) -> Any:
        """Allocates host CPU memory, page-locked (pinned) for DMA transfers if supported."""
        import torch
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=False)

    def empty_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: Any,
        device: Optional[Any] = None,
    ) -> Any:
        """Allocates an uninitialized device tensor."""
        import torch
        dev = "cpu" if device is None else device
        return torch.empty(shape, dtype=dtype, device=dev)
