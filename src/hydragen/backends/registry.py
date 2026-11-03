# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/backends/registry.py - Dynamic Hardware Plugin Discovery & Registry.

Discovers and binds the optimal native execution libraries (.so/.dll) for the host
hardware environment:
1. Introspects hardware: NVIDIA Ampere/Ada/Hopper (CUDA), AMD ROCm, or Host CPU.
2. Discovers compiled plugins in standard search roots (lib/, build/, installed package).
3. Automatically falls back to high-performance portable CPU implementations if no
   compatible accelerator library is present.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Dict, List, Optional, Type

import torch

from hydragen.backends.base import (
    DeviceCapability,
    DeviceType,
    HardwareBackend,
    IPCBackend,
    MoEBackend,
    PagedAttentionBackend,
)
from hydragen.backends.cpu import CPUHardwareBackend, CPUPagedAttentionBackend
from hydragen.backends.cuda import CUDAHardwareBackend, CUDAPagedAttentionBackend
from hydragen.backends.ipc import FallbackIPCBackend, NativeIPCBackend


class BackendRegistry:
    """Singleton registry and factory for Hydragen dynamic hardware plugins."""

    _instance: Optional[BackendRegistry] = None
    _paged_kv_lib: Optional[ctypes.CDLL] = None
    _hardware_backend: Optional[HardwareBackend] = None

    def __init__(self):
        self._custom_backends: Dict[str, HardwareBackend] = {}

    @classmethod
    def get_instance(cls) -> BackendRegistry:
        if cls._instance is None:
            cls._instance = BackendRegistry()
        return cls._instance

    @classmethod
    def discover_paged_attention_library(cls) -> Optional[ctypes.CDLL]:
        """Discovers and dynamically loads the optimal PagedAttention shared library."""
        if cls._paged_kv_lib is not None:
            return cls._paged_kv_lib

        from hydragen.lib_loader import load_native_library

        # Priority 1: Hardware-specific naming (e.g. sm_86, sm_80)
        candidates = []
        if torch.cuda.is_available():
            try:
                major, minor = torch.cuda.get_device_capability(0)
                arch_tag = f"sm_{major}{minor}"
                candidates.extend([
                    f"libpaged_kv_cuda_{arch_tag}.so",
                    f"libpaged_kv_{arch_tag}.so",
                    "libpaged_kv_cuda.so",
                ])
            except Exception:
                pass

        # Priority 2: Standard shared library names
        candidates.extend([
            "libpaged_kv.so",
            "libpaged_kv.dll",
        ])

        for cand in candidates:
            lib = load_native_library(cand, env_var="LIBPAGED_KV_SO")
            if lib is not None:
                cls._paged_kv_lib = lib
                return cls._paged_kv_lib

        return None

    @classmethod
    def detect_device_type(cls) -> DeviceType:
        """Determines the primary hardware acceleration target."""
        forced = os.environ.get("HYDRAGEN_DEVICE")
        if forced:
            forced_lower = forced.lower()
            if forced_lower in ("cuda", "gpu"):
                return DeviceType.CUDA
            elif forced_lower in ("rocm", "hip"):
                return DeviceType.ROCM
            elif forced_lower == "cpu":
                return DeviceType.CPU

        if torch.cuda.is_available():
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                return DeviceType.ROCM
            return DeviceType.CUDA

        return DeviceType.CPU

    @classmethod
    def get_hardware_backend(cls, force_device: Optional[DeviceType] = None) -> HardwareBackend:
        """Returns the active HardwareBackend for the current platform."""
        target_device = force_device or cls.detect_device_type()

        if target_device == DeviceType.CUDA and torch.cuda.is_available():
            paged_lib = cls.discover_paged_attention_library()
            return CUDAHardwareBackend(paged_kv_lib=paged_lib)

        return CPUHardwareBackend()

    @classmethod
    def get_paged_attention_backend(
        cls, force_device: Optional[DeviceType] = None
    ) -> PagedAttentionBackend:
        """Returns the optimal PagedAttentionBackend."""
        hw = cls.get_hardware_backend(force_device=force_device)
        backend = hw.get_paged_attention_backend()
        if backend.is_available():
            return backend
        # Fall back to CPU if native backend unavailable
        return CPUPagedAttentionBackend()

    @classmethod
    def get_ipc_backend(cls) -> IPCBackend:
        """Returns the optimal IPCBackend."""
        from hydragen.backends.ipc import NativeIPCBackend
        native_ipc = NativeIPCBackend()
        if native_ipc.is_available():
            return native_ipc
        return FallbackIPCBackend()


# Convenience top-level functions
def get_hardware_backend(force_device: Optional[DeviceType] = None) -> HardwareBackend:
    return BackendRegistry.get_hardware_backend(force_device=force_device)


def get_paged_attention_backend(force_device: Optional[DeviceType] = None) -> PagedAttentionBackend:
    return BackendRegistry.get_paged_attention_backend(force_device=force_device)


def get_ipc_backend() -> IPCBackend:
    return BackendRegistry.get_ipc_backend()
