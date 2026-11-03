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
from hydragen.backends.intel import (
    IntelHardwareBackend,
    IntelMoEBackend,
    IntelPagedAttentionBackend,
)
from hydragen.backends.ipc import FallbackIPCBackend, NativeIPCBackend


class _CDeviceCaps(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_name", ctypes.c_char * 128),
        ("compute_major", ctypes.c_int32),
        ("compute_minor", ctypes.c_int32),
        ("total_vram_bytes", ctypes.c_uint64),
        ("free_vram_bytes", ctypes.c_uint64),
        ("num_multiprocessors", ctypes.c_int32),
        ("has_dp4a", ctypes.c_int32),
        ("has_cp_async", ctypes.c_int32),
        ("has_wmma", ctypes.c_int32),
        ("has_avx512", ctypes.c_int32),
        ("has_dpas", ctypes.c_int32),
        ("has_amx", ctypes.c_int32),
    ]


class _CPluginDescriptor(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int32),
        ("plugin_name", ctypes.c_char * 64),
        ("target_architecture", ctypes.c_char * 32),
        ("build_timestamp", ctypes.c_char * 32),
        ("caps", _CDeviceCaps),
        ("attention_ops", ctypes.c_void_p),
    ]


class BackendRegistry:
    """Singleton registry and factory for Hydragen dynamic hardware plugins."""

    _instance: Optional[BackendRegistry] = None
    _paged_kv_lib: Optional[ctypes.CDLL] = None
    _hardware_backend: Optional[HardwareBackend] = None
    _plugin_desc: Optional[Dict[str, Any]] = None

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

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            candidates.extend([
                "libpaged_kv_intel.so",
                "libpaged_kv_intel.dll",
                "libpaged_kv_sycl.so",
                "libpaged_kv_sycl.dll",
            ])

        # Priority 2: Standard shared library names
        candidates.extend([
            "libpaged_kv.so",
            "libpaged_kv.dll",
            "libpaged_kv_intel.so",
            "libpaged_kv_intel.dll",
        ])

        for cand in candidates:
            lib = load_native_library(cand, env_var="LIBPAGED_KV_SO")
            if lib is not None:
                cls._paged_kv_lib = lib
                # Check for standard cross-platform C-ABI plugin descriptor
                if hasattr(lib, "hydragen_get_descriptor"):
                    try:
                        lib.hydragen_get_descriptor.restype = ctypes.POINTER(_CPluginDescriptor)
                        desc_ptr = lib.hydragen_get_descriptor()
                        if desc_ptr:
                            raw = desc_ptr.contents
                            name_str = raw.plugin_name.decode("utf-8", errors="replace")
                            arch_str = raw.target_architecture.decode("utf-8", errors="replace")
                            cls._plugin_desc = {
                                "name": name_str,
                                "arch": arch_str,
                                "abi_version": raw.abi_version,
                                "has_dp4a": bool(raw.caps.has_dp4a),
                                "has_cp_async": bool(raw.caps.has_cp_async),
                                "has_dpas": bool(raw.caps.has_dpas),
                                "has_amx": bool(raw.caps.has_amx),
                            }
                            print(f"[hydragen] Loaded native C-ABI hardware plugin: {name_str} (arch={arch_str}, ABI=v{raw.abi_version})")
                    except Exception:
                        pass
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
            elif forced_lower in ("intel", "xpu", "arc", "sycl"):
                return DeviceType.INTEL
            elif forced_lower == "cpu":
                return DeviceType.CPU

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return DeviceType.INTEL

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

        if target_device in (DeviceType.INTEL, DeviceType.XPU):
            paged_lib = cls.discover_paged_attention_library()
            return IntelHardwareBackend(intel_lib=paged_lib)

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
