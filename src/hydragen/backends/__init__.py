# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/backends/__init__.py - Dynamic Hardware Plugin Modularization Package."""

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
from hydragen.backends.registry import (
    BackendRegistry,
    get_hardware_backend,
    get_ipc_backend,
    get_paged_attention_backend,
)
from hydragen.backends.rocm import (
    ROCMHardwareBackend,
    ROCMMoEBackend,
    ROCMPagedAttentionBackend,
)

__all__ = [
    "DeviceType",
    "DeviceCapability",
    "PagedAttentionBackend",
    "MoEBackend",
    "IPCBackend",
    "HardwareBackend",
    "CPUHardwareBackend",
    "CPUPagedAttentionBackend",
    "CUDAHardwareBackend",
    "CUDAPagedAttentionBackend",
    "IntelHardwareBackend",
    "IntelPagedAttentionBackend",
    "IntelMoEBackend",
    "ROCMHardwareBackend",
    "ROCMPagedAttentionBackend",
    "ROCMMoEBackend",
    "NativeIPCBackend",
    "FallbackIPCBackend",
    "BackendRegistry",
    "get_hardware_backend",
    "get_paged_attention_backend",
    "get_ipc_backend",
]
