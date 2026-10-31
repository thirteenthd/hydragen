# serve/strata_gguf.py
"""Option A Strata GGUF Loader and On-The-Fly GPU Dequantizer.

Directly bridges Strata's GSQ-RCO-IQ3_XXS / IQ2_XS / IQ4_NL quantized model shards
into vLLM's PagedExpertManager and Triton Fused MoE kernels without requiring
offline model format conversion to SafeTensors.
"""

from __future__ import annotations

import ctypes
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES


# 1. Custom GGML Quantization Types for Strata GSQ/IQ/Trellis Models
CUSTOM_STRATA_TYPES: Dict[int, Tuple[str, int, int]] = {
    42: ("Q2_0_STRATA", 64, 18),
    40: ("NVFP4_STRATA", 64, 36),
    39: ("MXFP4_STRATA", 32, 17),
    143: ("PTQ1_0", 128, 28),
    144: ("TQ2_T", 128, 34),
    145: ("TQK6", 128, 26),
    146: ("TQK7", 128, 30),
    147: ("TQP5", 128, 22),
    148: ("TQP6", 128, 26),
    149: ("TQP7", 128, 30),
}


class _MockGGMLType:
    def __init__(self, name: str, val: int):
        self.name = name
        self.value = val

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<GGMLQuantizationType.{self.name}: {self.value}>"


def register_strata_gguf_types() -> None:
    """Register Strata-specific quantization types into python-gguf tables."""
    for val, (name, elems, bytes_per_blk) in CUSTOM_STRATA_TYPES.items():
        m = _MockGGMLType(name, val)
        GGMLQuantizationType._value2member_map_[val] = m
        if not hasattr(GGMLQuantizationType, name):
            setattr(GGMLQuantizationType, name, m)
        GGML_QUANT_SIZES[m] = (elems, bytes_per_blk)
        GGML_QUANT_SIZES[val] = (elems, bytes_per_blk)


# Run registration on module import
register_strata_gguf_types()


def find_default_lib_path() -> str:
    """Locate libstrata_iq_kernels.so dynamically across standard install locations."""
    if "STRATA_IQ_KERNELS_LIB" in os.environ and os.path.exists(os.environ["STRATA_IQ_KERNELS_LIB"]):
        return os.environ["STRATA_IQ_KERNELS_LIB"]
    
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "libstrata_iq_kernels.so"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "libstrata_iq_kernels.so"),
        os.path.join(os.getcwd(), "libstrata_iq_kernels.so"),
        os.path.join(os.getcwd(), "serve", "libstrata_iq_kernels.so"),
        "/home/thirteenthd/libstrata_iq_kernels.so",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return candidates[0]


class StrataCUDAKernelBridge:
    """Ctypes bridge to compiled Strata CUDA kernels (libstrata_iq_kernels.so)."""

    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path or find_default_lib_path()
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(
                f"Strata CUDA kernel library not found at: {self.lib_path}. "
                "Ensure libstrata_iq_kernels.so is placed in the serve/ directory or set STRATA_IQ_KERNELS_LIB."
            )

        self._cdll = ctypes.CDLL(self.lib_path)

        # void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream, const float* scale)
        self._iq_dequant_f16 = self._cdll._ZN6strata7kernels14iq_dequant_f16EiPKvlPtPvPKf
        self._iq_dequant_f16.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._iq_dequant_f16.restype = None

        # void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream, const float* tail)
        self._iq_dequant_gu_f16 = self._cdll._ZN6strata7kernels17iq_dequant_gu_f16EiPKvS2_llPtPvPKf
        self._iq_dequant_gu_f16.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._iq_dequant_gu_f16.restype = None

    def dequant_f16(
        self,
        ggml_type: int,
        src_ptr: int,
        num_elements: int,
        dst_ptr: int,
        cuda_stream: Any = None,
        scale_ptr: Optional[int] = None,
    ) -> None:
        """Dequantize contiguous blocks directly into destination FP16 tensor."""
        stream_val = cuda_stream.cuda_stream if cuda_stream is not None else None
        self._iq_dequant_f16(
            int(ggml_type),
            ctypes.c_void_p(src_ptr),
            int(num_elements),
            ctypes.c_void_p(dst_ptr),
            ctypes.c_void_p(stream_val),
            ctypes.c_void_p(scale_ptr) if scale_ptr else None,
        )


class StrataGGUFMoELoader:
    """Loads weights from Strata GGUF model shards and streams experts on-the-fly."""

    def __init__(
        self,
        shard_path: str,
        kernel_lib_path: Optional[str] = None,
    ):
        self.shard_path = shard_path
        self.bridge = StrataCUDAKernelBridge(kernel_lib_path)
        register_strata_gguf_types()

        self.reader = gguf.GGUFReader(shard_path)
        self.tensor_map = {t.name: t for t in self.reader.tensors}

    def get_layer_expert_metadata(self, layer_idx: int) -> Dict[str, Any]:
        """Inspect and return expert dimensions and quantization types for a layer."""
        gate_name = f"blk.{layer_idx}.ffn_gate_exps.weight"
        down_name = f"blk.{layer_idx}.ffn_down_exps.weight"

        gate_t = self.tensor_map[gate_name]
        down_t = self.tensor_map[down_name]

        n_embd = int(gate_t.shape[0])
        n_ff = int(gate_t.shape[1])
        n_exp = int(gate_t.shape[2])
        n_elems = n_embd * n_ff

        gate_type = int(gate_t.tensor_type)
        down_type = int(down_t.tensor_type)

        bytes_gate = (n_elems // 256) * 74 if gate_type == 17 else (n_elems // 256) * 98
        bytes_down = (n_elems // 32) * 18 if down_type == 20 else (n_elems // 64) * 18

        return {
            "n_embd": n_embd,
            "n_ff": n_ff,
            "n_exp": n_exp,
            "n_elems": n_elems,
            "gate_type": gate_type,
            "down_type": down_type,
            "bytes_gate": bytes_gate,
            "bytes_down": bytes_down,
        }

    def dequantize_expert_to_gpu(
        self,
        layer_idx: int,
        expert_idx: int,
        dst_gate_fp16: torch.Tensor,
        dst_down_fp16: torch.Tensor,
        staging_gate_u8: torch.Tensor,
        staging_down_u8: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Fetch raw expert from host mmap, DMA to GPU staging, and dequantize to FP16."""
        meta = self.get_layer_expert_metadata(layer_idx)
        gate_t = self.tensor_map[f"blk.{layer_idx}.ffn_gate_exps.weight"]
        down_t = self.tensor_map[f"blk.{layer_idx}.ffn_down_exps.weight"]

        bg = meta["bytes_gate"]
        bd = meta["bytes_down"]

        # 0-copy slice from mmap buffer
        g_raw = np.frombuffer(gate_t.data, dtype=np.uint8, count=bg, offset=expert_idx * bg)
        d_raw = np.frombuffer(down_t.data, dtype=np.uint8, count=bd, offset=expert_idx * bd)

        # Async PCIe DMA
        staging_gate_u8.copy_(torch.from_numpy(np.copy(g_raw)), non_blocking=True)
        staging_down_u8.copy_(torch.from_numpy(np.copy(d_raw)), non_blocking=True)

        # GPU Dequantize
        self.bridge.dequant_f16(
            meta["gate_type"],
            staging_gate_u8.data_ptr(),
            meta["n_elems"],
            dst_gate_fp16.data_ptr(),
            cuda_stream=stream,
        )
        self.bridge.dequant_f16(
            meta["down_type"],
            staging_down_u8.data_ptr(),
            meta["n_elems"],
            dst_down_fp16.data_ptr(),
            cuda_stream=stream,
        )
