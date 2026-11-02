# SPDX-License-Identifier: Apache-2.0
"""Strata GGUF Loader and On-The-Fly GPU Dequantizer Bridge.

Bridges Strata's GSQ-RCO-IQ3_XXS / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_S / IQ4_NL /
Q2_0_STRATA (type 42) and K-quant GGUF model shards into PagedExpertManager and
Triton Fused MoE kernels using libstrata_iq_kernels.so.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any, Dict, Optional, Tuple

import gguf
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
import numpy as np
import torch


# Custom GGML Quantization Types for Strata GSQ/IQ/Trellis Models
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


class _MockGGMLType(int):
    def __new__(cls, name: str, val: int):
        obj = super().__new__(cls, val)
        obj.name = name
        obj.value = val
        return obj

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


register_strata_gguf_types()


def _find_default_lib_path() -> str:
    env_path = os.environ.get("STRATA_IQ_KERNELS_SO")
    if env_path and os.path.exists(env_path):
        return env_path

    from pathlib import Path
    candidates = [
        Path(__file__).parent / "server" / "libstrata_iq_kernels.so",
        Path(__file__).parent.parent.parent / "serve" / "libstrata_iq_kernels.so",
        Path.cwd() / "serve" / "libstrata_iq_kernels.so",
        Path.cwd() / "libstrata_iq_kernels.so",
        Path(__file__).parent / "server" / "strata_iq_kernels.dll",
        Path.cwd() / "strata_iq_kernels.dll",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return env_path or "libstrata_iq_kernels.so"


class StrataCUDAKernelBridge:
    """Ctypes bridge to compiled Strata CUDA kernels (libstrata_iq_kernels.so / .dll)."""

    @classmethod
    def get_default_path(cls) -> str:
        return _find_default_lib_path()

    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path or self.get_default_path()
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(
                f"Strata CUDA kernel library not found at: {self.lib_path}. "
                "Set STRATA_IQ_KERNELS_SO or place libstrata_iq_kernels.so in your server/ directory."
            )

        try:
            self._cdll = ctypes.CDLL(self.lib_path)
        except OSError:
            # Fallback when running cross-platform or on Windows host without native DLL
            self._cdll = None

        if self._cdll is None:
            self._iq_row_bytes = lambda t, n: int(n * 0.4)
            self._dequant_f16_rows = None
            self._dequant_bf16_rows = None
            self._iq_dequant_f16 = None
            self._iq_dequant_gu_f16 = None
            self._route_prior_top10 = None
            self._router_top10 = None
            self._dequant_q2_0_strata_f16 = None
            self._dequant_q2_0_strata_direct_f16 = None
            self._dequant_f16_rows_host = None
            self._expert_blob_info = None
            self._expert_scratch_bytes = lambda *args: 4194304
            self._native_expert_grouped = None
            self._moe_combine_multi = None
            return

        # size_t iq_row_bytes(int ggml_type, int64_t n)
        self._iq_row_bytes = self._cdll._ZN6strata7kernels12iq_row_bytesEil
        self._iq_row_bytes.argtypes = [ctypes.c_int, ctypes.c_int64]
        self._iq_row_bytes.restype = ctypes.c_size_t

        # Universal FP16 dequantizer (supports IQ1..IQ4, Q2_0_STRATA=42, Q4_K..Q6_K, Q8_0):
        # void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out, void* stream)
        self._dequant_f16_rows = getattr(
            self._cdll, "_ZN6strata7kernels11dequant_f16EiPKvlllPtPv", None
        )
        if self._dequant_f16_rows is not None:
            self._dequant_f16_rows.argtypes = [
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._dequant_f16_rows.restype = None

        # Universal BF16 dequantizer:
        # void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out, void* stream)
        self._dequant_bf16_rows = getattr(
            self._cdll, "_ZN6strata7kernels12dequant_bf16EiPKvlllPtPv", None
        )
        if self._dequant_bf16_rows is not None:
            self._dequant_bf16_rows.argtypes = [
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._dequant_bf16_rows.restype = None

        # Contiguous IQ dequantizer:
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

        # Fused gate+up interleaved FP16 dequantizer:
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

        # Cache-conditioned top-10 router (STRATA_ROUTE_PRIOR):
        # void route_prior_top10(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights, int n_tok, void* stream)
        self._route_prior_top10 = getattr(
            self._cdll, "_ZN6strata7kernels17route_prior_top10EPKfPKifPiPfiPv", None
        )
        if self._route_prior_top10 is not None:
            self._route_prior_top10.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            self._route_prior_top10.restype = None

        # Unbiased top-k router:
        # void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights, void* stream)
        self._router_top10 = getattr(
            self._cdll, "_ZN6strata7kernels12router_top10EPKfiiiPiPfPv", None
        )
        if self._router_top10 is not None:
            self._router_top10.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._router_top10.restype = None

        # Direct compressed GGUF MoE execution (Option 2: zero BF16/FP16 materialization)
        self._expert_blob_info = getattr(self._cdll, "strata_vllm_expert_blob_info", None)
        if self._expert_blob_info is not None:
            self._expert_blob_info.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
            ]
            self._expert_blob_info.restype = None

        self._expert_scratch_bytes = getattr(self._cdll, "strata_vllm_expert_scratch_bytes", None)
        if self._expert_scratch_bytes is not None:
            self._expert_scratch_bytes.argtypes = [ctypes.c_int64, ctypes.c_int64]
            self._expert_scratch_bytes.restype = ctypes.c_uint64

        self._native_expert_grouped = getattr(self._cdll, "strata_vllm_native_expert_grouped", None)
        if self._native_expert_grouped is not None:
            self._native_expert_grouped.argtypes = [
                ctypes.c_int,      # gu_type
                ctypes.c_int,      # d_type
                ctypes.c_int64,    # n_embd
                ctypes.c_int64,    # n_ff
                ctypes.c_int64,    # n_tokens
                ctypes.c_void_p,   # grp_ptr
                ctypes.c_void_p,   # grp_start
                ctypes.c_void_p,   # n_groups
                ctypes.c_void_p,   # ent_dst
                ctypes.c_void_p,   # ent_tok
                ctypes.c_int64,    # cap_groups
                ctypes.c_int64,    # cap_entries
                ctypes.c_void_p,   # x_f32
                ctypes.c_void_p,   # x_q8_1
                ctypes.c_void_p,   # scratch
                ctypes.c_void_p,   # out
                ctypes.c_void_p,   # stream
            ]
            self._native_expert_grouped.restype = None

        self._moe_combine_multi = getattr(self._cdll, "strata_vllm_moe_combine_multi", None)
        if self._moe_combine_multi is not None:
            self._moe_combine_multi.argtypes = [
                ctypes.c_void_p,   # parts
                ctypes.c_void_p,   # weights
                ctypes.c_void_p,   # shared
                ctypes.c_void_p,   # output
                ctypes.c_int64,    # n_embd
                ctypes.c_int64,    # k
                ctypes.c_int,      # n_tok
                ctypes.c_void_p,   # stream
            ]
            self._moe_combine_multi.restype = None

    def expert_blob_info(self, gu_type: int, d_type: int, n_embd: int, n_ff: int) -> Dict[str, int]:
        """Return exact byte offsets of [gate | up | down] in a compressed GGUF expert blob."""
        if self._expert_blob_info is not None:
            arr = (ctypes.c_uint64 * 5)()
            self._expert_blob_info(int(gu_type), int(d_type), int(n_embd), int(n_ff), ctypes.byref(arr))
            return {
                "gu_row": int(arr[0]),
                "d_row": int(arr[1]),
                "up_off": int(arr[2]),
                "down_off": int(arr[3]),
                "blob_bytes": int(arr[4]),
            }
        gu_row = self.row_bytes(gu_type, n_embd)
        d_row = self.row_bytes(d_type, n_ff)
        up_off = gu_row * n_ff
        down_off = up_off * 2
        blob_bytes = down_off + d_row * n_embd
        return {
            "gu_row": gu_row,
            "d_row": d_row,
            "up_off": up_off,
            "down_off": down_off,
            "blob_bytes": blob_bytes,
        }

    def expert_scratch_bytes(self, cap_entries: int, n_ff: int) -> int:
        if self._expert_scratch_bytes is not None:
            return int(self._expert_scratch_bytes(int(cap_entries), int(n_ff)))
        return int(cap_entries) * (int(n_ff) * 8 + 4096)

    @staticmethod
    def _resolve_stream_ptr(cuda_stream: Any = None) -> int:
        if cuda_stream is not None:
            return int(cuda_stream.cuda_stream)
        return int(torch.cuda.current_stream().cuda_stream)

    def native_expert_grouped_forward(
        self,
        gu_type: int,
        d_type: int,
        n_embd: int,
        n_ff: int,
        n_tokens: int,
        grp_ptr: torch.Tensor,
        grp_start: torch.Tensor,
        n_groups: torch.Tensor,
        ent_dst: torch.Tensor,
        ent_tok: torch.Tensor,
        cap_groups: int,
        cap_entries: int,
        x_f32: torch.Tensor,
        x_q8_1: torch.Tensor,
        scratch: torch.Tensor,
        out_parts: torch.Tensor,
        cuda_stream: Any = None,
    ) -> None:
        """Execute Strata's native INT8 grouped MoE directly on compressed GGUF VRAM blobs."""
        if self._native_expert_grouped is None:
            raise RuntimeError("strata_vllm_native_expert_grouped symbol not loaded in libstrata_iq_kernels.so")
        stream_val = self._resolve_stream_ptr(cuda_stream)
        self._native_expert_grouped(
            int(gu_type),
            int(d_type),
            int(n_embd),
            int(n_ff),
            int(n_tokens),
            ctypes.c_void_p(grp_ptr.data_ptr()),
            ctypes.c_void_p(grp_start.data_ptr()),
            ctypes.c_void_p(n_groups.data_ptr()),
            ctypes.c_void_p(ent_dst.data_ptr()),
            ctypes.c_void_p(ent_tok.data_ptr()),
            int(cap_groups),
            int(cap_entries),
            ctypes.c_void_p(x_f32.data_ptr()),
            ctypes.c_void_p(x_q8_1.data_ptr()),
            ctypes.c_void_p(scratch.data_ptr()),
            ctypes.c_void_p(out_parts.data_ptr()),
            ctypes.c_void_p(stream_val),
        )

    def moe_combine_multi(
        self,
        parts: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
        n_embd: int,
        top_k: int,
        n_tok: int,
        shared: Optional[torch.Tensor] = None,
        cuda_stream: Any = None,
    ) -> None:
        """Weighted reduction over top-k expert outputs on GPU."""
        stream_val = self._resolve_stream_ptr(cuda_stream)
        if self._moe_combine_multi is not None and stream_val != 0 and 1 <= top_k <= 15:
            self._moe_combine_multi(
                ctypes.c_void_p(parts.data_ptr()),
                ctypes.c_void_p(weights.data_ptr()),
                ctypes.c_void_p(shared.data_ptr()) if shared is not None else None,
                ctypes.c_void_p(output.data_ptr()),
                int(n_embd),
                int(top_k),
                int(n_tok),
                ctypes.c_void_p(stream_val),
            )
        else:
            reshaped = parts.view(n_tok, top_k, n_embd)
            weighted = (reshaped * weights.view(n_tok, top_k, 1)).sum(dim=1)
            if shared is not None:
                weighted = weighted + shared
            output.copy_(weighted)

    def row_bytes(self, ggml_type: int, cols: int) -> int:
        """Return exact byte length of one row of `cols` elements for `ggml_type`."""
        rb = int(self._iq_row_bytes(int(ggml_type), int(cols)))
        if rb > 0:
            return rb
        if ggml_type == 30 or ggml_type == 1:  # BF16 / FP16
            return cols * 2
        if ggml_type == 0:  # FP32
            return cols * 4
        if ggml_type in GGML_QUANT_SIZES:
            elems, b = GGML_QUANT_SIZES[ggml_type]
            return (cols // elems) * b
        raise ValueError(f"Unsupported ggml_type={ggml_type} for cols={cols}")

    def dequant_rows_f16(
        self,
        ggml_type: int,
        src_ptr: int,
        rows: int,
        cols: int,
        dst_ptr: int,
        row0: int = 0,
        cuda_stream: Any = None,
    ) -> None:
        """Universal row-based GPU dequantization into FP16 (handles IQ*, Q2_0_STRATA=42, Q*_K)."""
        stream_val = self._resolve_stream_ptr(cuda_stream)
        if self._dequant_f16_rows is not None:
            self._dequant_f16_rows(
                int(ggml_type),
                ctypes.c_void_p(src_ptr),
                int(row0),
                int(rows),
                int(cols),
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(stream_val),
            )
        else:
            self.dequant_f16(
                ggml_type, src_ptr, rows * cols, dst_ptr, cuda_stream=cuda_stream
            )

    def dequant_f16(
        self,
        ggml_type: int,
        src_ptr: int,
        num_elements: int,
        dst_ptr: int,
        cuda_stream: Any = None,
        scale_ptr: Optional[int] = None,
        cols: int = 256,
    ) -> None:
        """Dequantize contiguous blocks directly into destination FP16 tensor."""
        stream_val = self._resolve_stream_ptr(cuda_stream)
        # Use universal dequant_f16_rows when type is Q2_0_STRATA (42) or K-quant
        if ggml_type in (2, 6, 7, 8, 11, 12, 13, 14, 20, 23, 42) and self._dequant_f16_rows is not None:
            rows = max(1, num_elements // cols)
            cols_eff = num_elements // rows
            self._dequant_f16_rows(
                int(ggml_type),
                ctypes.c_void_p(src_ptr),
                0,
                int(rows),
                int(cols_eff),
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(stream_val),
            )
            return

        self._iq_dequant_f16(
            int(ggml_type),
            ctypes.c_void_p(src_ptr),
            int(num_elements),
            ctypes.c_void_p(dst_ptr),
            ctypes.c_void_p(stream_val),
            ctypes.c_void_p(scale_ptr) if scale_ptr else None,
        )

    def route_prior_top10(
        self,
        logits: torch.Tensor,
        res_layer: torch.Tensor,
        prior_lambda: float = 0.8,
        cuda_stream: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Strata's GPU cache-conditioned top-10 router (STRATA_ROUTE_PRIOR).

        Args:
            logits: [n_tok, 512] float32 router logits on CUDA.
            res_layer: [512] int32 residency table on CUDA (>= 0 if in VRAM, -1 if in RAM).
            prior_lambda: Logit boost for VRAM-resident experts (default 0.8).
        Returns:
            (topk_ids [n_tok, 10] int32, topk_weights [n_tok, 10] float32)
        """
        n_tok = logits.shape[0]
        ids = torch.empty((n_tok, 10), dtype=torch.int32, device=logits.device)
        weights = torch.empty((n_tok, 10), dtype=torch.float32, device=logits.device)
        stream_val = self._resolve_stream_ptr(cuda_stream)
        if self._route_prior_top10 is not None and logits.shape[1] == 512:
            self._route_prior_top10(
                ctypes.c_void_p(logits.data_ptr()),
                ctypes.c_void_p(res_layer.data_ptr()),
                ctypes.c_float(prior_lambda),
                ctypes.c_void_p(ids.data_ptr()),
                ctypes.c_void_p(weights.data_ptr()),
                int(n_tok),
                ctypes.c_void_p(stream_val),
            )
        else:
            # PyTorch fallback for arbitrary expert counts
            probs = torch.softmax(logits, dim=-1)
            biased = logits + (res_layer >= 0).float().unsqueeze(0) * prior_lambda
            _, top_idx = torch.topk(biased, k=min(10, logits.shape[1]), dim=-1)
            ids = top_idx.to(torch.int32)
            gathered = torch.gather(probs, -1, top_idx)
            weights = gathered / gathered.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return ids, weights


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
        self._layer_meta_cache: Dict[int, Dict[str, Any]] = {}

    def get_layer_expert_metadata(self, layer_idx: int) -> Dict[str, Any]:
        """Inspect and return expert dimensions and quantization types for any layer (0..47)."""
        if layer_idx in self._layer_meta_cache:
            return self._layer_meta_cache[layer_idx]

        gate_name = f"blk.{layer_idx}.ffn_gate_exps.weight"
        up_name = f"blk.{layer_idx}.ffn_up_exps.weight"
        down_name = f"blk.{layer_idx}.ffn_down_exps.weight"

        gate_t = self.tensor_map[gate_name]
        up_t = self.tensor_map[up_name]
        down_t = self.tensor_map[down_name]

        n_embd = int(gate_t.shape[0])
        n_ff = int(gate_t.shape[1])
        n_exp = int(gate_t.shape[2])
        n_elems = n_embd * n_ff

        gate_type = int(gate_t.tensor_type)
        up_type = int(up_t.tensor_type)
        down_type = int(down_t.tensor_type)

        bytes_gate = self.bridge.row_bytes(gate_type, n_embd) * n_ff
        bytes_up = self.bridge.row_bytes(up_type, n_embd) * n_ff
        bytes_down = self.bridge.row_bytes(down_type, n_ff) * n_embd

        meta = {
            "layer_idx": layer_idx,
            "n_embd": n_embd,
            "n_ff": n_ff,
            "n_exp": n_exp,
            "n_elems": n_elems,
            "gate_type": gate_type,
            "up_type": up_type,
            "down_type": down_type,
            "bytes_gate": bytes_gate,
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
            "up_off": bytes_gate,
            "down_off": bytes_gate + bytes_up,
            "total_expert_bytes": bytes_gate + bytes_up + bytes_down,
        }
        self._layer_meta_cache[layer_idx] = meta
        return meta

    def dma_compressed_expert_to_gpu(
        self,
        layer_idx: int,
        expert_idx: int,
        dst_blob_u8: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> int:
        """DMA raw compressed GGUF expert blocks [gate | up | down] directly into VRAM slot.

        Performs ZERO FP16/BF16 dequantization (1.54 - 1.69 MB per expert instead of 9.83 MB).
        """
        meta = self.get_layer_expert_metadata(layer_idx)
        gate_t = self.tensor_map[f"blk.{layer_idx}.ffn_gate_exps.weight"]
        up_t = self.tensor_map[f"blk.{layer_idx}.ffn_up_exps.weight"]
        down_t = self.tensor_map[f"blk.{layer_idx}.ffn_down_exps.weight"]

        bg = meta["bytes_gate"]
        bu = meta["bytes_up"]
        bd = meta["bytes_down"]

        g_raw = np.frombuffer(gate_t.data, dtype=np.uint8, count=bg, offset=expert_idx * bg)
        u_raw = np.frombuffer(up_t.data, dtype=np.uint8, count=bu, offset=expert_idx * bu)
        d_raw = np.frombuffer(down_t.data, dtype=np.uint8, count=bd, offset=expert_idx * bd)

        dst_blob_u8[:bg].copy_(torch.from_numpy(np.copy(g_raw)), non_blocking=True)
        dst_blob_u8[bg : bg + bu].copy_(torch.from_numpy(np.copy(u_raw)), non_blocking=True)
        dst_blob_u8[bg + bu : bg + bu + bd].copy_(torch.from_numpy(np.copy(d_raw)), non_blocking=True)
        return bg + bu + bd

    def load_router_gate(
        self, layer_idx: int, device: torch.device = torch.device("cuda:0")
    ) -> torch.Tensor:
        """Load `blk.{layer_idx}.ffn_gate_inp.weight` as [n_exp, n_embd] float32 on GPU."""
        t = self.tensor_map[f"blk.{layer_idx}.ffn_gate_inp.weight"]
        n_embd = int(t.shape[0])
        n_exp = int(t.shape[1])
        ttype = int(t.tensor_type)
        if ttype == 30:  # BF16
            raw_u16 = np.frombuffer(t.data, dtype=np.uint16, count=n_embd * n_exp)
            bf16 = torch.from_numpy(np.copy(raw_u16)).view(torch.bfloat16).reshape(n_exp, n_embd)
            return bf16.to(device=device, dtype=torch.float32)
        elif ttype == 0:  # FP32
            raw_f32 = np.frombuffer(t.data, dtype=np.float32, count=n_embd * n_exp)
            return torch.from_numpy(np.copy(raw_f32)).reshape(n_exp, n_embd).to(device=device)
        else:
            rb = self.bridge.row_bytes(ttype, n_embd) * n_exp
            raw = torch.from_numpy(np.copy(np.frombuffer(t.data, dtype=np.uint8, count=rb))).to(device)
            dst = torch.empty((n_exp, n_embd), dtype=torch.float16, device=device)
            self.bridge.dequant_rows_f16(ttype, raw.data_ptr(), n_exp, n_embd, dst.data_ptr())
            return dst.float()

    def dequantize_token_embedding(
        self, token_ids: torch.Tensor, device: torch.device = torch.device("cuda:0")
    ) -> torch.Tensor:
        """Dequantize rows `token_ids` of `token_embd.weight` to [n_tokens, n_embd] float16 on GPU."""
        t = self.tensor_map["token_embd.weight"]
        n_embd = int(t.shape[0])
        vocab_size = int(t.shape[1])
        ttype = int(t.tensor_type)
        row_b = self.bridge.row_bytes(ttype, n_embd)

        flat_ids = token_ids.detach().cpu().tolist()
        n_tok = len(flat_ids)
        packed = np.empty((n_tok, row_b), dtype=np.uint8)
        for i, tid in enumerate(flat_ids):
            clamped = max(0, min(vocab_size - 1, int(tid)))
            packed[i] = np.frombuffer(t.data, dtype=np.uint8, count=row_b, offset=clamped * row_b)

        gpu_raw = torch.from_numpy(packed).to(device=device, non_blocking=True)
        dst = torch.empty((n_tok, n_embd), dtype=torch.float16, device=device)
        self.bridge.dequant_rows_f16(ttype, gpu_raw.data_ptr(), n_tok, n_embd, dst.data_ptr())
        return dst

    def dequantize_expert_to_gpu(
        self,
        layer_idx: int,
        expert_idx: int,
        dst_gate_fp16: torch.Tensor,
        dst_down_fp16: torch.Tensor,
        staging_gate_u8: torch.Tensor,
        staging_down_u8: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
        dst_up_fp16: Optional[torch.Tensor] = None,
        staging_up_u8: Optional[torch.Tensor] = None,
    ) -> int:
        """Fetch raw expert from host mmap, DMA to GPU staging, and dequantize to FP16.

        Returns the number of compressed bytes transferred over PCIe.
        """
        meta = self.get_layer_expert_metadata(layer_idx)
        gate_t = self.tensor_map[f"blk.{layer_idx}.ffn_gate_exps.weight"]
        down_t = self.tensor_map[f"blk.{layer_idx}.ffn_down_exps.weight"]

        bg = meta["bytes_gate"]
        bd = meta["bytes_down"]
        n_embd = meta["n_embd"]
        n_ff = meta["n_ff"]

        # 0-copy slice from mmap buffer
        g_raw = np.frombuffer(gate_t.data, dtype=np.uint8, count=bg, offset=expert_idx * bg)
        d_raw = np.frombuffer(down_t.data, dtype=np.uint8, count=bd, offset=expert_idx * bd)

        # Async PCIe DMA
        staging_gate_u8[:bg].copy_(torch.from_numpy(np.copy(g_raw)), non_blocking=True)
        staging_down_u8[:bd].copy_(torch.from_numpy(np.copy(d_raw)), non_blocking=True)

        # Universal GPU Dequantize (handles IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_NL, Q2_0_STRATA=42)
        self.bridge.dequant_rows_f16(
            meta["gate_type"],
            staging_gate_u8.data_ptr(),
            n_ff,
            n_embd,
            dst_gate_fp16.data_ptr(),
            cuda_stream=stream,
        )
        self.bridge.dequant_rows_f16(
            meta["down_type"],
            staging_down_u8.data_ptr(),
            n_embd,
            n_ff,
            dst_down_fp16.data_ptr(),
            cuda_stream=stream,
        )
        transferred = bg + bd

        if dst_up_fp16 is not None and staging_up_u8 is not None:
            up_t = self.tensor_map[f"blk.{layer_idx}.ffn_up_exps.weight"]
            bu = meta["bytes_up"]
            u_raw = np.frombuffer(up_t.data, dtype=np.uint8, count=bu, offset=expert_idx * bu)
            staging_up_u8[:bu].copy_(torch.from_numpy(np.copy(u_raw)), non_blocking=True)
            self.bridge.dequant_rows_f16(
                meta["up_type"],
                staging_up_u8.data_ptr(),
                n_ff,
                n_embd,
                dst_up_fp16.data_ptr(),
                cuda_stream=stream,
            )
            transferred += bu

        return transferred
