# SPDX-License-Identifier: Apache-2.0
"""hydragen/paged_kv.py - Python interface for C++ PagedAttention KV Block Manager.

Implements PagedAttention virtual block tables, Copy-on-Write prefix sharing,
dynamic block allocation, and fragmentation minimization.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PagedKVStats:
    total_blocks: int
    free_blocks: int
    allocated_blocks: int
    shared_cow_blocks: int
    active_sequences: int
    total_memory_bytes: int
    used_memory_bytes: int
    fragmentation_ratio: float


class _CPagedKVStats(ctypes.Structure):
    _fields_ = [
        ("total_blocks", ctypes.c_int32),
        ("free_blocks", ctypes.c_int32),
        ("allocated_blocks", ctypes.c_int32),
        ("shared_cow_blocks", ctypes.c_int32),
        ("active_sequences", ctypes.c_int32),
        ("total_memory_bytes", ctypes.c_uint64),
        ("used_memory_bytes", ctypes.c_uint64),
        ("fragmentation_ratio", ctypes.c_double),
    ]


class PagedKVManager:
    """Manages virtual memory pages for multi-sequence KV caching."""

    def __init__(
        self,
        block_size: int = 16,
        num_blocks: int = 4096,
        num_layers: int = 48,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype_bytes: int = 1,
    ):
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype_bytes = int(dtype_bytes)

        self._native_lib = self._load_native_lib()
        self._handle = None

        if self._native_lib is not None:
            self._handle = self._native_lib.paged_kv_create(
                self.block_size,
                self.num_blocks,
                self.num_layers,
                self.num_kv_heads,
                self.head_dim,
                self.dtype_bytes,
            )
        else:
            # High-performance pure-Python fallback implementation for zero-dependency portability
            self._free_blocks: List[int] = list(range(self.num_blocks - 1, -1, -1))
            self._block_refcount: Dict[int, int] = {i: 0 for i in range(self.num_blocks)}
            self._seq_tables: Dict[int, List[int]] = {}
            self._seq_tokens: Dict[int, int] = {}

    def _load_native_lib(self) -> Optional[ctypes.CDLL]:
        from hydragen.lib_loader import load_native_library

        lib = load_native_library("libpaged_kv.so", env_var="LIBPAGED_KV_SO")
        if lib is not None:
            try:
                self._setup_c_api(lib)
                return lib
            except Exception:
                pass
        return None

    def _setup_c_api(self, lib: ctypes.CDLL):
        lib.paged_kv_create.restype = ctypes.c_void_p
        lib.paged_kv_create.argtypes = [
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ]
        lib.paged_kv_destroy.argtypes = [ctypes.c_void_p]
        lib.paged_kv_allocate_seq.restype = ctypes.c_int32
        lib.paged_kv_allocate_seq.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
        lib.paged_kv_append_token.restype = ctypes.c_int32
        lib.paged_kv_append_token.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.paged_kv_fork_seq.restype = ctypes.c_int32
        lib.paged_kv_fork_seq.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
        lib.paged_kv_free_seq.restype = ctypes.c_int32
        lib.paged_kv_free_seq.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.paged_kv_get_num_tokens.restype = ctypes.c_int32
        lib.paged_kv_get_num_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.paged_kv_get_num_blocks.restype = ctypes.c_int32
        lib.paged_kv_get_num_blocks.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.paged_kv_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CPagedKVStats)]
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
                ctypes.c_float, ctypes.c_int32, ctypes.c_void_p,
            ]

    def allocate_sequence(self, seq_id: int, initial_tokens: int) -> bool:
        """Allocate physical 16-token pages for a sequence."""
        if self._handle is not None:
            return bool(self._native_lib.paged_kv_allocate_seq(self._handle, seq_id, initial_tokens))

        if seq_id in self._seq_tables:
            return False
        needed = max(1, (initial_tokens + self.block_size - 1) // self.block_size)
        if len(self._free_blocks) < needed:
            return False

        blocks = []
        for _ in range(needed):
            b = self._free_blocks.pop()
            self._block_refcount[b] = 1
            blocks.append(b)

        self._seq_tables[seq_id] = blocks
        self._seq_tokens[seq_id] = initial_tokens
        return True

    def append_token(self, seq_id: int) -> bool:
        """Append one generated token, write-faulting if current page is shared (Copy-on-Write)."""
        if self._handle is not None:
            return bool(self._native_lib.paged_kv_append_token(self._handle, seq_id))

        if seq_id not in self._seq_tables:
            return False

        cur_toks = self._seq_tokens[seq_id]
        blocks = self._seq_tables[seq_id]
        cap = len(blocks) * self.block_size

        if cur_toks < cap:
            last_blk = blocks[-1]
            if self._block_refcount[last_blk] > 1:
                # Copy-on-Write fault!
                if not self._free_blocks:
                    return False
                new_blk = self._free_blocks.pop()
                self._block_refcount[new_blk] = 1
                self._block_refcount[last_blk] -= 1
                blocks[-1] = new_blk
            self._seq_tokens[seq_id] += 1
            return True

        if not self._free_blocks:
            return False
        new_blk = self._free_blocks.pop()
        self._block_refcount[new_blk] = 1
        blocks.append(new_blk)
        self._seq_tokens[seq_id] += 1
        return True

    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> bool:
        """Fork sequence using Copy-on-Write (zero memory duplicate)."""
        if self._handle is not None:
            return bool(self._native_lib.paged_kv_fork_seq(self._handle, parent_seq_id, child_seq_id))

        if parent_seq_id not in self._seq_tables or child_seq_id in self._seq_tables:
            return False

        parent_blks = self._seq_tables[parent_seq_id]
        for b in parent_blks:
            self._block_refcount[b] += 1

        self._seq_tables[child_seq_id] = list(parent_blks)
        self._seq_tokens[child_seq_id] = self._seq_tokens[parent_seq_id]
        return True

    def free_sequence(self, seq_id: int) -> bool:
        """Free a sequence and recycle physical pages with ref_count == 0."""
        if self._handle is not None:
            return bool(self._native_lib.paged_kv_free_seq(self._handle, seq_id))

        if seq_id not in self._seq_tables:
            return False

        for b in self._seq_tables[seq_id]:
            self._block_refcount[b] -= 1
            if self._block_refcount[b] == 0:
                self._free_blocks.append(b)

        del self._seq_tables[seq_id]
        del self._seq_tokens[seq_id]
        return True

    def get_block_table(self, seq_id: int) -> List[int]:
        if self._handle is not None:
            n_blocks = self._native_lib.paged_kv_get_num_blocks(self._handle, seq_id)
            buf = (ctypes.c_int32 * n_blocks)()
            self._native_lib.paged_kv_export_tables(self._handle, (ctypes.c_int32 * 1)(seq_id), 1, n_blocks, buf)
            return list(buf)

        return list(self._seq_tables.get(seq_id, []))

    def get_stats(self) -> PagedKVStats:
        if self._handle is not None:
            raw = _CPagedKVStats()
            self._native_lib.paged_kv_get_stats(self._handle, ctypes.byref(raw))
            return PagedKVStats(
                total_blocks=raw.total_blocks,
                free_blocks=raw.free_blocks,
                allocated_blocks=raw.allocated_blocks,
                shared_cow_blocks=raw.shared_cow_blocks,
                active_sequences=raw.active_sequences,
                total_memory_bytes=raw.total_memory_bytes,
                used_memory_bytes=raw.used_memory_bytes,
                fragmentation_ratio=round(raw.fragmentation_ratio, 4),
            )

        block_bytes = self.block_size * self.num_layers * 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes
        allocated = self.num_blocks - len(self._free_blocks)
        cow = sum(1 for b in self._block_refcount.values() if b > 1)
        tot_tok = sum(self._seq_tokens.values())
        tot_cap = sum(len(blks) * self.block_size for blks in self._seq_tables.values())
        frag = (tot_cap - tot_tok) / tot_cap if tot_cap > 0 else 0.0

        return PagedKVStats(
            total_blocks=self.num_blocks,
            free_blocks=len(self._free_blocks),
            allocated_blocks=allocated,
            shared_cow_blocks=cow,
            active_sequences=len(self._seq_tables),
            total_memory_bytes=self.num_blocks * block_bytes,
            used_memory_bytes=allocated * block_bytes,
            fragmentation_ratio=round(frag, 4),
        )

    def allocate_blocks(self, seq_id: int, num_tokens: int) -> List[int]:
        success = self.allocate_sequence(seq_id, num_tokens)
        if not success:
            return []
        if self._handle is not None:
            n_blocks = self._native_lib.paged_kv_get_num_blocks(self._handle, seq_id)
            return list(range(n_blocks))
        return list(self._seq_tables.get(seq_id, []))

    def stats(self) -> Dict[str, Any]:
        s = self.get_stats()
        return {
            "total_blocks": s.total_blocks,
            "free_blocks": s.free_blocks,
            "used_blocks": s.allocated_blocks,
            "utilization": round(s.allocated_blocks / max(1, s.total_blocks), 3),
            "block_size": self.block_size,
        }

    def paged_attention_decode(
        self,
        out,
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale: Optional[float] = None,
        is_int8_kv: bool = False,
        stream: Optional[Any] = None,
    ):
        """Single-token decode attention across paged KV blocks."""
        num_seqs = q.shape[0]
        num_heads = q.shape[1]
        head_dim = q.shape[2]
        num_kv_heads = k_cache.shape[1]
        max_blocks = block_tables.shape[1]
        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)

        is_cuda = hasattr(q, "is_cuda") and q.is_cuda
        if self._native_lib is not None and is_cuda and hasattr(self._native_lib, "paged_attention_decode"):
            stream_ptr = stream.cuda_stream if hasattr(stream, "cuda_stream") else (stream or 0)
            self._native_lib.paged_attention_decode(
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(q.data_ptr()),
                ctypes.c_void_p(k_cache.data_ptr()),
                ctypes.c_void_p(v_cache.data_ptr()),
                ctypes.c_void_p(block_tables.data_ptr()),
                ctypes.c_void_p(context_lens.data_ptr()),
                int(num_seqs), int(num_heads), int(num_kv_heads), int(head_dim),
                int(self.block_size), int(max_blocks),
                float(scale), int(1 if is_int8_kv else 0),
                ctypes.c_void_p(stream_ptr)
            )
            return out

        # Torch / CPU fallback
        import torch
        group_size = num_heads // num_kv_heads
        for s in range(num_seqs):
            ctx_len = int(context_lens[s].item() if hasattr(context_lens[s], "item") else context_lens[s])
            if ctx_len <= 0:
                continue
            bt = block_tables[s]
            k_list, v_list = [], []
            for t_idx in range(ctx_len):
                b_idx = t_idx // self.block_size
                b_off = t_idx % self.block_size
                p_blk = int(bt[b_idx].item() if hasattr(bt[b_idx], "item") else bt[b_idx])
                k_list.append(k_cache[p_blk, :, b_off, :])
                v_list.append(v_cache[p_blk, :, b_off, :])
            k_seq = torch.stack(k_list, dim=0).permute(1, 0, 2).float()
            v_seq = torch.stack(v_list, dim=0).permute(1, 0, 2).float()
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
        out,
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens=None,
        scale: Optional[float] = None,
        is_int8_kv: bool = False,
        stream: Optional[Any] = None,
    ):
        """Causal prefill attention across paged KV blocks with Reverse Query-Tile Attention Scheduling (LPT)."""
        num_seqs = q.shape[0]
        query_len = q.shape[1]
        num_heads = q.shape[2]
        head_dim = q.shape[3]
        num_kv_heads = k_cache.shape[1]
        max_blocks = block_tables.shape[1]
        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)

        is_cuda = hasattr(q, "is_cuda") and q.is_cuda
        if self._native_lib is not None and is_cuda and hasattr(self._native_lib, "paged_attention_prefill"):
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
                int(head_dim), int(self.block_size), int(max_blocks),
                float(scale), int(1 if is_int8_kv else 0),
                ctypes.c_void_p(stream_ptr)
            )
            return out

        # Torch / CPU fallback with causal masking
        import torch
        group_size = num_heads // num_kv_heads
        for s in range(num_seqs):
            past_len = int(context_lens[s].item() if (context_lens is not None and hasattr(context_lens[s], "item")) else (context_lens[s] if context_lens is not None else 0))
            bt = block_tables[s]
            total_tokens = past_len + query_len

            k_list, v_list = [], []
            for t_idx in range(total_tokens):
                b_idx = t_idx // self.block_size
                b_off = t_idx % self.block_size
                p_blk = int(bt[b_idx].item() if hasattr(bt[b_idx], "item") else bt[b_idx])
                k_list.append(k_cache[p_blk, :, b_off, :])
                v_list.append(v_cache[p_blk, :, b_off, :])
            k_all = torch.stack(k_list, dim=0).permute(1, 0, 2).float()
            v_all = torch.stack(v_list, dim=0).permute(1, 0, 2).float()
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

    def __del__(self):
        if self._handle is not None and self._native_lib is not None:
            self._native_lib.paged_kv_destroy(self._handle)
            self._handle = None
