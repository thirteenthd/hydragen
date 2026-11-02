# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/core/block_manager.py - Paged KV Block Manager & Virtual Page Tables.

Manages virtual block tables, Copy-on-Write prefix sharing, and dynamic block allocation
for TensorRT-LLM and Hydragen execution runtimes.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

try:
    from hydragen.paged_kv import PagedKVManager, PagedKVStats
except ImportError:
    PagedKVManager = None  # type: ignore
    PagedKVStats = None  # type: ignore


class TRTPagedBlockManager:
    """Virtual block manager simulating TensorRT-LLM's C++ KVCacheManager."""

    def __init__(self, block_size: int = 16, max_blocks: int = 2048):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.free_blocks: List[int] = list(range(max_blocks - 1, -1, -1))
        self.block_ref_counts: Dict[int, int] = {b: 0 for b in range(max_blocks)}
        self.seq_block_tables: Dict[int, List[int]] = {}
        self.lock = threading.Lock()

    def allocate_blocks(self, seq_id: int, num_tokens: int) -> List[int]:
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        with self.lock:
            allocated = []
            for _ in range(needed_blocks):
                if not self.free_blocks:
                    break
                block_id = self.free_blocks.pop()
                self.block_ref_counts[block_id] = 1
                allocated.append(block_id)
            self.seq_block_tables[seq_id] = allocated
            return allocated

    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> bool:
        """Copy-on-Write sequence forking (prefix cache sharing)."""
        with self.lock:
            if parent_seq_id not in self.seq_block_tables:
                return False
            parent_blocks = self.seq_block_tables[parent_seq_id]
            for b in parent_blocks:
                self.block_ref_counts[b] += 1
            self.seq_block_tables[child_seq_id] = list(parent_blocks)
            return True

    def free_sequence(self, seq_id: int) -> None:
        with self.lock:
            if seq_id not in self.seq_block_tables:
                return
            for b in self.seq_block_tables.pop(seq_id):
                self.block_ref_counts[b] = max(0, self.block_ref_counts.get(b, 1) - 1)
                if self.block_ref_counts[b] == 0:
                    self.free_blocks.append(b)

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            used = self.max_blocks - len(self.free_blocks)
            return {
                "total_blocks": self.max_blocks,
                "used_blocks": used,
                "free_blocks": len(self.free_blocks),
                "utilization": round(used / max(1, self.max_blocks), 3),
                "block_size": self.block_size,
            }


def create_block_manager(
    block_size: int = 16,
    max_blocks: int = 4096,
    num_layers: int = 48,
    num_kv_heads: int = 8,
    head_dim: int = 128,
) -> Any:
    """Factory function to initialize native C++ PagedKVManager or pure-Python fallback."""
    if PagedKVManager is not None:
        try:
            mgr = PagedKVManager(
                block_size=block_size,
                num_blocks=max_blocks,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            is_native = getattr(mgr, "_handle", None) is not None
            print(f"[hydragen:core] Initialized PagedKVManager (Native C++ backend: {is_native})")
            return mgr
        except Exception as e:
            print(f"[hydragen:core] PagedKVManager init fallback note: {e}")

    return TRTPagedBlockManager(block_size=block_size, max_blocks=max_blocks)


__all__ = [
    "TRTPagedBlockManager",
    "PagedKVManager",
    "PagedKVStats",
    "create_block_manager",
]
