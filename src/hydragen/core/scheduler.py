# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/core/scheduler.py - Iteration-Level In-Flight Batch Scheduler.

Provides continuous slot admission, prompt prefix cache tracking, and concurrency
telemetry for the Hydragen TRT-LLM hybrid engine.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class InFlightScheduler:
    """Manages iteration-level slot admission and prefix sharing across concurrent requests."""

    def __init__(self, batch_slots: int = 4, kv_manager: Optional[Any] = None):
        self.batch = int(batch_slots)
        self.kv_manager = kv_manager
        self.slot_live: List[Optional[Dict[str, Any]]] = [None] * self.batch
        self.slot_held: List[List[int]] = [[] for _ in range(self.batch)]
        self.lock = threading.Lock()
        self._seq_counter = 0

        # Prefix cache for prompt token reuse calculation
        self.cached_prompts: List[List[int]] = []
        self.prompt_cache_lock = threading.Lock()

    def match_prefix_cache(self, ids: List[int]) -> int:
        """Find the length of the longest matching prefix among recently executed prompts."""
        with self.prompt_cache_lock:
            best_match = 0
            for cached in self.cached_prompts:
                m = 0
                max_check = min(len(cached), len(ids))
                while m < max_check and cached[m] == ids[m]:
                    m += 1
                if m > best_match:
                    best_match = m
            return best_match

    def record_prompt_cache(self, ids: List[int], generated_ids: List[int]) -> None:
        """Record completed request prompt + generation into prefix cache."""
        full = ids + generated_ids
        with self.prompt_cache_lock:
            self.cached_prompts.append(full)
            if len(self.cached_prompts) > 64:
                self.cached_prompts.pop(0)

    def acquire_slot(
        self,
        prompt_len: int,
        prompt_read: int,
        cancel: threading.Event,
    ) -> Tuple[int, int]:
        """In-Flight Batching admission: acquire slot and virtual KV page blocks."""
        with self.lock:
            self._seq_counter += 1
            seq_id = self._seq_counter

        while not cancel.is_set():
            with self.lock:
                for i in range(self.batch):
                    if self.slot_live[i] is None:
                        blocks_count = 0
                        if self.kv_manager is not None:
                            blocks = self.kv_manager.allocate_blocks(seq_id, prompt_len)
                            blocks_count = len(blocks) if blocks else 0

                        self.slot_live[i] = {
                            "seq_id": seq_id,
                            "prompt": prompt_len,
                            "read": prompt_read,
                            "generated": 0,
                            "start_time": time.time(),
                            "state": "prefill",
                            "blocks": blocks_count,
                        }
                        return i, seq_id
            time.sleep(0.005)
        return -1, -1

    def release_slot(self, slot_idx: int, seq_id: int) -> None:
        """Release slot and free virtual KV page blocks."""
        if self.kv_manager is not None:
            self.kv_manager.free_sequence(seq_id)
        with self.lock:
            if 0 <= slot_idx < len(self.slot_live):
                self.slot_live[slot_idx] = None

    def mark_decoding(self, slot_idx: int) -> None:
        """Transition slot state from prefill to decoding on first token."""
        with self.lock:
            if 0 <= slot_idx < len(self.slot_live) and self.slot_live[slot_idx]:
                self.slot_live[slot_idx]["state"] = "decoding"
                self.slot_live[slot_idx]["first_token"] = time.time()

    def update_generated(self, slot_idx: int, count: int) -> None:
        """Update count of generated tokens for active decoding slot."""
        with self.lock:
            if 0 <= slot_idx < len(self.slot_live) and self.slot_live[slot_idx]:
                self.slot_live[slot_idx]["generated"] = count

    def slots_view(self) -> List[Dict[str, Any]]:
        """Return slot snapshot telemetry for monitor UI."""
        with self.lock:
            result = []
            now = time.time()
            for i in range(self.batch):
                item = self.slot_live[i]
                if item is None:
                    result.append({"slot": i, "state": "idle", "held_tokens": len(self.slot_held[i])})
                else:
                    t_first = item.get("first_token")
                    gen_toks = item.get("generated", 0)
                    tok_s = None
                    if t_first and item.get("state") == "decoding" and now > t_first:
                        dt = now - t_first
                        if dt >= 0.05 and gen_toks > 0:
                            tok_s = round(gen_toks / dt, 1)
                    t_start = item.get("start_time", now)
                    result.append({
                        "slot": i,
                        "state": item.get("state", "decoding"),
                        "prompt_tokens": item.get("prompt_total", item.get("prompt", 0)),
                        "prompt_read": item.get("read"),
                        "generated": gen_toks,
                        "elapsed_s": round(now - t_start, 1),
                        "tok_s": tok_s,
                        "prefill_tok_s": item.get("prefill_tok_s"),
                    })
            return result


__all__ = ["InFlightScheduler"]
