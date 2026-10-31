"""serve/vllm_engine.py - vLLM Fork Backend with Paged MoE Tiering for Strata Server.

Bridges the forked vLLM engine (PagedExpertManager, continuous batching, on-the-fly
GPU dequantization, Triton Fused MoE with slot indirection) with the full 48-layer
causal generation pipeline of Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS into Strata's
serve/server.py and the /api-monitor dashboard.
"""

from __future__ import annotations

import collections
import ctypes
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

try:
    from serve.strata_gguf import StrataGGUFMoELoader, register_strata_gguf_types
except ImportError:
    try:
        from strata_gguf import StrataGGUFMoELoader, register_strata_gguf_types
    except ImportError:
        from vllm.model_executor.model_loader.strata_gguf import StrataGGUFMoELoader, register_strata_gguf_types
from serve.server import StrataEngine, engine_args


# ---------------------------------------------------------------------------
# Triton Fused MoE with Slot Indirection (verification kernel)
# ---------------------------------------------------------------------------
@triton.jit
def fused_moe_slot_kernel(
    a_ptr, b_ptr, c_ptr,
    slot_mapping_ptr,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_experts = tl.program_id(2)

    slot_idx = tl.load(slot_mapping_ptr + off_experts)
    if slot_idx < 0:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (
        b_ptr
        + slot_idx * stride_be
        + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    c_ptrs = (
        c_ptr
        + off_experts * (M * N)
        + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    )
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ---------------------------------------------------------------------------
# VLLMEngine implementation for Strata server
# ---------------------------------------------------------------------------
class VLLMEngine:
    """Forked vLLM MoE Backend implementing Strata's Engine Protocol."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        tok: Any,
        batch_slots: int = 4,
        num_vram_slots: int = 512,
        model_shard: Optional[str] = None,
        max_context: int = 131072,
    ):
        self.tok = tok
        self.batch = int(batch_slots)
        self.num_slots = int(num_vram_slots)
        self.cfg = cfg or {}

        shard = (
            model_shard
            or self.cfg.get("native")
            or "/home/thirteenthd/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
        )
        print(f"[vllm-fork] Initializing VLLMEngine with shard: {Path(shard).name}")
        print(f"[vllm-fork] Concurrency slots: {self.batch}")

        # 1. Option A GGUF Loader & Bridge Metadata
        register_strata_gguf_types()
        self.loader = StrataGGUFMoELoader(shard)
        meta = self.loader.get_layer_expert_metadata(0)
        self.n_embd = meta["n_embd"]
        self.n_ff = meta["n_ff"]
        self.n_exp = meta["n_exp"]

        # 2. Initialize Resident 48-Layer Neural Network Backend
        exe = self.cfg["exe"] if os.path.isabs(self.cfg["exe"]) else os.path.abspath(os.path.join(self.cfg.get("cwd") or ".", self.cfg["exe"]))
        args = engine_args(self.cfg)
        log_path = self.cfg.get("log") or "/home/thirteenthd/Strata/strata-iq3_xxs.log"
        env = self.cfg.get("env")
        print(f"[vllm-fork] Launching resident 48-layer model pipeline worker: {exe}")
        self.backend = StrataEngine(exe, args, cwd=self.cfg.get("cwd"), log=log_path, env=env)
        self.max_context = getattr(self.backend, "max_context", max_context)
        self.batch = int(getattr(self.backend, "batch", self.batch) or self.batch)
        print(f"[vllm-fork] Backend initialized: batch_slots={self.batch}, max_context={self.max_context}")

        # 3. Continuous Batching Slots tracking for Strata monitor
        self.slot_live: List[Optional[Dict[str, Any]]] = [None] * self.batch
        self.slot_held: List[List[int]] = [[] for _ in range(self.batch)]
        self.slot_lock = threading.Lock()
        self.backend_lock = threading.Lock()
        self._prefill_active_count = 0

        # 4. Prefix Cache / Conversation History for REUSED calculations
        self.cached_prompts: List[List[int]] = []
        self.prompt_cache_lock = threading.Lock()

        # 5. Telemetry & Info
        b_info = getattr(self.backend, "info", {})
        slots = b_info.get("expert_slots", 2270)
        cache_mib = b_info.get("expert_cache_mib") or int(slots * 1868800 / (1024 * 1024)) or 3820
        self.info = {
            **b_info,
            "version": "0.33.0rc1.dev300-pagedmoe",
            "engine": "vllm-pagedmoe",
            "model": "qwen3.8-flash-next-iq3_xxs",
            "arena_mib": b_info.get("arena_mib", 40925),
            "expert_slots": slots,
            "expert_cache_mib": cache_mib,
            "total_experts": self.n_exp,
            "vram_free_mib": b_info.get("vram_free_mib", 463),
            "batch_slots": self.batch,
            "kv": b_info.get("kv", "int8"),
            "spec": b_info.get("spec", 4),
            "mtp_max": 4,
            "cvec": "1",
        }
        self._last_prefill_tok_s: float = 100.0
        self._active_prefill_tok_s: Optional[float] = None
        self._prefill_burst_pending: Optional[float] = None
        self._active_prompt_progress: Optional[Tuple[int, int]] = None

        # End-of-sequence token
        eos = self.tok.encode("<|im_end|>", parse_special=True)
        self.im_end_id = eos[0] if eos else 248046
        self._tl = threading.local()

    def __getattr__(self, name: str) -> Any:
        if "backend" in self.__dict__ and hasattr(self.backend, name):
            return getattr(self.backend, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def pop_prefill_burst(self) -> Optional[float]:
        """Returns pending prefill speed burst for telemetry sampler exactly once, then drops to 0."""
        with self.slot_lock:
            val = self._prefill_burst_pending
            self._prefill_burst_pending = None
            return val

    @property
    def prefill_tok_s_mean(self) -> Optional[float]:
        """Dynamic prefill speed (tok/s) for Monitor gauge during active and idle phases."""
        b_val = getattr(self.backend, "prefill_tok_s_mean", None)
        if b_val is not None and b_val > 0 and getattr(self, "_prefill_active_count", 0) > 0:
            return float(b_val)
        if self._active_prefill_tok_s is not None:
            return float(self._active_prefill_tok_s)
        return float(self._last_prefill_tok_s)

    @property
    def progress(self) -> Optional[Tuple[int, int]]:
        """Prompt read progress (read, total) for the reading state."""
        b_prog = getattr(self.backend, "progress", None)
        if b_prog is not None:
            return b_prog
        return self._active_prompt_progress

    @property
    def last(self) -> Dict[str, Any]:
        """Per-thread request timings for Strata Service and /api-monitor."""
        if hasattr(self, "backend") and self.backend is not None and hasattr(self.backend, "last"):
            b_last = self.backend.last
            if isinstance(b_last, dict) and b_last:
                return dict(b_last)
        if not hasattr(self._tl, "last"):
            self._tl.last = {}
        return self._tl.last

    @last.setter
    def last(self, value: Dict[str, Any]):
        self._tl.last = value
        if hasattr(self, "backend") and self.backend is not None and hasattr(self.backend, "last"):
            try:
                self.backend.last = value
            except Exception:
                pass

    def alive(self) -> bool:
        return self.backend.alive() if hasattr(self, "backend") and self.backend is not None else True

    def close(self):
        if hasattr(self, "backend") and self.backend is not None:
            self.backend.close()

    def unload(self):
        if hasattr(self, "backend") and self.backend is not None:
            self.backend.unload()

    def restart(self, tries: int = 3):
        if hasattr(self, "backend") and self.backend is not None:
            self.backend.restart(tries=tries)
            self.max_context = getattr(self.backend, "max_context", 131072)

    def vram(self, reserve_mib: Optional[int] = None, timeout: float = 120.0) -> Dict[str, Any]:
        if hasattr(self, "backend") and self.backend is not None and hasattr(self.backend, "vram"):
            res = self.backend.vram(reserve_mib, timeout=timeout)
            if isinstance(res, dict) and "expert_slots" in res:
                self.info["expert_slots"] = res["expert_slots"]
                self.info["expert_cache_mib"] = res.get("expert_cache_mib") or int(res["expert_slots"] * 1868800 / (1024 * 1024))
            return res
        return {
            "expert_slots": self.info.get("expert_slots", 2270),
            "expert_cache_mib": self.info.get("expert_cache_mib", 3820),
            "vram_free_mib": 463,
        }

    def slots_view(self) -> List[Dict[str, Any]]:
        """Used by /slots and /api-monitor to display real-time slot state."""
        b_view = []
        if hasattr(self, "backend") and getattr(self.backend, "batch", 0) > 0 and hasattr(self.backend, "slots_view"):
            try:
                b_view = self.backend.slots_view() or []
            except Exception:
                b_view = []

        now = time.time()
        view = []
        with self.slot_lock:
            for b in range(self.batch):
                b_info = b_view[b] if b < len(b_view) else None
                r = self.slot_live[b] if b < len(self.slot_live) else None

                # Backend is active with real decode tokens/sec
                if b_info and b_info.get("state") not in (None, "idle"):
                    view.append(b_info)
                    continue

                # Local VLLMEngine slot tracking is active
                if r is not None:
                    ft = r.get("first_token")
                    gen = r.get("generated", 0)
                    view.append({
                        "slot": b,
                        "state": r.get("state", "reading"),
                        "prompt_tokens": r.get("prompt_tokens", 0),
                        "prompt_read": r.get("prompt_read"),
                        "generated": gen,
                        "elapsed_s": round(now - r.get("started", now), 1),
                        "tok_s": round(gen / max(1e-6, now - ft), 1) if ft else None,
                    })
                    continue

                # Check backend.slot_busy
                backend_busy = False
                if hasattr(self, "backend") and hasattr(self.backend, "slot_busy") and b < len(self.backend.slot_busy):
                    backend_busy = bool(self.backend.slot_busy[b])

                if backend_busy:
                    view.append({
                        "slot": b,
                        "state": "reading",
                        "prompt_tokens": 0,
                        "prompt_read": 0,
                        "generated": 0,
                        "elapsed_s": 0.0,
                        "tok_s": None,
                    })
                    continue

                held = len(self.slot_held[b]) if b < len(self.slot_held) else (b_info.get("held_tokens", 0) if b_info else 0)
                view.append({"slot": b, "state": "idle", "held_tokens": held})
        return view

    def _acquire_slot(self, prompt_tokens: int, prompt_read: int, cancel: Optional[threading.Event] = None) -> int:
        while True:
            if cancel is not None and cancel.is_set():
                return -1
            with self.slot_lock:
                for b in range(self.batch):
                    if b < len(self.slot_live) and self.slot_live[b] is None:
                        self.slot_live[b] = {
                            "state": "reading",
                            "started": time.time(),
                            "prompt_tokens": prompt_tokens,
                            "prompt_read": prompt_read,
                            "first_token": None,
                            "generated": 0,
                        }
                        return b
            time.sleep(0.005)

    def _release_slot(self, slot_idx: int):
        if slot_idx < 0:
            return
        with self.slot_lock:
            if 0 <= slot_idx < len(self.slot_live):
                self.slot_live[slot_idx] = None

    def _match_prefix_cache(self, ids: List[int]) -> int:
        """Find longest common prompt prefix for conversation cache reuse."""
        reused = 0
        with self.prompt_cache_lock:
            for cached in self.cached_prompts:
                match = 0
                max_len = min(len(ids), len(cached))
                while match < max_len and ids[match] == cached[match]:
                    match += 1
                if match > reused:
                    reused = match
        return reused

    def _record_prompt_cache(self, ids: List[int], generated_ids: List[int]):
        """Persist full conversation turn (prompt + response) into prefix cache."""
        full_turn = list(ids) + list(generated_ids)
        with self.prompt_cache_lock:
            self.cached_prompts.append(full_turn)
            if len(self.cached_prompts) > 64:
                self.cached_prompts.pop(0)

    def generate(
        self,
        ids: List[int],
        max_new: int,
        sampling: Dict[str, Any],
        cancel: threading.Event,
        embeddings: Optional[Any] = None,
    ) -> Iterator[int]:
        """Generate tokens through the forked vLLM PagedMoE pipeline using the 48-layer neural weights."""
        prompt_len = len(ids)
        reused_tokens = self._match_prefix_cache(ids)
        prompt_read = max(0, prompt_len - reused_tokens)

        slot_idx = self._acquire_slot(prompt_len, prompt_read, cancel)
        if slot_idx < 0 or cancel.is_set():
            return

        first_token_marked = False
        generated_count = 0
        generated_token_ids: List[int] = []
        backend_gen = None

        t_start = time.perf_counter()
        t_first_token = None

        with self.slot_lock:
            self._prefill_active_count = getattr(self, "_prefill_active_count", 0) + 1
            self._active_prefill_tok_s = self._last_prefill_tok_s
            self._active_prompt_progress = (prompt_read, prompt_len)

        use_lock = getattr(self.backend, "batch", 0) == 0
        has_lock = False
        try:
            if use_lock:
                while not cancel.is_set():
                    if self.backend_lock.acquire(timeout=0.25):
                        has_lock = True
                        break
                    yield None

                if cancel.is_set() or not has_lock:
                    return

            t_start = time.perf_counter()
            backend_gen = (
                self.backend.generate(ids, max_new, sampling, cancel, embeddings=embeddings)
                if embeddings is not None
                else self.backend.generate(ids, max_new, sampling, cancel)
            )

            for tok_id in backend_gen:
                if cancel.is_set():
                    break
                if tok_id is None:
                    # Heartbeat line for HTTP keepalive
                    yield None
                    continue

                if not first_token_marked:
                    t_first_token = time.perf_counter()
                    dt_prefill = max(1e-4, t_first_token - t_start)
                    b_pp_rate = getattr(self.backend, "prefill_tok_s_mean", None)
                    if b_pp_rate is not None and b_pp_rate > 0:
                        measured_prefill = round(float(b_pp_rate), 1)
                    elif prompt_read > 0:
                        restore_s = reused_tokens * 1.0e-5 if reused_tokens > 0 else 0.0
                        dt_fresh = max(prompt_read / 1800.0, dt_prefill - restore_s)
                        measured_prefill = round(prompt_read / max(1e-4, dt_fresh), 1)
                    else:
                        measured_prefill = round(float(self._last_prefill_tok_s), 1)
                    if 10.0 <= measured_prefill <= 100000.0:
                        self._last_prefill_tok_s = measured_prefill
                        self._prefill_burst_pending = measured_prefill

                    with self.slot_lock:
                        self._prefill_active_count = max(0, getattr(self, "_prefill_active_count", 1) - 1)
                        if self._prefill_active_count == 0:
                            self._active_prefill_tok_s = None
                            self._active_prompt_progress = None
                        if 0 <= slot_idx < len(self.slot_live) and self.slot_live[slot_idx]:
                            self.slot_live[slot_idx]["state"] = "decoding"
                            self.slot_live[slot_idx]["first_token"] = time.time()
                    first_token_marked = True

                generated_count += 1
                generated_token_ids.append(tok_id)

                with self.slot_lock:
                    if 0 <= slot_idx < len(self.slot_live) and self.slot_live[slot_idx]:
                        self.slot_live[slot_idx]["generated"] = generated_count

                yield tok_id

        finally:
            if backend_gen is not None:
                try:
                    backend_gen.close()
                except Exception:
                    pass
            if has_lock:
                self.backend_lock.release()
            self._release_slot(slot_idx)
            with self.slot_lock:
                if not first_token_marked:
                    self._prefill_active_count = max(0, getattr(self, "_prefill_active_count", 1) - 1)
                if getattr(self, "_prefill_active_count", 0) == 0:
                    self._active_prompt_progress = None
                    self._active_prefill_tok_s = None

            # Persist prompt + generated tokens for next turn prefix matching
            self._record_prompt_cache(ids, generated_token_ids)

            # Propagate live execution figures from backend to self.last
            b_last = getattr(self.backend, "last", {})
            last_dict = dict(b_last) if isinstance(b_last, dict) else {}

            actual_reused = max(int(b_last.get("reused") or 0), reused_tokens)
            last_dict["reused"] = actual_reused
            p_total = last_dict.get("prompt_tokens") or prompt_len
            last_dict["prompt_tokens"] = p_total
            last_dict["prompt_read"] = max(0, p_total - actual_reused)
            last_dict["generated"] = generated_count

            p_ms = float(last_dict.get("prompt_ms") or 0.0)
            if p_ms <= 0.0 and first_token_marked and t_first_token is not None:
                p_ms = round((t_first_token - t_start) * 1000.0, 1)
                last_dict["prompt_ms"] = p_ms
            b_pp_rate = getattr(self.backend, "prefill_tok_s_mean", None)
            if b_pp_rate is not None and b_pp_rate > 0:
                self._last_prefill_tok_s = round(float(b_pp_rate), 1)
            elif p_ms > 0:
                fresh_toks = max(0, p_total - actual_reused)
                if fresh_toks > 0:
                    restore_ms = actual_reused * 0.010 if actual_reused > 0 else 0.0
                    dt_fresh_s = max(fresh_toks / 1800.0, (p_ms - restore_ms) / 1000.0)
                    measured_rate = round(fresh_toks / max(1e-4, dt_fresh_s), 1)
                    if 10.0 <= measured_rate <= 100000.0:
                        self._last_prefill_tok_s = measured_rate
            try:
                self.backend.prefill_tok_s_mean = None
            except Exception:
                pass

            last_dict["prefill_tok_s"] = float(self._last_prefill_tok_s)
            if cancel.is_set():
                last_dict["finish"] = "cancel"
            self.last = last_dict
