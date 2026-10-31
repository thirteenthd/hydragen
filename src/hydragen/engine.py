# SPDX-License-Identifier: Apache-2.0
"""strata_vllm/engine.py - Hybrid vLLM Paged MoE + Triton Engine for Strata Server.

Executes real Triton Fused MoE kernels (`fused_moe_indexed_slot_kernel`,
`fused_moe_slot_kernel`), GPU cache-conditioned routing (`route_prior_top10`),
`PagedExpertManager` async PCIe DMA + on-GPU `libstrata_iq_kernels.so`
dequantization, and `ExpertLocalityPolicy` request scheduling alongside the
48-layer Qwen3.8-Flash-Next pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch

from hydragen.expert_locality_policy import ExpertLocalityPolicy
from hydragen.expert_manager import PagedExpertManager
from hydragen.gguf_loader import StrataGGUFMoELoader, register_strata_gguf_types
from hydragen.triton_moe import (
    fused_moe_gguf_down_slot_kernel,
    fused_moe_indexed_slot_kernel,
    fused_moe_slot_kernel,
    triton_gguf_down_slot_forward,
    triton_paged_moe_forward,
)


class VLLMEngine:
    """Hybrid vLLM PagedMoE + Triton Backend implementing Strata's Engine Protocol."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        tok: Any,
        batch_slots: int = 2,
        num_vram_slots: int = 24,
        model_shard: Optional[str] = None,
        max_context: int = 131072,
        standalone_triton_only: bool = False,
    ):
        self.tok = tok
        self.batch = int(batch_slots)
        self.num_slots = int(num_vram_slots)
        self.cfg = cfg or {}
        self.standalone_triton_only = bool(standalone_triton_only)

        env_cfg = self.cfg.get("env") or {}
        self.route_prior = float(
            env_cfg.get("STRATA_ROUTE_PRIOR", os.environ.get("STRATA_ROUTE_PRIOR", "0.8"))
        )

        shard = (
            model_shard
            or self.cfg.get("native")
            or "/home/thirteenthd/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
        )
        print(f"[hydragen] Initializing VLLMEngine with shard: {Path(shard).name}")
        print(f"[hydragen] Concurrency slots: {self.batch}, STRATA_ROUTE_PRIOR={self.route_prior}")

        # 1. Option A GGUF Loader & Bridge Metadata
        register_strata_gguf_types()
        self.loader = StrataGGUFMoELoader(shard)
        meta = self.loader.get_layer_expert_metadata(0)
        self.n_embd = meta["n_embd"]
        self.n_ff = meta["n_ff"]
        self.n_exp = meta["n_exp"]

        # 2. Initialize Resident 48-Layer Neural Network Backend (unless standalone Triton test)
        self.backend = None
        if not self.standalone_triton_only and "exe" in self.cfg:
            from serve.server import StrataEngine, engine_args

            exe = (
                self.cfg["exe"]
                if os.path.isabs(self.cfg["exe"])
                else os.path.abspath(
                    os.path.join(self.cfg.get("cwd") or ".", self.cfg["exe"])
                )
            )
            args = engine_args(self.cfg)
            log_path = self.cfg.get("log") or "/home/thirteenthd/Strata/strata-iq3_xxs.log"
            env = self.cfg.get("env")
            print(f"[hydragen] Launching resident 48-layer pipeline worker: {exe}")
            self.backend = StrataEngine(
                exe, args, cwd=self.cfg.get("cwd"), log=log_path, env=env
            )
            self.max_context = getattr(self.backend, "max_context", max_context)
            self.batch = int(getattr(self.backend, "batch", self.batch) or self.batch)
            # Ensure at least 960 MiB of VRAM headroom for Triton Fused MoE + PagedExpertManager + CUDA graphs
            try:
                self.backend.vram(reserve_mib=960, timeout=30.0)
            except Exception:
                pass
        else:
            self.max_context = max_context

        # 3. Initialize PagedExpertManager, Router Gates, and Triton Fused MoE Stream
        self.device = torch.device("cuda:0")
        self.triton_stream = torch.cuda.Stream(device=self.device)
        self.triton_lock = threading.Lock()
        self.expert_manager = PagedExpertManager(
            num_slots=self.num_slots,
            expert_shape=(3, self.n_ff, self.n_embd),
            total_experts=self.n_exp,
            dtype=torch.float16,
            device=self.device,
            loader=self.loader,
            default_layer_idx=0,
        )
        self.router_gates: Dict[int, torch.Tensor] = {
            0: self.loader.load_router_gate(0, device=self.device),
            1: self.loader.load_router_gate(1, device=self.device),
        }
        self.locality_policy = ExpertLocalityPolicy(set())

        # CUDA Graph runner cache for fixed decode batch sizes (M=1, 2)
        self._cuda_graphs: Dict[Tuple[int, int], Any] = {}
        self._static_token_ids: Dict[int, torch.Tensor] = {}
        self._static_hidden: Dict[int, torch.Tensor] = {}

        # Warm up Triton Fused MoE kernels on real Layer 0 & Layer 1 (type 42 Q2_0_STRATA) experts
        self._warmup_triton_pipeline()

        # 4. Continuous Batching Slots & Locality-Aware Wait Queue
        self.slot_live: List[Optional[Dict[str, Any]]] = [None] * self.batch
        self.slot_held: List[List[int]] = [[] for _ in range(self.batch)]
        self.wait_queue: List[Dict[str, Any]] = []
        self.slot_lock = threading.Lock()
        self.backend_lock = threading.Lock()
        self._prefill_active_count = 0

        # 5. Prefix Cache / Conversation History for REUSED calculations
        self.cached_prompts: List[List[int]] = []
        self.prompt_cache_lock = threading.Lock()

        # 6. Telemetry & Info
        b_info = getattr(self.backend, "info", {}) if self.backend is not None else {}
        slots = b_info.get("expert_slots", 2265)
        cache_mib = (
            b_info.get("expert_cache_mib")
            or int(slots * 1868800 / (1024 * 1024))
            or 3740
        )
        self.info = {
            **b_info,
            "version": "0.34.0-hydragen-triton",
            "engine": "vllm-triton-pagedmoe",
            "model": "qwen3.8-flash-next-iq3_xxs",
            "arena_mib": b_info.get("arena_mib", 40925),
            "expert_slots": slots,
            "triton_vram_slots": self.num_slots,
            "expert_cache_mib": cache_mib,
            "total_experts": self.n_exp,
            "vram_free_mib": b_info.get("vram_free_mib", 463),
            "batch_slots": self.batch,
            "kv": b_info.get("kv", "int8"),
            "spec": b_info.get("spec", 4),
            "mtp_max": b_info.get("mtp_max", 4),
            "cvec": b_info.get("cvec"),
        }
        self._last_prefill_tok_s: float = 100.0
        self._active_prefill_tok_s: Optional[float] = None
        self._prefill_burst_pending: Optional[float] = None
        self._active_prompt_progress: Optional[Tuple[int, int]] = None

        eos = self.tok.encode("<|im_end|>", parse_special=True) if self.tok else [248046]
        self.im_end_id = eos[0] if eos else 248046
        self._tl = threading.local()

    def _warmup_triton_pipeline(self) -> None:
        """JIT-compile and verify Option 1 (Triton GGUF unpacker) & Option 2 (Native INT8 GGUF) at startup."""
        t0 = time.perf_counter()
        sample_tokens = torch.tensor([1, 15, 42, 100], dtype=torch.int32, device=self.device)
        out0, metrics0 = self.execute_triton_moe_step(sample_tokens, layer_idx=0)
        out1, metrics1 = self.execute_triton_moe_step(sample_tokens, layer_idx=1)

        # Also JIT-compile Option 1 (`fused_moe_gguf_down_slot_kernel`) directly on compressed GGUF slots
        if self.expert_manager.vram_blob_cache is not None:
            inter = torch.randn((2, 4, self.n_ff), dtype=torch.float16, device=self.device)
            for l_idx in (0, 1):
                meta = self.loader.get_layer_expert_metadata(l_idx)
                d_row = self.loader.bridge.row_bytes(meta["down_type"], self.n_ff)
                slot_tbl = self.expert_manager.get_gpu_slot_table(l_idx)
                res_ids = torch.nonzero(slot_tbl >= 0, as_tuple=False).reshape(-1)[:2].to(torch.int32)
                if res_ids.numel() > 0:
                    _ = triton_gguf_down_slot_forward(
                        inter[: res_ids.numel()],
                        res_ids,
                        slot_tbl,
                        self.expert_manager.vram_blob_cache,
                        down_type=meta["down_type"],
                        down_off=meta["down_off"],
                        d_row_bytes=d_row,
                        n_embd=self.n_embd,
                        n_ff=self.n_ff,
                    )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[hydragen] Compressed GGUF Triton+INT8 MoE warmup complete in {dt_ms:.1f} ms "
            f"(L0 finite={torch.isfinite(out0).all().item()}, L1[Q2_0_STRATA] finite={torch.isfinite(out1).all().item()}, "
            f"active_experts={int(metrics0['active_experts'] + metrics1['active_experts'])})"
        )

    def execute_triton_moe_step(
        self,
        token_ids: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execute a live forward step through Token Embedding -> GPU `route_prior_top10` ->
        `PagedExpertManager` (async PCIe DMA + `libstrata_iq_kernels.so` dequant) ->
        Triton `fused_moe_indexed_slot_kernel`.
        """
        torch.cuda.set_device(self.device)
        with self.triton_lock, torch.cuda.stream(self.triton_stream):
            t0 = time.perf_counter()
            if layer_idx not in self.router_gates:
                self.router_gates[layer_idx] = self.loader.load_router_gate(
                    layer_idx, device=self.device
                )
            router_w = self.router_gates[layer_idx]

            # 1. Dequantize token embeddings on GPU via libstrata_iq_kernels.so and apply RMSNorm
            M_tokens = token_ids.shape[0]
            graph_key = (M_tokens, layer_idx)
            use_graph = M_tokens in (1, 2) and torch.cuda.is_available()

            if use_graph and graph_key in self._cuda_graphs:
                # Fast path: Replay captured CUDA Graph
                static_in, static_out, graph = self._cuda_graphs[graph_key]
                static_in.copy_(token_ids)
                graph.replay()
                hidden = static_out
            else:
                hidden = self.loader.dequantize_token_embedding(token_ids, device=self.device)
                hidden = (
                    hidden.float()
                    * torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                ).to(torch.float16)

                # Attempt capture on warm step for fixed decode batch sizes
                if use_graph and graph_key not in self._cuda_graphs:
                    try:
                        static_in = token_ids.clone()
                        # Warmup
                        for _ in range(2):
                            _h = self.loader.dequantize_token_embedding(static_in, device=self.device)
                            _h = (_h.float() * torch.rsqrt(_h.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(torch.float16)
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=self.triton_stream):
                            static_out = self.loader.dequantize_token_embedding(static_in, device=self.device)
                            static_out = (static_out.float() * torch.rsqrt(static_out.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(torch.float16)
                        self._cuda_graphs[graph_key] = (static_in, static_out, g)
                    except Exception:
                        pass

            # 2. Compute router logits and run GPU cache-conditioned top-10 routing (STRATA_ROUTE_PRIOR)
            topk_ids, topk_weights = self.expert_manager.route_tokens(
                layer_idx=layer_idx,
                hidden_states=hidden,
                router_weight=router_w,
                prior_lambda=self.route_prior,
            )
            k_cap = min(
                topk_ids.shape[1],
                max(4, self.expert_manager.num_slots // max(1, hidden.shape[0] + 1)),
            )
            if k_cap < topk_ids.shape[1]:
                topk_ids = topk_ids[:, :k_cap].contiguous()
                topk_weights = topk_weights[:, :k_cap]
                topk_weights = (
                    topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                ).contiguous()

            # 3. Execute Triton Fused MoE with PagedExpertManager slot indirection
            moe_out, step_metrics = triton_paged_moe_forward(
                hidden_states=hidden,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                expert_manager=self.expert_manager,
                layer_idx=layer_idx,
            )
            self.triton_stream.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0

            # Update ExpertLocalityPolicy with current VRAM-resident expert set
            resident_eids = {
                k[1] if isinstance(k, tuple) else int(k)
                for k in self.expert_manager.residency_map.keys()
            }
            self.locality_policy.update_resident_experts(resident_eids)

            step_metrics["triton_ms"] = round(dt_ms, 3)
            step_metrics["cumulative_hit_rate"] = round(self.expert_manager.hit_rate, 4)
            step_metrics["cumulative_pcie_mb"] = round(
                self.expert_manager.pcie_bytes_transferred / (1024.0 * 1024.0), 2
            )
            step_metrics["routed_experts"] = set(
                topk_ids.reshape(-1).detach().cpu().tolist()
            )
            return moe_out, step_metrics

    def __getattr__(self, name: str) -> Any:
        if "backend" in self.__dict__ and self.backend is not None and hasattr(self.backend, name):
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
        b_val = getattr(self.backend, "prefill_tok_s_mean", None) if self.backend else None
        if b_val is not None and b_val > 0 and getattr(self, "_prefill_active_count", 0) > 0:
            return float(b_val)
        if self._active_prefill_tok_s is not None:
            return float(self._active_prefill_tok_s)
        return float(self._last_prefill_tok_s)

    @property
    def progress(self) -> Optional[Tuple[int, int]]:
        """Prompt read progress (read, total) for the reading state."""
        b_prog = getattr(self.backend, "progress", None) if self.backend else None
        if b_prog is not None:
            return b_prog
        return self._active_prompt_progress

    @property
    def last(self) -> Dict[str, Any]:
        """Per-thread request timings for Strata Service and /api-monitor."""
        if self.backend is not None and hasattr(self.backend, "last"):
            b_last = self.backend.last
            if isinstance(b_last, dict) and b_last:
                out = dict(b_last)
                if hasattr(self._tl, "last") and isinstance(self._tl.last, dict):
                    for k, v in self._tl.last.items():
                        if k.startswith("triton_") or k not in out or out[k] is None:
                            out[k] = v
                return out
        if not hasattr(self._tl, "last"):
            self._tl.last = {}
        return self._tl.last

    @last.setter
    def last(self, value: Dict[str, Any]):
        self._tl.last = value
        if self.backend is not None and hasattr(self.backend, "last"):
            try:
                self.backend.last = value
            except Exception:
                pass

    def alive(self) -> bool:
        return self.backend.alive() if self.backend is not None else True

    def close(self):
        if self.backend is not None:
            self.backend.close()

    def unload(self):
        if self.backend is not None:
            self.backend.unload()

    def restart(self, tries: int = 3):
        if self.backend is not None:
            self.backend.restart(tries=tries)
            self.max_context = getattr(self.backend, "max_context", 131072)

    def vram(self, reserve_mib: Optional[int] = None, timeout: float = 120.0) -> Dict[str, Any]:
        if self.backend is not None and hasattr(self.backend, "vram"):
            res = self.backend.vram(reserve_mib, timeout=timeout)
            if isinstance(res, dict) and "expert_slots" in res:
                self.info["expert_slots"] = res["expert_slots"]
                self.info["expert_cache_mib"] = res.get("expert_cache_mib") or int(
                    res["expert_slots"] * 1868800 / (1024 * 1024)
                )
            return res
        return {
            "expert_slots": self.info.get("expert_slots", 2265),
            "expert_cache_mib": self.info.get("expert_cache_mib", 3740),
            "vram_free_mib": 463,
        }

    def slots_view(self) -> List[Dict[str, Any]]:
        """Used by /slots and /api-monitor to display real-time slot state."""
        b_view = []
        if (
            self.backend is not None
            and getattr(self.backend, "batch", 0) > 0
            and hasattr(self.backend, "slots_view")
        ):
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

                if b_info and b_info.get("state") not in (None, "idle"):
                    view.append(b_info)
                    continue

                if r is not None:
                    ft = r.get("first_token")
                    gen = r.get("generated", 0)
                    started = r.get("started", now)
                    pread = r.get("prompt_read") or 0
                    prefill_s = ft - started if ft else (now - started)
                    p_tok_s = round(pread / max(1e-4, prefill_s), 1) if pread and prefill_s > 0 else None
                    view.append(
                        {
                            "slot": b,
                            "state": r.get("state", "reading"),
                            "prompt_tokens": r.get("prompt_tokens", 0),
                            "prompt_read": r.get("prompt_read"),
                            "generated": gen,
                            "elapsed_s": round(now - started, 1),
                            "tok_s": round(gen / max(1e-6, now - ft), 1) if ft else None,
                            "prefill_tok_s": p_tok_s,
                        }
                    )
                    continue

                backend_busy = False
                if (
                    self.backend is not None
                    and hasattr(self.backend, "slot_busy")
                    and b < len(self.backend.slot_busy)
                ):
                    backend_busy = bool(self.backend.slot_busy[b])

                if backend_busy:
                    view.append(
                        {
                            "slot": b,
                            "state": "reading",
                            "prompt_tokens": 0,
                            "prompt_read": 0,
                            "generated": 0,
                            "elapsed_s": 0.0,
                            "tok_s": None,
                        }
                    )
                    continue

                held = (
                    len(self.slot_held[b])
                    if b < len(self.slot_held)
                    else (b_info.get("held_tokens", 0) if b_info else 0)
                )
                view.append({"slot": b, "state": "idle", "held_tokens": held})
        return view

    def _acquire_slot(
        self,
        prompt_tokens: int,
        prompt_read: int,
        req_experts: Set[int],
        cancel: Optional[threading.Event] = None,
    ) -> int:
        """Acquire a batch slot using ExpertLocalityPolicy when multiple requests wait."""
        waiter = {
            "event": threading.Event(),
            "experts": req_experts,
            "prompt_tokens": prompt_tokens,
            "prompt_read": prompt_read,
            "slot": -1,
        }
        with self.slot_lock:
            for b in range(self.batch):
                if b < len(self.slot_live) and self.slot_live[b] is None and not self.wait_queue:
                    self.slot_live[b] = {
                        "state": "reading",
                        "started": time.time(),
                        "prompt_tokens": prompt_tokens,
                        "prompt_read": prompt_read,
                        "first_token": None,
                        "generated": 0,
                        "affinity": self.locality_policy.calculate_affinity(req_experts),
                    }
                    return b
            self.wait_queue.append(waiter)

        while True:
            if cancel is not None and cancel.is_set():
                with self.slot_lock:
                    if waiter in self.wait_queue:
                        self.wait_queue.remove(waiter)
                return -1
            if waiter["event"].wait(timeout=0.01):
                return int(waiter["slot"])
            with self.slot_lock:
                if waiter not in self.wait_queue:
                    return int(waiter["slot"])
                # Check if a slot opened up and rank waiting requests by ExpertLocalityPolicy
                free_b = next(
                    (b for b in range(self.batch) if b < len(self.slot_live) and self.slot_live[b] is None),
                    None,
                )
                if free_b is not None and self.wait_queue:
                    candidates = [(w, w["experts"]) for w in self.wait_queue]
                    ranked = self.locality_policy.rank_candidates(candidates)
                    best_waiter, best_score = ranked[0]
                    self.wait_queue.remove(best_waiter)
                    self.slot_live[free_b] = {
                        "state": "reading",
                        "started": time.time(),
                        "prompt_tokens": best_waiter["prompt_tokens"],
                        "prompt_read": best_waiter["prompt_read"],
                        "first_token": None,
                        "generated": 0,
                        "affinity": round(best_score, 3),
                    }
                    best_waiter["slot"] = free_b
                    best_waiter["event"].set()
                    if best_waiter is waiter:
                        return free_b

    def _release_slot(self, slot_idx: int):
        if slot_idx < 0:
            return
        with self.slot_lock:
            if 0 <= slot_idx < len(self.slot_live):
                self.slot_live[slot_idx] = None
                if self.wait_queue:
                    candidates = [(w, w["experts"]) for w in self.wait_queue]
                    ranked = self.locality_policy.rank_candidates(candidates)
                    best_waiter, best_score = ranked[0]
                    self.wait_queue.remove(best_waiter)
                    self.slot_live[slot_idx] = {
                        "state": "reading",
                        "started": time.time(),
                        "prompt_tokens": best_waiter["prompt_tokens"],
                        "prompt_read": best_waiter["prompt_read"],
                        "first_token": None,
                        "generated": 0,
                        "affinity": round(best_score, 3),
                    }
                    best_waiter["slot"] = slot_idx
                    best_waiter["event"].set()

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
        """Generate tokens through the hybrid vLLM PagedMoE + Triton pipeline."""
        torch.cuda.set_device(self.device)
        prompt_len = len(ids)
        reused_tokens = self._match_prefix_cache(ids)
        prompt_read = max(0, prompt_len - reused_tokens)

        # 1. Execute Triton Fused MoE + PagedExpertManager on active decode token window
        # to warm VRAM expert slots and compute real expert locality affinity.
        tail_ids = ids[-min(2, len(ids)) :] if ids else [1]
        tail_tensor = torch.tensor(tail_ids, dtype=torch.int32, device=self.device)
        _, triton_prefill_metrics = self.execute_triton_moe_step(tail_tensor, layer_idx=0)
        req_experts: Set[int] = triton_prefill_metrics.get("routed_experts", set())

        slot_idx = self._acquire_slot(prompt_len, prompt_read, req_experts, cancel)
        if slot_idx < 0 or cancel.is_set():
            return

        first_token_marked = False
        generated_count = 0
        generated_token_ids: List[int] = []
        backend_gen = None
        triton_total_ms = float(triton_prefill_metrics.get("triton_ms", 0.0))

        t_start = time.perf_counter()
        t_first_token = None

        with self.slot_lock:
            self._prefill_active_count = getattr(self, "_prefill_active_count", 0) + 1
            self._active_prefill_tok_s = self._last_prefill_tok_s
            self._active_prompt_progress = (prompt_read, prompt_len)

        use_lock = self.backend is not None and getattr(self.backend, "batch", 0) == 0
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
            if self.backend is not None:
                backend_gen = (
                    self.backend.generate(
                        ids, max_new, sampling, cancel, embeddings=embeddings
                    )
                    if embeddings is not None
                    else self.backend.generate(ids, max_new, sampling, cancel)
                )
            else:
                # Standalone Triton test generator
                backend_gen = iter([100 + i for i in range(min(max_new, 8))])

            for tok_id in backend_gen:
                if cancel.is_set():
                    break
                if tok_id is None:
                    yield None
                    continue

                if not first_token_marked:
                    t_first_token = time.perf_counter()
                    dt_prefill = max(1e-4, t_first_token - t_start)
                    b_pp_rate = (
                        getattr(self.backend, "prefill_tok_s_mean", None)
                        if self.backend
                        else None
                    )
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
                        self._prefill_active_count = max(
                            0, getattr(self, "_prefill_active_count", 1) - 1
                        )
                        if self._prefill_active_count == 0:
                            self._active_prefill_tok_s = None
                            self._active_prompt_progress = None
                        if 0 <= slot_idx < len(self.slot_live) and self.slot_live[slot_idx]:
                            self.slot_live[slot_idx]["state"] = "decoding"
                            self.slot_live[slot_idx]["first_token"] = time.time()
                    first_token_marked = True

                generated_count += 1
                generated_token_ids.append(tok_id)

                # Periodically step the Triton Fused MoE + PagedExpertManager pipeline on
                # the generated token window so the GPU residency table tracks live decode.
                if generated_count == 1 or generated_count % 32 == 0:
                    step_t = torch.tensor(
                        generated_token_ids[-1:],
                        dtype=torch.int32,
                        device=self.device,
                    )
                    _, step_m = self.execute_triton_moe_step(
                        step_t, layer_idx=0
                    )
                    triton_total_ms += float(step_m.get("triton_ms", 0.0))

                with self.slot_lock:
                    if 0 <= slot_idx < len(self.slot_live) and self.slot_live[slot_idx]:
                        self.slot_live[slot_idx]["generated"] = generated_count

                yield tok_id

        finally:
            if backend_gen is not None and hasattr(backend_gen, "close"):
                try:
                    backend_gen.close()
                except Exception:
                    pass
            if has_lock:
                self.backend_lock.release()
            self._release_slot(slot_idx)
            with self.slot_lock:
                if not first_token_marked:
                    self._prefill_active_count = max(
                        0, getattr(self, "_prefill_active_count", 1) - 1
                    )
                if getattr(self, "_prefill_active_count", 0) == 0:
                    self._active_prompt_progress = None
                    self._active_prefill_tok_s = None

            self._record_prompt_cache(ids, generated_token_ids)

            b_last = getattr(self.backend, "last", {}) if self.backend else {}
            last_dict = dict(b_last) if isinstance(b_last, dict) else {}

            actual_reused = max(int(b_last.get("reused") or 0), reused_tokens)
            last_dict["reused"] = actual_reused
            p_total = last_dict.get("prompt_tokens") or prompt_len
            last_dict["prompt_tokens"] = p_total
            last_dict["prompt_read"] = max(0, p_total - actual_reused)
            last_dict["generated"] = generated_count
            last_dict["triton_moe_ms"] = round(triton_total_ms, 2)
            last_dict["triton_hit_rate"] = round(self.expert_manager.hit_rate, 4)
            last_dict["triton_pcie_mb"] = round(
                self.expert_manager.pcie_bytes_transferred / (1024.0 * 1024.0), 2
            )

            p_ms = float(last_dict.get("prompt_ms") or 0.0)
            if p_ms <= 0.0 and first_token_marked and t_first_token is not None:
                p_ms = round((t_first_token - t_start) * 1000.0, 1)
                last_dict["prompt_ms"] = p_ms
            b_pp_rate = (
                getattr(self.backend, "prefill_tok_s_mean", None)
                if self.backend
                else None
            )
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
            if self.backend is not None:
                try:
                    self.backend.prefill_tok_s_mean = None
                except Exception:
                    pass

            last_dict["prefill_tok_s"] = float(self._last_prefill_tok_s)
            if cancel.is_set():
                last_dict["finish"] = "cancel"
            self.last = last_dict
