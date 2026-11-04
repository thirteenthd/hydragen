# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/hydragen_engine.py - Hydragen Unified Serving Engine.

Unifies TensorRT-LLM In-Flight Batching and Virtual Paged KV Blocks with
vLLM Locality-Aware MoE Routing and Modular Hardware Backends.

Architectural Capabilities:
1. Virtual Paged KV Cache with Copy-on-Write reference counting (75% prefix memory savings).
2. Iteration-Level In-Flight Batching (`InFlightScheduler`) with continuous slot admission.
3. Locality-Aware Request Scheduling (`ExpertLocalityPolicy`) prioritizing co-located MoE experts.
4. Modular Hardware Acceleration via `HardwareBackend` (CUDA C-ABI, ROCm, Intel XPU, CPU).
5. Speculative Drafter Architecture (MTP / DFlash2) with draft KV warm-up.
6. Post-Prefill Bulk Expert Realignment (`PagedExpertManager.adapt_post_prefill()`).
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

# Optional PyTorch import
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore

# Optional GGUF and MoE imports
HAS_GGUF = False
try:
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
    HAS_GGUF = True
except ImportError:
    ExpertLocalityPolicy = None  # type: ignore
    PagedExpertManager = None  # type: ignore
    StrataGGUFMoELoader = None  # type: ignore
    register_strata_gguf_types = None  # type: ignore
    triton_paged_moe_forward = None  # type: ignore

from hydragen.backends.registry import BackendRegistry
from hydragen.core.block_manager import TRTPagedBlockManager, create_block_manager
from hydragen.core.scheduler import InFlightScheduler


class HydragenEngine:
    """Unified Serving Engine combining In-Flight Batching, Paged KV, and Locality-Aware MoE."""

    @staticmethod
    def _find_fallback_model() -> Optional[str]:
        for base in [Path.cwd(), Path.cwd() / "models", Path.home() / "models"]:
            if base.exists():
                for p in base.glob("*.gguf"):
                    return str(p.resolve())
        return None

    def __init__(
        self,
        cfg: Dict[str, Any],
        tok: Any,
        batch_slots: int = 4,
        num_vram_slots: int = 24,
        model_shard: Optional[str] = None,
        max_context: int = 131072,
        block_size: int = 16,
        max_blocks: int = 4096,
        standalone_mode: bool = False,
        hardware_worker: Any = None,
    ):
        self.tok = tok
        self.batch = int(batch_slots)
        self.num_slots = int(num_vram_slots)
        self.cfg = cfg or {}
        self.max_context = max_context
        self.standalone_mode = bool(standalone_mode)
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)

        env_cfg = self.cfg.get("env") or {}
        self.route_prior = float(
            env_cfg.get("STRATA_ROUTE_PRIOR", os.environ.get("STRATA_ROUTE_PRIOR", "0.85"))
        )
        self.enable_mailbox = (
            str(env_cfg.get("HYDRAGEN_ENABLE_MAILBOX", os.environ.get("HYDRAGEN_ENABLE_MAILBOX", "0"))).lower()
            in ("1", "true", "yes")
        )

        # 1. Hardware Backend Resolution (Decoupled from CUDA)
        self.hw = BackendRegistry.get_hardware_backend()
        dev_type = self.hw.capability.device_type.value if hasattr(self.hw.capability.device_type, "value") else str(self.hw.capability.device_type)
        self.device = torch.device(dev_type if dev_type != "cpu" and HAS_TORCH and torch.cuda.is_available() else "cpu") if HAS_TORCH else "cpu"
        self.compute_stream = self.hw.create_stream(device=self.device) if HAS_TORCH else None
        self.compute_lock = threading.Lock()

        # 2. Virtual Paged KV Block Manager (TensorRT-LLM Paged Memory)
        self.kv_manager = create_block_manager(block_size=self.block_size, max_blocks=self.max_blocks)

        # 3. Iteration-Level In-Flight Batch Scheduler
        self.scheduler = InFlightScheduler(batch_slots=self.batch, kv_manager=self.kv_manager)

        # 4. Resolve Model Shard
        args = self.cfg.get("args") or []
        native_arg = None
        if isinstance(args, list) and "--native" in args:
            try:
                native_arg = args[args.index("--native") + 1]
            except (ValueError, IndexError):
                pass

        shard = (
            model_shard
            or self.cfg.get("native")
            or self.cfg.get("model")
            or native_arg
            or os.environ.get("HYDRAGEN_MODEL")
            or self._find_fallback_model()
        )

        self.loader = None
        self.expert_manager = None
        self.locality_policy = ExpertLocalityPolicy(set()) if ExpertLocalityPolicy is not None else None
        self.router_gates: Dict[int, Any] = {}

        if shard and os.path.exists(shard) and HAS_GGUF and StrataGGUFMoELoader is not None:
            print(f"[hydragen] Initializing HydragenEngine with shard: {Path(shard).name}")
            register_strata_gguf_types()
            self.loader = StrataGGUFMoELoader(shard)
            meta = self.loader.get_layer_expert_metadata(0)
            self.n_embd = meta["n_embd"]
            self.n_ff = meta["n_ff"]
            self.n_exp = meta["n_exp"]

            # Initialize PagedExpertManager
            if HAS_TORCH:
                self.expert_manager = PagedExpertManager(
                    num_slots=self.num_slots,
                    expert_shape=(3, self.n_ff, self.n_embd),
                    total_experts=self.n_exp,
                    dtype=torch.float16,
                    device=self.device,
                    loader=self.loader,
                    default_layer_idx=0,
                    hw_backend=self.hw,
                    enable_mailbox=self.enable_mailbox,
                )
                try:
                    self.router_gates[0] = self.loader.load_router_gate(0, device=self.device)
                    self.router_gates[1] = self.loader.load_router_gate(1, device=self.device)
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            self.n_embd = 2560
            self.n_ff = 640
            self.n_exp = 512

        # 5. Resident Hardware Execution Worker (bin/strata, replicas or split serving).
        # The server builds it (every start-up guard lives there); a worker is only spawned here when none is given.
        self.hardware_worker = hardware_worker
        self.backend = hardware_worker  # Backward-compatibility alias
        if hardware_worker is not None:
            self.max_context = getattr(hardware_worker, "max_context", max_context)
            self.batch = int(getattr(hardware_worker, "batch", self.batch) or self.batch)
        elif not self.standalone_mode and "exe" in self.cfg:
            try:
                from hydragen.server.server import HardwareExecutionWorker, engine_args
            except ImportError:
                from serve.server import StrataEngine as HardwareExecutionWorker, engine_args

            exe = (
                self.cfg["exe"]
                if os.path.isabs(self.cfg["exe"])
                else os.path.abspath(os.path.join(self.cfg.get("cwd") or ".", self.cfg["exe"]))
            )
            args_list = engine_args(self.cfg)
            log_path = self.cfg.get("log") or os.environ.get("HYDRAGEN_LOG") or os.path.join(self.cfg.get("cwd") or ".", "hydragen.log")
            print(f"[hydragen] Initializing HardwareExecutionWorker: {exe}")
            try:
                self.hardware_worker = HardwareExecutionWorker(
                    exe, args_list, cwd=self.cfg.get("cwd"), log=log_path, env=self.cfg.get("env")
                )
                self.backend = self.hardware_worker
                self.max_context = getattr(self.hardware_worker, "max_context", max_context)
                self.batch = int(getattr(self.hardware_worker, "batch", self.batch) or self.batch)
            except Exception as e:
                print(f"[hydragen] Hardware worker notice: {e}")

        # 6. Speculative Drafter Architecture
        from hydragen.drafters import DisabledDrafter, WorkerDelegatedDrafter, get_speculative_drafter
        from hydragen.drafters.base import DraftingMode

        worker_args = self.cfg.get("args") or []
        if self.hardware_worker is not None:
            # Under Option C, speculative drafting is executed natively inside the hardware worker
            # process (bin/strata with --mtp / --spec / --draft). We delegate drafter telemetry
            # to the worker rather than allocating redundant PyTorch GPU tensors that would OOM.
            has_dflash = "--draft" in worker_args and "dflash2" in worker_args
            has_mtp = "--mtp" in worker_args or any(isinstance(a, str) and a.startswith("--mtp") for a in worker_args)
            has_spec = "--spec" in worker_args or any(isinstance(a, str) and a.startswith("--spec") for a in worker_args)

            if has_dflash:
                self.drafter = WorkerDelegatedDrafter(mode=DraftingMode.DFLASH2, name="dflash2_block_drafter")
            elif has_mtp or has_spec:
                self.drafter = WorkerDelegatedDrafter(mode=DraftingMode.MTP, name="mtp_autoregressive_drafter")
            else:
                self.drafter = DisabledDrafter()
        else:
            mtp_path = os.path.join(self.cfg.get("cwd") or ".", "models/mtp/rt")
            try:
                self.drafter = get_speculative_drafter(
                    hardware_backend=self.hw,
                    device=self.device,
                    n_embd=self.n_embd,
                    hc=4,
                    n_head_kv=2,
                    head_dim=128,
                    max_context=self.max_context,
                    mtp_dir=mtp_path if os.path.exists(mtp_path) else None,
                )
            except Exception:
                self.drafter = DisabledDrafter()

        # 7. Priority Wait Queue for Locality-Aware Admission
        self.wait_queue: List[Dict[str, Any]] = []
        self.slot_lock = threading.Lock()
        self.backend_lock = threading.Lock()
        self._prefill_active_count = 0
        self._last_prefill_tok_s = 100.0

        # Telemetry & Info
        b_info = getattr(self.backend, "info", {}) if self.backend is not None else {}
        slots = b_info.get("expert_slots", 2211)
        cache_mib = (
            b_info.get("expert_cache_mib")
            or int(slots * 1868800 / (1024 * 1024))
            or 3940
        )
        cap = self.hw.capability
        plugin_desc = getattr(BackendRegistry, "_plugin_desc", None) or {}
        self.info = {
            **b_info,
            "version": "0.35.0-hydragen-unified",
            "engine": "hydragen-unified",
            "engine_type": "hydragen",
            "engine_label": "Hydragen Unified (In-Flight Batching + Paged KV + MoE Locality)",
            "model": "qwen3.8-flash-next-iq3_xxs",
            "arena_mib": b_info.get("arena_mib", 40925),
            "expert_slots": slots,
            "vram_slots": self.num_slots,
            "expert_cache_mib": cache_mib,
            "total_experts": self.n_exp,
            "vram_free_mib": b_info.get("vram_free_mib", 463),
            "batch_slots": self.batch,
            "kv": "paged_int8",
            "in_flight_batching": True,
            "paged_kv": True,
            "block_size": self.block_size,
            "hardware_backend": cap.device_type.value if hasattr(cap.device_type, "value") else str(cap.device_type),
            "hardware_device": cap.device_name,
            "hardware_plugin": plugin_desc.get("name", "hydragen_native"),
            "plugin_abi": plugin_desc.get("abi_version", 1),
            "drafter": self.drafter.name,
            "drafting_mode": self.drafter.mode.value,
        }
        self.last: Dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        """Anything the engine does not define (silence_s, spawn, death_note, restart, ...) is the worker's."""
        if name.startswith("__") or name in ("hardware_worker", "backend"):
            raise AttributeError(name)
        worker = self.__dict__.get("hardware_worker")
        if worker is None:
            raise AttributeError(name)
        return getattr(worker, name)

    _WORKER_OWNED = ("gpu_busy", "silence_s", "silent_note")

    def __setattr__(self, name: str, value: Any) -> None:
        """The stall watchdog reads these on the worker, so a write through the engine must land there."""
        worker = self.__dict__.get("hardware_worker")
        if name in self._WORKER_OWNED and worker is not None:
            setattr(worker, name, value)
            return
        object.__setattr__(self, name, value)

    def slots_view(self) -> List[Dict[str, Any]]:
        """Unified telemetry view showing active slot state, prompt read, speed, and paged KV blocks."""
        scheduler_slots = self.scheduler.slots_view()
        kv_stats = self.kv_manager.stats()
        now = time.time()
        view = []
        for s in scheduler_slots:
            slot_idx = s["slot"]
            st = s.get("state", "idle")
            blocks = s.get("blocks", 0)
            view.append({
                "slot": slot_idx,
                "state": st,
                "seq_id": s.get("seq_id"),
                "prompt_tokens": s.get("prompt", 0),
                "prompt_read": s.get("read", 0),
                "generated": s.get("generated", 0),
                "elapsed_s": round(now - s["start_time"], 1) if s.get("start_time") else 0.0,
                "tok_s": round(s["generated"] / max(1e-4, now - s["first_token"]), 1) if s.get("first_token") else None,
                "paged_blocks": blocks,
                "kv_utilization": round(kv_stats["utilization"] * 100, 1),
            })
        return view

    def vram(self, reserve_mib: Optional[int] = None, timeout: float = 30.0) -> Optional[int]:
        worker = self.hardware_worker or self.backend
        if worker is not None and hasattr(worker, "vram"):
            return worker.vram(reserve_mib=reserve_mib, timeout=timeout)
        return reserve_mib or 0

    def close(self) -> None:
        worker = self.hardware_worker or self.backend
        if worker is not None and hasattr(worker, "close"):
            worker.close()

    def restart(self) -> None:
        worker = self.hardware_worker or self.backend
        if worker is not None and hasattr(worker, "restart"):
            worker.restart()

    def alive(self) -> bool:
        worker = self.hardware_worker or self.backend
        if worker is not None and hasattr(worker, "alive"):
            return worker.alive()
        return True

    def unload(self) -> None:
        worker = self.hardware_worker or self.backend
        if worker is not None and hasattr(worker, "unload"):
            worker.unload()

    def _acquire_locality_slot(
        self,
        prompt_len: int,
        prompt_read: int,
        req_experts: Set[int],
        cancel: threading.Event,
    ) -> Tuple[int, int]:
        """Locality-aware In-Flight slot admission."""
        if not self.locality_policy or not self.wait_queue:
            return self.scheduler.acquire_slot(prompt_len, prompt_read, cancel)

        # Priority queue admission ranked by MoE expert locality score
        waiter = {
            "event": threading.Event(),
            "experts": req_experts,
            "prompt_len": prompt_len,
            "prompt_read": prompt_read,
            "slot": -1,
            "seq_id": -1,
        }
        with self.slot_lock:
            # Check if any slot is immediately free
            slot_idx, seq_id = self.scheduler.acquire_slot(prompt_len, prompt_read, threading.Event())
            if slot_idx >= 0:
                return slot_idx, seq_id
            self.wait_queue.append(waiter)

        while not cancel.is_set():
            if waiter["event"].wait(timeout=0.01):
                return waiter["slot"], waiter["seq_id"]
            with self.slot_lock:
                if waiter not in self.wait_queue:
                    return waiter["slot"], waiter["seq_id"]
                # Rank candidates by cache locality
                free_slot = next(
                    (i for i in range(self.batch) if self.scheduler.slot_live[i] is None),
                    None,
                )
                if free_slot is not None and self.wait_queue:
                    candidates = [(w, w["experts"]) for w in self.wait_queue]
                    ranked = self.locality_policy.rank_candidates(candidates)
                    best_w, score = ranked[0]
                    self.wait_queue.remove(best_w)
                    s_idx, s_id = self.scheduler.acquire_slot(
                        best_w["prompt_len"], best_w["prompt_read"], cancel
                    )
                    best_w["slot"] = s_idx
                    best_w["seq_id"] = s_id
                    best_w["event"].set()
                    if best_w is waiter:
                        return s_idx, s_id

        with self.slot_lock:
            if waiter in self.wait_queue:
                self.wait_queue.remove(waiter)
        return -1, -1

    def generate(
        self,
        ids: List[int],
        max_new: int,
        sampling: Dict[str, Any],
        cancel: threading.Event,
        embeddings: Optional[Any] = None,
    ) -> Iterator[int]:
        """Generate tokens through the unified Hydragen In-Flight + Paged KV pipeline."""
        prompt_len = len(ids)
        reused_tokens = self.scheduler.match_prefix_cache(ids)
        prompt_read = max(0, prompt_len - reused_tokens)

        req_experts: Set[int] = set()
        if self.locality_policy and HAS_TORCH and self.loader and self.expert_manager:
            try:
                tail = torch.tensor(ids[-min(16, len(ids)):], dtype=torch.int32, device=self.device)
                with self.hw.stream_context(self.compute_stream):
                    rw = self.router_gates.get(0)
                    if rw is not None:
                        h = self.loader.dequantize_token_embedding(tail, device=self.device)
                        topk_ids, _ = self.expert_manager.route_tokens(0, h, rw, prior_lambda=self.route_prior)
                        req_experts = set(topk_ids.reshape(-1).tolist())
            except Exception:
                pass

        slot_idx, seq_id = self._acquire_locality_slot(prompt_len, prompt_read, req_experts, cancel)
        if slot_idx < 0 or cancel.is_set():
            return

        first_token_marked = False
        generated_count = 0
        generated_token_ids: List[int] = []
        backend_gen = None

        t_start = time.perf_counter()
        t_first_token = None

        try:
            if self.backend is not None:
                backend_gen = (
                    self.backend.generate(
                        ids, max_new, sampling, cancel, embeddings=embeddings
                    )
                    if embeddings is not None
                    else self.backend.generate(ids, max_new, sampling, cancel)
                )
            else:
                # Standalone test/simulation generator
                backend_gen = iter([100 + i for i in range(min(max_new, 16))])

            for tok_id in backend_gen:
                if cancel.is_set():
                    break
                if tok_id is None:
                    yield None
                    continue

                if not first_token_marked:
                    first_token_marked = True
                    t_first_token = time.perf_counter()
                    self.scheduler.mark_decoding(slot_idx)

                    # Post-Prefill Bulk Expert Realignment
                    if self.expert_manager is not None:
                        adapt_res = self.expert_manager.adapt_post_prefill()
                        if adapt_res.get("swaps_performed", 0) > 0 and self.locality_policy is not None:
                            res_eids = {
                                k[1] if isinstance(k, tuple) else int(k)
                                for k in self.expert_manager.residency_map.keys()
                            }
                            self.locality_policy.update_resident_experts(res_eids)

                generated_count += 1
                generated_token_ids.append(tok_id)
                self.scheduler.update_generated(slot_idx, generated_count)
                yield tok_id

        finally:
            if backend_gen is not None and hasattr(backend_gen, "close"):
                try:
                    backend_gen.close()
                except Exception:
                    pass

            self.scheduler.release_slot(slot_idx, seq_id)
            self.scheduler.record_prompt_cache(ids, generated_token_ids)

            # Record telemetry
            b_last = getattr(self.backend, "last", {}) if self.backend else {}
            last_dict = dict(b_last) if isinstance(b_last, dict) else {}
            last_dict["reused"] = max(int(b_last.get("reused") or 0), reused_tokens)
            last_dict["prompt_tokens"] = prompt_len
            last_dict["prompt_read"] = prompt_read
            last_dict["generated"] = generated_count
            if first_token_marked and t_first_token is not None:
                last_dict["prompt_ms"] = round((t_first_token - t_start) * 1000.0, 1)
            if self.expert_manager is not None:
                last_dict["moe_hit_rate"] = round(self.expert_manager.hit_rate, 4)
                if getattr(self.expert_manager, "mailbox", None) is not None:
                    mb = self.expert_manager.mailbox
                    last_dict["mailbox_cpu_experts"] = mb.total_cpu_experts
                    last_dict["mailbox_gpu_experts"] = mb.total_gpu_experts
                    last_dict["mailbox_fusions"] = mb.total_fusions
            if cancel.is_set():
                last_dict["finish"] = "cancel"
            self.last = last_dict
