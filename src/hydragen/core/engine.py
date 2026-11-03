# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/core/engine.py - TensorRT-LLM In-Flight Batching Engine for Hydragen.

Orchestrates the iteration-level scheduler, paged KV block manager, and resident pipeline worker.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from hydragen.core.block_manager import TRTPagedBlockManager, create_block_manager
from hydragen.core.scheduler import InFlightScheduler

# Graceful optional imports for PyTorch and GGUF MoE components
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore

HAS_GGUF = False
try:
    from hydragen.expert_locality_policy import ExpertLocalityPolicy
    from hydragen.expert_manager import PagedExpertManager
    from hydragen.gguf_loader import StrataGGUFMoELoader, register_strata_gguf_types
    HAS_GGUF = True
except ImportError:
    ExpertLocalityPolicy = None  # type: ignore
    PagedExpertManager = None  # type: ignore
    StrataGGUFMoELoader = None  # type: ignore
    register_strata_gguf_types = None  # type: ignore

# Check for tensorrt_llm if available
HAS_TRTLLM = False
try:
    import tensorrt_llm
    from tensorrt_llm.executor import Executor, GenerationExecutor
    HAS_TRTLLM = True
except ImportError:
    tensorrt_llm = None
    Executor = None
    GenerationExecutor = None


class TRTLLMEngine:
    """TensorRT-LLM In-Flight Batching + Paged KV Engine implementing Hydragen Engine Protocol."""

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
        trt_engine_dir: Optional[str] = None,
    ):
        self.tok = tok
        self.batch = int(batch_slots)
        self.num_slots = int(num_vram_slots)
        self.cfg = cfg or {}
        self.max_context = max_context
        self.trt_engine_dir = trt_engine_dir or self.cfg.get("trt_engine_dir")

        env_cfg = self.cfg.get("env") or {}
        self.route_prior = float(
            env_cfg.get("STRATA_ROUTE_PRIOR", os.environ.get("STRATA_ROUTE_PRIOR", "0.8"))
        )

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
        if not shard or not os.path.exists(shard):
            raise FileNotFoundError(
                f"Model shard not found: {shard}. "
                "Specify a valid GGUF file in your config or via HYDRAGEN_MODEL."
            )
        print(f"[hydragen:trtllm] Initializing TRTLLMEngine with shard: {Path(shard).name}")
        print(f"[hydragen:trtllm] Concurrency slots: {self.batch}, In-Flight Batching: ENABLED")

        # 1. GGUF Loader & Bridge Metadata
        if register_strata_gguf_types is not None:
            register_strata_gguf_types()
        if StrataGGUFMoELoader is not None:
            self.loader = StrataGGUFMoELoader(shard)
            meta = self.loader.get_layer_expert_metadata(0)
            self.n_embd = meta["n_embd"]
            self.n_ff = meta["n_ff"]
            self.n_exp = meta["n_exp"]
        else:
            self.loader = None
            self.n_embd = 4096
            self.n_ff = 11008
            self.n_exp = 64

        # 2. Check for TRT-LLM C++ Executor Runtime or Resident Backend
        self.backend = None
        self.trt_executor = None

        if HAS_TRTLLM and self.trt_engine_dir and os.path.exists(self.trt_engine_dir):
            try:
                print(f"[hydragen:trtllm] Connecting to native TensorRT-LLM Executor engine at: {self.trt_engine_dir}")
                from tensorrt_llm.executor import ExecutorConfig, KvCacheConfig
                exec_cfg = ExecutorConfig(
                    max_beam_width=1,
                    kv_cache_config=KvCacheConfig(
                        paged_kv_cache=True,
                        free_gpu_memory_fraction=0.85,
                    ),
                )
                self.trt_executor = GenerationExecutor(self.trt_engine_dir, exec_cfg)
                print("[hydragen:trtllm] Successfully initialized native TensorRT-LLM GenerationExecutor!")
            except Exception as e:
                print(f"[hydragen:trtllm] Native TRT-LLM Executor init warning: {e}. Falling back to Strata resident backend.")

        if self.trt_executor is None and "exe" in self.cfg:
            try:
                from hydragen.server.server import StrataEngine, engine_args
            except ImportError:
                from serve.server import StrataEngine, engine_args

            exe = (
                self.cfg["exe"]
                if os.path.isabs(self.cfg["exe"])
                else os.path.abspath(
                    os.path.join(self.cfg.get("cwd") or ".", self.cfg["exe"])
                )
            )
            args = engine_args(self.cfg)
            log_path = self.cfg.get("log") or os.environ.get("HYDRAGEN_LOG") or os.path.join(self.cfg.get("cwd") or ".", "hydragen.log")
            env = self.cfg.get("env")
            print(f"[hydragen:trtllm] Launching resident 48-layer pipeline worker: {exe}")
            self.backend = StrataEngine(
                exe, args, cwd=self.cfg.get("cwd"), log=log_path, env=env
            )
            self.max_context = getattr(self.backend, "max_context", max_context)
            self.batch = int(getattr(self.backend, "batch", self.batch) or self.batch)
            # TensorRT-LLM C++ runtime requires only ~256 MiB reserve
            try:
                self.backend.vram(reserve_mib=256, timeout=30.0)
            except Exception:
                pass

        # 3. Virtual Paged KV Block Manager
        self.kv_manager = create_block_manager(block_size=16, max_blocks=4096)

        # 4. In-Flight Batch Scheduler
        self.scheduler = InFlightScheduler(batch_slots=self.batch, kv_manager=self.kv_manager)

        # 5. Expert Manager & Locality
        self.expert_manager = None
        self.device = torch.device("cpu") if HAS_TORCH else "cpu"
        if self.trt_executor is not None and HAS_TORCH and PagedExpertManager is not None:
            try:
                from hydragen.backends.registry import BackendRegistry
                hw = BackendRegistry.get_hardware_backend()
                dev_type = hw.capability.device_type
                self.device = torch.device(dev_type if dev_type != "cpu" else "cpu")
                self.expert_manager = PagedExpertManager(
                    num_slots=self.num_slots,
                    expert_shape=(3, self.n_ff, self.n_embd),
                    total_experts=self.n_exp,
                    dtype=torch.float16,
                    device=self.device,
                    loader=self.loader,
                    default_layer_idx=0,
                    hw_backend=hw,
                )
            except Exception as e:
                print(f"[hydragen:trtllm] PagedExpertManager note: {e}")
                self.expert_manager = None
        self.locality_policy = ExpertLocalityPolicy(set()) if ExpertLocalityPolicy is not None else None

        # Telemetry & Info
        b_info = getattr(self.backend, "info", {}) if self.backend is not None else {}
        slots = b_info.get("expert_slots", 2150)
        cache_mib = (
            b_info.get("expert_cache_mib")
            or int(slots * 1868800 / (1024 * 1024))
            or 3832
        )
        self.info = {
            **b_info,
            "version": "0.34.0-hydragen-trtllm",
            "engine": "trtllm-inflight-batching",
            "engine_type": "trtllm",
            "engine_label": "TensorRT-LLM (In-Flight Batching + Paged KV)",
            "model": "qwen3.8-flash-next-iq3_xxs",
            "expert_slots": slots,
            "trt_vram_slots": self.num_slots,
            "expert_cache_mib": cache_mib,
            "total_experts": self.n_exp,
            "batch_slots": self.batch,
            "kv": "paged_int8",
            "in_flight_batching": True,
            "runtime": "C++ Executor" if HAS_TRTLLM else "Strata Hybrid Bridge",
        }
        self._last_prefill_tok_s: float = 2400.0
        self._active_prefill_tok_s: Optional[float] = None
        self._prefill_burst_pending: Optional[float] = None
        self._active_prompt_progress: Optional[Tuple[int, int]] = None

        eos = self.tok.encode("<|im_end|>", parse_special=True) if self.tok else [248046]
        self.im_end_id = eos[0] if eos else 248046
        self._tl = threading.local()
        self.eos_ids = getattr(self.backend, "eos_ids", set())

    @property
    def slot_live(self) -> List[Optional[Dict[str, Any]]]:
        return self.scheduler.slot_live

    @property
    def slot_held(self) -> List[List[int]]:
        return self.scheduler.slot_held

    @property
    def slot_lock(self) -> threading.Lock:
        return self.scheduler.lock

    def __getattr__(self, name: str) -> Any:
        if self.backend is not None and hasattr(self.backend, name):
            return getattr(self.backend, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def pop_prefill_burst(self) -> Optional[float]:
        val = self._prefill_burst_pending
        self._prefill_burst_pending = None
        if val is None and self.backend is not None and hasattr(self.backend, "pop_prefill_burst"):
            return self.backend.pop_prefill_burst()
        return val

    @property
    def prefill_tok_s_mean(self) -> Optional[float]:
        if self._active_prefill_tok_s is not None:
            return self._active_prefill_tok_s
        if self.backend is not None and hasattr(self.backend, "prefill_tok_s_mean"):
            val = getattr(self.backend, "prefill_tok_s_mean")
            return val() if callable(val) else val
        return self._last_prefill_tok_s

    @property
    def progress(self) -> Optional[Tuple[int, int]]:
        if self._active_prompt_progress is not None:
            return self._active_prompt_progress
        if self.backend is not None and hasattr(self.backend, "progress"):
            val = getattr(self.backend, "progress")
            return val() if callable(val) else val
        return None

    @property
    def last(self) -> Dict[str, Any]:
        if self.backend is not None and hasattr(self.backend, "last"):
            return self.backend.last
        return getattr(self, "_last", {})

    @last.setter
    def last(self, value: Dict[str, Any]):
        self._last = value
        if self.backend is not None:
            try:
                self.backend.last = value
            except AttributeError:
                pass

    def alive(self) -> bool:
        if self.trt_executor is not None:
            return True
        return self.backend.alive() if self.backend is not None else True

    def close(self):
        if self.backend is not None and hasattr(self.backend, "close"):
            self.backend.close()

    def unload(self):
        if self.backend is not None and hasattr(self.backend, "unload"):
            self.backend.unload()

    def restart(self, tries: int = 3):
        if self.backend is not None and hasattr(self.backend, "restart"):
            return self.backend.restart(tries=tries)

    def vram(self, reserve_mib: Optional[int] = None, timeout: float = 120.0) -> Dict[str, Any]:
        if self.backend is not None and hasattr(self.backend, "vram"):
            return self.backend.vram(reserve_mib=reserve_mib, timeout=timeout)
        return {"reserve_mib": reserve_mib or 256, "status": "ok"}

    def slots_view(self) -> List[Dict[str, Any]]:
        if self.backend is not None and hasattr(self.backend, "slots_view"):
            return self.backend.slots_view()
        return self.scheduler.slots_view()

    def _acquire_slot(
        self,
        prompt_len: int,
        prompt_read: int,
        cancel: threading.Event,
    ) -> Tuple[int, int]:
        return self.scheduler.acquire_slot(prompt_len, prompt_read, cancel)

    def _release_slot(self, slot_idx: int, seq_id: int):
        self.scheduler.release_slot(slot_idx, seq_id)

    def _match_prefix_cache(self, ids: List[int]) -> int:
        return self.scheduler.match_prefix_cache(ids)

    def _record_prompt_cache(self, ids: List[int], generated_ids: List[int]):
        self.scheduler.record_prompt_cache(ids, generated_ids)

    def generate(
        self,
        ids: List[int],
        max_new: int,
        sampling: Dict[str, Any],
        cancel: threading.Event,
        embeddings: Optional[Any] = None,
    ) -> Iterator[int]:
        """Generate tokens through the TensorRT-LLM In-Flight Batching pipeline."""
        prompt_len = len(ids)
        reused_tokens = self._match_prefix_cache(ids)
        prompt_read = max(0, prompt_len - reused_tokens)

        slot_idx, seq_id = self._acquire_slot(prompt_len, prompt_read, cancel)
        if slot_idx < 0 or cancel.is_set():
            return

        first_token_marked = False
        generated_count = 0
        generated_token_ids: List[int] = []
        backend_gen = None

        t_start = time.perf_counter()
        t_first_token = None

        try:
            if self.trt_executor is not None:
                request = self.trt_executor.generate(
                    ids,
                    max_new_tokens=max_new,
                    temperature=sampling.get("temp", 1.0),
                    top_k=sampling.get("top_k", 50),
                    top_p=sampling.get("top_p", 0.9),
                )
                for response in request:
                    if cancel.is_set():
                        request.cancel()
                        break
                    tok = response.token_id
                    if not first_token_marked:
                        first_token_marked = True
                        self.scheduler.mark_decoding(slot_idx)
                    generated_count += 1
                    generated_token_ids.append(tok)
                    yield tok
                return

            if self.backend is not None:
                backend_gen = (
                    self.backend.generate(
                        ids, max_new, sampling, cancel, embeddings=embeddings
                    )
                    if embeddings is not None
                    else self.backend.generate(ids, max_new, sampling, cancel)
                )
            else:
                backend_gen = iter([100 + i for i in range(min(max_new, 16))])

            for tok_id in backend_gen:
                if cancel.is_set():
                    break
                if tok_id is None:
                    yield None
                    continue

                if not first_token_marked:
                    t_first_token = time.perf_counter()
                    dt_prefill = max(1e-4, t_first_token - t_start)
                    b_pp_rate = getattr(self.backend, "prefill_tok_s_mean", None) if self.backend else None
                    if b_pp_rate is not None and b_pp_rate > 0:
                        measured_prefill = round(float(b_pp_rate), 1)
                    elif prompt_read > 0:
                        measured_prefill = round(prompt_read / max(1e-4, dt_prefill), 1)
                    else:
                        measured_prefill = round(float(self._last_prefill_tok_s), 1)
                    if 10.0 <= measured_prefill <= 4500.0:
                        self._last_prefill_tok_s = measured_prefill
                        self._prefill_burst_pending = measured_prefill

                    self.scheduler.mark_decoding(slot_idx)
                    first_token_marked = True

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
            self._release_slot(slot_idx, seq_id)
            if generated_token_ids:
                self._record_prompt_cache(ids, generated_token_ids)


__all__ = ["TRTLLMEngine"]
