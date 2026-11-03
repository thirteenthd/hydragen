# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project.

"""PagedExpertManager: Tiered memory management and asynchronous expert hot-swapping for MoE models.

Supports two VRAM caching modes:
1. Compressed GGUF Blob Mode (`compressed_cache=True`, default when `loader` is attached):
   Stores raw quantized `[gate | up | down]` GGUF blocks directly in `vram_blob_cache`
   (`~1.54 - 1.69 MB` per expert instead of `9.83 MB` in BF16/FP16 — a 5.8x VRAM savings
   and zero intermediate dequantization buffer), consumed directly by both:
   - Option 1: In-kernel Triton GGUF unpacker (`fused_moe_gguf_down_slot_kernel`)
   - Option 2: Native CUDA INT8 `native_expert_grouped` (`dp4a` / `mma.sync`)
2. FP16/BF16 Arena Mode (`compressed_cache=False`, or synthetic host-pool tests):
   Stores `[num_slots, *expert_shape]` FP16/BF16 tensors in `vram_arena`.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple
import torch


class PagedExpertManager:
    """Manages dynamic allocation, slot indirection, and async PCIe DMA transfers
    for Mixture-of-Experts weights in compressed GGUF or FP16 arena format.
    """

    # Max compressed expert blob across all 48 layers of Qwen3.8-Flash-Next (<= 2.22 MB, aligned to 2.25 MB)
    MAX_COMPRESSED_BLOB_BYTES = 2304 * 1024

    def __init__(
        self,
        num_slots: int,
        expert_shape: Tuple[int, ...],
        total_experts: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda:0"),
        loader: Optional[Any] = None,
        default_layer_idx: int = 0,
        compressed_cache: Optional[bool] = None,
    ):
        self.num_slots = int(num_slots)
        self.expert_shape = tuple(expert_shape)
        self.total_experts = int(total_experts)
        self.dtype = dtype
        self.device = device
        self.loader = loader
        self.default_layer_idx = int(default_layer_idx)
        self.compressed_cache = (loader is not None) if compressed_cache is None else bool(compressed_cache)

        # Telemetry counters
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.pcie_bytes_transferred: int = 0
        self.routing_counts: Dict[Any, float] = {}
        self.total_swaps: int = 0

        # Create dedicated CUDA stream first so CUDA context & stream memory are accounted for
        self.dma_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

        if self.compressed_cache and self.loader is not None:
            self.max_blob_bytes = self.MAX_COMPRESSED_BLOB_BYTES
            if device.type == "cuda" and torch.cuda.is_available():
                free_b, _ = torch.cuda.mem_get_info(device)
                # Leave >=110 MiB for Triton/CUDA scratch workspaces
                usable_b = max(
                    8 * self.max_blob_bytes,
                    min(int(free_b * 0.65), free_b - 110 * 1024 * 1024),
                )
                max_fit_slots = max(8, int(usable_b // self.max_blob_bytes))
                if self.num_slots > max_fit_slots:
                    self.num_slots = max_fit_slots

            # Compressed L1 GPU VRAM Slot Pool: [num_slots, max_blob_bytes] uint8 (~1.75 MB/slot vs 9.83 MB FP16)
            self.vram_blob_cache: Optional[torch.Tensor] = torch.empty(
                (self.num_slots, self.max_blob_bytes),
                dtype=torch.uint8,
                device=device,
            )
            self.vram_arena: Optional[torch.Tensor] = None
            self.host_pool: Optional[torch.Tensor] = None
            self.staging_gate_u8 = None
            self.staging_up_u8 = None
            self.staging_down_u8 = None
        else:
            self.vram_blob_cache = None
            self.max_blob_bytes = 0
            # Auto-clamp num_slots to available free GPU memory when sharing GPU
            if self.loader is not None and torch.cuda.is_available():
                free_b, _ = torch.cuda.mem_get_info(device)
                elem_b = 2 if dtype in (torch.float16, torch.bfloat16) else 4
                slot_b = elem_b
                for d in self.expert_shape:
                    slot_b *= d
                usable_b = max(4 * slot_b, min(int(free_b * 0.50), free_b - 110 * 1024 * 1024))
                max_fit_slots = max(4, int(usable_b // slot_b))
                if self.num_slots > max_fit_slots:
                    self.num_slots = max_fit_slots

            # 1. Allocate L1 GPU VRAM Expert Arena [num_slots, *expert_shape]
            self.vram_arena = torch.empty(
                (self.num_slots, *self.expert_shape),
                dtype=dtype,
                device=device,
            )

            # 2. Allocate L2 Host Pool (when not backed by GGUF mmap loader) or staging buffers
            is_cuda = (device.type == "cuda" and torch.cuda.is_available())
            if self.loader is None:
                self.host_pool = torch.empty(
                    (self.total_experts, *self.expert_shape),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=is_cuda,
                )
                self.staging_gate_u8 = None
                self.staging_up_u8 = None
                self.staging_down_u8 = None

                # Milestone 4: Concurrent CPU+GPU Hybrid Expert Mailbox (opt-in via HYDRAGEN_ENABLE_MAILBOX=1)
                enable_box = (os.environ.get("HYDRAGEN_ENABLE_MAILBOX", "0").lower() in ("1", "true", "yes"))
                if enable_box:
                    try:
                        from hydragen.expert_mailbox import HybridExpertMailbox
                        self.mailbox = HybridExpertMailbox(device=device)
                    except Exception:
                        self.mailbox = None
                else:
                    self.mailbox = None
            else:
                self.host_pool = None
                self.mailbox = None
                max_proj_bytes = 1536 * 1024
                self.staging_gate_u8 = torch.empty(
                    (max_proj_bytes,), dtype=torch.uint8, device=device
                )
                self.staging_up_u8 = torch.empty(
                    (max_proj_bytes,), dtype=torch.uint8, device=device
                )
                self.staging_down_u8 = torch.empty(
                    (max_proj_bytes,), dtype=torch.uint8, device=device
                )

        # 3. GPU-resident Indirection Table for Triton / CUDA kernels
        # gpu_slot_table[expert_id] == slot_idx (or -1 if not resident)
        is_cuda_device = (self.device.type == "cuda" and torch.cuda.is_available())
        self.gpu_slot_table = torch.full(
            (self.total_experts,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.cpu_slot_table = torch.full(
            (self.total_experts,),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=is_cuda_device,
        )

        # Multi-layer indirection tables when multiplexing slots across layers
        self._layer_gpu_tables: Dict[int, torch.Tensor] = {
            self.default_layer_idx: self.gpu_slot_table
        }
        self._layer_cpu_tables: Dict[int, torch.Tensor] = {
            self.default_layer_idx: self.cpu_slot_table
        }

        # 4. Slot tracking and LRU eviction state
        # Key is `expert_id` (single-layer) or `(layer_idx, expert_id)` (multi-layer)
        self.residency_map: Dict[Any, int] = {}
        self.slot_to_expert: Dict[int, Any] = {}
        self.free_slots: List[int] = list(range(self.num_slots))
        self.lru_tracker: OrderedDict = OrderedDict()

    def _ensure_layer_tables(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self._layer_gpu_tables:
            is_cuda = (self.device.type == "cuda" and torch.cuda.is_available())
            self._layer_gpu_tables[layer_idx] = torch.full(
                (self.total_experts,), -1, dtype=torch.int32, device=self.device
            )
            self._layer_cpu_tables[layer_idx] = torch.full(
                (self.total_experts,), -1, dtype=torch.int32, device="cpu", pin_memory=is_cuda
            )
        return self._layer_gpu_tables[layer_idx], self._layer_cpu_tables[layer_idx]

    def get_gpu_slot_table(self, layer_idx: int = 0) -> torch.Tensor:
        return self._ensure_layer_tables(layer_idx)[0]

    def is_resident(self, expert_id: int, layer_idx: Optional[int] = None) -> bool:
        """Check if an expert is currently resident in VRAM."""
        if layer_idx is not None and (layer_idx, expert_id) in self.residency_map:
            return True
        return expert_id in self.residency_map

    def partition_experts(
        self, expert_ids: List[int], layer_idx: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """Partition expert IDs into resident (in VRAM) and missed (in host RAM)."""
        resident = []
        missed = []
        for eid in expert_ids:
            if self.is_resident(eid, layer_idx):
                resident.append(eid)
            else:
                missed.append(eid)
        return resident, missed

    def get_slot(self, expert_id: int) -> int:
        """Retrieve slot for resident expert and refresh its LRU timestamp."""
        if expert_id in self.residency_map:
            slot = self.residency_map[expert_id]
            self.lru_tracker.move_to_end(expert_id)
            return slot
        key = (self.default_layer_idx, expert_id)
        slot = self.residency_map[key]
        self.lru_tracker.move_to_end(key)
        return slot

    def get_slot_for_layer(self, layer_idx: int, expert_id: int) -> int:
        key = (layer_idx, expert_id)
        if key in self.residency_map:
            slot = self.residency_map[key]
            self.lru_tracker.move_to_end(key)
            return slot
        return self.get_slot(expert_id)

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return float(self.cache_hits) / float(total) if total > 0 else 1.0

    def _dma_expert_into_slot(self, layer_idx: int, expert_id: int, target_slot: int) -> None:
        if self.loader is not None:
            if self.compressed_cache and self.vram_blob_cache is not None:
                n_bytes = self.loader.dma_compressed_expert_to_gpu(
                    layer_idx,
                    expert_id,
                    self.vram_blob_cache[target_slot],
                    stream=self.dma_stream,
                )
                self.pcie_bytes_transferred += n_bytes
            else:
                has_up = self.vram_arena.shape[1] >= 3
                dst_gate = self.vram_arena[target_slot, 0]
                dst_up = self.vram_arena[target_slot, 1] if has_up else None
                dst_down = self.vram_arena[target_slot, 2 if has_up else 1]
                n_bytes = self.loader.dequantize_expert_to_gpu(
                    layer_idx,
                    expert_id,
                    dst_gate,
                    dst_down,
                    self.staging_gate_u8,
                    self.staging_down_u8,
                    stream=self.dma_stream,
                    dst_up_fp16=dst_up,
                    staging_up_u8=self.staging_up_u8 if has_up else None,
                )
                self.pcie_bytes_transferred += n_bytes
        elif self.host_pool is not None:
            self.vram_arena[target_slot].copy_(
                self.host_pool[expert_id],
                non_blocking=True,
            )
            self.pcie_bytes_transferred += (
                self.host_pool[expert_id].numel()
                * self.host_pool[expert_id].element_size()
            )

    def request_experts(
        self,
        required_expert_ids: List[int],
        locked_slots: Optional[Set[int]] = None,
    ) -> Dict[int, int]:
        """Ensures all required_expert_ids are resident in VRAM for default_layer_idx."""
        if locked_slots is None:
            locked_slots = set()

        missing = []
        for eid in required_expert_ids:
            if eid in self.residency_map:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
                missing.append(eid)

        table_updated = False
        locked_all = set(locked_slots)
        for eid in required_expert_ids:
            if eid in self.residency_map:
                locked_all.add(self.residency_map[eid])

        for expert_id in missing:
            if self.free_slots:
                target_slot = self.free_slots.pop()
            else:
                target_slot = self._evict_lru_victim(locked_all)
            locked_all.add(target_slot)

            with torch.cuda.stream(self.dma_stream):
                self._dma_expert_into_slot(self.default_layer_idx, expert_id, target_slot)

            # Update CPU mapping state
            self.residency_map[expert_id] = target_slot
            self.slot_to_expert[target_slot] = expert_id
            self.lru_tracker[expert_id] = None
            self.cpu_slot_table[expert_id] = target_slot
            table_updated = True

        if table_updated:
            with torch.cuda.stream(self.dma_stream):
                self.gpu_slot_table.copy_(self.cpu_slot_table, non_blocking=True)

        # Refresh LRU ordering for all accessed experts
        for eid in required_expert_ids:
            if eid in self.residency_map:
                self.lru_tracker.move_to_end(eid)

        return {eid: self.residency_map[eid] for eid in required_expert_ids}

    def request_layer_experts(
        self,
        layer_idx: int,
        required_expert_ids: List[int],
        locked_slots: Optional[Set[int]] = None,
    ) -> Dict[int, int]:
        """Multi-layer expert paging across all 48 layers sharing the VRAM slot pool.
        Always uses `(layer_idx, eid)` keys so single-layer and multi-layer keys never collide.
        """
        if locked_slots is None:
            locked_slots = set()

        gpu_tbl, cpu_tbl = self._ensure_layer_tables(layer_idx)
        missing = []
        for eid in required_expert_ids:
            key = (layer_idx, eid)
            if key in self.residency_map:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
                missing.append(eid)

        dirty_layers: Set[int] = set()
        locked_all = set(locked_slots)
        for eid in required_expert_ids:
            key = (layer_idx, eid)
            if key in self.residency_map:
                locked_all.add(self.residency_map[key])

        for eid in missing:
            if self.free_slots:
                target_slot = self.free_slots.pop()
            else:
                target_slot, evicted_layer = self._evict_lru_victim_multilayer(locked_all)
                if evicted_layer is not None:
                    dirty_layers.add(evicted_layer)
            locked_all.add(target_slot)

            with torch.cuda.stream(self.dma_stream):
                self._dma_expert_into_slot(layer_idx, eid, target_slot)

            key = (layer_idx, eid)
            self.residency_map[key] = target_slot
            self.slot_to_expert[target_slot] = key
            self.lru_tracker[key] = None
            cpu_tbl[eid] = target_slot
            dirty_layers.add(layer_idx)

        if dirty_layers:
            with torch.cuda.stream(self.dma_stream):
                for l_idx in dirty_layers:
                    g_t, c_t = self._ensure_layer_tables(l_idx)
                    g_t.copy_(c_t, non_blocking=True)

        for eid in required_expert_ids:
            key = (layer_idx, eid)
            if key in self.residency_map:
                self.lru_tracker.move_to_end(key)

        return {eid: self.residency_map[(layer_idx, eid)] for eid in required_expert_ids}

    def record_routing(
        self,
        expert_ids: Any,
        layer_idx: Optional[int] = None,
        weight: float = 1.0,
    ) -> None:
        """Record expert routing occurrences during prefill or decode."""
        if isinstance(expert_ids, torch.Tensor):
            expert_ids = expert_ids.detach().cpu().flatten().tolist()
        for eid in expert_ids:
            key = (layer_idx, int(eid)) if layer_idx is not None else int(eid)
            self.routing_counts[key] = self.routing_counts.get(key, 0.0) + float(weight)

    def adapt_post_prefill(
        self,
        routing_histogram: Optional[Dict[Any, float]] = None,
        max_swaps: Optional[int] = None,
        decay_factor: float = 0.95,
        threshold_ratio: float = 1.2,
    ) -> Dict[str, Any]:
        """Immediate Post-Prefill Bulk Expert Realignment (RustingStrata-inspired LPT-MoE).

        Immediately upon completing prompt prefill, inspects the cumulative gating routing
        histogram. Identifies candidate hotter experts whose prefill frequency outstrips
        currently resident cold experts, and executes an all-at-once bulk DMA transfer
        into VRAM slots before token 1 generation begins.

        Args:
            routing_histogram: Optional explicit routing counts dict. If None, uses `self.routing_counts`.
            max_swaps: Max expert slots to realign in this pass (None or 0 = unlimited / up to all slots).
            decay_factor: Factor to multiply counts by after adaptation to maintain recency bias.
            threshold_ratio: Minimum ratio (hot_count / cold_count) required to justify a swap.

        Returns:
            Dict containing `swaps_performed`, `dirty_layers`, and `elapsed_ms`.
        """
        import time
        t0 = time.perf_counter()

        counts = dict(routing_histogram) if routing_histogram is not None else dict(self.routing_counts)
        if not counts:
            return {"swaps_performed": 0, "elapsed_ms": 0.0, "dirty_layers": 0}

        # Apply decay to existing counts
        for k in self.routing_counts:
            self.routing_counts[k] *= decay_factor

        # Determine if we're operating with multi-layer keys (tuple) or single-layer keys (int)
        is_multilayer = any(isinstance(k, tuple) for k in self.residency_map.keys()) or any(isinstance(k, tuple) for k in counts.keys())

        # Collect candidate hot non-resident experts
        hot_candidates: List[Tuple[float, Any]] = []
        for key, cnt in counts.items():
            if cnt <= 0:
                continue
            if is_multilayer:
                if not isinstance(key, tuple):
                    norm_key = (self.default_layer_idx, int(key))
                else:
                    norm_key = (int(key[0]), int(key[1]))
                if norm_key not in self.residency_map:
                    hot_candidates.append((cnt, norm_key))
            else:
                eid = key[1] if isinstance(key, tuple) else int(key)
                if eid not in self.residency_map:
                    hot_candidates.append((cnt, eid))

        # Sort hot candidates descending by routing count
        hot_candidates.sort(key=lambda x: x[0], reverse=True)

        if not hot_candidates:
            return {"swaps_performed": 0, "elapsed_ms": (time.perf_counter() - t0) * 1000.0, "dirty_layers": 0}

        swaps_limit = self.num_slots if max_swaps is None or max_swaps <= 0 else min(self.num_slots, int(max_swaps))

        swaps_performed = 0
        dirty_layers: Set[int] = set()

        # Step 1: Use any remaining free slots first
        hot_idx = 0
        while self.free_slots and hot_idx < len(hot_candidates) and swaps_performed < swaps_limit:
            hot_cnt, hot_key = hot_candidates[hot_idx]
            hot_idx += 1
            slot = self.free_slots.pop()

            if is_multilayer:
                hot_layer, hot_eid = hot_key
                if self.dma_stream is not None:
                    with torch.cuda.stream(self.dma_stream):
                        self._dma_expert_into_slot(hot_layer, hot_eid, slot)
                else:
                    self._dma_expert_into_slot(hot_layer, hot_eid, slot)

                _, hot_cpu_tbl = self._ensure_layer_tables(hot_layer)
                hot_cpu_tbl[hot_eid] = slot
                dirty_layers.add(hot_layer)
                self.residency_map[hot_key] = slot
                self.slot_to_expert[slot] = hot_key
                self.lru_tracker[hot_key] = None
            else:
                hot_eid = hot_key
                if self.dma_stream is not None:
                    with torch.cuda.stream(self.dma_stream):
                        self._dma_expert_into_slot(self.default_layer_idx, hot_eid, slot)
                else:
                    self._dma_expert_into_slot(self.default_layer_idx, hot_eid, slot)

                self.cpu_slot_table[hot_eid] = slot
                self.residency_map[hot_eid] = slot
                self.slot_to_expert[slot] = hot_eid
                self.lru_tracker[hot_eid] = None

            swaps_performed += 1

        # Step 2: Swap remaining hot candidates with cold resident candidates
        if hot_idx < len(hot_candidates) and swaps_performed < swaps_limit:
            # Collect resident candidates and their routing counts
            cold_candidates: List[Tuple[float, Any, int]] = []
            for key, slot in list(self.residency_map.items()):
                cnt = counts.get(key, 0.0)
                cold_candidates.append((cnt, key, slot))

            # Sort cold candidates ascending by count (lowest first)
            cold_candidates.sort(key=lambda x: x[0])

            cold_idx = 0
            while hot_idx < len(hot_candidates) and cold_idx < len(cold_candidates) and swaps_performed < swaps_limit:
                hot_cnt, hot_key = hot_candidates[hot_idx]
                cold_cnt, cold_key, slot = cold_candidates[cold_idx]

                # Require hot_cnt >= threshold_ratio * cold_cnt + 1.0 to prevent churning equal/low-frequency experts
                if hot_cnt < (threshold_ratio * cold_cnt + 1.0):
                    break

                # Evict cold resident
                del self.residency_map[cold_key]
                if cold_key in self.lru_tracker:
                    del self.lru_tracker[cold_key]

                if is_multilayer:
                    cold_layer, cold_eid = cold_key
                    _, cold_cpu_tbl = self._ensure_layer_tables(cold_layer)
                    cold_cpu_tbl[cold_eid] = -1
                    dirty_layers.add(cold_layer)

                    hot_layer, hot_eid = hot_key
                    if self.dma_stream is not None:
                        with torch.cuda.stream(self.dma_stream):
                            self._dma_expert_into_slot(hot_layer, hot_eid, slot)
                    else:
                        self._dma_expert_into_slot(hot_layer, hot_eid, slot)

                    _, hot_cpu_tbl = self._ensure_layer_tables(hot_layer)
                    hot_cpu_tbl[hot_eid] = slot
                    dirty_layers.add(hot_layer)
                    self.residency_map[hot_key] = slot
                    self.slot_to_expert[slot] = hot_key
                    self.lru_tracker[hot_key] = None
                else:
                    cold_eid = cold_key
                    self.cpu_slot_table[cold_eid] = -1

                    hot_eid = hot_key
                    if self.dma_stream is not None:
                        with torch.cuda.stream(self.dma_stream):
                            self._dma_expert_into_slot(self.default_layer_idx, hot_eid, slot)
                    else:
                        self._dma_expert_into_slot(self.default_layer_idx, hot_eid, slot)

                    self.cpu_slot_table[hot_eid] = slot
                    self.residency_map[hot_eid] = slot
                    self.slot_to_expert[slot] = hot_eid
                    self.lru_tracker[hot_eid] = None

                swaps_performed += 1
                hot_idx += 1
                cold_idx += 1

        # Synchronize GPU indirection tables
        if swaps_performed > 0:
            if is_multilayer and dirty_layers:
                if self.dma_stream is not None:
                    with torch.cuda.stream(self.dma_stream):
                        for l_idx in dirty_layers:
                            g_tbl, c_tbl = self._ensure_layer_tables(l_idx)
                            g_tbl.copy_(c_tbl, non_blocking=True)
                        self.dma_stream.synchronize()
                else:
                    for l_idx in dirty_layers:
                        g_tbl, c_tbl = self._ensure_layer_tables(l_idx)
                        g_tbl.copy_(c_tbl)
            else:
                if self.dma_stream is not None:
                    with torch.cuda.stream(self.dma_stream):
                        self.gpu_slot_table.copy_(self.cpu_slot_table, non_blocking=True)
                        self.dma_stream.synchronize()
                else:
                    self.gpu_slot_table.copy_(self.cpu_slot_table)

        self.total_swaps += swaps_performed
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "swaps_performed": swaps_performed,
            "elapsed_ms": round(dt_ms, 2),
            "dirty_layers": len(dirty_layers),
        }

    def route_tokens(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        prior_lambda: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute router logits `hidden_states @ router_weight.T` and apply Strata's
        GPU `route_prior_top10` kernel conditioned on the layer's VRAM residency table.
        """
        x_f32 = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
        logits = torch.matmul(x_f32, router_weight.t())  # [M, 512]
        res_table = self.get_gpu_slot_table(layer_idx)
        if self.loader is not None and hasattr(self.loader, "bridge"):
            return self.loader.bridge.route_prior_top10(
                logits, res_table, prior_lambda=prior_lambda
            )
        probs = torch.softmax(logits, dim=-1)
        biased = logits + (res_table >= 0).float().unsqueeze(0) * prior_lambda
        _, top_idx = torch.topk(biased, k=min(10, logits.shape[1]), dim=-1)
        ids = top_idx.to(torch.int32)
        gathered = torch.gather(probs, -1, top_idx)
        weights = gathered / gathered.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return ids, weights

    def _evict_lru_victim(self, locked_slots: Set[int]) -> int:
        """Selects and evicts the least-recently-used expert not in locked_slots."""
        for victim_id in list(self.lru_tracker.keys()):
            slot = self.residency_map[victim_id]
            if slot not in locked_slots:
                del self.lru_tracker[victim_id]
                del self.residency_map[victim_id]
                del self.slot_to_expert[slot]
                if isinstance(victim_id, tuple):
                    v_layer, v_eid = victim_id
                    _, c_tbl = self._ensure_layer_tables(v_layer)
                    c_tbl[v_eid] = -1
                else:
                    self.cpu_slot_table[victim_id] = -1
                return slot
        raise RuntimeError("Expert Arena thrashing: All VRAM slots are currently locked!")

    def _evict_lru_victim_multilayer(
        self, locked_slots: Set[int]
    ) -> Tuple[int, Optional[int]]:
        for victim_key in list(self.lru_tracker.keys()):
            slot = self.residency_map[victim_key]
            if slot not in locked_slots:
                del self.lru_tracker[victim_key]
                del self.residency_map[victim_key]
                del self.slot_to_expert[slot]
                if isinstance(victim_key, tuple):
                    v_layer, v_eid = victim_key
                    _, c_tbl = self._ensure_layer_tables(v_layer)
                    c_tbl[v_eid] = -1
                    return slot, v_layer
                else:
                    self.cpu_slot_table[victim_key] = -1
                    return slot, self.default_layer_idx
        raise RuntimeError("Expert Arena thrashing: All VRAM slots are currently locked!")

    def synchronize(self, compute_stream: Optional[torch.cuda.Stream] = None):
        """Instructs the compute stream to wait for in-flight PCIe DMA transfers."""
        if compute_stream is None:
            compute_stream = torch.cuda.current_stream(self.device)
        compute_stream.wait_stream(self.dma_stream)
