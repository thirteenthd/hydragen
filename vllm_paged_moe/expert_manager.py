# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project.

"""PagedExpertManager: Tiered memory management and asynchronous expert hot-swapping for MoE models.

Manages an L1 VRAM slot pool and an L2 pinned host-memory pool, enabling models
with hundreds of sparse experts (e.g. Qwen Flash Next) to run efficiently on 
consumer GPUs (like the RTX 3080 Ti 12GB).
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple
import torch


class PagedExpertManager:
    """Manages dynamic allocation, slot indirection, and async PCIe DMA transfers
    for Mixture-of-Experts weights.
    """

    def __init__(
        self,
        num_slots: int,
        expert_shape: Tuple[int, ...],
        total_experts: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.num_slots = num_slots
        self.expert_shape = expert_shape
        self.total_experts = total_experts
        self.dtype = dtype
        self.device = device

        # 1. Allocate L1 GPU VRAM Expert Arena [num_slots, *expert_shape]
        self.vram_arena = torch.empty(
            (num_slots, *expert_shape),
            dtype=dtype,
            device=device,
        )

        # 2. Allocate L2 Pinned Host RAM Pool [total_experts, *expert_shape]
        self.host_pool = torch.empty(
            (total_experts, *expert_shape),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

        # 3. GPU-resident Indirection Table for Triton / CUTLASS kernels
        # slot_table[expert_id] == slot_idx (or -1 if not resident)
        self.gpu_slot_table = torch.full(
            (total_experts,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.cpu_slot_table = torch.full(
            (total_experts,),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        # 4. Dedicated CUDA stream for non-blocking PCIe DMA transfers
        self.dma_stream = torch.cuda.Stream(device=device)

        # 5. Slot tracking and LRU eviction state
        self.residency_map: Dict[int, int] = {}       # expert_id -> slot_idx
        self.slot_to_expert: Dict[int, int] = {}       # slot_idx -> expert_id
        self.free_slots: List[int] = list(range(num_slots))
        self.lru_tracker: OrderedDict = OrderedDict()

    def is_resident(self, expert_id: int) -> bool:
        """Check if an expert is currently resident in VRAM."""
        return expert_id in self.residency_map

    def get_slot(self, expert_id: int) -> int:
        """Retrieve slot for resident expert and refresh its LRU timestamp."""
        slot = self.residency_map[expert_id]
        self.lru_tracker.move_to_end(expert_id)
        return slot

    def request_experts(
        self,
        required_expert_ids: List[int],
        locked_slots: Optional[Set[int]] = None,
    ) -> Dict[int, int]:
        """Ensures all required_expert_ids are resident in VRAM.
        Issues asynchronous DMA transfers for missing experts on the DMA stream.
        Updates the GPU slot indirection table.
        """
        if locked_slots is None:
            locked_slots = set()

        missing = [eid for eid in required_expert_ids if eid not in self.residency_map]
        table_updated = False

        for expert_id in missing:
            if self.free_slots:
                target_slot = self.free_slots.pop()
            else:
                target_slot = self._evict_lru_victim(locked_slots)

            # Non-blocking host-to-device DMA copy
            with torch.cuda.stream(self.dma_stream):
                self.vram_arena[target_slot].copy_(
                    self.host_pool[expert_id],
                    non_blocking=True,
                )

            # Update CPU mapping state
            self.residency_map[expert_id] = target_slot
            self.slot_to_expert[target_slot] = expert_id
            self.lru_tracker[expert_id] = None
            self.cpu_slot_table[expert_id] = target_slot
            table_updated = True

        if table_updated:
            # Sync the indirection table to GPU memory asynchronously
            with torch.cuda.stream(self.dma_stream):
                self.gpu_slot_table.copy_(self.cpu_slot_table, non_blocking=True)

        # Refresh LRU ordering for all accessed experts
        for eid in required_expert_ids:
            if eid in self.residency_map:
                self.lru_tracker.move_to_end(eid)

        return {eid: self.residency_map[eid] for eid in required_expert_ids}

    def _evict_lru_victim(self, locked_slots: Set[int]) -> int:
        """Selects and evicts the least-recently-used expert not in locked_slots."""
        for victim_id in self.lru_tracker:
            slot = self.residency_map[victim_id]
            if slot not in locked_slots:
                del self.lru_tracker[victim_id]
                del self.residency_map[victim_id]
                del self.slot_to_expert[slot]
                self.cpu_slot_table[victim_id] = -1
                return slot
        raise RuntimeError("Expert Arena thrashing: All VRAM slots are currently locked!")

    def synchronize(self, compute_stream: Optional[torch.cuda.Stream] = None):
        """Instructs the compute stream to wait for in-flight PCIe DMA transfers."""
        if compute_stream is None:
            compute_stream = torch.cuda.current_stream(self.device)
        compute_stream.wait_stream(self.dma_stream)
