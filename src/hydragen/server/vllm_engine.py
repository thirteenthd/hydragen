# SPDX-License-Identifier: Apache-2.0
"""serve/vllm_engine.py - Hybrid vLLM Paged MoE + Triton Engine for Strata Server.

Re-exports and executes the full `strata_vllm` pipeline:
- Triton Fused MoE kernels (`fused_moe_indexed_slot_kernel`, `fused_moe_slot_kernel`)
- `PagedExpertManager` with async PCIe DMA + on-GPU `libstrata_iq_kernels.so` dequantization
- GPU cache-conditioned top-10 routing (`route_prior_top10`, `STRATA_ROUTE_PRIOR=0.8`)
- `ExpertLocalityPolicy` cache-affinity request scheduling
"""

from hydragen.engine import VLLMEngine
from hydragen.expert_locality_policy import ExpertLocalityPolicy
from hydragen.expert_manager import PagedExpertManager
from hydragen.gguf_loader import (
    StrataCUDAKernelBridge,
    StrataGGUFMoELoader,
    register_strata_gguf_types,
)
from hydragen.triton_moe import (
    fused_moe_indexed_slot_kernel,
    fused_moe_slot_kernel,
    triton_paged_moe_forward,
)

__all__ = [
    "ExpertLocalityPolicy",
    "PagedExpertManager",
    "StrataCUDAKernelBridge",
    "StrataGGUFMoELoader",
    "VLLMEngine",
    "fused_moe_indexed_slot_kernel",
    "fused_moe_slot_kernel",
    "register_strata_gguf_types",
    "triton_paged_moe_forward",
]
