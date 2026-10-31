# SPDX-License-Identifier: Apache-2.0
"""strata_vllm: Modular Hybrid vLLM + Triton Paged MoE Engine for Strata GGUF Models."""

from strata_vllm.cpu_fallback import CPUExpertFallbackExecutor
from strata_vllm.engine import VLLMEngine
from strata_vllm.expert_locality_policy import ExpertLocalityPolicy
from strata_vllm.expert_manager import PagedExpertManager
from strata_vllm.gguf_loader import (
    StrataCUDAKernelBridge,
    StrataGGUFMoELoader,
    register_strata_gguf_types,
)
from strata_vllm.triton_moe import (
    fused_moe_gguf_down_slot_kernel,
    fused_moe_indexed_slot_kernel,
    fused_moe_slot_kernel,
    native_gguf_paged_moe_forward,
    triton_gguf_down_slot_forward,
    triton_paged_moe_forward,
)
from strata_vllm.vllm_plugin import register_strata_vllm_plugin

__version__ = "0.35.0"

__all__ = [
    "CPUExpertFallbackExecutor",
    "ExpertLocalityPolicy",
    "PagedExpertManager",
    "StrataCUDAKernelBridge",
    "StrataGGUFMoELoader",
    "VLLMEngine",
    "fused_moe_gguf_down_slot_kernel",
    "fused_moe_indexed_slot_kernel",
    "fused_moe_slot_kernel",
    "native_gguf_paged_moe_forward",
    "register_strata_gguf_types",
    "register_strata_vllm_plugin",
    "triton_gguf_down_slot_forward",
    "triton_paged_moe_forward",
]
