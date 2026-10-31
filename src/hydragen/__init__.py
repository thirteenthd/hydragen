# SPDX-License-Identifier: Apache-2.0
"""hydragen: High-Concurrency Tiered PagedMoE Engine combining vLLM batching with Triton expert paging."""

from hydragen.cpu_fallback import CPUExpertFallbackExecutor
from hydragen.engine import VLLMEngine
from hydragen.expert_locality_policy import ExpertLocalityPolicy
from hydragen.expert_manager import PagedExpertManager
from hydragen.gguf_loader import (
    StrataCUDAKernelBridge,
    StrataGGUFMoELoader,
    register_strata_gguf_types,
)
from hydragen.triton_moe import (
    fused_moe_gguf_down_slot_kernel,
    fused_moe_indexed_slot_kernel,
    fused_moe_slot_kernel,
    native_gguf_paged_moe_forward,
    triton_gguf_down_slot_forward,
    triton_paged_moe_forward,
)
from hydragen.vllm_plugin import register_strata_vllm_plugin

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
