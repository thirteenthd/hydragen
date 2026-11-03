# SPDX-License-Identifier: Apache-2.0
"""hydragen: High-Concurrency Tiered PagedMoE Engine combining vLLM batching with Triton expert paging."""

__version__ = "0.35.0"

# Lazy export map (PEP 562) to ensure lightweight imports (CLI, server shims, core block managers)
# work instantly across Windows and Linux even before heavyweight frameworks (torch, triton) are loaded.
_EXPORT_MAP = {
    "CPUExpertFallbackExecutor": "hydragen.cpu_fallback",
    "HybridExpertMailbox": "hydragen.expert_mailbox",
    "VLLMEngine": "hydragen.engine",
    "ExpertLocalityPolicy": "hydragen.expert_locality_policy",
    "PagedExpertManager": "hydragen.expert_manager",
    "StrataCUDAKernelBridge": "hydragen.gguf_loader",
    "StrataGGUFMoELoader": "hydragen.gguf_loader",
    "register_strata_gguf_types": "hydragen.gguf_loader",
    "fused_moe_gguf_down_slot_kernel": "hydragen.triton_moe",
    "fused_moe_indexed_slot_kernel": "hydragen.triton_moe",
    "fused_moe_slot_kernel": "hydragen.triton_moe",
    "native_gguf_paged_moe_forward": "hydragen.triton_moe",
    "triton_gguf_down_slot_forward": "hydragen.triton_moe",
    "triton_paged_moe_forward": "hydragen.triton_moe",
    "register_strata_vllm_plugin": "hydragen.vllm_plugin",
    "TRTLLMEngine": "hydragen.trtllm_engine",
    "TRTPagedBlockManager": "hydragen.core.block_manager",
    "InFlightScheduler": "hydragen.core.scheduler",
}

__all__ = list(_EXPORT_MAP.keys()) + ["__version__"]


def __getattr__(name: str):
    if name in _EXPORT_MAP:
        import importlib
        mod = importlib.import_module(_EXPORT_MAP[name])
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

