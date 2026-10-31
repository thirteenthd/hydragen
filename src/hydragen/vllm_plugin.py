# SPDX-License-Identifier: Apache-2.0
"""Official vLLM General Plugin for Strata GGUF Paged MoE + Triton Execution.

Registered via `[project.entry-points."vllm.general_plugins"]` in `pyproject.toml`.
When vLLM starts up in any process (API server, EngineCore, or GPU worker),
`register_strata_vllm_plugin()` registers the `"strata_gguf"` quantization
method (`StrataGGUFQuantConfig` and `StrataPagedMoEMethod(FusedMoEMethodBase)`)
so `FusedMoE` layers execute directly through `PagedExpertManager` +
`libstrata_iq_kernels.so` + Triton `fused_moe_indexed_slot_kernel` without
patching upstream vLLM source files.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional
import torch

from hydragen.expert_manager import PagedExpertManager
from hydragen.gguf_loader import StrataGGUFMoELoader, register_strata_gguf_types
from hydragen.triton_moe import triton_paged_moe_forward


_SHARED_EXPERT_MANAGERS: Dict[str, PagedExpertManager] = {}


def get_or_create_shared_expert_manager(
    shard_path: str,
    num_slots: int = 128,
    n_ff: int = 640,
    n_embd: int = 2560,
    n_exp: int = 512,
    device: torch.device = torch.device("cuda:0"),
) -> PagedExpertManager:
    """Return a process-wide shared PagedExpertManager backed by the GGUF shard."""
    key = f"{shard_path}:{num_slots}:{device}"
    if key not in _SHARED_EXPERT_MANAGERS:
        loader = StrataGGUFMoELoader(shard_path)
        meta = loader.get_layer_expert_metadata(0)
        _SHARED_EXPERT_MANAGERS[key] = PagedExpertManager(
            num_slots=num_slots,
            expert_shape=(3, meta["n_ff"], meta["n_embd"]),
            total_experts=meta["n_exp"],
            dtype=torch.float16,
            device=device,
            loader=loader,
            default_layer_idx=0,
        )
    return _SHARED_EXPERT_MANAGERS[key]


def _extract_layer_idx(prefix: str) -> int:
    m = re.search(r"(?:layers|blk)\.(\d+)", prefix or "")
    return int(m.group(1)) if m else 0


def register_strata_vllm_plugin() -> None:
    """Entrypoint invoked automatically by `vllm.plugins.load_general_plugins()`."""
    register_strata_gguf_types()
    try:
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEConfig,
            FusedMoEQuantConfig,
        )
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )
        from vllm.model_executor.layers.quantization import (
            register_quantization_config,
        )
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizationConfig,
            QuantizeMethodBase,
        )
    except ImportError:
        # vLLM is not installed in the current Python environment
        return

    class StrataPagedMoEMethod(FusedMoEMethodBase):
        """vLLM FusedMoEMethodBase backed by Strata's PagedExpertManager + Triton."""

        def __init__(
            self,
            moe: FusedMoEConfig,
            shard_path: str,
            num_vram_slots: int = 128,
            layer_idx: int = 0,
        ):
            super().__init__(moe)
            self.shard_path = shard_path
            self.num_vram_slots = num_vram_slots
            self.layer_idx = layer_idx
            self.expert_manager: Optional[PagedExpertManager] = None

        def create_weights(
            self,
            layer: torch.nn.Module,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
        ) -> None:
            # Do NOT allocate [num_experts, 2*N, K] static GPU weights!
            # Instead bind to the shared tiered PagedExpertManager.
            self.expert_manager = get_or_create_shared_expert_manager(
                shard_path=self.shard_path,
                num_slots=self.num_vram_slots,
                n_ff=intermediate_size_per_partition,
                n_embd=hidden_size,
                n_exp=num_experts,
            )
            layer.expert_manager = self.expert_manager
            layer.strata_layer_idx = self.layer_idx

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            return

        def get_fused_moe_quant_config(
            self, layer: torch.nn.Module
        ) -> Optional[FusedMoEQuantConfig]:
            return FusedMoEQuantConfig.make()

        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts: Optional[Any] = None,
            shared_experts_input: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            assert self.expert_manager is not None
            routed_out, _ = triton_paged_moe_forward(
                hidden_states=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                expert_manager=self.expert_manager,
                layer_idx=self.layer_idx,
            )
            if shared_experts is not None:
                s_in = shared_experts_input if shared_experts_input is not None else x
                routed_out = routed_out + shared_experts(s_in)
            return routed_out

    @register_quantization_config("strata_gguf")
    class StrataGGUFQuantConfig(QuantizationConfig):
        """QuantizationConfig routing FusedMoE layers to StrataPagedMoEMethod."""

        def __init__(
            self,
            shard_path: Optional[str] = None,
            num_vram_slots: int = 128,
        ):
            super().__init__()
            self.shard_path = shard_path or os.environ.get(
                "STRATA_GGUF_SHARD",
                "/home/thirteenthd/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
            )
            self.num_vram_slots = int(
                os.environ.get("STRATA_VRAM_SLOTS", str(num_vram_slots))
            )

        def get_name(self) -> str:
            return "strata_gguf"

        def get_supported_act_dtypes(self) -> List[torch.dtype]:
            return [torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 75

        @staticmethod
        def get_config_filenames() -> List[str]:
            return []

        @classmethod
        def from_config(cls, config: Dict[str, Any]) -> "StrataGGUFQuantConfig":
            return cls(
                shard_path=config.get("shard_path"),
                num_vram_slots=int(config.get("num_vram_slots", 128)),
            )

        def get_quant_method(
            self,
            layer: torch.nn.Module,
            prefix: str,
        ) -> Optional[QuantizeMethodBase]:
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            if hasattr(layer, "moe_config"):
                layer_idx = _extract_layer_idx(prefix)
                return StrataPagedMoEMethod(
                    moe=layer.moe_config,
                    shard_path=self.shard_path,
                    num_vram_slots=self.num_vram_slots,
                    layer_idx=layer_idx,
                )
            return None
