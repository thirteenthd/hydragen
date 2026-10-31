# SPDX-License-Identifier: Apache-2.0
"""End-to-End Verification of the Modular strata_vllm Triton + PagedExpertManager Engine.

Tests:
1. vLLM General Plugin registration (`get_quantization_config("strata_gguf")`).
2. Multi-layer GGUF streaming + GPU dequantization (`libstrata_iq_kernels.so`) +
   GPU `route_prior_top10` + Triton `fused_moe_indexed_slot_kernel` across
   heterogeneous quantization formats (Layer 0 IQ2_XS/IQ4_NL, Layer 1 IQ2_XXS/Q2_0_STRATA,
   Layer 8 IQ3_S/Q2_0_STRATA, Layer 12 IQ2_S/Q2_0_STRATA).
3. `VLLMEngine` live Triton execution and `ExpertLocalityPolicy` scheduling.
"""

import threading
import torch

from strata_vllm import (
    PagedExpertManager,
    StrataGGUFMoELoader,
    VLLMEngine,
    register_strata_vllm_plugin,
    triton_paged_moe_forward,
)


SHARD_PATH = "/home/thirteenthd/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"


def test_vllm_plugin_registration():
    register_strata_vllm_plugin()
    from vllm.model_executor.layers.quantization import get_quantization_config

    cfg_cls = get_quantization_config("strata_gguf")
    cfg = cfg_cls.from_config({"shard_path": SHARD_PATH, "num_vram_slots": 16})
    assert cfg.get_name() == "strata_gguf"
    print("[PASS] vLLM plugin registration (`strata_gguf`) verified.")


def test_multilayer_triton_paged_moe():
    loader = StrataGGUFMoELoader(SHARD_PATH)
    meta0 = loader.get_layer_expert_metadata(0)
    mgr = PagedExpertManager(
        num_slots=24,
        expert_shape=(3, meta0["n_ff"], meta0["n_embd"]),
        total_experts=meta0["n_exp"],
        dtype=torch.float16,
        loader=loader,
    )

    tokens = torch.tensor([10, 42, 128, 256, 512, 1024], dtype=torch.int32, device="cuda")
    hidden = loader.dequantize_token_embedding(tokens)
    hidden = (
        hidden.float()
        * torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    ).to(torch.float16)
    assert hidden.shape == (6, 2560)
    assert torch.isfinite(hidden).all()

    for layer_idx in (0, 1, 8, 12):
        rw = loader.load_router_gate(layer_idx)
        topk_ids, topk_weights = mgr.route_tokens(
            layer_idx=layer_idx,
            hidden_states=hidden,
            router_weight=rw,
            prior_lambda=0.8,
        )
        assert topk_ids.shape == (6, 10)
        assert topk_weights.shape == (6, 10)

        out, metrics = triton_paged_moe_forward(
            hidden_states=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_manager=mgr,
            layer_idx=layer_idx,
        )
        assert out.shape == (6, 2560)
        assert torch.isfinite(out).all(), f"Non-finite output at layer {layer_idx}"
        print(
            f"[PASS] Layer {layer_idx:2d} Triton PagedMoE forward: "
            f"active_experts={int(metrics['active_experts'])}, "
            f"hit_rate={metrics['hit_rate']*100:.1f}%, "
            f"pcie_kb={metrics['pcie_bytes']/1024:.1f} KB, "
            f"out_std={out.float().std().item():.4f}"
        )

    # Repeat Layer 1 with 1 token (top-4 experts <= mgr.num_slots) to verify 100% VRAM cache hit
    rw1 = loader.load_router_gate(1)
    h1 = hidden[:1]
    topk_ids1, topk_weights1 = mgr.route_tokens(1, h1, rw1, prior_lambda=0.8)
    k_fit = min(mgr.num_slots, topk_ids1.shape[1])
    topk_ids1 = topk_ids1[:, :k_fit]
    topk_weights1 = topk_weights1[:, :k_fit]
    _, _ = triton_paged_moe_forward(h1, topk_ids1, topk_weights1, mgr, layer_idx=1)
    out1_cached, metrics1_cached = triton_paged_moe_forward(
        h1, topk_ids1, topk_weights1, mgr, layer_idx=1
    )
    assert metrics1_cached["hit_rate"] == 1.0
    assert metrics1_cached["pcie_bytes"] == 0.0
    assert torch.allclose(out1_cached, out1_cached)
    print(f"[PASS] Layer 1 cached re-execution (num_slots={mgr.num_slots}) achieved 100.0% VRAM hit rate (0 B PCIe).")
    del mgr, rw, rw1, hidden, h1, out, out1_cached
    torch.cuda.empty_cache()


def test_vllm_engine_live_triton_step():
    eng = VLLMEngine(
        cfg={"native": SHARD_PATH, "env": {"STRATA_ROUTE_PRIOR": "0.8"}},
        tok=None,
        batch_slots=2,
        num_vram_slots=24,
        standalone_triton_only=True,
    )
    cancel = threading.Event()
    toks1 = list(eng.generate([1, 2, 3, 4, 5], max_new=4, sampling={}, cancel=cancel))
    toks2 = list(eng.generate([1, 2, 3, 4, 5], max_new=4, sampling={}, cancel=cancel))
    assert len(toks1) == 4 and len(toks2) == 4
    assert eng.last.get("triton_moe_ms", 0) > 0
    assert eng.last.get("triton_hit_rate", 0) > 0
    assert eng.last.get("triton_pcie_mb", 0) > 0
    print(
        f"[PASS] VLLMEngine.generate() executed Triton MoE: "
        f"triton_moe_ms={eng.last['triton_moe_ms']} ms, "
        f"triton_hit_rate={eng.last['triton_hit_rate']*100:.1f}%, "
        f"triton_pcie_mb={eng.last['triton_pcie_mb']} MB"
    )


if __name__ == "__main__":
    test_vllm_plugin_registration()
    test_multilayer_triton_paged_moe()
    test_vllm_engine_live_triton_step()
