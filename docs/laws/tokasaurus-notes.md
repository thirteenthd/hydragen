# Tokasaurus (Stanford ScalingIntelligence blog, 2025)

Throughput-first batch inference engine (NOT agentic token-state caching).
- Small models: async+adaptive CPU "manager" that skips optional steps when the GPU input queue runs dry + *dynamic* Hydragen grouping: greedy DFS longest-shared-prefix detection before every forward.
- Big models: PP for non-NVLink nodes (3x vs vLLM/SGLang PP on 8xL40S); Async-TP via torch.compile on NVLink (only pays at 6k+ token batches).
- Pure Python + FlashInfer ops; Llama-3/Qwen-2 families only; no MoE expert offload, no hybrid linear attention, no PLE.

Relevance to Phase 2:
- They authored the Hydragen paper = the exact scheme Gate-1 kv1.cu proved (prefix-grouped attention + LSE recombine). Their DFS prefix-discovery + per-forward regrouping is the mining target for SGLang grouping heuristics.
- Confirms Hydragen win profile: attention-heavy small models / prefix-shared batches (batch sampling, shared system prompts). qwen4exp is 75% linear-attention, so prefix grouping pays only on the 12 full-attn layers + KV row sharing.
- "CPU overhead stalls the GPU" matches our pinned-tier/PCIe transport results; async manager ≈ tr10 async staging intuition.
- Not a serving target for qwen4exp; SGLang stays the base. Their up-to-2x on Large-Language-Monkeys prefix-shared sampling is our concurrency workload shape.
