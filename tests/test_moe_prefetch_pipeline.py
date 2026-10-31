import time
import torch
import sys
import os

sys.path.insert(0, os.path.expanduser("~/vllm"))
from vllm.v1.core.expert_manager import PagedExpertManager

class MoEPrefetchPipeline:
    def __init__(self, manager: PagedExpertManager):
        self.manager = manager
        self.pending_layer = None

    def prefetch(self, layer_idx: int, expert_ids: list):
        self.manager.request_experts(expert_ids)
        self.pending_layer = layer_idx

    def wait(self, compute_stream=None):
        self.manager.synchronize(compute_stream)
        self.pending_layer = None


def run_pipeline_benchmark():
    print("=== Benchmarking MoE Prefetch & Double-Buffering Pipeline ===")
    device = torch.device("cuda:0")
    compute_stream = torch.cuda.current_stream(device)

    # Simulate realistic MoE dimensions
    TOTAL_EXPERTS = 64
    SLOTS = 8
    # 15 MB expert (Qwen Flash Next fine-grained expert)
    EXPERT_SHAPE = (1024, 7680)  # ~15.3 MB at float16
    
    manager = PagedExpertManager(
        num_slots=SLOTS,
        expert_shape=EXPERT_SHAPE,
        total_experts=TOTAL_EXPERTS,
        dtype=torch.float16,
        device=device,
    )
    manager.host_pool.normal_()
    torch.cuda.synchronize()

    pipeline = MoEPrefetchPipeline(manager)

    # Simulate Attention computation workload: GEMM that takes ~1.5 ms
    attn_A = torch.randn(256, 4096, dtype=torch.float16, device=device)
    attn_B = torch.randn(4096, 4096, dtype=torch.float16, device=device)

    def simulate_attention():
        # Runs matrix multiplications representing QKV + Attention + O-proj
        for _ in range(4):
            _ = torch.matmul(attn_A, attn_B)

    # Warmup
    for _ in range(5):
        simulate_attention()
    torch.cuda.synchronize()

    NUM_LAYERS = 32
    ITERS = 20

    # -------------------------------------------------------------
    # CASE 1: SYNCHRONOUS SWAPPING (No overlap)
    # (Wait for DMA to finish, THEN do attention, THEN do MoE)
    # -------------------------------------------------------------
    print("\n[1] Running Synchronous Baseline (No Overlap)...")
    torch.cuda.synchronize()
    start_sync = time.perf_counter()

    for it in range(ITERS):
        for layer in range(NUM_LAYERS):
            # 2 missing experts per layer
            needed = [(it * 3 + layer * 2) % TOTAL_EXPERTS, (it * 3 + layer * 2 + 1) % TOTAL_EXPERTS]
            
            # Synchronous transfer (DMA + immediately wait)
            manager.request_experts(needed)
            manager.synchronize(compute_stream)
            
            # Then run attention
            simulate_attention()

    torch.cuda.synchronize()
    sync_time = time.perf_counter() - start_sync
    sync_ms_per_step = (sync_time / ITERS) * 1000.0

    # -------------------------------------------------------------
    # CASE 2: ASYNCHRONOUS OVERLAPPED PREFETCH PIPELINE
    # (Trigger DMA on dma_stream -> Run attention concurrently -> Wait for DMA)
    # -------------------------------------------------------------
    print("[2] Running Overlapped Prefetch Pipeline...")
    torch.cuda.synchronize()
    start_async = time.perf_counter()

    for it in range(ITERS):
        for layer in range(NUM_LAYERS):
            needed = [(it * 3 + layer * 2) % TOTAL_EXPERTS, (it * 3 + layer * 2 + 1) % TOTAL_EXPERTS]
            
            # Start prefetching missing experts on background DMA stream
            pipeline.prefetch(layer, needed)
            
            # GPU executes attention on compute stream WHILE PCIe is transferring weights!
            simulate_attention()
            
            # Synchronize right before MoE
            pipeline.wait(compute_stream)

    torch.cuda.synchronize()
    async_time = time.perf_counter() - start_async
    async_ms_per_step = (async_time / ITERS) * 1000.0

    # -------------------------------------------------------------
    # RESULTS
    # -------------------------------------------------------------
    print("\n" + "=" * 55)
    print(f"Synchronous Step Time:       {sync_ms_per_step:.2f} ms")
    print(f"Overlapped Prefetch Time:    {async_ms_per_step:.2f} ms")
    savings_ms = sync_ms_per_step - async_ms_per_step
    speedup = sync_ms_per_step / async_ms_per_step
    print(f"PCIe Latency Hidden:         {savings_ms:.2f} ms per iteration")
    print(f"Pipeline Speedup:            {speedup:.2f}x")
    print("=" * 55)

if __name__ == "__main__":
    run_pipeline_benchmark()
