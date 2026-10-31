import time
import torch
import sys
import os

sys.path.insert(0, os.path.expanduser("~/vllm"))
from vllm.v1.core.expert_manager import PagedExpertManager

def run_realistic_simulation():
    print("=== Realistic MoE Step Benchmark (90% Cache Hit Rate, Full VRAM Memory Read) ===")
    device = torch.device("cuda:0")
    compute_stream = torch.cuda.current_stream(device)

    # 3080 Ti VRAM setup:
    TOTAL_EXPERTS = 64
    SLOTS = 24  # Holds 24 hot experts in VRAM
    EXPERT_SHAPE = (1024, 7680)  # 15.3 MB
    
    manager = PagedExpertManager(
        num_slots=SLOTS,
        expert_shape=EXPERT_SHAPE,
        total_experts=TOTAL_EXPERTS,
        dtype=torch.float16,
        device=device,
    )
    manager.host_pool.normal_()
    torch.cuda.synchronize()

    # Pre-fill VRAM arena with initial hot experts 0..23
    manager.request_experts(list(range(24)))
    manager.synchronize()

    # Simulate realistic decode forward pass computation across 32 layers for batch=4:
    # 32 layers of GEMV projections: A [4, 4096] @ B [4096, 4096]
    decode_A = torch.randn(4, 4096, dtype=torch.float16, device=device)
    decode_B = torch.randn(4096, 4096, dtype=torch.float16, device=device)

    def simulate_layer_compute():
        # ~0.25 ms per layer -> ~8 ms across 32 layers
        for _ in range(8):
            _ = torch.matmul(decode_A, decode_B)

    # Warmup
    for _ in range(5):
        simulate_layer_compute()
    torch.cuda.synchronize()

    ITERS = 20

    # In realistic steady state (90% hit rate):
    # Missing experts are bounded within 0..63
    missing_experts_per_iter = [
        [(24 + it * 3) % 64, (25 + it * 3) % 64, (26 + it * 3) % 64]
        for it in range(ITERS)
    ]

    # CASE 1: Synchronous DMA (Wait for PCIe before doing decode pass)
    torch.cuda.synchronize()
    start_sync = time.perf_counter()
    for it in range(ITERS):
        needed = missing_experts_per_iter[it]
        # Synchronous: trigger transfer, wait immediately, then run 32 layers of compute
        manager.request_experts(needed)
        manager.synchronize(compute_stream)
        for _ in range(32):
            simulate_layer_compute()

    torch.cuda.synchronize()
    sync_ms = (time.perf_counter() - start_sync) / ITERS * 1000.0

    # Reset cache to initial 0..23
    manager.request_experts(list(range(24)))
    manager.synchronize()

    # CASE 2: Asynchronous Overlapped DMA (Start PCIe transfer, run compute concurrently)
    torch.cuda.synchronize()
    start_async = time.perf_counter()
    for it in range(ITERS):
        needed = missing_experts_per_iter[it]
        # Start async PCIe DMA on background dma_stream
        manager.request_experts(needed)
        # Immediately run compute on compute_stream (PCIe DMA runs in parallel!)
        for _ in range(32):
            simulate_layer_compute()
        # Synchronize at the end of the pass
        manager.synchronize(compute_stream)

    torch.cuda.synchronize()
    async_ms = (time.perf_counter() - start_async) / ITERS * 1000.0

    print("=" * 60)
    print(f"Synchronous Step Latency:          {sync_ms:.3f} ms")
    print(f"Overlapped Asynchronous Latency:   {async_ms:.3f} ms")
    pcie_saved = sync_ms - async_ms
    print(f"PCIe Transfer Latency Hidden:      {pcie_saved:.3f} ms")
    print(f"Throughput Improvement:            {(sync_ms / async_ms - 1.0) * 100:.1f}%")
    print(f"Token Generation Frequency:        {1000.0 / async_ms:.1f} iterations/sec")
    print("=" * 60)

if __name__ == "__main__":
    run_realistic_simulation()
