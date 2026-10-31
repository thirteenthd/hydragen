import time
import torch
import sys
import os

sys.path.insert(0, os.path.expanduser("~/vllm"))
from vllm.v1.core.expert_manager import PagedExpertManager

def run_lookahead_test():
    print("=== Testing Lookahead Prefetching (Layer L+1 Prefetched during Layer L) ===")
    device = torch.device("cuda:0")
    compute_stream = torch.cuda.current_stream(device)

    TOTAL_EXPERTS = 64
    SLOTS = 16
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

    # Realistic layer compute: simulating Attention + MoE compute (~1.2 ms per layer)
    mat_A = torch.randn(2048, 4096, dtype=torch.float16, device=device)
    mat_B = torch.randn(4096, 4096, dtype=torch.float16, device=device)

    def simulate_full_layer_compute():
        # ~1.2 ms of Tensor Core compute
        res = torch.matmul(mat_A, mat_B)
        res = torch.matmul(res, mat_B)
        return res

    # Warmup
    for _ in range(10):
        simulate_full_layer_compute()
    torch.cuda.synchronize()

    NUM_LAYERS = 32
    ITERS = 10
    
    # Pre-generate missing expert sequence (1 missing expert per layer)
    expert_seq = [[(it * 5 + l) % TOTAL_EXPERTS] for it in range(ITERS) for l in range(NUM_LAYERS)]

    # 1. SYNCHRONOUS: Each layer fetches its own expert synchronously, then computes
    torch.cuda.synchronize()
    start_sync = time.perf_counter()
    idx = 0
    for it in range(ITERS):
        for layer in range(NUM_LAYERS):
            needed = expert_seq[idx]
            idx += 1
            # Fetch and immediately wait
            manager.request_experts(needed)
            manager.synchronize(compute_stream)
            # Then compute
            simulate_full_layer_compute()

    torch.cuda.synchronize()
    sync_time = (time.perf_counter() - start_sync) / ITERS * 1000.0

    # 2. LOOKAHEAD PREFETCH: Layer L-1 triggers transfer for Layer L
    torch.cuda.synchronize()
    start_async = time.perf_counter()
    idx = 0
    for it in range(ITERS):
        # Prefetch first layer upfront
        manager.request_experts(expert_seq[idx])
        for layer in range(NUM_LAYERS):
            # Compute stream waits for current layer (which was prefetched in previous step)
            manager.synchronize(compute_stream)
            
            # Immediately trigger next layer prefetch while current layer computes!
            if layer + 1 < NUM_LAYERS:
                manager.request_experts(expert_seq[idx + 1])
            idx += 1
            
            # Compute current layer (overlaps with next layer DMA!)
            simulate_full_layer_compute()

    torch.cuda.synchronize()
    async_time = (time.perf_counter() - start_async) / ITERS * 1000.0

    print("=" * 55)
    print(f"Synchronous Latency:         {sync_time:.2f} ms")
    print(f"Lookahead Prefetched Latency: {async_time:.2f} ms")
    print(f"Latency Saved per Step:      {sync_time - async_time:.2f} ms")
    print(f"Throughput Speedup:          {sync_time / async_time:.2f}x")
    print("=" * 55)

if __name__ == "__main__":
    run_lookahead_test()
