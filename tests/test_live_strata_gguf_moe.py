import time
import ctypes
import numpy as np
import torch
import triton
import triton.language as tl
import gguf
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES

# 1. Custom GGUF Quantization Types
custom_types = {
    42: ("Q2_0", 64, 18),
    40: ("NVFP4", 64, 36),
    39: ("MXFP4", 32, 17),
    143: ("PTQ1_0", 128, 28),
    144: ("TQ2_T", 128, 34),
    145: ("TQK6", 128, 26),
    146: ("TQK7", 128, 30),
    147: ("TQP5", 128, 22),
    148: ("TQP6", 128, 26),
    149: ("TQP7", 128, 30),
}

class MockType:
    def __init__(self, name, val):
        self.name = name
        self.value = val
    def __str__(self):
        return self.name
    def __repr__(self):
        return f"<GGMLQuantizationType.{self.name}: {self.value}>"

for val, (name, elems, bytes_per_blk) in custom_types.items():
    m = MockType(name, val)
    GGMLQuantizationType._value2member_map_[val] = m
    if not hasattr(GGMLQuantizationType, name):
        setattr(GGMLQuantizationType, name, m)
    GGML_QUANT_SIZES[m] = (elems, bytes_per_blk)
    GGML_QUANT_SIZES[val] = (elems, bytes_per_blk)

# 2. Bind C dequantizer
lib = ctypes.CDLL("/home/thirteenthd/libstrata_iq_kernels.so")
dequant_f16 = lib._ZN6strata7kernels14iq_dequant_f16EiPKvlPtPvPKf
dequant_f16.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
dequant_f16.restype = None

# 3. Triton Fused MoE with Slot Indirection
@triton.jit
def fused_moe_slot_kernel(
    a_ptr, b_ptr, c_ptr,
    slot_mapping_ptr,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_experts = tl.program_id(2)

    slot_idx = tl.load(slot_mapping_ptr + off_experts)
    if slot_idx < 0:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (
        b_ptr
        + slot_idx * stride_be
        + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    c_ptrs = (
        c_ptr
        + off_experts * (M * N)
        + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    )
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# 4. Main Benchmark
def run_live_gguf_moe():
    print("=" * 70)
    print("Option A: Live GGUF Streaming MoE with On-The-Fly GPU Dequantization")
    print("=" * 70)

    path1 = '/home/thirteenthd/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf'
    print("Opening model shard 1...")
    t0 = time.perf_counter()
    reader = gguf.GGUFReader(path1)
    print(f"GGUF opened in {(time.perf_counter() - t0)*1000:.1f} ms")

    tensor_map = {t.name: t for t in reader.tensors}
    gate_t = tensor_map["blk.0.ffn_gate_exps.weight"]
    down_t = tensor_map["blk.0.ffn_down_exps.weight"]

    n_embd = int(gate_t.shape[0])  # 2560
    n_ff = int(gate_t.shape[1])    # 640
    n_exp = int(gate_t.shape[2])   # 512
    n_elems = int(n_embd * n_ff)   # 1,638,400

    bytes_gate_exp = int((n_elems // 256) * 74)  # IQ2_XS: 462.5 KB
    bytes_down_exp = int((n_elems // 32) * 18)   # IQ4_NL: 900.0 KB

    print(f"Model dimensions: n_embd={n_embd}, n_ff={n_ff}, n_experts={n_exp}")
    print(f"Raw weight size per expert: {(bytes_gate_exp + bytes_down_exp)/1024:.1f} KB")

    # Allocate VRAM Slot Arena
    num_slots = 16
    print(f"\nAllocating VRAM Slot Arena for {num_slots} resident experts...")
    slot_weights_gate = torch.zeros((num_slots, n_embd, n_ff), dtype=torch.float16, device="cuda")
    slot_weights_down = torch.zeros((num_slots, n_ff, n_embd), dtype=torch.float16, device="cuda")

    # GPU Slot Table: mapping [total_experts] -> slot_idx (-1 = not in VRAM)
    gpu_slot_table = torch.full((n_exp,), -1, dtype=torch.int32, device="cuda")

    # Staging buffers for DMA
    d_raw_gate = torch.empty(bytes_gate_exp, dtype=torch.uint8, device="cuda")
    d_raw_down = torch.empty(bytes_down_exp, dtype=torch.uint8, device="cuda")
    dma_stream = torch.cuda.Stream()

    # Pre-populate first 16 slots
    print("Populating initial 16 VRAM slots with real weights from GGUF...")
    t_pop0 = time.perf_counter()
    for s in range(num_slots):
        g_raw = np.frombuffer(gate_t.data, dtype=np.uint8, count=bytes_gate_exp, offset=s * bytes_gate_exp)
        d_raw = np.frombuffer(down_t.data, dtype=np.uint8, count=bytes_down_exp, offset=s * bytes_down_exp)
        
        d_raw_gate.copy_(torch.from_numpy(g_raw), non_blocking=False)
        d_raw_down.copy_(torch.from_numpy(d_raw), non_blocking=False)

        dequant_f16(17, d_raw_gate.data_ptr(), n_elems, slot_weights_gate[s].data_ptr(), dma_stream.cuda_stream, None)
        dequant_f16(20, d_raw_down.data_ptr(), n_elems, slot_weights_down[s].data_ptr(), dma_stream.cuda_stream, None)
        gpu_slot_table[s] = s

    torch.cuda.synchronize()
    print(f"Initial 16 slots populated in {(time.perf_counter() - t_pop0)*1000:.2f} ms!")

    # Simulate 50 token steps across 4 concurrent streams
    batch_size = 4
    top_k = 8
    print(f"\nSimulating {batch_size} concurrent streams, top-{top_k} routing for 50 steps...")

    # Activation input tensor: [batch_size, n_embd]
    x = torch.randn((batch_size, n_embd), dtype=torch.float16, device="cuda")
    out = torch.empty((n_exp, batch_size, n_ff), dtype=torch.float16, device="cuda")

    hits = 0
    misses = 0
    swaps = 0
    total_swapped_ms = 0.0

    step_times = []
    
    # Next LRU slot to evict
    next_evict_slot = 0
    slot_to_expert = list(range(num_slots))

    for step in range(50):
        t_step0 = time.perf_counter()

        # Simulate router selection: 8 experts per token with 85% locality
        active_experts = set()
        for b in range(batch_size):
            for k in range(top_k):
                if np.random.rand() < 0.85:
                    exp_id = int(np.random.choice(slot_to_expert))
                else:
                    exp_id = int(np.random.randint(0, n_exp))
                active_experts.add(exp_id)

        # Check hits vs misses
        for exp_id in active_experts:
            slot_id = int(gpu_slot_table[exp_id].item())
            if slot_id >= 0:
                hits += 1
            else:
                misses += 1
                # On-the-fly hot swap from GGUF into next slot!
                evicted_exp = slot_to_expert[next_evict_slot]
                gpu_slot_table[evicted_exp] = -1

                t_swap0 = time.perf_counter()
                g_raw = np.frombuffer(gate_t.data, dtype=np.uint8, count=bytes_gate_exp, offset=exp_id * bytes_gate_exp)
                d_raw_gate.copy_(torch.from_numpy(g_raw), non_blocking=False)
                dequant_f16(17, d_raw_gate.data_ptr(), n_elems, slot_weights_gate[next_evict_slot].data_ptr(), dma_stream.cuda_stream, None)

                gpu_slot_table[exp_id] = next_evict_slot
                slot_to_expert[next_evict_slot] = exp_id
                next_evict_slot = (next_evict_slot + 1) % num_slots
                swaps += 1
                total_swapped_ms += (time.perf_counter() - t_swap0) * 1000

        # Run Triton Fused MoE kernel with slot indirection
        grid = (
            int(triton.cdiv(batch_size, 16)),
            int(triton.cdiv(n_ff, 64)),
            int(n_exp),
        )
        fused_moe_slot_kernel[grid](
            x, slot_weights_gate, out,
            gpu_slot_table,
            int(x.stride(0)), int(x.stride(1)),
            int(slot_weights_gate.stride(0)), int(slot_weights_gate.stride(1)), int(slot_weights_gate.stride(2)),
            int(out.stride(1)), int(out.stride(2)),
            int(batch_size), int(n_ff), int(n_embd),
            BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        )
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t_step0) * 1000)

    avg_step_ms = np.mean(step_times)
    tok_per_sec = (batch_size * 50) / (sum(step_times) / 1000)
    hit_rate = hits / (hits + misses) * 100

    print("\n" + "=" * 70)
    print("LIVE STREAMING BENCHMARK RESULTS (Option A - Real Strata Weights)")
    print("=" * 70)
    print(f"Total Steps:              50 steps")
    print(f"Concurrent Streams (N):   {batch_size}")
    print(f"Total Tokens Generated:   {batch_size * 50}")
    print(f"Cache Hit Rate:           {hit_rate:.1f}% ({hits} hits, {misses} misses)")
    print(f"Dynamic PCIe Swaps:       {swaps} swaps (avg swap+dq: {total_swapped_ms/max(1,swaps):.3f} ms)")
    print(f"Average Step Latency:     {avg_step_ms:.2f} ms")
    print(f"Aggregate Throughput:     {tok_per_sec:.1f} tokens/second")
    print(f"Per-User Throughput:      {tok_per_sec / batch_size:.1f} tokens/second")
    print("=" * 70)

if __name__ == "__main__":
    run_live_gguf_moe()
