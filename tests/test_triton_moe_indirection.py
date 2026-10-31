import torch
import triton
import triton.language as tl
import sys
import os

sys.path.insert(0, os.path.expanduser("~/vllm"))
from vllm.v1.core.expert_manager import PagedExpertManager

@triton.jit
def paged_moe_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    slot_mapping_ptr,
    num_tokens_post_padded_ptr,
    # Dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    USE_SLOT_INDIRECTION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        return

    # --- MILESTONE 2 SLOT INDIRECTION ---
    if USE_SLOT_INDIRECTION:
        slot_idx = tl.load(slot_mapping_ptr + off_experts).to(tl.int64)
        if slot_idx == -1:
            return
        weight_expert_idx = slot_idx
    else:
        weight_expert_idx = off_experts
    # ------------------------------------

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + weight_expert_idx * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply routing weight
    topk_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
    accumulator = accumulator * topk_weight[:, None]

    # Write back to output C
    c_ptrs = c_ptr + (offs_token[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=token_mask[:, None])


def run_test():
    print("=== Testing Triton Fused MoE with Slot Indirection ===")
    device = torch.device("cuda:0")
    
    # Model dimensions (BLOCK_SIZE_M must be >= 16 for Tensor Cores)
    TOTAL_EXPERTS = 16
    VRAM_SLOTS = 4
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    NUM_TOKENS = 16
    HIDDEN_K = 256
    INTERMEDIATE_N = 512
    TOP_K = 2

    torch.manual_seed(42)
    
    # 1. Input tokens A: [NUM_TOKENS, HIDDEN_K]
    A = torch.randn((NUM_TOKENS, HIDDEN_K), dtype=torch.float16, device=device)
    
    # Active experts: 4 out of 16
    active_expert_pool = [2, 5, 8, 11]
    
    # 2 blocks of 16 tokens: Block 0 uses expert 2, Block 1 uses expert 5
    M = 32  # 2 blocks of 16
    sorted_token_ids = torch.arange(M, dtype=torch.int32, device=device)
    # block 0 -> expert 2, block 1 -> expert 5
    expert_ids = torch.tensor([2, 5], dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.tensor([M], dtype=torch.int32, device=device)
    routing_weights = torch.ones(M, dtype=torch.float32, device=device)

    # 3. Create full weights B_full [TOTAL_EXPERTS, HIDDEN_K, INTERMEDIATE_N]
    B_full = torch.randn((TOTAL_EXPERTS, HIDDEN_K, INTERMEDIATE_N), dtype=torch.float16, device=device)

    # Output buffer for Baseline
    C_baseline = torch.zeros((M, INTERMEDIATE_N), dtype=torch.float16, device=device)

    grid = (2 * triton.cdiv(INTERMEDIATE_N, BLOCK_SIZE_N),)

    # RUN BASELINE (No indirection, full 16 experts in VRAM)
    dummy_slot_table = torch.empty(1, dtype=torch.int32, device=device)
    paged_moe_kernel[grid](
        A, B_full, C_baseline,
        routing_weights, sorted_token_ids, expert_ids, dummy_slot_table, num_tokens_post_padded,
        INTERMEDIATE_N, HIDDEN_K, M, M,
        A.stride(0), A.stride(1),
        B_full.stride(0), B_full.stride(1), B_full.stride(2),
        C_baseline.stride(0), C_baseline.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=1, top_k=TOP_K,
        USE_SLOT_INDIRECTION=False,
    )
    torch.cuda.synchronize()
    print("[1] Baseline execution (16 experts in VRAM) completed.")

    # NOW RUN WITH PAGED EXPERT MANAGER (Only 4 slots in VRAM)
    manager = PagedExpertManager(
        num_slots=VRAM_SLOTS,
        expert_shape=(HIDDEN_K, INTERMEDIATE_N),
        total_experts=TOTAL_EXPERTS,
        dtype=torch.float16,
        device=device,
    )
    # Populate host pool with identical weights from B_full
    manager.host_pool.copy_(B_full.cpu())
    
    # Request the 4 active experts needed for this batch
    mapping = manager.request_experts(active_expert_pool)
    manager.synchronize()
    print(f"[2] PagedExpertManager loaded active experts {active_expert_pool} into slots: {mapping}")

    # Output buffer for Paged MoE
    C_paged = torch.zeros((M, INTERMEDIATE_N), dtype=torch.float16, device=device)

    # Run Paged MoE (passing manager.vram_arena [4, K, N] and manager.gpu_slot_table [16])
    paged_moe_kernel[grid](
        A, manager.vram_arena, C_paged,
        routing_weights, sorted_token_ids, expert_ids, manager.gpu_slot_table, num_tokens_post_padded,
        INTERMEDIATE_N, HIDDEN_K, M, M,
        A.stride(0), A.stride(1),
        manager.vram_arena.stride(0), manager.vram_arena.stride(1), manager.vram_arena.stride(2),
        C_paged.stride(0), C_paged.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=1, top_k=TOP_K,
        USE_SLOT_INDIRECTION=True,
    )
    torch.cuda.synchronize()
    print("[3] Paged execution (only 4 slots in VRAM) completed.")

    # 4. Compare outputs
    max_diff = torch.max(torch.abs(C_baseline - C_paged)).item()
    print(f"Max absolute difference between Baseline and Paged MoE: {max_diff:.6f}")
    assert torch.allclose(C_baseline, C_paged, atol=1e-3, rtol=1e-3), "Outputs do not match!"
    print(" SUCCESS: Paged MoE with Slot Indirection matches Baseline EXACTLY!")

if __name__ == "__main__":
    run_test()
