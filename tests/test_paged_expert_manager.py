import torch
import sys
import os

sys.path.insert(0, os.path.expanduser("~/vllm"))
from vllm.v1.core.expert_manager import PagedExpertManager

def test_paged_expert_manager():
    print("Running PagedExpertManager Unit Test...")
    device = torch.device("cuda:0")
    
    # 4 VRAM slots, 16 total experts in model
    NUM_SLOTS = 4
    TOTAL_EXPERTS = 16
    EXPERT_SHAPE = (128, 256)
    
    manager = PagedExpertManager(
        num_slots=NUM_SLOTS,
        expert_shape=EXPERT_SHAPE,
        total_experts=TOTAL_EXPERTS,
        dtype=torch.float16,
        device=device,
    )
    
    # Fill each expert in host pool with its expert_id as a distinctive value
    for eid in range(TOTAL_EXPERTS):
        manager.host_pool[eid].fill_(float(eid + 1))
        
    print("[1] Testing initial allocation and free slots...")
    assert len(manager.free_slots) == 4
    assert len(manager.residency_map) == 0
    assert torch.all(manager.gpu_slot_table == -1)
    
    print("[2] Requesting initial experts [0, 1]...")
    mapping = manager.request_experts([0, 1])
    manager.synchronize()
    
    assert len(mapping) == 2
    assert manager.is_resident(0)
    assert manager.is_resident(1)
    assert not manager.is_resident(2)
    
    # Verify values transferred correctly
    slot_0 = mapping[0]
    slot_1 = mapping[1]
    assert torch.all(manager.vram_arena[slot_0] == 1.0)
    assert torch.all(manager.vram_arena[slot_1] == 2.0)
    assert manager.gpu_slot_table[0].item() == slot_0
    assert manager.gpu_slot_table[1].item() == slot_1
    
    print("[3] Filling remaining slots with [2, 3]...")
    mapping_2 = manager.request_experts([2, 3])
    manager.synchronize()
    assert len(manager.free_slots) == 0
    assert all(manager.is_resident(i) for i in range(4))
    
    print("[4] Testing LRU Eviction...")
    # Access order so far: 0, 1, 2, 3 -> LRU is 0!
    # Touch 0 so that 1 becomes the LRU victim
    manager.get_slot(0)
    
    # Now request expert 4 (must evict expert 1!)
    mapping_evict = manager.request_experts([4])
    manager.synchronize()
    
    assert manager.is_resident(4)
    assert manager.is_resident(0)
    assert not manager.is_resident(1), "Expert 1 should have been evicted!"
    assert manager.gpu_slot_table[1].item() == -1
    
    slot_4 = mapping_evict[4]
    assert torch.all(manager.vram_arena[slot_4] == 5.0)  # eid 4 was filled with 5.0
    
    print("[5] Testing Locked Slots Protection...")
    # Current resident: 0, 2, 3, 4
    # Lock slot for expert 2, request expert 5 -> ensure 2 is NOT evicted
    slot_2 = manager.get_slot(2)
    mapping_lock = manager.request_experts([5], locked_slots={slot_2})
    manager.synchronize()
    
    assert manager.is_resident(2), "Locked expert 2 must not be evicted!"
    assert manager.is_resident(5)
    
    print(" ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_paged_expert_manager()
