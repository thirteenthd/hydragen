import unittest
import torch
from hydragen.expert_manager import PagedExpertManager


class TestPostPrefillAdapt(unittest.TestCase):
    def setUp(self):
        use_cuda = False
        if torch.cuda.is_available():
            try:
                free_b, _ = torch.cuda.mem_get_info(0)
                if free_b >= 350 * 1024 * 1024:
                    use_cuda = True
            except Exception:
                pass
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

    def test_adapt_fills_free_slots_first(self):
        """Verify that adapt_post_prefill leverages free VRAM slots for hot experts."""
        mgr = PagedExpertManager(
            num_slots=8,
            expert_shape=(8, 8),
            total_experts=32,
            device=self.device,
            compressed_cache=False,
        )

        # Simulate prefill routing: experts 10, 11, 12, 13 receive frequent routing
        mgr.record_routing([10, 11, 12, 13], weight=15.0)

        # Run post-prefill adaptation
        res = mgr.adapt_post_prefill()
        self.assertEqual(res["swaps_performed"], 4)

        # All 4 hot experts should now be resident
        for eid in [10, 11, 12, 13]:
            self.assertIn(eid, mgr.residency_map)
            slot = mgr.residency_map[eid]
            self.assertEqual(mgr.cpu_slot_table[eid].item(), slot)
            if self.device.type == "cuda":
                self.assertEqual(mgr.gpu_slot_table[eid].item(), slot)

        # Requesting these experts should now achieve 100% cache hits
        hits_before = mgr.cache_hits
        misses_before = mgr.cache_misses
        mgr.request_experts([10, 11, 12, 13])
        self.assertEqual(mgr.cache_hits - hits_before, 4)
        self.assertEqual(mgr.cache_misses - misses_before, 0)

    def test_adapt_evicts_cold_residents_for_hot_candidates(self):
        """Verify that when slots are full, coldest residents are evicted for hot candidates."""
        mgr = PagedExpertManager(
            num_slots=4,
            expert_shape=(8, 8),
            total_experts=24,
            device=self.device,
            compressed_cache=False,
        )

        # Pre-fill all 4 slots with cold experts [0, 1, 2, 3]
        mgr.request_experts([0, 1, 2, 3])
        self.assertEqual(len(mgr.residency_map), 4)
        self.assertEqual(len(mgr.free_slots), 0)

        # Cold experts have low or zero counts
        mgr.record_routing([0], weight=1.0)
        mgr.record_routing([1], weight=2.0)
        # Experts 2 and 3 have 0 routing counts

        # Non-resident candidates 20 and 21 have high prefill routing
        mgr.record_routing([20, 21], weight=40.0)

        res = mgr.adapt_post_prefill(threshold_ratio=1.2)
        self.assertGreaterEqual(res["swaps_performed"], 2)

        # 20 and 21 must be resident now
        self.assertIn(20, mgr.residency_map)
        self.assertIn(21, mgr.residency_map)
        self.assertGreaterEqual(mgr.cpu_slot_table[20].item(), 0)
        self.assertGreaterEqual(mgr.cpu_slot_table[21].item(), 0)

        # Coldest experts (2 and 3) should have been evicted
        self.assertNotIn(2, mgr.residency_map)
        self.assertNotIn(3, mgr.residency_map)
        self.assertEqual(mgr.cpu_slot_table[2].item(), -1)
        self.assertEqual(mgr.cpu_slot_table[3].item(), -1)

    def test_threshold_ratio_prevents_churn(self):
        """Verify that threshold_ratio stops swaps when candidate advantage is negligible."""
        mgr = PagedExpertManager(
            num_slots=2,
            expert_shape=(8, 8),
            total_experts=16,
            device=self.device,
            compressed_cache=False,
        )
        mgr.request_experts([0, 1])

        # Resident counts: 0 -> 10.0, 1 -> 10.0
        mgr.record_routing([0, 1], weight=10.0)

        # Non-resident candidate 5 has count 11.0
        # 11.0 < (1.2 * 10.0 + 1.0) = 13.0 -> swap must be rejected
        mgr.record_routing([5], weight=11.0)

        res = mgr.adapt_post_prefill(threshold_ratio=1.2)
        self.assertEqual(res["swaps_performed"], 0)
        self.assertIn(0, mgr.residency_map)
        self.assertIn(1, mgr.residency_map)
        self.assertNotIn(5, mgr.residency_map)

    def test_multilayer_adaptation(self):
        """Verify multi-layer expert realignment and dirty layer tracking."""
        mgr = PagedExpertManager(
            num_slots=6,
            expert_shape=(8, 8),
            total_experts=16,
            device=self.device,
            compressed_cache=False,
        )

        # Route multi-layer experts
        mgr.record_routing([2, 3], layer_idx=0, weight=25.0)
        mgr.record_routing([4, 5], layer_idx=1, weight=30.0)

        res = mgr.adapt_post_prefill()
        self.assertEqual(res["swaps_performed"], 4)
        self.assertGreaterEqual(res["dirty_layers"], 2)

        self.assertIn((0, 2), mgr.residency_map)
        self.assertIn((0, 3), mgr.residency_map)
        self.assertIn((1, 4), mgr.residency_map)
        self.assertIn((1, 5), mgr.residency_map)


if __name__ == "__main__":
    unittest.main()
