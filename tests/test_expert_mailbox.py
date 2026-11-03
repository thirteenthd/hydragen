import unittest
import torch
import torch.nn.functional as F

from hydragen.expert_mailbox import HybridExpertMailbox
from hydragen.expert_manager import PagedExpertManager
from hydragen.triton_moe import triton_paged_moe_forward


class TestExpertMailbox(unittest.TestCase):
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

    def test_mailbox_cpu_expert_projection(self):
        """Verify that CPU expert projection correctly calculates SwiGLU MoE forward pass."""
        mailbox = HybridExpertMailbox(max_workers=2, device=torch.device("cpu"))
        M, K, N_ff = 4, 32, 64
        x = torch.randn((M, K), dtype=torch.float32)

        # Expert weights: [3, N_ff, K] for Gate, Up, Down
        w_gate = torch.randn((N_ff, K), dtype=torch.float32)
        w_up = torch.randn((N_ff, K), dtype=torch.float32)
        w_down = torch.randn((N_ff, K), dtype=torch.float32)
        weights = torch.stack([w_gate, w_up, w_down], dim=0)

        # Ground truth mathematical formulation
        g = torch.matmul(x, w_gate.t())
        u = torch.matmul(x, w_up.t())
        inter = F.silu(g) * u
        expected = torch.matmul(inter, w_down)

        actual = mailbox.compute_cpu_expert(
            x_cpu=x,
            host_expert_weights=weights,
            has_up=True,
            N_ff=N_ff,
            K=K,
        )

        self.assertEqual(actual.shape, (M, K))
        self.assertTrue(torch.allclose(expected, actual, atol=1e-4, rtol=1e-4))
        mailbox.close()

    def test_mailbox_dispatch_and_gather(self):
        """Verify asynchronous dispatch to worker pool and gather into target tensor."""
        mailbox = HybridExpertMailbox(max_workers=2, device=self.device)
        M, K, N_ff = 2, 16, 32
        x = torch.randn((M, K), dtype=torch.float16)

        total_experts = 4
        host_pool = torch.randn((total_experts, 3, N_ff, K), dtype=torch.float16)

        missed_items = [(0, 1), (1, 3)]  # compact 0 -> expert 1, compact 1 -> expert 3
        futures = mailbox.dispatch_missed_experts(
            x_cpu=x.cpu(),
            missed_items=missed_items,
            host_pool=host_pool,
            has_up=True,
            N_ff=N_ff,
            K=K,
        )
        self.assertEqual(len(futures), 2)

        target_gpu = torch.zeros((2, M, K), dtype=torch.float16, device=self.device)
        mailbox.gather_and_merge(target_gpu, futures)

        # Verify target is populated and non-zero
        self.assertTrue(torch.isfinite(target_gpu).all())
        self.assertFalse((target_gpu == 0).all())
        self.assertEqual(mailbox.total_fusions, 1)
        mailbox.close()

    def test_hybrid_moe_forward_with_mailbox(self):
        """Verify end-to-end triton_paged_moe_forward with mixed resident + missed experts."""
        if self.device.type != "cuda":
            self.skipTest("Triton MoE kernel requires CUDA device")

        M, K, N_ff = 4, 32, 64
        total_experts = 8
        num_slots = 2  # Only 2 resident slots!

        x = torch.randn((M, K), dtype=torch.float16, device=self.device)
        topk_ids = torch.tensor([[0, 2], [1, 3], [0, 1], [2, 3]], dtype=torch.int32, device=self.device)
        topk_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]], dtype=torch.float32, device=self.device)

        mgr = PagedExpertManager(
            num_slots=num_slots,
            expert_shape=(3, N_ff, K),
            total_experts=total_experts,
            dtype=torch.float16,
            device=self.device,
            compressed_cache=False,
            enable_mailbox=True,
        )
        # Pre-fill weights in host pool
        torch.manual_seed(42)
        mgr.host_pool.copy_(torch.randn_like(mgr.host_pool))

        # Make experts 0 and 1 resident in the 2 slots
        mgr.request_experts([0, 1])
        self.assertTrue(mgr.is_resident(0))
        self.assertTrue(mgr.is_resident(1))
        self.assertFalse(mgr.is_resident(2))
        self.assertFalse(mgr.is_resident(3))

        # Run forward pass: experts 0, 1 are resident (GPU), experts 2, 3 are misses (CPU mailbox)
        out, metrics = triton_paged_moe_forward(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_manager=mgr,
            layer_idx=0,
        )

        self.assertEqual(out.shape, (M, K))
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(mgr.mailbox.total_cpu_experts, 0)
        self.assertGreater(mgr.mailbox.total_gpu_experts, 0)
        mgr.mailbox.close()


if __name__ == "__main__":
    unittest.main()
