# SPDX-License-Identifier: Apache-2.0
"""tests/test_mtp_preconditioner.py - Unit tests for Milestone 5 (MTP Prompt Pre-Conditioning)."""

import unittest
import torch
from hydragen.mtp_preconditioner import MtpDraftPreconditioner, rms_norm


class TestMtpPreconditioner(unittest.TestCase):
    def test_rms_norm(self):
        x = torch.randn(4, 64)
        out = rms_norm(x)
        self.assertEqual(out.shape, (4, 64))
        var = out.pow(2).mean(dim=-1)
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-3))

    def test_mtp_preconditioner_init(self):
        device = torch.device("cpu")
        precond = MtpDraftPreconditioner(
            n_embd=128,
            hc=4,
            n_head_kv=2,
            head_dim=32,
            max_context=1024,
            device=device,
        )
        self.assertEqual(precond.n_embd, 128)
        self.assertEqual(precond.hcn, 512)
        self.assertEqual(precond.kv_dim, 64)
        self.assertEqual(precond.draft_k.shape, (1024, 64))
        self.assertEqual(precond.draft_v.shape, (1024, 64))
        self.assertEqual(precond.current_pos, 0)
        self.assertIsNone(precond.saved_hidden)

    def test_mtp_preconditioner_warmup(self):
        device = torch.device("cpu")
        precond = MtpDraftPreconditioner(
            n_embd=128,
            hc=4,
            n_head_kv=2,
            head_dim=32,
            max_context=256,
            device=device,
        )

        T = 20
        tokens = torch.randn(T, 128, dtype=torch.float16, device=device)
        hidden = torch.randn(T, 512, dtype=torch.float16, device=device)

        res = precond.warmup_prompt(tokens, hidden)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["preconditioned_tokens"], T - 1)
        self.assertEqual(res["current_kv_pos"], T - 1)
        self.assertIsNotNone(precond.saved_hidden)
        self.assertEqual(precond.saved_hidden.shape, (1, 512))

        self.assertTrue(torch.isfinite(precond.draft_k[: T - 1]).all())
        self.assertTrue(torch.isfinite(precond.draft_v[: T - 1]).all())
        self.assertFalse(torch.allclose(precond.draft_k[: T - 1], torch.zeros(T - 1, 64, dtype=torch.float16)))

    def test_mtp_preconditioner_first_decode(self):
        device = torch.device("cpu")
        precond = MtpDraftPreconditioner(
            n_embd=128,
            hc=4,
            n_head_kv=2,
            head_dim=32,
            max_context=256,
            device=device,
        )

        T = 10
        tokens = torch.randn(T, 128, dtype=torch.float16, device=device)
        hidden = torch.randn(T, 512, dtype=torch.float16, device=device)

        precond.warmup_prompt(tokens, hidden)
        pos_before = precond.current_pos

        first_decode_tok = torch.randn(1, 128, dtype=torch.float16, device=device)
        k_step, v_step = precond.prepare_first_decode_step(first_decode_tok)

        self.assertEqual(k_step.shape, (1, 64))
        self.assertEqual(v_step.shape, (1, 64))
        self.assertEqual(precond.current_pos, pos_before + 1)

        k_slice, v_slice = precond.get_context_slice(10)
        self.assertEqual(k_slice.shape, (10, 64))
        self.assertEqual(v_slice.shape, (10, 64))

    def test_mtp_preconditioner_reset(self):
        device = torch.device("cpu")
        precond = MtpDraftPreconditioner(
            n_embd=64,
            hc=2,
            n_head_kv=1,
            head_dim=32,
            max_context=64,
            device=device,
        )
        tokens = torch.randn(8, 64, dtype=torch.float16, device=device)
        precond.warmup_prompt(tokens)
        self.assertEqual(precond.current_pos, 7)
        self.assertIsNotNone(precond.saved_hidden)

        precond.reset()
        self.assertEqual(precond.current_pos, 0)
        self.assertIsNone(precond.saved_hidden)
        self.assertTrue(torch.allclose(precond.draft_k, torch.zeros_like(precond.draft_k)))


if __name__ == "__main__":
    unittest.main()
