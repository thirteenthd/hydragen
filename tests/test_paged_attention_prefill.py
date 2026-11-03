# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Paged Attention Prefill with Reverse Query-Tile Attention Scheduling (LPT)."""

import math
import unittest
import torch

from hydragen.paged_kv import PagedKVManager


class TestPagedAttentionPrefill(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.block_size = 16
        self.num_blocks = 32
        self.num_layers = 2
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 128
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.mgr = PagedKVManager(
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

    def test_prefill_causal_correctness_against_reference(self):
        """Verifies that paged_attention_prefill matches an exact causal reference."""
        num_seqs = 1
        query_len = 8
        seq_id = 101

        # Allocate blocks in paged KV manager
        self.mgr.allocate_sequence(seq_id, query_len)
        block_list = self.mgr.get_block_table(seq_id)
        self.assertTrue(len(block_list) >= 1)

        block_tables = torch.tensor([block_list + [0] * (8 - len(block_list))], dtype=torch.int32)
        context_lens = torch.tensor([0], dtype=torch.int32)

        # Create synthetic Q, K_cache, V_cache
        q = torch.randn(num_seqs, query_len, self.num_heads, self.head_dim, dtype=torch.float32)
        k_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim, dtype=torch.float32)
        v_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim, dtype=torch.float32)

        out_paged = torch.zeros_like(q)
        self.mgr.paged_attention_prefill(
            out=out_paged,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=self.scale,
            is_int8_kv=False,
        )

        # Compute exact reference causal attention
        # Extract flat K and V for the query_len tokens
        k_flat = torch.zeros(self.num_kv_heads, query_len, self.head_dim)
        v_flat = torch.zeros(self.num_kv_heads, query_len, self.head_dim)
        for t in range(query_len):
            b_idx = t // self.block_size
            b_off = t % self.block_size
            p_blk = block_list[b_idx]
            k_flat[:, t, :] = k_cache[p_blk, :, b_off, :]
            v_flat[:, t, :] = v_cache[p_blk, :, b_off, :]

        # Reference attention
        group = self.num_heads // self.num_kv_heads
        out_ref = torch.zeros_like(q)
        for t in range(query_len):
            for h in range(self.num_heads):
                kv_h = h // group
                q_vec = q[0, t, h] * self.scale
                # Dot product with visible keys [0 .. t]
                scores = torch.matmul(k_flat[kv_h, :t + 1], q_vec)
                probs = torch.softmax(scores, dim=-1)
                out_ref[0, t, h] = torch.matmul(probs, v_flat[kv_h, :t + 1])

        # Assert equivalence
        max_diff = torch.max(torch.abs(out_paged - out_ref)).item()
        self.assertLess(max_diff, 1e-4, f"Prefill output diverged from causal reference: {max_diff}")

    def test_past_context_continuation(self):
        """Verifies prefill attention with existing past context in KV cache."""
        num_seqs = 1
        past_len = 16
        query_len = 4
        seq_id = 102

        # Allocate blocks for total 20 tokens
        self.mgr.allocate_sequence(seq_id, past_len + query_len)
        block_list = self.mgr.get_block_table(seq_id)

        block_tables = torch.tensor([block_list + [0] * (8 - len(block_list))], dtype=torch.int32)
        context_lens = torch.tensor([past_len], dtype=torch.int32)

        q = torch.randn(num_seqs, query_len, self.num_heads, self.head_dim, dtype=torch.float32)
        k_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim, dtype=torch.float32)
        v_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim, dtype=torch.float32)

        out = torch.zeros_like(q)
        self.mgr.paged_attention_prefill(
            out=out,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=self.scale,
        )

        # Output shape must match query shape
        self.assertEqual(out.shape, q.shape)
        # Verify no NaN or Inf
        self.assertFalse(torch.isnan(out).any().item())
        self.assertFalse(torch.isinf(out).any().item())

    def test_multi_sequence_prefill_batch(self):
        """Verifies batched prefill across multiple concurrent sequences."""
        num_seqs = 2
        query_len = 6
        seq1, seq2 = 201, 202

        self.mgr.allocate_sequence(seq1, query_len)
        self.mgr.allocate_sequence(seq2, query_len)

        bl1 = self.mgr.get_block_table(seq1)
        bl2 = self.mgr.get_block_table(seq2)
        max_b = max(len(bl1), len(bl2), 4)

        block_tables = torch.tensor([
            bl1 + [0] * (max_b - len(bl1)),
            bl2 + [0] * (max_b - len(bl2)),
        ], dtype=torch.int32)
        context_lens = torch.tensor([0, 0], dtype=torch.int32)

        q = torch.randn(num_seqs, query_len, self.num_heads, self.head_dim, dtype=torch.float32)
        k_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim, dtype=torch.float32)
        v_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim, dtype=torch.float32)

        out = torch.zeros_like(q)
        self.mgr.paged_attention_prefill(
            out=out,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=self.scale,
        )

        self.assertEqual(out.shape, (num_seqs, query_len, self.num_heads, self.head_dim))
        self.assertFalse(torch.isnan(out).any().item())


if __name__ == "__main__":
    unittest.main()
