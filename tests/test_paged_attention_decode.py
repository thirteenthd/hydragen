# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Paged Attention Decode with Clamped Branchless KV Fetch."""

import math
import unittest
import torch

from hydragen.paged_kv import PagedKVManager


class TestPagedAttentionDecode(unittest.TestCase):
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

    def test_decode_correctness_against_exact_reference(self):
        """Verifies single-token decode attention matches mathematical reference across partial and full blocks."""
        for ctx_len in [1, 7, 16, 23, 32]:
            seq_id = 500 + ctx_len
            self.mgr.allocate_sequence(seq_id, ctx_len)
            block_list = self.mgr.get_block_table(seq_id)
            self.assertTrue(len(block_list) >= 1)

            block_tables = torch.tensor([block_list + [0] * (8 - len(block_list))], dtype=torch.int32)
            context_lens = torch.tensor([ctx_len], dtype=torch.int32)

            # 1 decode query token
            q = torch.randn(1, self.num_heads, self.head_dim, dtype=torch.float32)
            k_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim, dtype=torch.float32)
            v_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim, dtype=torch.float32)

            out_paged = torch.zeros_like(q)
            self.mgr.paged_attention_decode(
                out=out_paged,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_tables,
                context_lens=context_lens,
                scale=self.scale,
                is_int8_kv=False,
            )

            # Mathematical reference
            k_flat = torch.zeros(self.num_kv_heads, ctx_len, self.head_dim)
            v_flat = torch.zeros(self.num_kv_heads, ctx_len, self.head_dim)
            for t in range(ctx_len):
                b_idx = t // self.block_size
                b_off = t % self.block_size
                p_blk = block_list[b_idx]
                k_flat[:, t, :] = k_cache[p_blk, :, b_off, :]
                v_flat[:, t, :] = v_cache[p_blk, :, b_off, :]

            group = self.num_heads // self.num_kv_heads
            out_ref = torch.zeros_like(q)
            for h in range(self.num_heads):
                kv_h = h // group
                qs = q[0, h] * self.scale
                scores = torch.matmul(k_flat[kv_h], qs)
                probs = torch.softmax(scores, dim=-1)
                out_ref[0, h] = torch.matmul(probs, v_flat[kv_h])

            max_diff = torch.max(torch.abs(out_paged - out_ref)).item()
            self.assertLess(max_diff, 1e-4, f"Decode diverged at ctx_len={ctx_len}: {max_diff}")

    def test_decode_clamped_branchless_arithmetic_equivalence(self):
        """Verifies that clamping index and masking invalid slots to -1e9 strictly preserves exact softmax results."""
        # Simulated single block with 9 valid tokens out of 16
        tokens_in_block = 9
        q_vec = torch.randn(self.head_dim) * self.scale
        k_block = torch.randn(self.block_size, self.head_dim)
        v_block = torch.randn(self.block_size, self.head_dim)

        # 1. Truncated reference
        ref_scores = torch.matmul(k_block[:tokens_in_block], q_vec)
        ref_probs = torch.softmax(ref_scores, dim=-1)
        ref_out = torch.matmul(ref_probs, v_block[:tokens_in_block])

        # 2. Clamped branchless simulation:
        # Loop runs for all 16 slots with clamped memory index and -1e9 mask
        clamped_scores = []
        clamped_v = []
        for t in range(self.block_size):
            t_clamped = min(t, tokens_in_block - 1)
            mask = 0.0 if t < tokens_in_block else -1e9
            score = torch.dot(k_block[t_clamped], q_vec) + mask
            clamped_scores.append(score)
            clamped_v.append(v_block[t_clamped])

        clamped_scores_t = torch.stack(clamped_scores)
        clamped_v_t = torch.stack(clamped_v)
        clamped_probs = torch.softmax(clamped_scores_t, dim=-1)
        clamped_out = torch.matmul(clamped_probs, clamped_v_t)

        diff = torch.max(torch.abs(ref_out - clamped_out)).item()
        self.assertLess(diff, 1e-5, f"Clamped branchless attention diverged: {diff}")
        # Verify that probability on masked slots is strictly zero
        self.assertLess(torch.sum(clamped_probs[tokens_in_block:]).item(), 1e-7)

    def test_decode_int8_kv(self):
        """Verifies INT8 quantized KV cache decode attention."""
        ctx_len = 19
        seq_id = 601
        self.mgr.allocate_sequence(seq_id, ctx_len)
        block_list = self.mgr.get_block_table(seq_id)

        block_tables = torch.tensor([block_list + [0] * (8 - len(block_list))], dtype=torch.int32)
        context_lens = torch.tensor([ctx_len], dtype=torch.int32)

        q = torch.randn(1, self.num_heads, self.head_dim, dtype=torch.float32)
        # INT8 range -128..127
        k_cache = torch.randint(-64, 64, (self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim), dtype=torch.int8)
        v_cache = torch.randint(-64, 64, (self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim), dtype=torch.int8)

        out = torch.zeros_like(q)
        self.mgr.paged_attention_decode(
            out=out,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=self.scale,
            is_int8_kv=True,
        )

        self.assertEqual(out.shape, (1, self.num_heads, self.head_dim))
        self.assertFalse(torch.isnan(out).any().item())
        self.assertFalse(torch.isinf(out).any().item())

    def test_multi_sequence_batched_decode(self):
        """Verifies batched decode across multiple sequences with divergent context lengths."""
        num_seqs = 3
        lens = [4, 16, 25]
        sids = [701, 702, 703]
        for sid, l in zip(sids, lens):
            self.mgr.allocate_sequence(sid, l)

        max_b = 4
        tables = []
        for sid in sids:
            bl = self.mgr.get_block_table(sid)
            tables.append(bl + [0] * (max_b - len(bl)))

        block_tables = torch.tensor(tables, dtype=torch.int32)
        context_lens = torch.tensor(lens, dtype=torch.int32)

        q = torch.randn(num_seqs, self.num_heads, self.head_dim, dtype=torch.float32)
        k_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim, dtype=torch.float32)
        v_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim, dtype=torch.float32)

        out = torch.zeros_like(q)
        self.mgr.paged_attention_decode(
            out=out,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=self.scale,
        )

        self.assertEqual(out.shape, (num_seqs, self.num_heads, self.head_dim))
        self.assertFalse(torch.isnan(out).any().item())


if __name__ == "__main__":
    unittest.main()
