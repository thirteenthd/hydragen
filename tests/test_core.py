# SPDX-License-Identifier: Apache-2.0
"""Test suite for modular hydragen.core components."""

import os
import sys
import threading
import unittest

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from hydragen.core.block_manager import TRTPagedBlockManager, create_block_manager
from hydragen.core.scheduler import InFlightScheduler
from hydragen.core.engine import TRTLLMEngine
import hydragen.trtllm_engine as facade_engine


class TestCoreBlockManager(unittest.TestCase):
    def test_block_allocation_and_free(self):
        mgr = TRTPagedBlockManager(block_size=16, max_blocks=64)
        self.assertEqual(mgr.stats()["free_blocks"], 64)
        self.assertEqual(mgr.stats()["used_blocks"], 0)

        # Allocate 35 tokens -> 3 blocks (ceil(35/16))
        blocks = mgr.allocate_blocks(seq_id=1, num_tokens=35)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(mgr.stats()["used_blocks"], 3)
        self.assertEqual(mgr.stats()["free_blocks"], 61)

        # Fork sequence (Copy-on-Write)
        ok = mgr.fork_sequence(parent_seq_id=1, child_seq_id=2)
        self.assertTrue(ok)
        self.assertEqual(mgr.stats()["used_blocks"], 3)  # Blocks are shared

        # Free parent sequence
        mgr.free_sequence(seq_id=1)
        self.assertEqual(mgr.stats()["used_blocks"], 3)  # Child still references them

        # Free child sequence
        mgr.free_sequence(seq_id=2)
        self.assertEqual(mgr.stats()["used_blocks"], 0)
        self.assertEqual(mgr.stats()["free_blocks"], 64)


class TestCoreScheduler(unittest.TestCase):
    def test_in_flight_scheduler_slot_lifecycle(self):
        mgr = TRTPagedBlockManager(block_size=16, max_blocks=32)
        scheduler = InFlightScheduler(batch_slots=2, kv_manager=mgr)

        cancel = threading.Event()
        slot0, seq0 = scheduler.acquire_slot(prompt_len=30, prompt_read=30, cancel=cancel)
        self.assertEqual(slot0, 0)
        self.assertEqual(seq0, 1)

        views = scheduler.slots_view()
        self.assertEqual(len(views), 2)
        self.assertEqual(views[0]["slot"], 0)
        self.assertEqual(views[0]["state"], "prefill")
        self.assertEqual(views[1]["state"], "idle")

        scheduler.mark_decoding(slot0)
        scheduler.update_generated(slot0, 5)

        views = scheduler.slots_view()
        self.assertEqual(views[0]["state"], "decoding")
        self.assertEqual(views[0]["generated"], 5)

        slot1, seq1 = scheduler.acquire_slot(prompt_len=20, prompt_read=10, cancel=cancel)
        self.assertEqual(slot1, 1)
        self.assertEqual(seq1, 2)

        scheduler.release_slot(slot0, seq0)
        self.assertIsNone(scheduler.slot_live[0])

        scheduler.release_slot(slot1, seq1)
        self.assertIsNone(scheduler.slot_live[1])

    def test_prefix_cache_matching(self):
        scheduler = InFlightScheduler(batch_slots=2)
        prompt = [10, 20, 30, 40, 50]
        gen = [60, 70]
        scheduler.record_prompt_cache(prompt, gen)

        match_len = scheduler.match_prefix_cache([10, 20, 30, 99])
        self.assertEqual(match_len, 3)

        match_len_full = scheduler.match_prefix_cache([10, 20, 30, 40, 50, 60])
        self.assertEqual(match_len_full, 6)


class TestFacadeReExports(unittest.TestCase):
    def test_re_exports_match(self):
        self.assertIs(facade_engine.TRTLLMEngine, TRTLLMEngine)
        self.assertIs(facade_engine.TRTPagedBlockManager, TRTPagedBlockManager)
        self.assertIs(facade_engine.InFlightScheduler, InFlightScheduler)


if __name__ == "__main__":
    unittest.main()
