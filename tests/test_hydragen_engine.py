# SPDX-License-Identifier: Apache-2.0
"""tests/test_hydragen_engine.py - Unit test suite for unified HydragenEngine."""

import os
import sys
import threading
import time
import unittest

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from hydragen.hydragen_engine import HydragenEngine
from hydragen.core.block_manager import TRTPagedBlockManager
from hydragen.core.scheduler import InFlightScheduler


class MockTokenizer:
    def encode(self, text, parse_special=True):
        return [101, 102, 103]


class TestHydragenEngine(unittest.TestCase):
    def setUp(self):
        self.tok = MockTokenizer()
        self.cfg = {
            "batch": 4,
            "max_context": 32768,
            "env": {"STRATA_ROUTE_PRIOR": "0.85"},
        }

    def test_hydragen_engine_initialization(self):
        eng = HydragenEngine(
            self.cfg,
            self.tok,
            batch_slots=4,
            num_vram_slots=24,
            standalone_mode=True,
        )
        self.assertEqual(eng.batch, 4)
        self.assertEqual(eng.info["engine"], "hydragen-unified")
        self.assertEqual(eng.info["engine_type"], "hydragen")
        self.assertTrue(eng.info["in_flight_batching"])
        self.assertTrue(eng.info["paged_kv"])
        self.assertEqual(eng.info["block_size"], 16)
        self.assertIsNotNone(eng.kv_manager)
        self.assertIsNotNone(eng.scheduler)

    def test_paged_kv_block_allocation_and_cow(self):
        eng = HydragenEngine(
            self.cfg,
            self.tok,
            batch_slots=4,
            standalone_mode=True,
            block_size=16,
            max_blocks=128,
        )
        cancel = threading.Event()
        # Acquire slot 0 with 32 prompt tokens -> 2 blocks
        s0, seq0 = eng.scheduler.acquire_slot(prompt_len=32, prompt_read=32, cancel=cancel)
        self.assertEqual(s0, 0)
        self.assertGreater(seq0, 0)

        stats = eng.kv_manager.stats()
        self.assertEqual(stats["used_blocks"], 2)

        # Fork sequence into child sequence (Copy-on-Write)
        ok = eng.kv_manager.fork_sequence(parent_seq_id=seq0, child_seq_id=999)
        self.assertTrue(ok)
        self.assertEqual(eng.kv_manager.stats()["used_blocks"], 2)  # Shared blocks

        # Free sequence
        eng.scheduler.release_slot(s0, seq0)
        self.assertIsNone(eng.scheduler.slot_live[s0])

        eng.kv_manager.free_sequence(999)
        self.assertEqual(eng.kv_manager.stats()["used_blocks"], 0)

    def test_slots_view_telemetry(self):
        eng = HydragenEngine(
            self.cfg,
            self.tok,
            batch_slots=2,
            standalone_mode=True,
        )
        views = eng.slots_view()
        self.assertEqual(len(views), 2)
        self.assertEqual(views[0]["state"], "idle")
        self.assertEqual(views[1]["state"], "idle")
        self.assertEqual(views[0]["paged_blocks"], 0)

    def test_standalone_token_generation_single_slot(self):
        eng = HydragenEngine(
            self.cfg,
            self.tok,
            batch_slots=2,
            standalone_mode=True,
        )
        cancel = threading.Event()
        gen = eng.generate([10, 20, 30], max_new=8, sampling={}, cancel=cancel)
        tokens = list(gen)
        self.assertGreater(len(tokens), 0)
        self.assertEqual(eng.last["prompt_tokens"], 3)
        self.assertEqual(eng.last["generated"], len(tokens))

    def test_multi_slot_concurrent_generation(self):
        """Verify parallel generation across 4 concurrent slots without interference."""
        eng = HydragenEngine(
            self.cfg,
            self.tok,
            batch_slots=4,
            standalone_mode=True,
        )
        results = [None] * 4
        exceptions = []

        def _client_worker(slot_idx: int):
            try:
                cancel = threading.Event()
                prompt = [1000 + slot_idx * 50 + i for i in range(16)]
                gen = eng.generate(prompt, max_new=16, sampling={}, cancel=cancel)
                toks = list(gen)
                results[slot_idx] = toks
            except Exception as e:
                exceptions.append(e)

        threads = [threading.Thread(target=_client_worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(exceptions), 0, f"Exceptions encountered: {exceptions}")
        for i in range(4):
            self.assertIsNotNone(results[i], f"Slot {i} did not finish generation")
            self.assertGreater(len(results[i]), 0)

        # All slots must be cleanly returned to idle and KV blocks freed
        self.assertEqual(eng.kv_manager.stats()["used_blocks"], 0)
        for s in eng.slots_view():
            self.assertEqual(s["state"], "idle")


if __name__ == "__main__":
    unittest.main()
