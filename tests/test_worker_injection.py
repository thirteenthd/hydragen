# SPDX-License-Identifier: Apache-2.0
"""tests/test_worker_injection.py - HydragenEngine wraps a server-built hardware worker."""

import os
import sys
import threading
import unittest

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from hydragen.hydragen_engine import HydragenEngine


class FakeWorker:
    batch = 3
    max_context = 4096
    info = {"arena_mib": 1, "expert_slots": 7}
    spawn = ("exe", 0)
    gpu_busy = None
    silence_s = 300.0

    def __init__(self):
        self.restarted = False
        self.closed = False

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for i in range(max_new):
            yield 500 + i

    def restart(self):
        self.restarted = True

    def close(self):
        self.closed = True

    def death_note(self):
        return "fake"


class TestWorkerInjection(unittest.TestCase):
    def setUp(self):
        self.worker = FakeWorker()
        self.eng = HydragenEngine({}, None, batch_slots=4, hardware_worker=self.worker)

    def test_batch_and_context_follow_worker(self):
        self.assertEqual(self.eng.batch, 3)
        self.assertEqual(self.eng.max_context, 4096)
        self.assertIs(self.eng.hardware_worker, self.worker)

    def test_unknown_attributes_come_from_worker(self):
        self.assertEqual(self.eng.spawn, ("exe", 0))
        self.assertEqual(self.eng.death_note(), "fake")
        self.eng.restart()
        self.assertTrue(self.worker.restarted)

    def test_watchdog_attributes_land_on_worker(self):
        self.eng.gpu_busy = lambda: True
        self.eng.silence_s = 42.0
        self.assertTrue(self.worker.gpu_busy())
        self.assertEqual(self.worker.silence_s, 42.0)
        self.assertNotIn("gpu_busy", self.eng.__dict__)

    def test_generate_goes_through_worker(self):
        toks = list(self.eng.generate([1, 2, 3], 5, {}, threading.Event()))
        self.assertEqual(toks, [500, 501, 502, 503, 504])
        self.assertEqual(self.eng.last["generated"], 5)

    def test_missing_attribute_still_raises(self):
        with self.assertRaises(AttributeError):
            self.eng.no_such_thing

    def test_no_worker_means_no_passthrough(self):
        eng = HydragenEngine({}, None, batch_slots=2, standalone_mode=True)
        with self.assertRaises(AttributeError):
            eng.spawn

    def test_worker_drafter_delegation_mtp(self):
        cfg = {"args": ["--spec", "4", "--mtp", "models/mtp/rt"]}
        eng = HydragenEngine(cfg, None, hardware_worker=FakeWorker())
        self.assertEqual(eng.info["drafter"], "mtp_autoregressive_drafter")
        self.assertEqual(eng.info["drafting_mode"], "mtp")

    def test_worker_drafter_delegation_dflash2(self):
        cfg = {"args": ["--draft", "dflash2"]}
        eng = HydragenEngine(cfg, None, hardware_worker=FakeWorker())
        self.assertEqual(eng.info["drafter"], "dflash2_block_drafter")
        self.assertEqual(eng.info["drafting_mode"], "dflash2")


if __name__ == "__main__":
    unittest.main()
