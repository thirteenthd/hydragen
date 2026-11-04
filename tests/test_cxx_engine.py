# SPDX-License-Identifier: Apache-2.0
"""tests/test_cxx_engine.py - Unit test suite for Native C++ HydragenEngine."""

import os
import sys
import unittest

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from hydragen.core.native_engine import NativeHydragenEngineWrapper


class TestNativeCxxEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.has_native_lib = NativeHydragenEngineWrapper.load_library()

    def test_native_cxx_engine_lifecycle(self):
        if not self.has_native_lib:
            self.skipTest("libhydragen_core.so not compiled on this platform yet")

        eng = NativeHydragenEngineWrapper(batch_slots=4, block_size=16, max_blocks=128, mock_worker=True)
        self.assertTrue(eng.is_alive())

        telem = eng.get_telemetry()
        self.assertEqual(telem.get("engine"), "hydragen_native")
        self.assertEqual(telem.get("max_slots"), 4)
        self.assertEqual(telem.get("active_slots"), 0)

        # Submit request 101
        slot = eng.submit(req_id=101, prompt_ids=[1, 2, 3, 4], max_new=4)
        self.assertGreater(slot, 0)

        telem = eng.get_telemetry()
        self.assertEqual(telem.get("active_slots"), 1)

        # Step until finished
        steps = 0
        finished = False
        tokens = []
        while steps < 10 and not finished:
            ret, req_id, tok_id, fin = eng.step()
            self.assertEqual(ret, 1)
            self.assertEqual(req_id, 101)
            tokens.append(tok_id)
            if fin:
                finished = True
                break
            steps += 1

        self.assertTrue(finished)
        self.assertEqual(len(tokens), 4)

        # Verify active slots returned to 0
        telem = eng.get_telemetry()
        self.assertEqual(telem.get("active_slots"), 0)

        eng.close()
        self.assertFalse(eng.is_alive())


if __name__ == "__main__":
    unittest.main()
