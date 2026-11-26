"""serve/test_kcap_admission.py - P5-5C.2: the engine's "KCAP <k_active> <max_k> <live_slots> <full_slots>"
line gates StrataEngine.pick_slot (no process, no GPU): the pump consumes KCAP (it never enters a request's
lines), pick_slot refuses past k_active, a raised K wakes the waiters, and BT/BDONE/DONE routing is untouched.

    python -m unittest serve.test_kcap_admission -v
"""
from __future__ import annotations

import queue
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.server import StrataEngine  # noqa: E402


class FakeProc:
    """Stands in for the engine process: stdout yields the scripted lines, then EOF."""

    def __init__(self, lines):
        # a real stdout yields lines that end with their newline; the scripts here may leave it off
        self.stdout = iter(l if l.endswith("\n") else l + "\n" for l in lines)
        self.returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass


def engine_with_lines(lines, n=4):
    """A StrataEngine with the admission state of a started one, a fake stdout, and the pump run over it."""
    e = StrataEngine("missing-executable", [], lazy=True)
    e.batch = n
    e.slot_q = [queue.Queue() for _ in range(n)]
    e.slot_busy = [False] * n
    e.slot_t_free = [0.0] * n
    e.slot_claims, e.slot_releasing, e._orphan_seen = set(), set(), {}
    e.slot_cv = threading.Condition()
    e.k_active, e.k_max, e.pool_live_slots, e.pool_full_slots = n, n, 0, 0
    e.lines = queue.Queue()
    e.gen = 1
    e.up = True
    e.proc = FakeProc(lines)
    e._pump()                       # the scripted lines, then EOF (ended=True, None published)
    return e


class KcapParseTest(unittest.TestCase):
    def test_kcap_updates_state(self):
        e = engine_with_lines(["KCAP 3 8 2400 2880", "KCAP 2 8 2100 2880"], n=8)
        self.assertEqual(e.k_active, 2)
        self.assertEqual(e.k_max, 8)
        self.assertEqual(e.pool_live_slots, 2100)
        self.assertEqual(e.pool_full_slots, 2880)

    def test_kcap_clamped_to_wrapper_slots(self):
        e = engine_with_lines(["KCAP 9 8 2400 2880"], n=4)      # an engine saying more than the wrapper has
        self.assertEqual(e.k_active, 4)

    def test_kcap_two_fields_only(self):
        e = engine_with_lines(["KCAP 3 8"], n=4)
        self.assertEqual(e.k_active, 3)
        self.assertEqual(e.pool_live_slots, 0)                  # absent fields leave the old values

    def test_kcap_malformed_changes_nothing(self):
        e = engine_with_lines(["KCAP x y", "KCAP 2 8"], n=4)
        self.assertEqual(e.k_active, 2)                         # the good line applied; the bad one did not
        e2 = engine_with_lines(["KCAP", "KCAP 2"], n=4)         # too few fields: nothing parses
        self.assertEqual(e2.k_active, 4)                        # the default stands

    def test_kcap_never_reaches_a_request(self):
        e = engine_with_lines(["READY 131072 stop hold holdb", "KCAP 2 4 100 200", "DONE"])
        got = []
        while True:
            line = e.lines.get()
            if line is None:
                break
            got.append(line)
        self.assertEqual(got, ["READY 131072 stop hold holdb\n", "DONE\n"])


class KcapRoutingTest(unittest.TestCase):
    def test_bt_bdone_untouched_by_kcap(self):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.batch = 4
        e.slot_q = [queue.Queue() for _ in range(4)]
        e.slot_busy = [False] * 4
        e.slot_busy[1] = True                                   # slot 1 is claimed: its frames route
        e.slot_claims, e.slot_releasing, e._orphan_seen = {1}, set(), {}
        e.slot_cv = threading.Condition()
        e.k_active, e.k_max, e.pool_live_slots, e.pool_full_slots = 4, 4, 0, 0
        e.lines = queue.Queue()
        e.gen = 1
        e.up = True
        e.proc = FakeProc(["BT 1 5", "KCAP 2 4 100 200", "BDONE 1 stop 10.0", "DONE"])
        e._pump()
        routed = [e.slot_q[1].get_nowait() for _ in range(2)]
        self.assertEqual(routed, ["BT 1 5\n", "BDONE 1 stop 10.0\n"])
        # the other slots' queues hold only the EOF sentinel the pump publishes on close
        for b in (0, 2, 3):
            self.assertIsNone(e.slot_q[b].get_nowait())
            self.assertTrue(e.slot_q[b].empty())
        self.assertIsNone(e.slot_q[1].get_nowait())             # slot 1's EOF sentinel, then nothing
        self.assertEqual(e.k_active, 2)
        got = []
        while True:
            line = e.lines.get()
            if line is None:
                break
            got.append(line)
        self.assertEqual(got, ["DONE\n"])


class PickSlotGateTest(unittest.TestCase):
    def setUp(self):
        self.e = StrataEngine("missing-executable", [], lazy=True)
        self.e.batch = 4
        self.e.slot_q = [queue.Queue() for _ in range(4)]
        self.e.slot_busy = [False] * 4
        self.e.slot_t_free = [0.0] * 4
        self.e.slot_claims, self.e.slot_releasing, self.e._orphan_seen = set(), set(), {}
        self.e.slot_cv = threading.Condition()
        self.e.slot_order = [0, 1, 2, 3]
        self.e.slot_group = [0, 0, 0, 0]
        self.e.slot_groups = 1
        self.e.slot_gs = 0
        self.e.slot_held = [[] for _ in range(4)]
        self.e.slot_used = [0.0] * 4
        self.e.lines = queue.Queue()
        self.e.k_active, self.e.k_max = 4, 4

    def pick(self):
        with self.e.slot_cv:
            return self.e.pick_slot([1, 2, 3])

    def test_no_gate_at_default_k(self):
        self.e.slot_busy = [True, True, True, False]
        self.assertEqual(self.pick(), 3)                        # k_active == batch: behaves as before

    def test_gate_refuses_past_k_active(self):
        self.e.k_active = 2
        self.e.slot_busy = [True, True, False, False]
        self.assertIsNone(self.pick())                          # two busy == K: the free slots wait

    def test_gate_opens_when_a_slot_frees(self):
        self.e.k_active = 2
        self.e.slot_busy = [True, True, False, False]
        self.assertIsNone(self.pick())
        self.e.slot_busy[1] = False
        self.assertIsNotNone(self.pick())

    def test_raised_k_wakes_a_waiter(self):
        self.e.k_active = 1
        self.e.slot_busy = [True, False, False, False]
        got = []

        def waiter():
            with self.e.slot_cv:
                while True:
                    slot = self.e.pick_slot([1, 2, 3])
                    if slot is not None:
                        got.append(slot)
                        return
                    self.e.slot_cv.wait(timeout=5.0)

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        self.assertFalse(t.is_alive() and got)                  # it is waiting: no slot yet
        self.e.proc = FakeProc(["KCAP 3 4 100 200"])            # the engine raises K
        self.e._pump()                                          # the pump notifies slot_cv
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(got), 1)                           # the waiter woke and took a slot


if __name__ == "__main__":
    unittest.main()
