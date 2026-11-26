"""serve/test_kcap_admission.py - P5-5C.2: the engine's "KCAP <k_active> <max_k> <live_slots> <full_slots>"
line gates StrataEngine.pick_slot (no process, no GPU): the pump consumes KCAP (it never enters a request's
lines), pick_slot refuses past k_active, a raised K wakes the waiters, and BT/BDONE/DONE routing is untouched.

    python -m unittest serve.test_kcap_admission -v
"""
from __future__ import annotations

import collections
import queue
import sys
import threading
import time
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


class FakeBatchProc:
    """A fake batch engine process (P5-5C.5): its stdin answers BGEN lines.  An UNATTACHED slot gets the
    engine's memory-limit refusal ("ERR BGEN: slot N session unavailable (memory limit)"); an ATTACHED
    slot gets DONE + BADM on the control lines and BT/BDONE on its own queue (the pump is not running -
    the fake routes the lines the way the pump would).  It records every slot it refused and accepted."""

    def __init__(self, e, attached):
        self.e, self.attached = e, set(attached)
        self.refused, self.accepted = [], []
        self.returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass

    @property
    def stdin(self):
        return self

    def write(self, text):
        line = text.strip()
        if line.startswith("BGEN "):
            b = int(line.split()[1])
            if b not in self.attached:
                self.refused.append(b)
                self.e.lines.put(f"ERR BGEN: slot {b} session unavailable (memory limit)")
            else:
                self.accepted.append(b)
                self.e.lines.put("DONE 1 3 0.0 0.0 stop")
                self.e.lines.put(f"BADM {b} 1")
                self.e.slot_q[b].put(f"BT {b} 65")
                self.e.slot_q[b].put(f"BDONE {b} 1 stop 1.0")

    def flush(self):
        pass


def batched_engine(n=8, attached=()):
    """A StrataEngine with the full admission state of a started batch engine and a fake process that
    answers BGEN lines (no process, no GPU): the P5-5C.5 queueing harness."""
    e = StrataEngine("missing-executable", [], lazy=True)
    e.batch = n
    e.info = {}
    e.slot_q = [queue.Queue() for _ in range(n)]
    e.slot_busy = [False] * n
    e.slot_t_free = [0.0] * n
    e.slot_claims, e.slot_releasing, e._orphan_seen = set(), set(), {}
    e.slot_cv = threading.Condition()
    e.slot_order = list(range(n))
    e.slot_group = [0] * n
    e.slot_groups, e.slot_gs = 1, 0
    e.slot_held = [[] for _ in range(n)]
    e.slot_prompt = [[] for _ in range(n)]
    e.slot_used = [0.0] * n
    e.slot_live = [None] * n
    e._recent_reads = collections.deque(maxlen=2 * n)
    e._mem_backoff = {}
    e._slot_attached = set(attached)
    e.k_active, e.k_max, e.pool_live_slots, e.pool_full_slots = n, n, 0, 0
    e.lines = queue.Queue()
    e.gen = 1
    e.up, e.ended = True, False
    e.waiting, e.admitting, e.wait_lens, e.ctl_epoch = 0, 0, [], 0
    e.arrivals = collections.deque(maxlen=64)
    e.ctl = threading.Lock()
    e.wlock = threading.Lock()
    e._ctl_erred = False
    e._yielded = None
    e.progress, e.progress_ms, e.reused = None, 0, 0
    e.proc = FakeBatchProc(e, attached)
    e.alive = lambda: e.up and not e.ended
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


class AttachedPreferenceTest(unittest.TestCase):
    """P5-5C.5: with a memory backoff live, pick_slot prefers a free slot the engine already
    ATTACHED (reusing it costs 0 new host RAM and 0 new VRAM) over an unattached empty one."""

    def setUp(self):
        self.e = batched_engine(4)

    def pick(self):
        with self.e.slot_cv:
            return self.e.pick_slot([1, 2, 3])

    def test_prefers_attached_under_backoff(self):
        self.e._slot_attached = {2}
        self.e._mem_backoff = {0: time.monotonic() + 30.0, 1: time.monotonic() + 30.0}
        self.assertEqual(self.pick(), 2)                        # attached 2 before unattached 3

    def test_no_preference_without_backoff(self):
        self.e._slot_attached = {2}
        # no memory backoff at all: the old empty/least-recent order stands (slot 0 first)
        self.assertEqual(self.pick(), 0)

    def test_expired_backoff_restores_the_old_order(self):
        self.e._slot_attached = {2}
        self.e._mem_backoff = {0: time.monotonic() - 1.0}       # expired: nothing is "under pressure"
        self.assertEqual(self.pick(), 0)

    def test_attached_only_when_free(self):
        self.e._slot_attached = {1}
        self.e.slot_busy[1] = True                              # the attached slot is taken: normal pick
        self.e._mem_backoff = {0: time.monotonic() + 30.0}
        self.assertEqual(self.pick(), 2)


class MemoryRefusalQueueTest(unittest.TestCase):
    """P5-5C.5: the §245 ms01 burst, in one request.  Seven UNATTACHED slots answer
    "BGEN: slot N session unavailable (memory limit)" while one ATTACHED slot is busy/releasing:
    the request must NOT raise GpuBusy (0 503s) - it waits on slot_cv - and completes on the
    attached slot the moment it frees (re-admission costs 0 new RAM/VRAM)."""

    def test_zero_503_and_completes_on_the_attached_slot(self):
        e = batched_engine(8, attached={0})
        e.slot_busy[0] = True                                   # a request is finishing on slot 0
        e.slot_releasing = {0}                                  # ... in the background BDONE wait
        cancel = threading.Event()
        got, err = [], []

        def drive():
            try:
                for t in e.generate_batched([1, 2, 3], 10, None, cancel):
                    if t is not None:
                        got.append(t)
            except BaseException as ex:                         # noqa: BLE001 - the test reads it
                err.append(ex)

        t = threading.Thread(target=drive, daemon=True)
        t.start()
        # the request walks the unattached slots: every one is refused for the memory limit, and each
        # refusal puts that slot in the 30 s backoff.  refused never increments (slot 0 is in flight),
        # so BGEN_REFUSALS_MAX is never burned and no GpuBusy is raised.
        deadline = time.time() + 20.0
        while time.time() < deadline and len(e.proc.refused) < 7:
            time.sleep(0.05)
        self.assertEqual(sorted(e.proc.refused), [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(e.proc.accepted, [])                   # nothing admitted yet
        self.assertEqual(err, [])                               # no GpuBusy, no other failure
        self.assertTrue(t.is_alive())                           # it waits on slot_cv, not erroring
        # the attached slot frees (its BDONE wait ends): the waiter wakes and lands there
        with e.slot_cv:
            e.slot_busy[0] = False
            e.slot_releasing.discard(0)
            e.slot_t_free[0] = time.time()
            e.slot_cv.notify_all()
        t.join(timeout=10.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(err, [])                               # zero 503s end to end
        self.assertEqual(e.proc.accepted, [0])                  # admitted on the attached slot
        self.assertEqual(got, [65])                             # its reply came through
        self.assertIn(0, e._slot_attached)                      # and it stays attached for the next one

    def test_genuine_zero_capacity_still_raises_after_the_budget(self):
        # nothing in flight, nothing attached: the machine truly fits no slot - the refusal budget
        # still applies and the request ends as GpuBusy (a 503), as before.  (waiting=2: with a queue
        # beside it the request takes the batch path instead of the solo GEN one.)
        e = batched_engine(8)
        e.waiting = 2
        cancel = threading.Event()
        err = []

        def drive():
            try:
                for _ in e.generate_batched([1, 2, 3], 10, None, cancel):
                    pass
            except BaseException as ex:                         # noqa: BLE001
                err.append(ex)

        t = threading.Thread(target=drive, daemon=True)
        t.start()
        t.join(timeout=20.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(e.proc.refused), 7)                # budget 6: the 7th refusal ends it
        self.assertEqual(len(err), 1)
        self.assertIsInstance(err[0], server.GpuBusy)


if __name__ == "__main__":
    unittest.main()
