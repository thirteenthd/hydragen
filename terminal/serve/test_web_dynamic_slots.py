"""serve/test_web_dynamic_slots.py - P5-5D: the Monitor's dynamic active-slot cards (no GPU, no process).

Backend: /metrics' live payload carries the dynamic auto-batch cap and unified pool (k_active / k_max /
pool_live_slots / pool_full_slots / min_stream_tps); slots_view() carries `releasing` (a slot whose reply ended
and whose background BDONE wait still owns it -> state "detaching") and the admission's restored-vs-generated
counts (restored_tokens / restored_from, wired from the RESUME line via _control's resume_sink and the engine
log's parked / disk restore events).

Frontend: app.js renders a slot card only while its slot is utilized, an idle summary tile when nothing runs,
the K_active header badge, and the restored-vs-generated breakdown - checked by running the page's own code
under node with DOM stand-ins, plus source-level assertions and `node --check`.

    python -m unittest serve.test_web_dynamic_slots -v
"""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine  # noqa: E402
from serve.test_kcap_admission import batched_engine  # noqa: E402  (the fake BGEN-answering engine)

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "serve" / "web"
NODE = shutil.which("node")


def batch_engine(args=(), n=4):
    """A StrataEngine with the admission state of a started one and no process (as serve/test_kcap_admission)."""
    e = StrataEngine("missing-executable", list(args), lazy=True)
    e.batch = n
    e.slot_q = [queue.Queue() for _ in range(n)]
    e.slot_busy = [False] * n
    e.slot_t_free = [0.0] * n
    e.slot_held = [[] for _ in range(n)]
    e.slot_used = [0.0] * n
    e.slot_live = [None] * n
    e.slot_claims, e.slot_releasing, e._orphan_seen = set(), set(), {}
    e.slot_cv = threading.Condition()
    e.k_active, e.k_max, e.pool_live_slots, e.pool_full_slots = n, n, 0, 0
    e.lines = queue.Queue()
    e.gen = 1
    e.up = True
    e._ctl_mode = "batch"
    e.alive = lambda: e.up
    return e


def live_entry(slot, state, generated=0, restored=0, source=None, shared=0):
    now = time.time()
    return {"slot": slot, "request_id": None, "state": state, "prompt_tokens": 1500, "prompt_read": 0,
            "generated": generated, "started": now - 2.0, "first_token": (now - 1.0) if state == "decoding" else None,
            "restored_tokens": restored, "restored_from": source, "_shared_held": shared}


class MinStreamTpsTest(unittest.TestCase):
    def test_configured_min_stream_tps_is_kept_on_the_engine(self):
        self.assertEqual(StrataEngine("x", ["--min-stream-tps", "25"], lazy=True).min_stream_tps, 25.0)

    def test_unset_or_broken_is_zero(self):
        self.assertEqual(StrataEngine("x", [], lazy=True).min_stream_tps, 0.0)
        self.assertEqual(StrataEngine("x", ["--min-stream-tps"], lazy=True).min_stream_tps, 0.0)
        self.assertEqual(StrataEngine("x", ["--min-stream-tps", "auto"], lazy=True).min_stream_tps, 0.0)


class SlotsViewTest(unittest.TestCase):
    def setUp(self):
        self.e = batch_engine(n=4)
        self.e.slot_live[0] = live_entry(0, "decoding", generated=30, restored=1200, source="parked")
        self.e.slot_live[1] = live_entry(1, "reading")
        self.e.slot_busy[2] = True                       # reply ended, the BDONE wait owns it: live is already gone
        self.e.slot_releasing = {2}
        self.e.slot_held[2] = [1, 2, 3]

    def view(self):
        return {s["slot"]: s for s in self.e.slots_view()}

    def test_decoding_slot_carries_the_restored_breakdown(self):
        s = self.view()[0]
        self.assertEqual(s["state"], "decoding")
        self.assertFalse(s["releasing"])
        self.assertEqual(s["restored_tokens"], 1200)
        self.assertEqual(s["restored_from"], "parked")

    def test_a_slot_without_reuse_has_zero_and_none(self):
        s = self.view()[1]
        self.assertEqual(s["state"], "reading")
        self.assertEqual(s["restored_tokens"], 0)
        self.assertIsNone(s["restored_from"])
        self.assertFalse(s["releasing"])

    def test_a_releasing_slot_shows_detaching(self):
        s = self.view()[2]
        self.assertTrue(s["releasing"])
        self.assertEqual(s["state"], "detaching")
        self.assertEqual(s["held_tokens"], 3)

    def test_an_idle_slot_still_lists_the_new_keys(self):
        s = self.view()[3]
        self.assertEqual(s["state"], "idle")
        self.assertFalse(s["releasing"])
        self.assertEqual(s["restored_tokens"], 0)
        self.assertIsNone(s["restored_from"])

    def test_a_disk_restore_event_names_the_source(self):
        self.e._restore_events = (1200, time.time(), None, None)      # ParkDiskLog read back exactly this many
        self.assertEqual(self.view()[0]["restored_from"], "disk")

    def test_a_stale_event_does_not(self):
        self.e._restore_events = (1200, time.time() - self.e.RESTORE_MATCH_S - 1, None, None)
        self.assertEqual(self.view()[0]["restored_from"], "parked")   # the admission's own answer stands

    def test_in_slot_reuse_stays_cache(self):
        self.e.slot_live[0] = live_entry(0, "decoding", generated=30, restored=1200, source="cache", shared=1200)
        self.e._restore_events = (1200, time.time(), None, None)
        self.assertEqual(self.view()[0]["restored_from"], "cache")


class ResumeSinkTest(unittest.TestCase):
    """The admission's RESUME line reaches its own slot card: _control's resume_sink keeps the count per
    exchange (self.reused is shared between the threads that run the slots)."""

    def test_resume_lands_in_the_sink(self):
        e = batch_engine(n=4)
        e.lines.put("RESUME 1200\n")
        e.lines.put("BADM 0 1\n")
        sink = []
        for _ in e._control(threading.Event(), lambda t: None, lines_q=e.lines, resume_sink=sink):
            pass
        self.assertEqual(sink, [1200])
        self.assertEqual(e.reused, 1200)                # the Context gauge's field is untouched

    def test_no_sink_changes_nothing_extra(self):
        e = batch_engine(n=4)
        e.lines.put("RESUME 64\n")
        e.lines.put("BADM 0 1\n")
        for _ in e._control(threading.Event(), lambda t: None, lines_q=e.lines):
            pass
        self.assertEqual(e.reused, 64)


class MetricsLiveTest(unittest.TestCase):
    def setUp(self):
        self.e = batch_engine(args=["--min-stream-tps", "25"], n=4)
        self.e.k_active, self.e.k_max = 2, 8            # what the engine's KCAP line published
        self.e.pool_live_slots, self.e.pool_full_slots = 2100, 2880
        self.e.slot_live[0] = live_entry(0, "decoding", generated=30, restored=1200, source="parked")
        self.e.slot_busy[2] = True
        self.e.slot_releasing = {2}
        self.svc = Service(self.e, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))

    def test_live_payload_carries_the_dynamic_cap_and_pool(self):
        live = self.svc.metrics()["live"]
        self.assertEqual(live["k_active"], 2)
        self.assertEqual(live["k_max"], 8)
        self.assertEqual(live["pool_live_slots"], 2100)
        self.assertEqual(live["pool_full_slots"], 2880)
        self.assertEqual(live["min_stream_tps"], 25.0)

    def test_slots_in_live_carry_the_new_fields(self):
        slots = {s["slot"]: s for s in self.svc.metrics()["live"]["slots"]}
        self.assertEqual(slots[0]["restored_tokens"], 1200)
        self.assertEqual(slots[0]["restored_from"], "parked")
        self.assertFalse(slots[0]["releasing"])
        self.assertTrue(slots[2]["releasing"])
        self.assertEqual(slots[2]["state"], "detaching")

    def test_the_parked_conversation_views_stay_intact(self):
        m = self.svc.metrics()
        for key in ("conversation_cache", "park_disk"):
            self.assertIn(key, m)
        self.assertIn("parked", m["conversation_cache"])
        self.assertIn("restores", m["park_disk"])

    def test_a_non_batch_engine_gets_no_parallel_fields(self):
        e = StrataEngine("x", [], lazy=True)
        e.alive = lambda: False
        svc = Service(e, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        live = svc.metrics()["live"]
        for key in ("k_active", "k_max", "pool_live_slots", "pool_full_slots", "min_stream_tps"):
            self.assertNotIn(key, live)                 # the panel only exists for a batch engine


class BgenRefusalGhostTest(unittest.TestCase):
    """P5-5D.9: a refused BGEN must not leave its live card behind.  The admission sets slot_live[slot] to
    state "reading", prompt_read 0 just before sending the BGEN; while the request backs off or naps on
    slot_cv after a memory refusal, that ghost showed in /metrics as a "Reading (0%)" card and inflated
    the Active count (the "6 / 3 Active" burst)."""

    def test_memory_refusal_clears_slot_live_before_the_nap(self):
        e = batched_engine(8, attached={0})
        e.slot_busy[0] = True                       # one slot in flight: the request naps, never burns the budget
        e.slot_releasing = {0}
        cancel = threading.Event()
        err = []

        def drive():
            try:
                for _ in e.generate_batched([1, 2, 3], 10, None, cancel):
                    pass
            except BaseException as ex:             # noqa: BLE001 - the test reads it
                err.append(ex)

        t = threading.Thread(target=drive, daemon=True)
        t.start()
        deadline = time.time() + 10.0
        while time.time() < deadline and not e.proc.refused:
            time.sleep(0.02)
        self.assertTrue(e.proc.refused, "the fake engine never refused a BGEN")
        refused = e.proc.refused[0]
        time.sleep(0.3)                             # the handler sits in its 1 s slot_cv nap: the ghost window
        self.assertIsNone(e.slot_live[refused])     # cleared before the nap, not only at the retry tail
        svc = Service(e, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        ghosts = [s for s in svc.metrics()["live"]["slots"] if s["state"] == "reading"]
        self.assertEqual(ghosts, [])                # no ghost "Reading (0%)" card in /metrics
        cancel.set()
        with e.slot_cv:
            e.slot_cv.notify_all()
        t.join(timeout=10.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(err, [])                   # the cancelled wait ends clean, no GpuBusy


JS_HARNESS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const start = src.indexOf("function calcConcurrencyAndQueue(live, eng, hw)");
const end = src.indexOf("/* The machine's PCIe link");
if (start < 0 || end < 0) { console.error("markers not found"); process.exit(2); }
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "\u2013" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
function mkEl(tag) {
  let html = "";
  return { tag, id: "", className: "", hidden: false, style: {}, dataset: {}, children: [],
    get innerHTML() { return html; },
    set innerHTML(v) { html = v; this.children.length = 0; },   // like the real DOM: setting it drops the children
    appendChild(c) { this.children.push(c); return c; } };
}
const scenarios = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const out = {};
for (const [name, sc] of Object.entries(scenarios)) {
  const els = {};
  const $ = (id) => (els[id] ||= Object.assign(mkEl("div"), {id}));
  const document = { createElement: (t) => mkEl(t) };
  const f = new Function("fmt", "$", "document",
    src.slice(start, end) + "; return {calcConcurrencyAndQueue, renderConcurrency, slotIsUtilized};");
  const calc = f(fmt, $, document);
  const calcOut = calc.calcConcurrencyAndQueue(sc.frames[0][0], sc.frames[0][1], sc.frames[0][2]);
  for (const [live, eng, hw] of sc.frames) calc.renderConcurrency(live, eng, hw);
  const grid = $("slots-bars-grid");
  out[name] = {
    keys: grid.children.map((el) => el.dataset.key || el.id),
    html: grid.children.map((el) => el.innerHTML),
    calc: calcOut,
    els: Object.fromEntries(Object.entries(els).map(([k, v]) =>
      [k, {text: v.textContent, html: v.innerHTML, cls: v.className, hidden: v.hidden, width: v.style.width}])),
  };
}
console.log(JSON.stringify(out));
"""

IDLE = {"slot": 0, "state": "idle", "releasing": False, "held_tokens": 0, "idle_s": 5.0,
        "restored_tokens": 0, "restored_from": None}


def slots(states):
    return [dict(IDLE, slot=i, **kw) for i, (st, kw) in enumerate(states)]


def base_live(**kw):
    live = {"state": "idle", "queued": 0, "parallel": 4, "running": 0, "outside_slots": 0,
            "waiting": 0, "admitting": 0, "k_active": 2, "k_max": 8,
            "pool_live_slots": 2100, "pool_full_slots": 2880, "min_stream_tps": 25.0,
            "slots": slots([("idle", {})] * 4)}
    return {**live, **kw}


ENG = {"batch_slots": 4}


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class WebDynamicSlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (WEB / "app.js").read_text(encoding="utf-8")
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(JS_HARNESS)
            cls.harness = f.name

    @classmethod
    def tearDownClass(cls):
        Path(cls.harness).unlink(missing_ok=True)

    def run_scenarios(self, scenarios):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(scenarios, f)
            path = f.name
        try:
            done = subprocess.run([NODE, self.harness, str(WEB / "app.js"), path],
                                  capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
        finally:
            Path(path).unlink(missing_ok=True)
        return json.loads(done.stdout)

    def test_nothing_running_shows_one_idle_summary_tile(self):
        out = self.run_scenarios({"s": {"frames": [[base_live(), ENG, {}]]}})["s"]
        self.assertEqual(out["keys"], ["slot-idle-summary"])
        html = out["html"][0]
        self.assertIn("Idle · Auto Cap K=2 of 8", html)
        self.assertIn("SLA ≥ 25 t/s", html)
        self.assertRegex(html, r"Experts 2[,.]100 / 2[,.]880")
        self.assertIn("Ready for direct admission", html)
        self.assertEqual(out["els"]["slots-active-badge"]["text"], "0 / 2 Active (Max 8) · SLA 25 t/s")

    def test_only_utilized_slots_get_a_card(self):
        live = base_live(state="generating", running=1,
                         slots=slots([("decoding", {"state": "decoding", "tok_s": 61.2, "generated": 30,
                                                    "restored_tokens": 1200, "restored_from": "parked"}),
                                      ("idle", {}), ("idle", {}), ("idle", {})]))
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertEqual(out["keys"], ["slot-0"])       # the three standby slots have no card
        self.assertEqual(out["els"]["st-badge-0"]["text"], "Decoding")
        self.assertEqual(out["els"]["slot-decode-0"]["html"], "61.2<small>t/s</small>")

    def test_a_detaching_slot_keeps_its_card(self):
        live = base_live(slots=slots([("idle", {}), ("idle", {}),
                                      ("detaching", {"releasing": True, "held_tokens": 500}), ("idle", {})]))
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertEqual(out["keys"], ["slot-2"])
        self.assertEqual(out["els"]["st-badge-2"]["text"], "Detaching")

    def test_restored_vs_generated_breakdown(self):
        live = base_live(state="generating", running=1,
                         slots=slots([("decoding", {"state": "decoding", "tok_s": 61.2, "generated": 30,
                                                    "restored_tokens": 1200, "restored_from": "parked"}),
                                      ("idle", {}), ("idle", {}), ("idle", {})]))
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertEqual(out["els"]["slot-restored-badge-0"]["text"], "Restored · RAM")   # P5-5D.5: RAM, not "parked"
        self.assertFalse(out["els"]["slot-restored-badge-0"]["hidden"])
        html0 = out["html"][0]
        self.assertLess(html0.index("slot-restored-badge-0"), html0.index("st-badge-0"))  # restored pill sits left
        self.assertIn("is-split", out["els"]["slot-track-0"]["cls"])
        self.assertEqual(out["els"]["slot-restored-fill-0"]["width"], "98%")   # 1200 of 1230
        self.assertEqual(out["els"]["slot-fill-0"]["width"], "2%")             # 30 of 1230
        self.assertIn("has-output", out["els"]["slot-fill-0"]["cls"])          # P5-5D.4: the green tip stays >= 6px
        self.assertEqual(out["els"]["slot-tokens-0"]["text"], "1,200 restored · 30 generated")

    def test_a_restored_read_splits_the_prompt_bar(self):
        # P5-5D.6: a restored conversation reading its next prompt: [dark-green restored 83%]
        # [light-blue newly-read 5%] on the prompt's own scale, "33,134 restored · 35,000 / 40,000 read".
        live = base_live(state="reading", running=1,
                         slots=slots([("reading", {"state": "reading", "prompt_tokens": 40000, "prompt_read": 35000,
                                                   "restored_tokens": 33134, "restored_from": "parked"}),
                                      ("idle", {}), ("idle", {}), ("idle", {})]))
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertIn("is-split", out["els"]["slot-track-0"]["cls"])
        self.assertEqual(out["els"]["slot-restored-fill-0"]["width"], "83%")     # 33,134 of 40,000
        self.assertEqual(out["els"]["slot-fill-0"]["width"], "5%")               # 1,866 newly read of 40,000
        self.assertEqual(out["els"]["slot-fill-0"]["cls"], "slot-tile__bar-fill is-reading has-output")
        self.assertEqual(out["els"]["slot-tokens-0"]["text"], "33,134 restored · 35,000 / 40,000 read")

    def test_a_disk_restore_reads_as_disk(self):
        live = base_live(state="generating", running=1,
                         slots=slots([("decoding", {"state": "decoding", "generated": 5,
                                                    "restored_tokens": 900, "restored_from": "disk"}),
                                      ("idle", {}), ("idle", {}), ("idle", {})]))
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertEqual(out["els"]["slot-restored-badge-0"]["text"], "Restored · Storage")

    def test_a_cache_restore_reads_as_ram(self):
        # P5-5D.9: warm in-slot reuse (restored_from "cache") is KV already in RAM like a parked restore:
        # the badge must read "Restored · RAM", not a bare "Restored" (the slot-0 report that looked unlabelled).
        live = base_live(state="generating", running=1,
                         slots=slots([("decoding", {"state": "decoding", "generated": 30,
                                                    "restored_tokens": 1200, "restored_from": "cache"}),
                                      ("idle", {}), ("idle", {}), ("idle", {})]))
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertEqual(out["els"]["slot-restored-badge-0"]["text"], "Restored · RAM")

    def test_a_waiting_reading_slot_shows_no_prefill_rate(self):
        # P5-5D.9: slot 1 waits at 0 / 900 toks while slot 0 prefills: the waiting card shows "–", never
        # the other slot's live rate (live.prefill_tok_s_mean mirrored onto every reading card).
        live = base_live(state="reading", running=2, prefill_tok_s_mean=800,
                         slots=slots([("reading", {"state": "reading", "prompt_tokens": 900, "prompt_read": 450}),
                                      ("reading", {"state": "reading", "prompt_tokens": 900, "prompt_read": 0}),
                                      ("idle", {}), ("idle", {})]))
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertEqual(out["els"]["slot-prefill-1"]["text"], "–")
        self.assertIn("800", out["els"]["slot-prefill-0"]["html"])   # the slot that IS reading still shows the rate

    def test_the_active_badge_never_shows_a_smaller_denominator(self):
        # P5-5D.9: K_active stepped down to 3 while four admitted streams are still finishing: the badge
        # must never read "4 / 3 Active" - the denominator is at least the streams shown.
        live = base_live(state="generating", running=4, k_active=3,
                         slots=slots([("decoding", {"state": "decoding", "generated": 10}),
                                      ("decoding", {"state": "decoding", "generated": 20}),
                                      ("decoding", {"state": "decoding", "generated": 30}),
                                      ("decoding", {"state": "decoding", "generated": 40})]))
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertEqual(out["els"]["slots-active-badge"]["text"], "4 / 4 Active (Max 8) · SLA 25 t/s")

    def test_a_burst_past_k_active_queues(self):
        live = base_live(state="generating", running=2, waiting=3,
                         slots=slots([("decoding", {"state": "decoding", "generated": 10}),
                                      ("decoding", {"state": "decoding", "generated": 20}),
                                      ("idle", {}), ("idle", {})]))
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertEqual(out["calc"]["queuedCount"], 3)     # counted against k_active=2 (the old maxSlots=4 math
        self.assertEqual(out["calc"]["kActive"], 2)         # gave 1: the cap the engine published, not the wrapper slots)
        self.assertEqual(out["calc"]["kMax"], 8)
        self.assertEqual(out["keys"], ["slot-0", "slot-1", "slot-queue-tile"])
        self.assertEqual(out["els"]["slots-active-badge"]["text"], "2 / 2 Active (Max 8) · SLA 25 t/s")

    def test_queued_requests_never_count_as_active_slots(self):
        # P5-5D.4 regression: live.running also counts requests waiting behind K_active, so the old
        # max(inSlotsCount, rawRunning) read "3 / 1 Active (Max 8)" while one slot card was rendered.
        # The badge must match the cards: one decoding slot -> 1 / 1 Active, the waiters into the queue.
        live = base_live(state="generating", running=3, waiting=2, k_active=1,
                         slots=slots([("decoding", {"state": "decoding", "generated": 10}),
                                      ("idle", {}), ("idle", {}), ("idle", {})]))
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertEqual(out["calc"]["curConcurrency"], 1)
        self.assertGreaterEqual(out["calc"]["queuedCount"], 2)
        self.assertEqual(out["keys"], ["slot-0", "slot-queue-tile"])
        self.assertEqual(out["els"]["slots-active-badge"]["text"], "1 / 1 Active (Max 8) · SLA 25 t/s")

    def test_a_solo_gen_request_gets_a_solo_card(self):
        # P5-5D.2: N=1 runs on the solo GEN path (admit_slot -1): running=1, state generating, every slot idle
        live = base_live(state="generating", running=1, tok_s=42.5, generated=17, reused=900)
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertEqual(out["keys"], ["slot-solo"])          # no idle summary while a solo request is live
        self.assertEqual(out["els"]["st-badge-solo"]["text"], "Decoding")
        self.assertIn("42.5", out["els"]["slot-decode-solo"]["html"])
        self.assertEqual(out["els"]["slot-tokens-solo"]["text"], "900 restored · 17 generated")
        self.assertEqual(out["els"]["slot-restored-badge-solo"]["text"], "Restored · RAM")   # P5-5D.9: in-RAM reuse
        self.assertIn("Slot 1", out["html"][0])               # P5-5D.7: the card is named Slot 1, not "Solo"

    def test_a_solo_read_shows_its_prompt_progress(self):
        live = base_live(state="reading", running=1, prompt_total=900, prompt_read=450)
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertEqual(out["keys"], ["slot-solo"])
        self.assertEqual(out["els"]["st-badge-solo"]["text"], "Reading")
        self.assertEqual(out["els"]["slot-phase-solo"]["text"], "Reading prompt (50%)")

    def test_the_cards_change_dynamically_between_polls(self):
        idle_live = base_live()
        busy_live = base_live(state="reading", running=1,
                              slots=slots([("reading", {"state": "reading", "prompt_tokens": 900, "prompt_read": 450}),
                                           ("idle", {}), ("idle", {}), ("idle", {})]))
        out = self.run_scenarios({"s": {"frames": [[idle_live, ENG, {}],
                                                   [busy_live, ENG, {}],
                                                   [idle_live, ENG, {}]]}})["s"]
        self.assertEqual(out["keys"], ["slot-idle-summary"])   # the last frame is idle again: the card is gone
        self.assertEqual(out["els"]["slot-phase-0"]["text"], "Reading prompt (50%)")  # (updated while it was up)


JS_HARNESS_PD = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const start = src.indexOf("function renderParkDisk(d) {");
const end = src.indexOf("function renderTotals(t) {");
if (start < 0 || end < 0) { console.error("markers not found"); process.exit(2); }
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "\u2013" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const gb = (b, d = 1) => (b == null ? "\u2013" : fmt(b / 1073741824, d));
const facts = () => {};
const since = () => "just now";
const scenarios = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const out = {};
for (const [name, d] of Object.entries(scenarios)) {
  const els = {};
  function mkEl(tag) {
    let html = "";
    return { tag, id: "", className: "", hidden: false, style: {}, dataset: {}, children: [],
      get innerHTML() { return html; },
      set innerHTML(v) { html = v; this.children.length = 0; },
      appendChild(c) { this.children.push(c); return c; } };
  }
  const $ = (id) => (els[id] ||= Object.assign(mkEl("div"), {id}));
  const document = { createElement: (t) => mkEl(t) };
  const f = new Function("fmt", "gb", "facts", "since", "$", "document",
    src.slice(start, end) + "; return {renderParkDisk};");
  f(fmt, gb, facts, since, $, document).renderParkDisk(d);
  out[name] = {text: $("pd-mem-text").textContent, width: $("pd-mem-bar").style.width};
}
console.log(JSON.stringify(out));
"""


def pd_state(**kw):
    d = {"enabled": True, "bytes": 0, "budget_mib": None, "disk_avail_bytes": 0, "entries": 0,
         "skips": 0, "refuses": 0, "fails": 0, "spills": 0, "drops": 0, "restores": 0,
         "v2": False, "chunks": 0, "chunk_bytes": 0, "saved_mib": 0, "min_free_mib": 1024,
         "last_tokens": None}
    return {**d, **kw}


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class WebParkDiskBar(unittest.TestCase):
    """P5-5D.3: the ParkDisk utilization bar fills against the real usable space, never a fake 100%."""

    def run_pd(self, scenarios):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(scenarios, f)
            path = f.name
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(JS_HARNESS_PD)
            harness = f.name
        try:
            done = subprocess.run([NODE, harness, str(WEB / "app.js"), path],
                                  capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
        finally:
            Path(path).unlink(missing_ok=True)
            Path(harness).unlink(missing_ok=True)
        return json.loads(done.stdout)

    def test_nothing_parked_on_a_roomy_disk_is_a_zero_bar(self):
        out = self.run_pd({"s": pd_state(disk_avail_bytes=400 * 1073741824)})["s"]
        self.assertEqual(out["width"], "0%")
        self.assertEqual(out["text"], "0.0 / 400.0 GB available")

    def test_a_small_spill_fills_a_small_share(self):
        out = self.run_pd({"s": pd_state(bytes=512 * 1048576, disk_avail_bytes=400 * 1073741824)})["s"]
        self.assertAlmostEqual(float(out["width"].rstrip("%")), 0.125, places=3)
        self.assertEqual(out["text"], "0.5 / 400.0 GB available")

    def test_no_known_capacity_never_fakes_a_full_bar(self):
        out = self.run_pd({"s": pd_state(bytes=5 * 1073741824, disk_avail_bytes=0)})["s"]
        self.assertEqual(out["width"], "0%")                        # never the old fake 100% green
        self.assertEqual(out["text"], "5.0 GB")

    def test_a_budget_still_wins(self):
        out = self.run_pd({"s": pd_state(bytes=2 * 1073741824, budget_mib=4096,
                                         disk_avail_bytes=400 * 1073741824)})["s"]
        self.assertEqual(out["width"], "50%")
        self.assertEqual(out["text"], "2.0 / 4.0 GB")


JS_HARNESS_RES = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const start = src.indexOf("function renderResFacts(live, eng) {");
const end = src.indexOf("function facts(el, rows) {");
if (start < 0 || end < 0) { console.error("markers not found"); process.exit(2); }
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "\u2013" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "\u2013" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const facts = (el, rows) => { el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join(""); };
const scenarios = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const out = {};
for (const [name, sc] of Object.entries(scenarios)) {
  const els = {};
  function mkEl(tag) {
    let html = "";
    return { tag, id: "", className: "", hidden: false, style: {}, dataset: {}, children: [],
      get innerHTML() { return html; },
      set innerHTML(v) { html = v; this.children.length = 0; },
      appendChild(c) { this.children.push(c); return c; } };
  }
  const $ = (id) => (els[id] ||= Object.assign(mkEl("div"), {id}));
  const document = { createElement: (t) => mkEl(t) };
  const f = new Function("fmt", "ctxfmt", "facts", "$", "document",
    src.slice(start, end) + "; return {renderResFacts};");
  f(fmt, ctxfmt, facts, $, document).renderResFacts(sc.live, sc.eng);
  out[name] = {html: $("res-facts").innerHTML};
}
console.log(JSON.stringify(out));
"""


def res_frame(live, eng):
    return {"live": live, "eng": eng}


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class WebResFactsStrip(unittest.TestCase):
    """P5-5D.8: the Resource Stats card's config strip - context window, decode floor, concurrency cap."""

    def run_res(self, scenarios):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(scenarios, f)
            path = f.name
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(JS_HARNESS_RES)
            harness = f.name
        try:
            done = subprocess.run([NODE, harness, str(WEB / "app.js"), path],
                                  capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
        finally:
            Path(path).unlink(missing_ok=True)
            Path(harness).unlink(missing_ok=True)
        return json.loads(done.stdout)

    def test_the_strip_shows_the_active_config(self):
        out = self.run_res({"s": res_frame(
            {"min_stream_tps": 0.0, "k_active": 3, "k_max": 8},
            {"max_context": 131072, "batch_slots": 8, "kv": "q4_0"})})["s"]
        html = out["html"]
        self.assertIn("Context window", html)
        self.assertIn("128K (131,072 tok)", html)
        self.assertIn("Auto (0 tok/s)", html)          # the default floor reads as auto, not as a bare 0
        self.assertIn("K=3 / 8 slots", html)
        self.assertIn("4-bit", html)

    def test_a_configured_floor_shows_its_number(self):
        out = self.run_res({"s": res_frame(
            {"min_stream_tps": 25.0, "k_active": 3, "k_max": 8},
            {"max_context": 131072})})["s"]
        self.assertIn("25 tok/s", out["html"])
        self.assertNotIn("Auto", out["html"])

    def test_no_dynamic_cap_falls_back_to_the_static_slots(self):
        out = self.run_res({"s": res_frame({"min_stream_tps": 0.0}, {"max_context": 8192, "batch_slots": 4})})["s"]
        self.assertIn("8K (8,192 tok)", out["html"])
        self.assertIn("4 slots", out["html"])
        self.assertNotIn("K=", out["html"])


class IdleDetachHeldTokensTest(unittest.TestCase):
    """P5-5C.9: the engine's `slot N detached idle` log line clears that slot's stale held_tokens (and its
    releasing flag) in /metrics: the engine freed the slot's KV, so the wrapper must stop naming the old
    conversation as held (the stuck 73,809 on a freed slot)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "engine.log"
        self.log.write_text("", encoding="utf-8")
        self.e = batch_engine(n=4)
        self.e.log_path, self.e.log_start = str(self.log), 0
        self.svc = Service(self.e, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))

    def tearDown(self):
        self.tmp.cleanup()

    def slots(self):
        return {s["slot"]: s for s in self.svc.metrics()["live"]["slots"]}

    def write(self, line):
        with self.log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def test_an_idle_detach_clears_the_stale_held_tokens(self):
        self.e.slot_held[1] = list(range(73809))
        self.e.slot_releasing = {1}
        s = self.slots()[1]
        self.assertEqual(s["held_tokens"], 73809)              # before: the stale number on a freed slot
        self.write("strata serve: conversation cache: slot 1 detached idle (73809 tokens to the parked tier)")
        s = self.slots()[1]
        self.assertEqual(s["held_tokens"], 0)
        self.assertFalse(s["releasing"])
        self.assertEqual(s["state"], "idle")

    def test_a_busy_slots_held_stands(self):
        # a fresh admission re-took the slot between the engine's detach and this poll: its held is not stale
        self.e.slot_held[2] = list(range(500))
        self.e.slot_busy[2] = True
        self.write("strata serve: conversation cache: slot 2 detached idle (500 tokens dropped (not parked))")
        self.assertEqual(self.slots()[2]["held_tokens"], 500)

    def test_a_detach_is_applied_once(self):
        self.e.slot_held[3] = list(range(100))
        self.write("strata serve: conversation cache: slot 3 detached idle (100 tokens dropped (not parked))")
        self.assertEqual(self.slots()[3]["held_tokens"], 0)
        self.e.slot_held[3] = list(range(40))                  # a new conversation lands after the detach
        self.assertEqual(self.slots()[3]["held_tokens"], 40)   # later polls must not clear it again

    def test_a_park_line_is_not_a_detach(self):
        self.e.slot_held[0] = list(range(1024))
        self.write("strata serve: conversation cache: parked 1024 tokens (slot 0 idle 30 s); parked=1 bytes=4096 evictions=0")
        self.assertEqual(self.slots()[0]["held_tokens"], 1024)

    def test_the_arena_returned_line_is_the_same_detach_not_a_second_one(self):
        # the engine writes both lines per detach; only the first counts, so a held set after the
        # event's second line is never cleared by a later poll
        self.e.slot_held[1] = list(range(100))
        self.write("strata serve: conversation cache: slot 1 detached idle (100 tokens dropped (not parked))")
        self.assertEqual(self.slots()[1]["held_tokens"], 0)
        self.write("strata serve: conversation cache: slot 1 detached idle: session arena returned (0.31 GiB back)")
        self.e.slot_held[1] = list(range(60))                  # a new conversation lands after the event
        self.assertEqual(self.slots()[1]["held_tokens"], 60)


class ParkDiskAvailTest(unittest.TestCase):
    """P5-5D.3: /metrics' park_disk carries disk_avail_bytes: the filesystem's free space minus the
    keep-free floor, plus what the tier already holds (the bar's denominator)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "engine.log"
        self.e = batch_engine(n=2)
        self.e.log_path, self.e.log_start = str(self.log), 0
        self.svc = Service(self.e, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))

    def tearDown(self):
        self.tmp.cleanup()

    def on_line(self, pd_dir, budget="free-space only", keep=1024):
        self.log.write_text(
            f"strata serve: park disk: ON {pd_dir} (index 0 files, 0.0 GiB, 0 refused, rebuilt in 5 ms; "
            f"budget {budget}, keep-free {keep} MiB)\n", encoding="utf-8")

    def test_avail_is_real_free_minus_the_keep_free_floor(self):
        self.on_line(self.tmp.name)
        d = self.svc.metrics()["park_disk"]
        self.assertTrue(d["enabled"])
        want = max(0, shutil.disk_usage(self.tmp.name).free - 1024 * 1048576)
        self.assertAlmostEqual(d["disk_avail_bytes"], want, delta=100 * 1048576)   # free moves between the two reads

    def test_a_gone_folder_reports_only_what_is_held(self):
        self.on_line(str(Path(self.tmp.name) / "not-a-real-folder"))
        self.assertEqual(self.svc.metrics()["park_disk"]["disk_avail_bytes"], 0)

    def test_a_budgeted_tier_reports_it_too(self):
        self.on_line(self.tmp.name, budget="4096 MiB")
        d = self.svc.metrics()["park_disk"]
        self.assertEqual(d["budget_mib"], 4096)
        self.assertIn("disk_avail_bytes", d)

    def test_a_disabled_tier_has_no_capacity_field(self):
        self.log.write_text("strata serve: park disk: DISABLED\n", encoding="utf-8")
        self.assertNotIn("disk_avail_bytes", self.svc.metrics()["park_disk"])


class PageSource(unittest.TestCase):
    """The dynamic rendering must stay in the page even where node is not installed to run it."""

    @classmethod
    def setUpClass(cls):
        cls.app = (WEB / "app.js").read_text(encoding="utf-8")

    def test_the_visibility_filter(self):
        self.assertIn("function slotIsUtilized(s)", self.app)
        self.assertIn('s.state === "detaching" || s.releasing === true', self.app)
        self.assertIn("slots.filter(slotIsUtilized)", self.app)

    def test_the_k_active_header(self):
        self.assertIn("const kActive = (live && live.k_active > 0) ? live.k_active : maxSlots;", self.app)
        self.assertIn("const kMax = (live && live.k_max > 0) ? live.k_max : maxSlots;", self.app)
        # P5-5D.9: the denominator is at least the streams shown (never "4 / 3 Active")
        self.assertIn("const capShown = Math.max(kActive, curConcurrency);", self.app)
        self.assertIn("${curConcurrency} / ${capShown} Active (Max ${kMax})", self.app)

    def test_the_restored_breakdown_rendering(self):
        self.assertIn('"Restored · RAM"', self.app)
        self.assertIn('"Restored · Storage"', self.app)
        self.assertIn("restored · ${fmt(gen)} generated", self.app)
        self.assertIn("slot-tile__bar-track is-split", self.app)
        # P5-5D.5: the restored pill is emitted before the state pill, matching the bar's left-to-right chunks
        self.assertLess(self.app.index('id="slot-restored-badge-${i}"'), self.app.index('id="st-badge-${i}"'))

    def test_the_idle_summary_tile(self):
        self.assertIn("`Idle · Auto Cap K=${kActive} of ${kMax}`", self.app)
        self.assertIn("Ready for direct admission", self.app)

    def test_the_pool_bar_is_live(self):
        self.assertIn("if (live && live.pool_full_slots > 0)", self.app)
        self.assertIn("slots live", self.app)

    def test_the_parked_restore_views_are_untouched(self):
        # the upstream Strata monitor-tab parked-restore and restored-vs-generated behavior stays wired
        self.assertIn("renderConvCache(m.conversation_cache)", self.app)
        self.assertIn("renderParkDisk(m.park_disk)", self.app)
        self.assertIn('["Parked / restored"', self.app)
        self.assertIn('["Read back"', self.app)

    def test_the_resource_facts_strip_is_wired_into_render_monitor(self):
        # P5-5D.8: renderMonitor populates the strip; the strip's own rendering is exercised under node
        # (WebResFactsStrip), so here only the wiring and the row labels need to hold in the source.
        self.assertIn("renderResFacts(live, eng);", self.app)
        self.assertIn('["Context window"', self.app)
        self.assertIn('"Auto (0 tok/s)"', self.app)
        self.assertIn("`K=${fmt(live.k_active)} / ${fmt(live.k_max)} slots`", self.app)


class PageStyles(unittest.TestCase):
    """P5-5D.4: one colour per stream state across every graph and badge, the bar track flex in every
    state (the prefill bar was pushed out of its own track by the block layout), and the Resource Stats
    card in place of the Context fill gauge."""

    @classmethod
    def setUpClass(cls):
        cls.css = (WEB / "app.css").read_text(encoding="utf-8")
        cls.html = (WEB / "index.html").read_text(encoding="utf-8")

    def rule(self, sel):
        i = self.css.index(sel)
        return self.css[i:self.css.index("}", i)]

    def test_the_bar_track_is_flex_in_every_state(self):
        self.assertIn("display: flex", self.rule(".slot-tile__bar-track {"))
        self.assertIn("flex-shrink: 0", self.rule(".slot-tile__bar-fill {"))
        self.assertNotIn(".slot-tile__bar-track.is-split", self.css)   # flex no longer depends on is-split

    def test_one_colour_per_stream_state(self):
        for sel in (".slot-tile__bar-fill.is-reading", ".slot-tile__badge.tone-reading"):
            self.assertIn("#38bdf8", self.rule(sel))                   # prefill: light blue
        self.assertIn("#7dd3fc", self.rule(".slot-tile__bar-fill.is-reading"))
        for sel in (".slot-tile__bar-fill.is-restored", ".slot-tile__badge.tone-restored"):
            self.assertIn("#15803d", self.rule(sel))                   # restored: dark green, never blue
        self.assertIn("#166534", self.rule(".slot-tile__bar-fill.is-restored"))
        for sel in (".slot-tile__bar-fill.is-generated", ".slot-tile__bar-fill.is-generating",
                    ".slot-tile__badge.tone-generating"):
            self.assertIn("#4ade80", self.rule(sel))                   # decoding: light green
        self.assertIn("#22c55e", self.rule(".slot-tile__bar-fill.is-generated"))
        self.assertIn("min-width: 6px", self.rule(".slot-tile__bar-fill.is-generated.has-output"))

    def test_the_resource_stats_card_replaced_the_gauge(self):
        self.assertIn('<span class="card-title">Resource Stats</span>', self.html)
        for gone in ("ctx-gauge", "ctx-legend", "ctx-note"):
            self.assertNotIn(gone, self.html)

    def test_the_resource_facts_strip_sits_in_the_card(self):
        # P5-5D.8: #res-facts lives inside the Resource Stats card, below the three bars and before the
        # next card, and the CSS keeps it clear of the temperature bar.
        title = self.html.index('<span class="card-title">Resource Stats</span>')
        facts = self.html.index('<dl class="facts" id="res-facts"></dl>')
        temp = self.html.index('id="temp-bar"')
        next_card = self.html.index('class="st-card req-card"')
        self.assertLess(temp, facts)
        self.assertLess(facts, next_card)
        self.assertLess(title, facts)
        self.assertIn("margin-top", self.rule("#res-facts {"))


if __name__ == "__main__":
    unittest.main()
