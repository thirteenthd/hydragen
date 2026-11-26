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
        self.assertEqual(out["els"]["slot-restored-badge-0"]["text"], "Restored · parked")
        self.assertFalse(out["els"]["slot-restored-badge-0"]["hidden"])
        self.assertIn("is-split", out["els"]["slot-track-0"]["cls"])
        self.assertEqual(out["els"]["slot-restored-fill-0"]["width"], "98%")   # 1200 of 1230
        self.assertEqual(out["els"]["slot-fill-0"]["width"], "2%")             # 30 of 1230
        self.assertEqual(out["els"]["slot-tokens-0"]["text"], "1,200 restored · 30 generated")

    def test_a_disk_restore_reads_as_disk(self):
        live = base_live(state="generating", running=1,
                         slots=slots([("decoding", {"state": "decoding", "generated": 5,
                                                    "restored_tokens": 900, "restored_from": "disk"}),
                                      ("idle", {}), ("idle", {}), ("idle", {})]))
        out = self.run_scenarios({"s": {"frames": [[live, ENG, {}]]}})["s"]
        self.assertEqual(out["els"]["slot-restored-badge-0"]["text"], "Restored · disk")

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
        self.assertIn("${curConcurrency} / ${kActive} Active (Max ${kMax})", self.app)

    def test_the_restored_breakdown_rendering(self):
        self.assertIn('"Restored · parked"', self.app)
        self.assertIn('"Restored · disk"', self.app)
        self.assertIn("restored · ${fmt(gen)} generated", self.app)
        self.assertIn("slot-tile__bar-track is-split", self.app)

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


if __name__ == "__main__":
    unittest.main()
