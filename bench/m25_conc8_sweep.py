#!/usr/bin/env python3
# m25_conc8_sweep.py - CLIENT-SIDE deep-context concurrency sweep vs the live canonical server.
#
# Every number is measured at the client wall clock around SSE streaming (first content
# chunk arrival = TTFT; decode rate from streamed piece arrival times). Engine-side
# timers are deliberately NOT used - they hide queueing, network and park/restore waits.
#
# Arms:
#   natural : C simultaneous single-shot sessions, each a UNIQUE deep prompt (~100k tok).
#             Worst case for prefix caching: zero sharing between sessions.
#   harness : C agent-harness-style sessions x T turns. Turn 1 = shared system head
#             (~66k tok, byte-identical across sessions) + a unique history; later turns
#             append the assistant reply + ~1k new tokens, with randomized "think" gaps.
#             Reuse is expected via the parked conversation cache; evictions show up as
#             re-prefill spikes IN THE CLIENT NUMBERS, which is the point.
#
# A host-RAM guard (MemAvailable) pauses NEW request starts locally below a floor - it
# never touches the engine. Needle QC: every prompt embeds a unique code the model must
# repeat; mismatches are counted so depth/contamination failures are visible.
#
# Usage: python3 m25_conc8_sweep.py --arm natural --C 4 [--target 100000] [--turns 6] [--quick]
import argparse, json, os, random, statistics as st, sys, threading, time, urllib.request

CFG_PATH = os.path.expanduser(os.environ.get(
    "HG_CFG", "~/hydragen-experimental/strata-iq3xxs-m25-0213-ctx131k-lan-cow.json"))
CFG = json.load(open(CFG_PATH))
KEY = CFG.get("api_key", "")
PORT = CFG.get("port", 8181)
MODEL = CFG.get("model_name", "qwen3.8-flash-next-iq3_xxs-m25")
BASE = "http://127.0.0.1:%d" % PORT
HDR = ("You are the operations agent for distributed systems team {seed}. The runbook text follows. "
       "Rule 1: when asked for this team's incident code, repeat it exactly as written. "
       "Rule 2: only the code stated in this team's runbook is valid; never invent codes. ")
WORDS = ("kernel allocator thread segment page fault eviction window quota radix tensor expert router "
         "batch slot stream pinned cache token layer gradient weight matrix latency queue dispatch commit "
         "prefill ring anchor resident mirror journal shard fence barrier drain lease extent").split()

LOW_KB, HIGH_KB = 1_500_000, 2_500_000  # MemAvailable guard: pause under LOW, resume over HIGH
guard_lock = threading.Lock()
guard = {"allow": True, "pauses": 0, "min_avail_kb": 1 << 40}

def mem_avail_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return 1 << 40

def guard_thread(stop):
    while not stop.is_set():
        kb = mem_avail_kb()
        with guard_lock:
            guard["min_avail_kb"] = min(guard["min_avail_kb"], kb)
            if kb < LOW_KB and guard["allow"]:
                guard["allow"] = False; guard["pauses"] += 1
                print("[guard] MemAvailable %.1f GiB low - holding new requests" % (kb/2**20), flush=True)
            elif kb > HIGH_KB and not guard["allow"]:
                guard["allow"] = True
                print("[guard] resumed (MemAvailable %.1f GiB)" % (kb/2**20), flush=True)
        time.sleep(3)

def wait_guard():
    while True:
        with guard_lock:
            if guard["allow"]:
                return
        time.sleep(2)

def api_get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def ask_once(content):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": 8, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    return api_post("/v1/chat/completions", body)

def api_post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)

def mk_text(seed, nlines):
    rnd = random.Random(seed * 7919 + 13)
    parts = []
    for i in range(nlines):
        parts.append("[%d-%05d] %s\n" % (seed, i, " ".join(rnd.choice(WORDS) for _ in range(14))))
    return "".join(parts)

def mk_prompt(seed, nlines, code_pos, q):
    rnd = random.Random(seed * 7919 + 13)
    code = "INC%d%s" % (seed, "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(7)))
    code_line = ("[%d-rule] This team's incident code is %s. It appears exactly once; "
                 "repeat it verbatim when asked.\n" % (seed, code))
    parts = [HDR.format(seed=seed), mk_text(seed, nlines)]
    parts.insert(1 + min(code_pos, len(parts) - 1), code_line)
    parts.append("\nAnswer directly, no preamble: first line = this team's incident code exactly. "
                 "Then a numbered list of %d diagnostic steps. For EACH step write 3-4 full sentences "
                 "explaining what to check, what a failure looks like, and the exact remediation. "
                 "Be exhaustive; do not stop early.\n" % q)
    return "".join(parts), code

def calibrate(target, seed):
    p2 = mk_prompt(seed, 300, 150, 1)[0]; p3 = mk_prompt(seed, 1000, 500, 1)[0]
    t2 = ask_once(p2)["usage"]["prompt_tokens"]; t3 = ask_once(p3)["usage"]["prompt_tokens"]
    per = (t3 - t2) / 700.0
    return max(50, int((target - 120) / per)), t3, per

def sse(messages, max_tokens):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    wait_guard()
    t0 = time.time(); t_first = None; pieces = []; usage = None
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = (d.get("choices") or [{}])[0]
            piece = (ch.get("delta") or {}).get("content") or ch.get("text") or ""
            if piece:
                if t_first is None:
                    t_first = time.time()
                pieces.append((time.time(), piece))
    t_done = time.time()
    ntok = usage.get("completion_tokens", len(pieces)) if usage else len(pieces)
    gaps = [b[0]-a[0] for a, b in zip(pieces, pieces[1:])]
    content = "".join(p for _, p in pieces)
    return {"t_start": t0, "ttft": (t_first - t0) if t_first else None,
            "wall": t_done - t0, "decode": (ntok - 1) / max(1e-6, (pieces[-1][0] - pieces[0][0])) if len(pieces) > 1 else None,
            "max_gap": max(gaps) if gaps else None, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached_tokens": (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0),
            "completion_tokens": ntok, "content_head": content[:120]}

def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = min(len(xs)-1, int(round(p/100*(len(xs)-1))))
    return xs[k]

def classify(r, arm):
    """c/w/d taxonomy from EXACT reuse counts (usage.prompt_tokens_details.cached_tokens),
    not TTFT bands (queueing behind other 100k prefills corrupts TTFT at C>=4).
      cold               : first turn of a session (or any natural request) - full prefill paid
      warm               : later turn, prefix reuse landed (>=1k tokens cached)
      deep_warm_reprefill: later turn but reuse missed (parked eviction / lost prefix)
      warm_hit           : natural arm only - unexpected cross-run prefix hit (artifact flag)
    """
    cached = r.get("cached_tokens") or 0
    if arm == "natural":
        return "warm_hit" if cached > 1000 else "cold"
    if r.get("turn") == 0:
        return "cold"
    return "warm" if cached > 1000 else "deep_warm_reprefill"

def run_natural(C, target, out):
    nlines, _, _ = calibrate(target, 1000)
    print("[natural] C=%d calibrated nlines=%d" % (C, nlines), flush=True)
    res = {}
    def w(c):
        seed = 7000 + 100 * C + c
        prompt, code = mk_prompt(seed, nlines, (c*997) % nlines, 12)
        r = sse([{"role": "user", "content": prompt}], 2048)
        r["needle_ok"] = code.lower() in r["content_head"].lower(); r["code"] = code
        res[c] = r
    ts = [threading.Thread(target=w, args=(c,)) for c in range(C)]
    t0 = time.time()
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.time() - t0
    return res, wall

def run_harness(C, turns, out):
    head_lines, _, _ = calibrate(66000, 0)
    hist_lines, _, _ = calibrate(30000, 2000)
    head_text = mk_text(0, head_lines)          # byte-identical across sessions
    print("[harness] C=%d turns=%d head_lines=%d hist_lines=%d" % (C, turns, head_lines, hist_lines), flush=True)
    sess = {}
    def w(c):
        seed = 9000 + 100 * C + c
        prompt, code = mk_prompt(seed, hist_lines, (c*613) % hist_lines, 4)
        msgs = [{"role": "system", "content": head_text}, {"role": "user", "content": prompt}]
        rows = []
        for t in range(turns):
            if t > 0:
                rnd = random.Random(seed * 31 + t)
                msgs.append({"role": "user", "content": "Follow-up %d context:\n%s\nSame rules; restate the code first, then 3 detailed next steps (3-4 sentences each; do not stop early)."
                                                    % (t, mk_text(seed * 100 + t, 70))})
            r = sse(msgs, 1600)
            r["turn"] = t
            rows.append(r)
            if t == 0:
                r["needle_ok"] = code.lower() in r["content_head"].lower()
            msgs.append({"role": "assistant", "content": r["content_head"] + ("..." if r["completion_tokens"] > 40 else "")})
            wait_guard()
            time.sleep(4 + random.Random(seed + t).random() * 11)
        sess[c] = rows
    ts = [threading.Thread(target=w, args=(c,)) for c in range(C)]
    t0 = time.time()
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.time() - t0
    return sess, wall

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["natural", "harness"])
    ap.add_argument("--C", type=int, required=True)
    ap.add_argument("--target", type=int, default=100000)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.target = 12000; a.turns = 2
    tag = "%s_C%d" % (a.arm, a.C)
    out_path = os.path.join(os.path.expanduser(os.environ.get(
        "HG_OUT", "~/hydragen-experimental")), "conc8_%s.jsonl" % tag)
    m0 = api_get("/metrics")
    stop = threading.Event(); g = threading.Thread(target=guard_thread, args=(stop,)); g.start()
    t_wall0 = time.time()
    if a.arm == "natural":
        data, wall = run_natural(a.C, a.target, out_path)
        rows = list(data.values())
    else:
        data, wall = run_harness(a.C, a.turns, out_path)
        rows = [r for rs in data.values() for r in rs]
    stop.set(); g.join()
    with open(out_path, "w") as f:
        json.dump(data, f, default=str)
    ttfts = [r["ttft"] for r in rows if r["ttft"] is not None]
    decs = [r["decode"] for r in rows if r["decode"]]
    tot_out = sum(r["completion_tokens"] for r in rows)
    tot_in = sum(r["prompt_tokens"] or 0 for r in rows)
    needles = sum(1 for r in rows if r.get("needle_ok") is not None)
    needle_ok = sum(1 for r in rows if r.get("needle_ok"))
    def rnd(x, d=2):
        return round(x, d) if x is not None else None
    per = {}
    for r in rows:
        k = classify(r, a.arm)
        d = per.setdefault(k, {"n": 0, "out": 0, "ttft": [], "decode": [], "prefill": 0})
        d["n"] += 1; d["out"] += r["completion_tokens"]
        d["prefill"] += (r["prompt_tokens"] or 0) - (r.get("cached_tokens") or 0)
        if r["ttft"] is not None: d["ttft"].append(r["ttft"])
        if r.get("decode"): d["decode"].append(r["decode"])
    per_class = {}
    for k, v in per.items():
        per_class[k] = {"n": v["n"], "out_tokens": v["out"], "prefill_tokens": v["prefill"],
                        "ttft_p50": rnd(pct(v["ttft"], 50)), "ttft_p95": rnd(pct(v["ttft"], 95)),
                        "decode_p50": rnd(pct(v["decode"], 50), 1),
                        "decode_min": rnd(min(v["decode"]), 1) if v["decode"] else None}
    summary = {"tag": tag, "wall_s": round(wall, 1), "requests": len(rows),
               "prompt_tokens": tot_in, "completion_tokens": tot_out,
               "agg_decode_tok_s": round(tot_out / wall, 1),
               "ttft_p50": rnd(pct(ttfts, 50)), "ttft_p95": rnd(pct(ttfts, 95)),
               "ttft_max": rnd(max(ttfts)) if ttfts else None,
               "decode_p50": rnd(pct(decs, 1), 1), "decode_min": rnd(min(decs), 1) if decs else None,
               "decode_p10": rnd(pct(decs, 10), 1), "decode_p90": rnd(pct(decs, 90), 1),
               "n_decode_samples": len(decs),
               "max_stall_gap": round(max([r["max_gap"] or 0 for r in rows]), 2),
               "needle": "%d/%d" % (needle_ok, needles),
               "per_class": per_class,
               "guard": {"pauses": guard["pauses"], "min_memavail_gib": round(guard["min_avail_kb"]/2**20, 1)}}
    with open(out_path.replace(".jsonl", ".summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("SUMMARY " + json.dumps(summary), flush=True)

if __name__ == "__main__":
    main()
