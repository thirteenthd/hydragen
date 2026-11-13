#!/usr/bin/env python3
# Fairness probe: B1 warmup x2, then two B=4 bursts with per-token SSE arrival times.
# Goal: decompose the B4 wall spread (4.3-25.9 s in §33) into admission/prefill stagger
# vs decode-step starvation. Run ON the reference host against :30099.
import json, urllib.request as u, time, concurrent.futures as cf, sys

URL = "http://127.0.0.1:30099/generate"

def mk(p, stream=False):
    d = {"text": p, "sampling_params": {"max_new_tokens": 16, "temperature": 0, "ignore_eos": True}}
    if stream:
        d["stream"] = True
    return json.dumps(d).encode()

def req(b):
    r = u.Request(URL, data=b, headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(u.urlopen(r, timeout=1200))
    return round(time.time() - t0, 1), d.get("text", "")

def req_stream(b, t_wall0):
    r = u.Request(URL, data=b, headers={"Content-Type": "application/json"})
    t0 = time.time()
    arrivals = []          # (t_rel_start, ntok_cumulative)
    prev_n = 0
    for line in u.urlopen(r, timeout=1200):
        s = line.decode().strip()
        if not s.startswith("data:"):
            continue
        try:
            d = json.loads(s[5:])
        except Exception:
            continue
        n = d.get("meta_info", {}).get("completion_tokens", 0)
        if n > prev_n:
            arrivals.append((round(time.time() - t0, 2), n - prev_n))
            prev_n = n
    return t0 - t_wall0, time.time() - t0, arrivals

b = mk("The capital of France is")
for i in (1, 2):
    t, txt = req(b)
    print("B1_WARM%d secs=" % i, t, "text=", repr(txt[:40]), flush=True)

prompts = ("The capital of France is", "The capital of Germany is",
           "The capital of Italy is", "The capital of Spain is")
for burst in (1, 2):
    bodies = [mk(p, stream=True) for p in prompts]
    t0 = time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda b: req_stream(b, t0), bodies))
    wall = time.time() - t0
    print("BURST", burst, "wall=%.1f agg=%.2f tok/s" % (wall, 64 / wall), flush=True)
    for i, (start_lag, dur, arr) in enumerate(res):
        gaps = []
        prev_t = 0.0
        for t, n in arr:
            gaps.append((round(t - prev_t, 2), n))
            prev_t = t
        top = sorted(gaps, reverse=True)[:3]
        print("  S%d first_tok=%.2f last=%.2f ntok=%d biggest_gaps=%s"
              % (i, arr[0][0] if arr else -1, arr[-1][0] if arr else -1,
                 sum(n for _, n in arr), top), flush=True)
