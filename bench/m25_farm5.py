import json, random, time, urllib.request, concurrent.futures as cf, sys
MODEL = "qwen3.8-flash-next-iq3_xxs-m25"; URL = "http://127.0.0.1:8181/v1/chat/completions"
SHARED = ("You are the operations agent for a distributed systems team. Follow the runbook exactly. "
          "When asked for diagnostics, cite the numbered procedure steps you used. Never invent log lines. " * 25)
WORDS = ("kernel allocator thread segment page fault eviction window quota radix tensor expert router "
         "batch slot stream pinned cache token layer gradient weight matrix latency queue dispatch commit").split()
def mk_prompt(seed):
    rnd = random.Random(seed)
    parts = [SHARED, f"\nRunbook section {seed}: historical incident notes follow. \n"]
    for i in range(900):
        line = " ".join(rnd.choice(WORDS) for _ in range(14))
        parts.append(f"[{seed}-{i:04d}] {line}\n")
    parts.append("\nUsing the runbook rules above, answer directly. List every prime number between 200 and 900 in ascending order, one per line, with no other commentary.")
    return "".join(parts)
def ask(p, mt):
    t0 = time.time()
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": mt,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=3600))
    ti = r.get("timings", {})
    return (time.time() - t0, ti.get("predicted_per_second", 0), r["usage"].get("completion_tokens", 0),
            ti.get("draft_n_accepted", None), r["choices"][0]["message"]["content"])
TAG = sys.argv[1] if len(sys.argv) > 1 else "arm"
# FARM shape: every concurrent stream asks with the SAME 22.5k-token conversation (the agent-farm case).
for n in (1, 2, 3, 4):
    sid = 700 + n            # one farm conversation per level; C1 warms it, later levels fork/restore it
    prompt = mk_prompt(sid)
    prompts = [prompt] * n
    t0 = time.time()
    with cf.ThreadPoolExecutor(n) as ex:
        futs = [ex.submit(ask, p, 800) for p in prompts]
        res = [f.result() for f in futs]
    wall = time.time() - t0
    tot = sum(r[2] for r in res)
    decs = [round(r[1], 1) for r in res]
    acc = [r[3] for r in res]
    ok = all(("211" in r[4] and "223" in r[4] and "887" in r[4]) for r in res)   # head/mid/tail primes of the 200-900 list
    print("C%d: wall=%.1fs agg=%.1f tok/s decode=%s accepted=%s ntok=%s qc_primes=%s" %
          (n, wall, tot / wall, decs, acc, [r[2] for r in res], ok), flush=True)
