import json, random, time, urllib.request, sys
MODEL = "qwen3.8-flash-next-iq3_xxs-m25"; URL = "http://127.0.0.1:8181/v1/chat/completions"
UNIT = ("You are the operations agent for a distributed systems team. Follow the runbook exactly. "
        "When asked for diagnostics, cite the numbered procedure steps you used. Never invent log lines. ")
SHARED = UNIT * 1900   # ~66.5k-token shared head (the sibling-subagent case at 100k depth)
WORDS = ("kernel allocator thread segment page fault eviction window quota radix tensor expert router "
         "batch slot stream pinned cache token layer gradient weight matrix latency queue dispatch commit").split()
def mk_prompt(seed):
    rnd = random.Random(seed)
    parts = [SHARED, f"\nRunbook section {seed}: historical incident notes follow. \n"]
    for i in range(1200):
        line = " ".join(rnd.choice(WORDS) for _ in range(14))
        parts.append(f"[{seed}-{i:04d}] {line}\n")
    parts.append("\nUsing the runbook rules above, answer directly. List every prime number between 200 and 900 "
                 "in ascending order, one per line, with no other commentary.")
    return "".join(parts)
def ask(p, mt):
    t0 = time.time()
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": mt,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=3600))
    return (time.time() - t0, r["choices"][0]["message"]["content"])
def mem():
    d = dict(l.split(":") for l in open("/proc/meminfo") if ":" in l)
    gb = lambda k: int(d[k].split()[0]) // 1048576
    return f"avail={gb('MemAvailable')}GiB"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TAG = sys.argv[2] if len(sys.argv) > 2 else "ladder"
prompts = {s: mk_prompt(900 + s) for s in range(N)}
print(f"LADDER100k {TAG}: {N} sibling conversations, ~99k tokens each, shared head", flush=True)
for s in range(N):
    w, c = ask(prompts[s], 800)
    ok = "211" in c and "223" in c and "887" in c
    print(f"S{s}: wall={w:.1f}s qc={ok} {mem()}", flush=True)
bad = 0
for s in range(N):  # restore phase: parked sibling borrowed back; a corrupt view or budget eviction shows as wrong recall
    w, c = ask(prompts[s] + "\nNow list ONLY the first five primes from your answer above.", 40)
    ok = "211" in c and "223" in c and not "887" in c
    if not ok: bad += 1
    print(f"R{s}: wall={w:.1f}s qc={ok}", flush=True)
print(f"DONE {TAG}: restore_qc_fail={bad}", flush=True)
