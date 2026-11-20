#!/usr/bin/env python3
# kv-resident QUALITY bar battery: planted-fact needle recall at escalating depths,
# run against a live serve endpoint. Usage: needle_battery.py [base_url]
import json, sys, urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")

def fill(n_words, seed, before_fact, fact):
    # deterministic filler; fact is planted at the END of the filler
    body = " ".join(f"zq{i} qtx" for i in range(n_words))
    return before_fact + "\n" + body + "\n" + fact

NEEDLES = [
    # (filler_iterations, planted_fact, question) — ~8.1 tokens per iteration
    (2500,  "CODEX NOTE: the maintenance window token for rack 7 is FLAMINGO-7741.",
            "What is the maintenance window token for rack 7? Answer with just the token."),
    (6000,  "CODEX NOTE: the maintenance window token for rack 7 is MARBLE-2213.",
            "What is the maintenance window token for rack 7? Answer with just the token."),
    (12000, "CODEX NOTE: the maintenance window token for rack 7 is OSPREY-9152.",
            "What is the maintenance window token for rack 7? Answer with just the token."),
    (22000, "CODEX NOTE: the maintenance window token for rack 7 is QUARTZ-6038.",
            "What is the maintenance window token for rack 7? Answer with just the token."),
]

def ask(prompt, max_tokens=400):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}],
                         "max_tokens": max_tokens, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                d = json.loads(r.read())
            ch = (d.get("choices") or [{}])[0]
            content = (ch.get("message") or {}).get("content")
            if content is None:
                content = (ch.get("message") or {}).get("reasoning_content") or ""
            return content.strip(), d.get("usage", {}), ch.get("finish_reason")
        except Exception as e:
            if attempt: return "", {}, f"error:{e}"
    return "", {}, "unreachable"

score = 0
for w, fact, q in NEEDLES:
    prompt = fill(w, w, "You are a meticulous archivist. Context follows.", fact + "\n\n" + q)
    ans, usage, finish = ask(prompt)
    tok = fact.split()[-1].rstrip(".")
    ok = tok in ans.upper()
    score += ok
    print(f"depth={w:>7}w prompt={usage.get('prompt_tokens','?'):>7} finish={finish} pass={ok} want={tok} got={ans[:40]!r}")

def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        return json.loads(r.read())

m = metrics()
eng = m.get("engine", {})
park = m.get("park_disk", {})
conv = m.get("conversation_cache", {})
ov = [k for k in (eng.get("kv_overflow"), conv.get("overflow")) if k]
print(f"battery {score}/4  engine={eng.get('engine')} kv_resident={eng.get('kv_resident')} "
      f"overflow_fields={ {k: v for k, v in list(eng.items()) + list(conv.items()) if 'overflow' in k} }")
print(f"park_disk: entries={park.get('entries')} spills={park.get('spills')} restores={park.get('restores')} refuses={park.get('refuses')}")
sys.exit(0 if score >= 3 else 1)
