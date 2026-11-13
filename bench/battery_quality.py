#!/usr/bin/env python3
# battery_quality: substantive greedy Q&A spot-check for arm-to-arm parity comparison.
import json, time, urllib.request, sys
URL = "http://127.0.0.1:30099/v1/completions"
PROMPTS = [
    ("Q: What is 17 times 23?\nA:", 24),
    ("Explain in one sentence why the sky is blue:", 40),
    ("Write a haiku about RAM and caches:", 48),
    ("Translate to French: The server is starting up quickly.\n:", 32),
    ("def fib(n):\n", 64),
    ("Q: Which is larger, 9/10 or 10/11?\nA:", 32),
    ("List three prime numbers between 90 and 110, comma separated:", 32),
    ("The FASTA record starts with which character, and what does GTIN stand for?\nA:", 48),
]
def ask(p, mx):
    body = json.dumps({"model": "q4e", "prompt": p, "max_tokens": mx, "temperature": 0}).encode()
    r = urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}))
    return json.load(r)["choices"][0]["text"].replace("\n", "\\n")
tag = sys.argv[1] if len(sys.argv) > 1 else "arm"
for i, (p, mx) in enumerate(PROMPTS):
    t0 = time.time()
    out = ask(p, mx)
    print("QC%d [%.1fs] %s" % (i, time.time() - t0, out[:110]), flush=True)
print("QC_DONE tag=%s" % tag)
