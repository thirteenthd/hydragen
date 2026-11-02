# SPDX-License-Identifier: Apache-2.0
"""Test live HTTP chat completion on localhost:8080."""
import json
import time
import urllib.request

t0 = time.time()
payload = {
    "model": "qwen3.8-flash-next-iq3_xxs",
    "messages": [{"role": "user", "content": "What is the capital of France? Reply in 5 words or less."}],
    "max_tokens": 50,
}
req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))

dt = time.time() - t0
content = res["choices"][0]["message"]["content"]
tokens = res.get("usage", {}).get("completion_tokens", 0)
print(f"Content: {content.strip()}")
print(f"Tokens: {tokens}, Elapsed: {round(dt, 2)}s, Speed: {round(tokens / max(0.01, dt), 1)} tok/s")
print("LIVE INFERENCE SUCCESSFUL!")
