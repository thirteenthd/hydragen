import urllib.request
import json
import time
import concurrent.futures

url = "http://127.0.0.1:8080/v1/chat/completions"

def run_single(name, prompt_text="Count from 1 to 50 with commas.", max_tokens=60):
    payload = {
        "model": "qwen3.8-flash-next-iq3_xxs",
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True,
    }
    t0 = time.perf_counter()
    t_first = None
    toks = 0
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    delta = json.loads(line[6:])["choices"][0]["delta"]
                    c = delta.get("content") or delta.get("reasoning_content")
                    if c:
                        if t_first is None:
                            t_first = time.perf_counter()
                        toks += 1
                except:
                    pass
    t1 = time.perf_counter()
    ttft = (t_first - t0) if t_first else 0.0
    decode_s = (t1 - t_first) if t_first else 0.001
    spd = toks / decode_s
    return {"name": name, "tokens": toks, "ttft": ttft, "decode_s": decode_s, "tok_s": spd}

print("=== 1. SINGLE STREAM TEST ===")
r = run_single("single_stream")
print(f"Result: {r['tokens']} tokens in {r['decode_s']:.2f}s | TTFT: {r['ttft']:.3f}s | Decode Speed: {r['tok_s']:.1f} tok/s")

print("\n=== 2. CONCURRENT 2-STREAM TEST ===")
t_start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    f1 = ex.submit(run_single, "Stream-A")
    f2 = ex.submit(run_single, "Stream-B")
    r1 = f1.result()
    r2 = f2.result()
t_end = time.perf_counter()
total_toks = r1["tokens"] + r2["tokens"]
wall_time = t_end - t_start
agg_throughput = total_toks / wall_time

print(f"Stream-A: {r1['tokens']} tokens | TTFT: {r1['ttft']:.3f}s | Decode: {r1['tok_s']:.1f} tok/s")
print(f"Stream-B: {r2['tokens']} tokens | TTFT: {r2['ttft']:.3f}s | Decode: {r2['tok_s']:.1f} tok/s")
print(f"Total tokens generated: {total_toks}")
print(f"Wall-clock elapsed:     {wall_time:.2f}s")
print(f"Aggregate throughput:   {agg_throughput:.1f} tok/s")
