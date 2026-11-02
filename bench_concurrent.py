import urllib.request
import json
import time
import concurrent.futures

prompts = [
    "Write an extensive, highly detailed historical saga and lore of the Golden Key of Eldoria across twelve ancient dynasties.",
    "Write an extensive, highly detailed exploration of the mysterious voyages guided by the Silver Compass of the Northern Seas.",
    "Write an extensive, highly detailed cosmic chronicle of the astronomical secrets locked within the Bronze Astrolabe.",
    "Write an extensive, highly detailed mythological analysis of the terrifying ancient prophecies reflected in the Obsidian Mirror."
]

def run_prompt(idx, prompt, max_tokens=300):
    data = json.dumps({
        "model": "qwen3.8-flash-next-iq3_xxs",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }).encode("utf-8")
    
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8")
            res = json.loads(body)
            t1 = time.time()
            usage = res.get("usage", {})
            comp = usage.get("completion_tokens", 0)
            dur = t1 - t0
            tok_s = comp / dur if dur > 0 else 0
            text = res["choices"][0]["message"]["content"]
            return {"slot": idx, "tokens": comp, "duration": dur, "tok_s": tok_s, "error": None, "snippet": text[:80].strip()}
    except Exception as e:
        return {"slot": idx, "tokens": 0, "duration": time.time() - t0, "tok_s": 0, "error": str(e), "snippet": ""}

def run_concurrency(n_workers, max_tokens=300):
    print("=" * 60)
    print(f"Long-generation Concurrency Stress Test: C={n_workers} (max_tokens={max_tokens})")
    print("=" * 60)
    t_start = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_prompt, i, prompts[i % len(prompts)], max_tokens) for i in range(n_workers)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    t_total = time.time() - t_start
    
    total_tokens = sum(r["tokens"] for r in results)
    agg_tok_s = total_tokens / t_total if t_total > 0 else 0
    
    for r in sorted(results, key=lambda x: x["slot"]):
        status = f"OK ({r['tokens']} tok in {r['duration']:.2f}s, {r['tok_s']:.1f} tok/s)" if not r["error"] else f"ERR: {r['error']}"
        print(f"  Job {r['slot']}: {status}")
            
    print(f"--- Summary C={n_workers} ---")
    print(f"  Wall time:        {t_total:.2f} s")
    print(f"  Total tokens:     {total_tokens}")
    print(f"  Aggregate tok/s:  {agg_tok_s:.2f} tok/s\n")
    return agg_tok_s

if __name__ == '__main__':
    print("Running 4-slot long-generation test (max_tokens=300)...")
    run_concurrency(4, max_tokens=300)
