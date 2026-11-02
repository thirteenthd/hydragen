import urllib.request
import json

def print_monitor_stats():
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/api/requests")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("\nRecent API Monitor Records (with 2,348 Resident Experts):")
            for r in data.get("requests", [])[:6]:
                rid = r.get("id")
                state = r.get("state")
                u = r.get("usage", {})
                t = r.get("timings", {})
                out_tok = u.get("completion_tokens")
                prefill_s = t.get("prompt_per_second", 0)
                dec_s = t.get("predicted_per_second", 0)
                wall = r.get("wallclock_s", 0)
                print(f"  Req {rid}: state={state}, out={out_tok}, prefill={prefill_s:.1f} t/s, decode={dec_s:.1f} t/s, wall={wall:.2f}s")
    except Exception as e:
        print(f"Failed to fetch monitor stats: {e}")

if __name__ == '__main__':
    print_monitor_stats()
