import urllib.request
import json
import time

def query(prompt, max_tokens=50):
    url = "http://192.168.7.152:8080/v1/chat/completions"
    data = {
        "model": "qwen3.8-flash-next-iq3_xxs",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    m_before = json.loads(urllib.request.urlopen("http://192.168.7.152:8080/metrics").read().decode("utf-8"))["totals"]
    
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    t1 = time.perf_counter()
    
    m_after = json.loads(urllib.request.urlopen("http://192.168.7.152:8080/metrics").read().decode("utf-8"))["totals"]
    
    d_offered = m_after["drafts_offered"] - m_before["drafts_offered"]
    d_accepted = m_after["drafts_accepted"] - m_before["drafts_accepted"]
    acc = (d_accepted / d_offered * 100.0) if d_offered > 0 else 0.0
    wall_s = t1 - t0
    msg = resp["choices"][0]["message"]
    text = (msg.get("content") or msg.get("reasoning_content") or "")[:60].replace("\n", " ")
    print(f"Prompt ({len(prompt)} chars) -> Wall: {wall_s:.2f}s | Drafts: {d_accepted}/{d_offered} ({acc:.1f}%) | Snippet: {text}...")
    return acc, d_offered, d_accepted

print("Testing prompts on modelserver01...")
query("Say hello world in Python.", 30)
query("Below is the source code of a binary search tree in C++:\n```cpp\nclass Node {\npublic:\n  int key;\n  Node* left;\n  Node* right;\n};\n```\nPlease reproduce the BST class and add insertion.", 60)
query("Summarize the significance of the treaty of Westphalia in European geopolitical history in 50 words.", 60)
