import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/thirteenthd/hydragen/hydragen.json"
with open(path) as f:
    d = json.load(f)

args = d.get("args", [])
if "--kv-grow" not in args:
    args.append("--kv-grow")

d["args"] = args
with open(path, "w") as f:
    json.dump(d, f, indent=2)

print(f"Config at {path} updated with --kv-grow successfully.")
