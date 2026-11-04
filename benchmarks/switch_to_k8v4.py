import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/thirteenthd/hydragen/hydragen.json"
with open(path) as f:
    d = json.load(f)

args = d.get("args", [])
if "--kv" in args:
    idx = args.index("--kv")
    args[idx + 1] = "k8v4"
else:
    args.extend(["--kv", "k8v4"])

d["args"] = args

with open(path, "w") as f:
    json.dump(d, f, indent=2)

print("Updated hydragen.json to use --kv k8v4.")
