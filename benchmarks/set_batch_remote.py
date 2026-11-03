import json
import sys

batch_val = int(sys.argv[1]) if len(sys.argv) > 1 else 4
path = '/home/thirteenthd/hydragen/hydragen.json'
with open(path) as f:
    d = json.load(f)

# Update --batch in args
args = d.get("args", [])
if "--batch" in args:
    idx = args.index("--batch")
    args[idx + 1] = str(batch_val)
else:
    args.extend(["--batch", str(batch_val)])
d["args"] = args
d["parallel"] = batch_val

with open(path, 'w') as f:
    json.dump(d, f, indent=2)

print(f"Updated hydragen.json: --batch {batch_val}, parallel {batch_val}")
