import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/thirteenthd/hydragen/hydragen.json"
with open(path) as f:
    d = json.load(f)

args = d.get("args", [])

# Remove --kv-grow if present
args = [a for a in args if a != "--kv-grow"]

# Set --batch 2
if "--batch" in args:
    idx = args.index("--batch")
    args[idx + 1] = "2"
else:
    args.extend(["--batch", "2"])

# Set --vram-reserve-mib 500
if "--vram-reserve-mib" in args:
    idx = args.index("--vram-reserve-mib")
    args[idx + 1] = "500"
else:
    args.extend(["--vram-reserve-mib", "500"])

d["args"] = args
d["parallel"] = 2
d["vram_reserve_mib"] = 500

with open(path, "w") as f:
    json.dump(d, f, indent=2)

print("Updated hydragen.json to 2-way batching with 500 MiB reserve for maximum expert residency.")
