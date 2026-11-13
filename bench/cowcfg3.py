#!/usr/bin/env python3
# Derive a CoW-arm config from the canonical config: swap the engine binary to the
# CoW build, set an arm-specific log path, optionally override --batch.
# Reconstructed for the product tree (original lived in /tmp; functionally identical
# to the version used in CHANGES.md 107-111).
import json, sys
# usage: cowcfg3.py <tag> [batch]  -> writes ~/hydragen-experimental/strata-cow-<tag>.json
tag = sys.argv[1]; batch = sys.argv[2] if len(sys.argv) > 2 else None
home = __import__("os").path.expanduser("~")
src = f"{home}/hydragen-experimental/strata-iq3xxs-m25-0213-ctx131k.json"
cfg = json.load(open(src))
cfg["exe"] = f"{home}/strata-cow/build/strata"
args = cfg["args"]
if batch:
    while "--batch" in args:
        i = args.index("--batch"); del args[i:i+2]
    args += ["--batch", batch]
cfg["log"] = f"{home}/hydragen-experimental/logs/cow-{tag}.log"
out = f"{home}/hydragen-experimental/strata-cow-{tag}.json"
json.dump(cfg, open(out, "w"), indent=1)
print(out)
