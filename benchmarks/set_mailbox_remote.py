import json
import sys

val = sys.argv[1] if len(sys.argv) > 1 else "1"
path = '/home/thirteenthd/hydragen/hydragen.json'
with open(path) as f:
    d = json.load(f)
d.setdefault('env', {})['HYDRAGEN_ENABLE_MAILBOX'] = val
with open(path, 'w') as f:
    json.dump(d, f, indent=2)
print(f"HYDRAGEN_ENABLE_MAILBOX is now: {val}")
