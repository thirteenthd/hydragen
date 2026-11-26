#!/bin/bash
# P5 Track 5A admission-backpressure regression gate (docs/CHANGES.md §243).
# Same p4f matrix, but the wrapper from the REBASED tree (terminal/serve/server.py,
# 30s slot mem backoff, #1994) instead of the stale hydragen-experimental one that
# produced the §242 503s. Engine binary unchanged (build-p4-arm16; phase5 engine ==
# phase4 engine). Acceptance: bench_quick errors=0 at every N (incl. N>=4), probe222x
# 4/4 parallel + 4/4 serial, wrapper_503=0, refusals==backoffs 1:1, OOM=0.
# ms01 paths; run detached ON the host (setsid) so an interrupted ssh cannot eat it.
# PITFALL (§243): bench_quick.py / probe222x.py take the endpoint from the PORT env
# (defaults 8080/8181) - every bench line below sets PORT=8181 explicitly.
B=/home/thirteenthd/hydragen-experimental   # production tree (restore target)
R=/home/thirteenthd/hydragen-v0221          # rebased tree (phase5/v0.0.1)
G=/home/thirteenthd/g9
PD=$G/park-disk-p5
avail(){ awk '/^MemAvailable:/{print $2}' /proc/meminfo; }

kill_svc(){
  pkill -9 -f "serve/server.py --engine strata" 2>/dev/null
  pkill -9 -f "build-p4-arm16/hydragen" 2>/dev/null
  sleep 5
}

boot(){  # $1 = COW_SHARED value
  local cfg=$G/cfg-p5-$1.json
  python3 - "$1" "$cfg" <<'PY'
import json, sys
sh, cfg = sys.argv[1], sys.argv[2]
B_ = "/home/thirteenthd/hydragen-experimental"
G_ = "/home/thirteenthd/g9"
c = json.load(open(B_ + "/strata-mechdbg.json"))
c["exe"] = "/home/thirteenthd/hydragen-v0221/build-p4-arm16/hydragen"
args, skip = [], False
for a in c["args"]:
    if skip: skip = False; continue
    if a == "--pcie-frac": skip = True; continue
    if a == "--batch": args.append(a); args.append("8"); skip = True; continue
    args.append(a)
c["args"] = args
c["parallel"] = 8
c["env"]["STRATA_COW_SHARED"] = sh
c["env"]["STRATA_PARK_DISK"] = G_ + "/park-disk-p5"
c["env"]["STRATA_BATCH_YIELD_CHUNKS"] = "0"
c["env"]["CUDA_MODULE_LOADING"] = "LAZY"
c["log"] = G_ + "/logs/p5_engine_%s.log" % sh
json.dump(c, open(cfg, "w"), indent=1)
PY
  kill_svc
  for i in $(seq 1 40); do [ "$(avail)" -ge 26000000 ] && break; sleep 5; done
  rm -rf $PD; mkdir -p $PD
  ( cd $R && setsid nohup ~/hydragen-live-venv/bin/python terminal/serve/server.py --engine strata \
      --config $cfg --port 8181 >> $G/logs/p5_serve_$1.log 2>&1 < /dev/null & )
  for i in $(seq 1 150); do
    curl -s -m2 http://127.0.0.1:8181/v1/models | grep -q loaded && break
    pgrep -f "cfg-p5-$1.json" >/dev/null || { echo "SERVE_DIED arm=$1"; return 1; }
    sleep 10
  done
  curl -s -m2 http://127.0.0.1:8181/v1/models | grep -q loaded || { echo "BOOTFAIL arm=$1"; return 1; }
  echo "BOOTED arm=$1 $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
}

runbench(){
  echo "--- bench_quick arm=$1 $(date -u +%FT%TZ)"
  PORT=8181 BQ_NS=1,2,4,8 timeout 3600 python3 $G/bench_quick.py p5_$1 2>&1 | grep -E "N=|error"
  echo "--- probe222x parallel arm=$1 $(date -u +%FT%TZ)"
  PORT=8181 timeout 5400 python3 $G/probe222x.py parallel 9172 2>&1 | grep -E "qc=|PROBE222|Error"
  echo "--- probe222x serial arm=$1 $(date -u +%FT%TZ)"
  PORT=8181 timeout 5400 python3 $G/probe222x.py serial 9171 2>&1 | grep -E "qc=|PROBE222|Error"
}

evidence(){
  echo "=== evidence arm=$1"
  echo "OOM_lines=$(grep -c -i 'out of memory\|CUDA out of memory' $G/logs/p5_engine_$1.log 2>/dev/null)"
  echo "engine_refusals=$(grep -c 'session unavailable' $G/logs/p5_engine_$1.log 2>/dev/null)"
  echo "wrapper_backoffs=$(grep -c 'backing off' $G/logs/p5_serve_$1.log 2>/dev/null)"
  echo "wrapper_503=$(grep -c '503' $G/logs/p5_serve_$1.log 2>/dev/null)"
  echo "zombie_freed=$(grep -c 'marked busy with no request' $G/logs/p5_serve_$1.log 2>/dev/null)"
  grep -E "KV streaming: .*hit VRAM" $G/logs/p5_engine_$1.log | tail -1
  grep -E "joins-at-fetch served so far" $G/logs/p5_engine_$1.log | tail -1
}

mkdir -p $G/logs
for sh in 0 2048 8192; do
  boot $sh || { echo "ARM_FAIL arm=$sh continuing"; continue; }
  runbench $sh
  evidence $sh
done

echo "=== restoring production :8181 $(date -u +%FT%TZ)"
kill_svc
for i in $(seq 1 40); do [ "$(avail)" -ge 26000000 ] && break; sleep 5; done
( cd $B/strata && setsid nohup ~/hydragen-live-venv/bin/python serve/server.py --engine strata \
    --config $B/strata-mechdbg.json --port 8181 >> $B/logs/p5_restore.log 2>&1 < /dev/null & )
for i in $(seq 1 150); do
  curl -s -m2 http://127.0.0.1:8181/v1/models | grep -q loaded && break
  sleep 10
done
curl -s -m2 http://127.0.0.1:8181/v1/models | grep -q loaded && echo "PROD_RESTORED $(date -u +%FT%TZ)" || echo "PROD_RESTORE_FAIL"
echo "=== P5 5A DONE $(date -u +%FT%TZ)"
