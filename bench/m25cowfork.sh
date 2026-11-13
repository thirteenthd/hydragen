#!/usr/bin/env bash
# CoW arm runner: boot a CoW-engine arm with env knobs, wait for health, run the
# battery QC gate, and leave the arm up for benches. Teardown on next invocation.
# Reconstructed for the product tree from the version used in CHANGES.md 107-111.
#
# usage: m25cowfork.sh <tag> [ENV=VAL ...] [-- --batch N]
#   e.g. m25cowfork.sh forkb4 STRATA_COW_ARENA=1 STRATA_COW_ARENA_SLOTS=9216 \
#        STRATA_COW_SLOT_CELLS=4096 STRATA_COW_FORK=1 -- --batch 4
set -u
TAG="$1"; shift
ENVS=(); BATCH=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then
    shift
    if [ "${1:-}" = "--batch" ]; then BATCH="${2:-}"; shift 2; fi
    continue
  fi
  ENVS+=("$1"); shift
done
HOME_DIR=~; PY=$HOME_DIR/hydragen-live-venv/bin/python
CFG=$(python3 $HOME_DIR/hydragen-experimental/bench/cowcfg3.py "$TAG" ${BATCH:-})

# teardown: TERM, wait for no process AND no listener (fuser is not reliable), KILL escalation
pkill -f "serve/server[.]py" 2>/dev/null; pkill -x strata 2>/dev/null; pkill -x strata-host 2>/dev/null
for i in $(seq 1 15); do
  pgrep -f "serve/server[.]py" >/dev/null || ! (ss -ltn 2>/dev/null | grep -q ":8181 ") && break
  sleep 2
done
pkill -9 -f "serve/server[.]py" 2>/dev/null; sleep 1

# launch with arm env
env ${ENVS[@]+${ENVS[@]}} nohup $PY $HOME_DIR/hydragen-experimental/strata/serve/server.py \
  --engine strata --config "$CFG" --port 8181 \
  > $HOME_DIR/hydragen-experimental/logs/cow-$TAG.out 2>&1 &

# health poll
C=000
for i in $(seq 1 200); do sleep 2; C=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8181/v1/models 2>/dev/null); [ "$C" = "200" ] && break; done
echo "ARM:$TAG HEALTH:$C"
[ "$C" = "200" ] || { tail -30 $HOME_DIR/hydragen-experimental/logs/cow-$TAG.out; exit 1; }
python3 $HOME_DIR/hydragen-experimental/bench/battery_quality_m25.py "$TAG" 2>&1 | grep -E "QC[0-9]|QC_DONE" | head -10
