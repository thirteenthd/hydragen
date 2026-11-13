#!/usr/bin/env bash
# Canonical restart helper: teardown any serve/server[.]py, boot the canonical
# ctx-131k config on :8181, wait for health. Reconstructed for the product tree
# from the version used in CHANGES.md 95-111 (arm arg kept for parity: `canon`).
# usage: m25iso3.sh canon
set -u
HOME_DIR=~; PY=$HOME_DIR/hydragen-live-venv/bin/python
pkill -f "serve/server[.]py" 2>/dev/null; pkill -x strata 2>/dev/null; pkill -x strata-host 2>/dev/null
for i in $(seq 1 15); do
  if ! pgrep -f "serve/server[.]py" >/dev/null && ! (ss -ltn 2>/dev/null | grep -q ":8181 "); then break; fi
  sleep 2
done
pkill -9 -f "serve/server[.]py" 2>/dev/null; sleep 1
nohup $PY $HOME_DIR/hydragen-experimental/strata/serve/server.py \
  --engine strata --config $HOME_DIR/hydragen-experimental/strata-iq3xxs-m25-0213-ctx131k.json \
  --port 8181 > $HOME_DIR/hydragen-experimental/logs/canon-restart.log 2>&1 &
C=000
for i in $(seq 1 200); do sleep 2; C=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8181/v1/models 2>/dev/null); [ "$C" = "200" ] && break; done
echo "CANON HEALTH:$C"
