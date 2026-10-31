#!/usr/bin/env bash
# ============================================================================
# Hydragen Server Launcher (Linux / WSL2)
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
fi

if [ -d "$SCRIPT_DIR/bin" ]; then
    export LD_LIBRARY_PATH="$SCRIPT_DIR/bin:${LD_LIBRARY_PATH:-}"
    export PATH="$SCRIPT_DIR/bin:$PATH"
fi

CONFIG_FILE="hydragen.json"
if [ ! -f "$CONFIG_FILE" ]; then
    if [ -f "strata-iq3_xxs.json" ]; then
        CONFIG_FILE="strata-iq3_xxs.json"
    fi
fi

echo "[hydragen] Starting server using config: $CONFIG_FILE"
exec hydragen --config "$CONFIG_FILE" --port 8080 --api-monitor "$@"
