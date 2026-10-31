#!/usr/bin/env bash
# ============================================================================
# Launch Strata with vLLM PagedMoE Backend
# ============================================================================
PYTHON_BIN="$(which python3 || which python)"

CONFIG_FILE="strata-iq3_xxs.json"
if [ ! -f "$CONFIG_FILE" ]; then
    # Look for any existing strata-*.json config in cwd
    FOUND=$(ls strata-*.json 2>/dev/null | head -n 1)
    if [ -n "$FOUND" ]; then
        CONFIG_FILE="$FOUND"
    fi
fi

echo "[strata-vllm] Starting server using config: $CONFIG_FILE"
exec "$PYTHON_BIN" -m serve.server \
    --engine vllm \
    --config "$CONFIG_FILE" \
    --port 8080 \
    --api-monitor \
    --no-api-key "$@"
