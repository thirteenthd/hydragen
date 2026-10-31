#!/usr/bin/env bash
# ============================================================================
# Launch Strata with Hybrid vLLM + Triton PagedMoE Backend
# ============================================================================
if [ -x "/home/thirteenthd/vllm-env/bin/python" ]; then
    PYTHON_BIN="/home/thirteenthd/vllm-env/bin/python"
else
    PYTHON_BIN="$(which python3 || which python)"
fi

if [ -d "/home/thirteenthd/cuda-toolkit" ]; then
    export CUDA_HOME="/home/thirteenthd/cuda-toolkit"
    export PATH="/home/thirteenthd/cuda-toolkit/bin:$PATH"
    export LD_LIBRARY_PATH="/home/thirteenthd/cuda-toolkit/lib64:/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH}"
fi

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
