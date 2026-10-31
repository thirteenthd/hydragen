#!/usr/bin/env bash
# ============================================================================
# Hydragen One-Click Setup Script (Linux / WSL2)
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "      Hydragen Engine Automated Setup & Updater           "
echo "=========================================================="

# Check if this is an update run
if [ "$1" = "--update" ] || [ "$1" = "-u" ]; then
    echo "[+] Updating Hydragen from git repository..."
    if [ -d ".git" ]; then
        git pull || true
    fi
fi

# 1. Check Python (PyTorch/Triton/vLLM require Python >=3.10 and <=3.13)
export PATH="$HOME/.local/bin:$PATH"
PYTHON_CMD=""
CANDIDATES=("python3.12" "python3.11" "python3.10")
for uv_py in "$HOME"/.local/share/uv/python/*/bin/python3; do
    if [ -x "$uv_py" ]; then
        CANDIDATES+=("$uv_py")
    fi
done
CANDIDATES+=("python3" "python")

for cmd in "${CANDIDATES[@]}"; do
    if command -v "$cmd" &>/dev/null; then
        PY_VER=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
        PY_MAJOR=$("$cmd" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || true)
        PY_MINOR=$("$cmd" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || true)
        if [ "$PY_MAJOR" = "3" ] && [ "$PY_MINOR" -ge 10 ] && [ "$PY_MINOR" -le 13 ]; then
            PYTHON_CMD="$cmd"
            echo "[+] Detected Python $PY_VER ($cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[-] Error: Python >=3.10 and <=3.13 is required (PyTorch/vLLM do not support Python 3.14+ yet)."
    exit 1
fi

# 2. Check NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)
    echo "[+] Detected GPU: $GPU_NAME"
else
    echo "[!] Warning: nvidia-smi not found. CUDA GPU is required for acceleration."
fi

# 3. Virtual Environment Setup
VENV_DIR="$SCRIPT_DIR/.venv"
if [ -z "$VIRTUAL_ENV" ]; then
    if [ ! -d "$VENV_DIR" ]; then
        echo "[+] Creating virtual environment in $VENV_DIR..."
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
    echo "[+] Activating virtual environment..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    echo "[+] Using active virtual environment: $VIRTUAL_ENV"
fi

# 4. Install Dependencies & Hydragen
echo "[+] Installing Hydragen and dependencies..."
python -m pip install --upgrade pip
python -m pip install -e .

# 5. Check & Fetch Precompiled Worker Binary
BIN_DIR="$SCRIPT_DIR/bin"
mkdir -p "$BIN_DIR"
if [ ! -f "$BIN_DIR/strata" ]; then
    if [ -n "$STRATA_DIR" ] && [ -f "$STRATA_DIR/engine/strata" ]; then
        echo "[+] Linking existing Strata worker binary from $STRATA_DIR..."
        ln -sf "$STRATA_DIR/engine/strata" "$BIN_DIR/strata"
    elif [ -f "../Strata/engine/strata" ]; then
        echo "[+] Linking local Strata worker binary..."
        ln -sf "$(cd ../Strata && pwd)/engine/strata" "$BIN_DIR/strata"
    else
        echo "[+] Downloading precompiled Strata worker binary (v0.2.9 / engine 0.1.55)..."
        CUDA_VER="cuda12"
        if command -v nvcc &>/dev/null; then
            if nvcc --version | grep -q "release 13"; then
                CUDA_VER="cuda13"
            fi
        elif [ -d "/usr/local/cuda-13" ] || [ -d "/usr/local/cuda-13.1" ]; then
            CUDA_VER="cuda13"
        fi
        echo "[+] Selected CUDA target: $CUDA_VER"
        STRATA_DL_URL="https://github.com/Niko1221/Strata/releases/download/v0.2.9/strata-engine-0.1.55-linux-x86_64-cuda-${CUDA_VER}.tar.gz"
        echo "[+] Fetching $STRATA_DL_URL..."
        if curl -sL --fail "$STRATA_DL_URL" -o "$BIN_DIR/strata.tar.gz"; then
            tar -xzf "$BIN_DIR/strata.tar.gz" -C "$BIN_DIR"
            rm -f "$BIN_DIR/strata.tar.gz"
            echo "[+] Successfully unpacked Strata v0.2.9 worker into $BIN_DIR"
        else
            echo "[!] Auto-download failed. You can copy your compiled 'strata' binary into $BIN_DIR/strata."
        fi
        chmod +x "$BIN_DIR/strata" "$BIN_DIR/strata-vision" 2>/dev/null || true
    fi
fi

# 6. Generate Default Config if Missing
CONFIG_FILE="$SCRIPT_DIR/hydragen.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[+] Generating default hydragen.json template..."
    cat > "$CONFIG_FILE" << 'EOF'
{
  "exe": "bin/strata",
  "args": [
    "--expert-cache", "auto",
    "--prefill", "auto:16384",
    "--spec", "4",
    "--max-context", "131072",
    "--kv", "int8",
    "--vram-reserve-mib", "1152",
    "--pcie-frac", "0.20",
    "--spec-min-p", "0.45",
    "--adapt-async", "1",
    "--kv-resident", "32768",
    "--batch", "2",
    "--batch-mtp"
  ],
  "batch": 2,
  "max_context": 131072,
  "log": "hydragen.log"
}
EOF
fi

chmod +x "$SCRIPT_DIR/run.sh" 2>/dev/null || true

echo ""
echo "=========================================================="
echo "         Hydragen Setup Complete!"
echo "=========================================================="
echo "To start the server:"
echo "  ./run.sh"
echo "  (or: hydragen --config hydragen.json --port 8080 --api-monitor)"
echo ""
echo "To update in the future:"
echo "  ./setup.sh --update"
echo "=========================================================="
