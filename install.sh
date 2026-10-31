#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-.}"

echo "=========================================================="
echo "  Installing Strata vLLM PagedMoE Engine Drop-in"
echo "=========================================================="
echo "Target directory: $TARGET_DIR"

if [ ! -d "$TARGET_DIR/serve" ]; then
    echo "[-] Error: '$TARGET_DIR' does not look like a Strata repository (missing 'serve/' folder)."
    echo "    Usage: ./install.sh /path/to/Strata"
    exit 1
fi

echo "[1/4] Installing strata-vllm Python package & vLLM general plugin..."
if [ -x "/home/thirteenthd/vllm-env/bin/pip" ]; then
    /home/thirteenthd/vllm-env/bin/pip install -e "$SCRIPT_DIR" --no-build-isolation --no-deps || true
elif command -v pip3 &>/dev/null; then
    pip3 install -q -r "$SCRIPT_DIR/requirements-vllm.txt" || true
    pip3 install -e "$SCRIPT_DIR" --no-build-isolation --no-deps || true
elif command -v pip &>/dev/null; then
    pip install -q -r "$SCRIPT_DIR/requirements-vllm.txt" || true
    pip install -e "$SCRIPT_DIR" --no-build-isolation --no-deps || true
fi

echo "[2/4] Deploying engine bridge files to $TARGET_DIR/serve/..."
cp -vf "$SCRIPT_DIR/serve/vllm_engine.py" "$TARGET_DIR/serve/"
cp -vf "$SCRIPT_DIR/serve/strata_gguf.py" "$TARGET_DIR/serve/"
cp -vf "$SCRIPT_DIR/serve/libstrata_iq_kernels.so" "$TARGET_DIR/serve/"
chmod +x "$TARGET_DIR/serve/libstrata_iq_kernels.so" 2>/dev/null || true

echo "[3/4] Backing up and updating server and monitor files..."
for f in serve/server.py serve/telemetry.py serve/web/index.html serve/web/app.css serve/web/app.js; do
    if [ ! -f "$TARGET_DIR/$f.orig_backup" ]; then
        echo "  -> Backing up: $TARGET_DIR/$f.orig_backup"
        cp -vf "$TARGET_DIR/$f" "$TARGET_DIR/$f.orig_backup"
    fi
    mkdir -p "$(dirname "$TARGET_DIR/$f")"
    cp -vf "$SCRIPT_DIR/$f" "$TARGET_DIR/$f"
done

echo "[4/4] Deploying launch and uninstall scripts..."
cp -vn "$SCRIPT_DIR/run_vllm.sh" "$TARGET_DIR/" 2>/dev/null || true
chmod +x "$TARGET_DIR/run_vllm.sh" 2>/dev/null || true
cp -vf "$SCRIPT_DIR/uninstall.sh" "$TARGET_DIR/" 2>/dev/null || true
chmod +x "$TARGET_DIR/uninstall.sh" 2>/dev/null || true

echo ""
echo "=========================================================="
echo "  Installation Successful!"
echo "=========================================================="
echo "NOTE: Your original Strata setup is 100% preserved!"
echo "  - To run with vLLM engine:     ./run_vllm.sh (or --engine vllm)"
echo "  - To run stock Strata:         ./run-iq3_xxs.sh (or --engine strata)"
echo "  - To completely revert/remove: ./uninstall.sh"
echo "=========================================================="
