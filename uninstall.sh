#!/usr/bin/env bash
set -e

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo "  Reverting Strata to Original Stock Installation"
echo "=========================================================="
echo "Target directory: $TARGET_DIR"

echo "[1/3] Restoring original server and web files from backup..."
for f in serve/server.py serve/telemetry.py serve/web/index.html serve/web/app.css serve/web/app.js; do
    if [ -f "$TARGET_DIR/$f.orig_backup" ]; then
        echo "  -> Restoring $f"
        cp -vf "$TARGET_DIR/$f.orig_backup" "$TARGET_DIR/$f"
        rm -f "$TARGET_DIR/$f.orig_backup"
    elif [ -d "$TARGET_DIR/.git" ]; then
        (cd "$TARGET_DIR" && git checkout -- "$f" 2>/dev/null || true)
    fi
done

echo "[2/3] Removing vLLM drop-in bridge files..."
rm -f "$TARGET_DIR/serve/vllm_engine.py"
rm -f "$TARGET_DIR/serve/strata_gguf.py"
rm -f "$TARGET_DIR/serve/libstrata_iq_kernels.so"
rm -f "$TARGET_DIR/run_vllm.sh"

echo "[3/3] Cleaning bytecode caches..."
rm -rf "$TARGET_DIR/serve/__pycache__"

echo ""
echo "=========================================================="
echo "  Revert Complete! Strata is back to its original state."
echo "=========================================================="
