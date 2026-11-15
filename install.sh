#!/usr/bin/env bash
# HydraGen installer (Linux, NVIDIA).
# Builds the vendored engine FROM SOURCE - prebuilt Strata release binaries do not contain
# HydraGen's CoW chain (expert-slot refcounting + admission fork, host-tier parked-session
# dedup, park-disk v2 lineage tier + deep resume; see docs/FINAL_REPORT.md) - and sets up the
# Python venv for the serving front. The vendored engine is the canonical cow-m3b branch
# (M-C1c..M-C10c): this build IS the canonical engine, no arm switches.
#
# Needs: an NVIDIA driver, a CUDA toolkit (nvcc), a C++ toolchain, Python 3.10+.
#        CUDA toolkit: https://developer.nvidia.com/cuda-downloads
#        cmake + ninja come from pip (engine/requirements-build.txt); nothing else to install.
#
#   ./install.sh                       # sm_86 (RTX 3060 Ti / 3080 Ti - the target cards)
#   HYDRAGEN_CUDA_ARCH=89-real ./install.sh          # RTX 40-series
#   HYDRAGEN_CUDA_ARCH="86-real;89-real" ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${HYDRAGEN_CUDA_ARCH:-86-real}"

# Model blob location: --model-dir DIR (default: <this repo>/models). The generated
# config's --pack/--native/--ple-gguf/--mtp paths are rewritten to live under DIR.
# The expert-routing profile needs no download - it ships in this repo at
# engine/data/expert-profile.bin. See docs/MODELS.md for sources and the
# vendored Strata downloader (engine/setup.sh / engine/SETUP.bat).
MODEL_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model-dir) MODEL_DIR="${2:?--model-dir needs a directory}"; shift 2 ;;
    *) echo "install.sh: unknown argument: $1 (try --model-dir DIR)" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL_DIR" ] || MODEL_DIR="$PWD/models"
case "$MODEL_DIR" in /*) ;; *) MODEL_DIR="$PWD/$MODEL_DIR" ;; esac

# ------------------------------------------------------------------ CUDA
CUDA=""
for c in $(ls -d /usr/local/cuda /usr/local/cuda-* /opt/cuda 2>/dev/null | sort -V) "${CUDA_HOME:-}"; do
  [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && CUDA="$c"
done
if [ -z "$CUDA" ] && command -v nvcc >/dev/null 2>&1; then
  CUDA="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi
if [ -z "$CUDA" ]; then
  echo "install.sh: no nvcc found. Install a CUDA toolkit (https://developer.nvidia.com/cuda-downloads)" >&2
  echo "or point CUDA_HOME at one. cmake/ninja are handled for you." >&2
  exit 1
fi
echo "CUDA: $CUDA ($("$CUDA/bin/nvcc" --version | grep -o 'release [0-9.]*'))"

# ------------------------------------------------------------------ Python + venv
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && \
     "$c" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "install.sh: Python 3.10+ with venv+pip is required." >&2
  echo "  Debian/Ubuntu: sudo apt install python3 python3-venv python3-pip" >&2
  exit 1
fi
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip -q install -U pip
echo "Python: installing runtime and build packages into .venv ..."
.venv/bin/python -m pip -q install -r engine/requirements.txt -r engine/requirements-build.txt

# ------------------------------------------------------------------ engine build
# Upstream's cuda-compile preset flags (the release flags of packaging/build_cuda_linux.sh);
# the default arch is sm_86 because that is the card HydraGen was developed and measured on.
echo "Building the HydraGen engine (CUDA arch: $ARCH) ..."
".venv/bin/cmake" -S engine -B engine/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_PORTABLE=ON \
  -DSTRATA_Q6K_EXPERTS=ON \
  -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja"
".venv/bin/cmake" --build engine/build -j "$(nproc)"
BIN="$PWD/engine/build/strata"
[ -x "$BIN" ] || { echo "install.sh: build finished but engine/build/strata is missing" >&2; exit 1; }
echo "engine OK: $BIN"

# ------------------------------------------------------------------ local config
if [ ! -f hydragen-local.json ]; then
  sed -e "s#\"exe\": \"[^\"]*\"#\"exe\": \"$BIN\"#" \
      -e "s#\"log\": \"[^\"]*\"#\"log\": \"$PWD/logs/strata.log\"#" \
      configs/strata-iq3xxs-m25-0213-ctx131k.json > hydragen-local.json
  mkdir -p logs park-disk "$MODEL_DIR/packs" "$MODEL_DIR/models" "$MODEL_DIR/mtp"
  PARK_DISK_DIR="$PWD/park-disk" MODEL_DIR="$MODEL_DIR" REPO_DIR="$PWD" .venv/bin/python - <<'PYEOF'
import json, os
p = "hydragen-local.json"
cfg = json.load(open(p))
md, repo = os.environ["MODEL_DIR"], os.environ["REPO_DIR"]
gguf = os.path.join(md, "models", "IQ3_XXS", "IQ3_XXS")
targets = {
    "--pack": os.path.join(md, "packs", "iq3_xxs"),
    "--native": os.path.join(gguf, "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"),
    "--ple-gguf": os.path.join(gguf, "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"),
    "--expert-profile": os.path.join(repo, "engine", "data", "expert-profile.bin"),
    "--mtp": os.path.join(md, "mtp", "rt"),
}
args = cfg["args"]
for i, tok in enumerate(args):
    if tok in targets and i + 1 < len(args):
        args[i + 1] = targets[tok]
env = cfg.setdefault("env", {})
env["STRATA_PARK_COW"] = "1"
env["STRATA_PARK_DISK"] = os.environ["PARK_DISK_DIR"]
env.setdefault("STRATA_PARK_DISK_MIB", "163840")
env["STRATA_PARK_DISK_V2"] = "1"
env["STRATA_RESUME_DEEPEST"] = "1"
if "--conversation-cache-min-free-mib" not in args:
    args += ["--conversation-cache-min-free-mib", "512"]
if "--prompt-cache-tail" not in args:
    args += ["--prompt-cache-tail"]
json.dump(cfg, open(p, "w"), indent=1)
PYEOF
  echo "config created: hydragen-local.json (localhost:8181, canonical cow stack wired)"
  echo "model paths point under: $MODEL_DIR (override location: ./install.sh --model-dir DIR)"
fi

cat <<'EON'

Next steps
  1. Model blobs (~76 GB IQ3_XXS set). hydragen-local.json now expects them under the
     model dir printed above. Easiest is the vendored Strata setup - it downloads the
     pinned GGUFs (resumable, SHA-256-checked), builds the expert pack and fetches the
     MTP draft into that dir:
         bash engine/setup.sh --model IQ3_XXS --data-dir <model-dir>
     Or fetch them yourself and let setup reuse them:
         uvx hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --include "IQ3_XXS/*" --local-dir DIR
         bash engine/setup.sh --gguf-dir DIR
     Source table and licenses: docs/MODELS.md. The expert-routing profile is already
     in this repo (engine/data/expert-profile.bin) - nothing to download for it.
  2. Serve:
       .venv/bin/python engine/serve/server.py --engine strata --config hydragen-local.json
     Dashboard + Monitor (concurrency panel): http://127.0.0.1:8181/
     LAN serving: set "host": "0.0.0.0" and "api_key" in the config (the key gates the
     API; the dashboard page itself is open, so keep it on a trusted network).
  3. The canonical serving stack is already wired into hydragen-local.json:
       STRATA_PARK_COW=1        host-tier parked-session dedup (CoW parks) - ~40% less
                                parked RAM on agent workloads (docs/FINAL_REPORT.md)
       STRATA_PARK_DISK=DIR     the cross-boot park-disk tier under it (a park-disk/ dir
                                was created here); STRATA_PARK_DISK_MIB caps its budget
                                (default 163840 = 160 GiB - lower it on small disks)
       STRATA_PARK_DISK_V2=1    v2 writer + lineage-aware two-phase boot rebuild
       STRATA_RESUME_DEEPEST=1  a parked session resumes at its own deepest checkpoint,
                                together with the --prompt-cache-tail engine arg now in
                                the config's args list
       STRATA_COW_JOIN_HH, STRATA_COW_SHARED are closed-negative experiments; leave off.
     Merge gates (optional, from this tree): build with -DSTRATA_BUILD_CONVERSATION_TESTS=ON and run
     the conversation_cache_test and park_disk_boot_test targets.
EON
