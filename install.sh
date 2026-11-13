#!/usr/bin/env bash
# HydraGen installer (Linux, NVIDIA).
# Builds the vendored engine FROM SOURCE - prebuilt Strata release binaries do not contain
# HydraGen's CoW chain (expert-slot refcounting + admission fork, host-tier parked-session
# dedup; see docs/FINAL_REPORT.md) - and sets up the Python venv for the serving front.
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
  mkdir -p logs
  echo "config created: hydragen-local.json (localhost:8181)"
fi

cat <<'EON'

Next steps
  1. Model blobs. hydragen-local.json still points at the IQ3_XXS expert pack, native
     model GGUFs, PLE GGUF and MTP draft used for the published measurements - download
     them (README.md, Model blobs) or edit the paths in hydragen-local.json to your own.
  2. Serve:
       .venv/bin/python engine/serve/server.py --engine strata --config hydragen-local.json
     Dashboard + Monitor (concurrency panel): http://127.0.0.1:8181/
     LAN serving: set "host": "0.0.0.0" and "api_key" in the config (the key gates the
     API; the dashboard page itself is open, so keep it on a trusted network).
  3. Concurrency levers (env, or the config's "env" map):
       STRATA_PARK_COW=1        host-tier parked-session dedup - recommended for agent
                                workloads (long histories: ~40% less parked RAM, no more
                                multi-GB re-prefills; docs/FINAL_REPORT.md)
       admission fork (M-C1c)   inside the engine, always on: bigger expert cache under
                                batch load - no knob
       STRATA_COW_JOIN_HH, STRATA_COW_SHARED   closed-negative experiments; leave off
EON
