#!/usr/bin/env bash
# HydraGen installer (Linux, NVIDIA).
# Prefers a PREBUILT CUDA engine (this repo's GitHub Releases: RTX 30/40/50, fat
# sm_86/89/120 binary + bundled CUDA runtime libs, SHA-256-verified, provenance
# in runtime/BUILD_INFO) and falls back to building the vendored engine FROM
# SOURCE automatically when the download, the hash check, or the distro's glibc
# floor fails - or on request with --build-from-source. The prebuilt engine IS
# the canonical cow chain (expert-slot refcounting + admission fork, host-tier
# parked-session dedup, park-disk v2 lineage tier + deep resume; see
# docs/FINAL_REPORT.md) - same flags the source build uses.
#
# Source-build path needs: an NVIDIA driver, a CUDA toolkit (nvcc), a C++
# toolchain, Python 3.10+. CUDA toolkit: https://developer.nvidia.com/cuda-downloads
# cmake + ninja come from pip (terminal/requirements-build.txt); nothing else to install.
#
#   ./install.sh                       # prebuilt if available, else source (sm_86)
#   ./install.sh --build-from-source   # never run a downloaded binary; compile locally
#   ./install.sh --model IQ3_S         # wire the local config for the IQ3_S model tier (default: IQ3_XXS)
#   HYDRAGEN_CUDA_ARCH=89-real ./install.sh          # RTX 40-series
#   HYDRAGEN_CUDA_ARCH="86-real;89-real" ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${HYDRAGEN_CUDA_ARCH:-86-real}"
RELEASE_TAG="v0.0.1"

# Model blob location: --model-dir DIR (default: <this repo>/models). The generated
# config's --pack/--native/--ple-gguf/--mtp paths are rewritten to live under DIR.
# The expert-routing profile needs no download - it ships in this repo at
# terminal/data/expert-profile.bin. See docs/MODELS.md for sources and the
# vendored Strata downloader (setup.sh / terminal/setup.py).
MODEL_DIR=""
MODEL="IQ3_XXS"
BUILD_FROM_SOURCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="${2:?--model needs IQ3_XXS or IQ3_S}"; shift 2 ;;
    --model-dir) MODEL_DIR="${2:?--model-dir needs a directory}"; shift 2 ;;
    --build-from-source) BUILD_FROM_SOURCE=1; shift ;;
    *) echo "install.sh: unknown argument: $1 (try --model IQ3_XXS|IQ3_S, --model-dir DIR, --build-from-source)" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL_DIR" ] || MODEL_DIR="$PWD/models"
case "$MODEL_DIR" in /*) ;; *) MODEL_DIR="$PWD/$MODEL_DIR" ;; esac
case "$MODEL" in IQ3_XXS|IQ3_S) ;; *) echo "install.sh: --model must be IQ3_XXS (default) or IQ3_S (docs/MODELS.md)" >&2; exit 2 ;; esac

# --- environment sizing (auto-detected; override by editing hydragen-local.json) ---
DET_MIB=""
GPU_LINE=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2 -rn | head -1)
fi
if [ -n "$GPU_LINE" ]; then
  VRAM=$(echo "$GPU_LINE" | awk -F', ' '{print $2}')
  CC=$(echo "$GPU_LINE" | awk -F', ' '{print $3}')
  echo "install.sh: GPU detected: $GPU_LINE"
  [ "$VRAM" -lt 12000 ] 2>/dev/null && echo "install.sh: WARNING: largest GPU is ${VRAM} MiB - the shipped config targets 12 GB+ VRAM" >&2
  case "$CC" in 8.*|9.*|1[0-9].*) : ;; "") : ;; *) echo "install.sh: WARNING: compute capability $CC - this engine build targets Ampere (sm_80) or newer" >&2 ;; esac
else
  echo "install.sh: NOTE: nvidia-smi not found - install the NVIDIA driver before serving (skipping GPU sizing)" >&2
fi
RAM_MIB=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$RAM_MIB" -gt 0 ] 2>/dev/null; then
  DET_MIB=16384
  [ "$RAM_MIB" -lt 49152 ] && echo "install.sh: WARNING: ${RAM_MIB} MiB RAM detected - the documented floor is 64 GB; expect parked sessions to be re-prefilled more often" >&2
  echo "install.sh: RAM detected: ${RAM_MIB} MiB -> park-disk tier budget ${DET_MIB} MiB (Phase 5 default cap: a tier pinned above 16 GiB measurably decays steady-state throughput and slows boot - raise it only with several x that cap in system RAM)"
fi
FREE_MIB=$(df -Pm "$MODEL_DIR" 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "$FREE_MIB" ] && [ -n "$DET_MIB" ] && [ "$FREE_MIB" -lt $(( DET_MIB + 174080 )) ] 2>/dev/null; then
  echo "install.sh: NOTE: ${FREE_MIB} MiB free at $MODEL_DIR - model blobs need ~160 GiB plus the ${DET_MIB} MiB tier budget (the tier fills lazily; this is a now-check, not a hard gate)" >&2
fi

# ------------------------------------------------------------------ prebuilt engine (preferred)
# Fat CUDA binary (sm_86/89/120 = RTX 30/40/50) + the CUDA runtime libs it needs,
# SHA-256-verified against the release's SHA256SUMS.txt. Provenance (commit, build
# host, glibc floor) lands in runtime/. Any failure falls through to source build.
BIN=""
if [ "$BUILD_FROM_SOURCE" -eq 0 ]; then
  BASE="https://github.com/thirteenthd/hydragen/releases/download/$RELEASE_TAG"
  if curl -fsSL -o /tmp/hg-rel-sums.txt "$BASE/SHA256SUMS.txt" 2>/dev/null; then
    if curl -fsSL -o /tmp/hydragen-linux-cuda-x86_64.tar.gz "$BASE/hydragen-linux-cuda-x86_64.tar.gz" 2>/dev/null && \
       (cd /tmp && grep "hydragen-linux-cuda-x86_64.tar.gz" hg-rel-sums.txt | sha256sum -c - >/dev/null 2>&1); then
      mkdir -p runtime
      tar xzf /tmp/hydragen-linux-cuda-x86_64.tar.gz -C runtime
      chmod +x runtime/hydragen 2>/dev/null || true
      FLOOR=$(cat runtime/GLIBC_FLOOR 2>/dev/null || echo "")
      HAVE=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
      OLDER=$(printf '%s\n%s\n' "${FLOOR:-0}" "${HAVE:-0}" | sort -V | head -1)
      if [ -x runtime/hydragen ] && { [ -z "$FLOOR" ] || [ "$OLDER" = "$FLOOR" ]; }; then
        BIN="$PWD/runtime/hydragen"
        echo "install.sh: prebuilt engine installed (runtime/hydragen, SHA-256 verified):"
        sed 's/^/    /' runtime/BUILD_INFO 2>/dev/null || true
        echo "install.sh: this binary is unsigned like all community CUDA builds; the hash above"
        echo "            pins provenance. Prefer your own compile? rerun: ./install.sh --build-from-source"
      else
        echo "install.sh: NOTE: glibc $HAVE is older than this binary's build floor ($FLOOR) - falling back to a source build" >&2
      fi
    else
      echo "install.sh: NOTE: prebuilt download/hash check failed - falling back to a source build" >&2
    fi
    rm -f /tmp/hydragen-linux-cuda-x86_64.tar.gz /tmp/hg-rel-sums.txt
  else
    echo "install.sh: NOTE: no release manifest reachable (offline?) - falling back to a source build" >&2
  fi
fi

# ------------------------------------------------------------------ CUDA (source-build path)
CUDA=""
if [ -z "$BIN" ]; then
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
fi

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
.venv/bin/python -m pip -q install -r terminal/requirements.txt -r terminal/requirements-build.txt

# ------------------------------------------------------------------ engine build (fallback path)
# Upstream's cuda-compile preset flags (the release flags of tools/packaging/build_cuda_linux.sh);
# the default arch is sm_86 because that is the card HydraGen was developed and measured on.
if [ -z "$BIN" ]; then
echo "Building the HydraGen engine from source (CUDA arch: $ARCH) ..."
".venv/bin/cmake" -S engine -B engine/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_PORTABLE=ON \
  -DSTRATA_Q6K_EXPERTS=ON \
  -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_WINDOW_MAX_T=16 \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja"
".venv/bin/cmake" --build engine/build -j "$(nproc)"
BIN="$PWD/engine/build/hydragen"
[ -x "$BIN" ] || { echo "install.sh: build finished but engine/build/hydragen is missing" >&2; exit 1; }
echo "engine OK: $BIN"
fi

# ------------------------------------------------------------------ local config
if [ ! -f hydragen-local.json ]; then
  if [ "$MODEL" = "IQ3_S" ]; then CFG="configs/strata-iq3s-m25-0213-ctx131k.json"; else CFG="configs/strata-iq3xxs-m25-0213-ctx131k.json"; fi
  sed -e "s#\"exe\": \"[^\"]*\"#\"exe\": \"$BIN\"#" \
      -e "s#\"log\": \"[^\"]*\"#\"log\": \"$PWD/logs/strata.log\"#" \
      "$CFG" > hydragen-local.json
  mkdir -p logs park-disk "$MODEL_DIR/packs" "$MODEL_DIR/models" "$MODEL_DIR/mtp"
  PARK_DISK_DIR="$PWD/park-disk" MODEL_DIR="$MODEL_DIR" MODEL="$MODEL" REPO_DIR="$PWD" DET_MIB="$DET_MIB" .venv/bin/python - <<'PYEOF'
import json, os
p = "hydragen-local.json"
cfg = json.load(open(p))
md, repo = os.environ["MODEL_DIR"], os.environ["REPO_DIR"]
mdl = os.environ["MODEL"]
gguf = os.path.join(md, "models", mdl, mdl)
targets = {
    "--pack": os.path.join(md, "packs", mdl.lower()),
    "--native": os.path.join(gguf, f"Qwen3.8-Flash-Next-GSQ-RCO-{mdl}-00001-of-00002.gguf"),
    "--ple-gguf": os.path.join(gguf, f"Qwen3.8-Flash-Next-GSQ-RCO-{mdl}-00002-of-00002.gguf"),
    "--expert-profile": os.path.join(repo, "terminal", "data", "expert-profile.bin"),
    "--mtp": os.path.join(md, "mtp", "rt"),
}
# per-artifact learned-profile loop: the adaptive tier re-ranks experts from live traffic
# at runtime; --expert-profile-save persists that ranking so the next boot re-seeds from
# what THIS machine's traffic actually learned. The save file is keyed by the model blob
# name: change the artifact and the key changes, so a stale ranking can never abort the
# boot (an explicitly-given profile that fails to load is fatal upstream by design).
import hashlib
_pkey = hashlib.sha256(os.path.basename(targets["--native"]).encode()).hexdigest()[:10]
_learned = os.path.join(repo, f"expert-profile-learned-{_pkey}.bin")
if os.path.exists(_learned):
    targets["--expert-profile"] = _learned
args = cfg["args"]
for i, tok in enumerate(args):
    if tok in targets and i + 1 < len(args):
        args[i + 1] = targets[tok]
for i, tok in enumerate(args):
    if tok == "--expert-profile" and "--expert-profile-save" not in args:
        args[i + 2 : i + 2] = ["--expert-profile-save", _learned]
        break
cfg["cwd"] = repo
cfg["tokenizer"] = os.path.join(md, "packs", mdl.lower(), "tokenizer")
env = cfg.setdefault("env", {})
rt = os.path.join(repo, "runtime")
if os.path.isdir(os.path.join(rt, "lib")):  # prebuilt engine: bundled CUDA runtime libs
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.path.join(rt, "lib") + ((":" + prev) if prev else "")
env["HG_PARK_COW"] = "1"
env["HG_PARK_DISK"] = os.environ["PARK_DISK_DIR"]
env.setdefault("HG_PARK_DISK_MIB", os.environ.get("DET_MIB") or "16384")
env.setdefault("STRATA_ADMIT_SHORT_FIRST", "8192")
env["HG_PARK_DISK_V2"] = "1"
env["HG_RESUME_DEEPEST"] = "1"
env["HG_UNIFIED_POOL"] = "1"
env["STRATA_AUTO_BATCH"] = "1"
env["CUDA_MODULE_LOADING"] = "LAZY"
if "--conversation-cache-min-free-mib" not in args:
    args += ["--conversation-cache-min-free-mib", "512"]
if "--prompt-cache-tail" not in args:
    args += ["--prompt-cache-tail"]
json.dump(cfg, open(p, "w"), indent=1)
PYEOF
  echo "config created: hydragen-local.json (localhost:8181, canonical cow stack wired)"
  echo "model paths point under: $MODEL_DIR (tier: $MODEL; override location: ./install.sh --model-dir DIR --model IQ3_XXS|IQ3_S)"
fi

cat <<EON

Next steps
  1. Model blobs (~76 GB IQ3_XXS / ~84 GB IQ3_S set). hydragen-local.json now expects them under the
     model dir printed above. Easiest is the vendored Strata setup - it downloads the
     pinned GGUFs (resumable, SHA-256-checked), builds the expert pack and fetches the
     MTP draft into that dir:
         bash setup.sh --model $MODEL --data-dir <model-dir>
     Or fetch them yourself and let setup reuse them:
         uvx hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --include "$MODEL/*" --local-dir DIR
         bash setup.sh --gguf-dir DIR
     Source table and licenses: docs/MODELS.md. The expert-routing profile is already
     in this repo (terminal/data/expert-profile.bin) - nothing to download for it.
     --mtp is OPTIONAL: the shipped configs run with MTP OFF (lossless suffix drafting
     via --spec 4 --coupled-draft stays on). 12 GB cards default to MTP OFF for maximum
     expert residency; on 16 GB+ cards you can add "--mtp <model-dir>/mtp/rt" to the
     args for extra decode speed.
  2. Serve:
       .venv/bin/python terminal/serve/server.py --engine hydragen --config hydragen-local.json
     Dashboard + Monitor (concurrency panel): http://127.0.0.1:8181/
     LAN serving: set "host": "0.0.0.0" and "api_key" in the config (the key gates the
     API; the dashboard page itself is open, so keep it on a trusted network).
  3. The canonical serving stack is already wired into hydragen-local.json.
     Fork-introduced knobs carry an HG_ prefix so they are distinguishable from
     upstream Strata knobs (which keep their STRATA_ names); the legacy
     STRATA_* spelling of the HG_ knobs still works as a fallback.
       HG_PARK_COW=1            host-tier parked-session dedup (CoW parks) - ~40% less
                                parked RAM on agent workloads (docs/FINAL_REPORT.md)
       HG_PARK_DISK=DIR         the cross-boot park-disk tier under it (a park-disk/ dir
                                was created here); HG_PARK_DISK_MIB caps its budget
                                (default 16384 = 16 GiB, the Phase 5 cap: a tier pinned
                                above it measurably decays steady-state throughput and
                                slows boot - raise it only with several x that cap in RAM)
       HG_PARK_DISK_V2=1        v2 writer + lineage-aware two-phase boot rebuild
       HG_RESUME_DEEPEST=1      a parked session resumes at its own deepest checkpoint,
                                together with the --prompt-cache-tail engine arg now in
                                the config's args list
       HG_UNIFIED_POOL=1        one unified host pool for the arena + park tiers (Phase 5)
       STRATA_AUTO_BATCH=1      the batcher sizes itself from live traffic (Phase 5)
       CUDA_MODULE_LOADING=LAZY lazy CUDA module loading - faster boot (Phase 5)
       STRATA_ADMIT_SHORT_FIRST=8192  short turns (typical agent follow-ups) take the
                                prefill lane before big cold prompts; a long prompt that
                                waited STRATA_ADMIT_MAX_WAIT_S (30 s) counts as short,
                                so nothing starves  (upstream feature, measured here)
       The Phase 5 CoW-arena block (STRATA_COW_ARENA*, STRATA_COW_FORK, STRATA_COW_JOIN_HH,
       STRATA_SLOT_*, STRATA_BATCH_YIELD_CHUNKS) ships in the canonical configs' env;
       STRATA_COW_SHARED=0 and STRATA_ROUTE_TAIL_SKIP=0 stay pinned off there.
       HG_COW_SHARED is a closed-negative experiment; leave off.
     Merge gates (optional, from this tree): build with -DSTRATA_BUILD_CONVERSATION_TESTS=ON and run
     the conversation_cache_test and park_disk_boot_test targets.
EON
if [ ! -f "$MODEL_DIR/packs/iq3_xxs/tokenizer/vocab.json" ]; then
  echo >&2
  echo "install.sh: NOTE: $MODEL_DIR/packs/iq3_xxs/tokenizer/vocab.json does not exist yet - the pack build (step 1) is what" >&2
  echo "    exports the tokenizer, and serve refuses to start without it. Run step 1 before step 2." >&2
fi
