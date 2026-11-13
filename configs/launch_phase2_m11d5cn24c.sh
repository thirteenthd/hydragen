#!/bin/bash
# Phase-2 M9: native Strata grouped MoE over IQ blob store.
# mamba 2048/1024+1 = 3 slots/req, pool 12 -> cap 4. Device cache N=4.
set -e
export PATH=$HOME/sg-env/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export HG_PAGED_STORE_DIR=~/hg-store
export HG_PLE_HOSTSIDE=0
export HG_PLE_DIET=1
export HG_HOST_EMBED=1
export HG_HOST_HEAD=1
export HG_HEAD_FP8=1
export HG_MOE_SHIM=hydragen_moe_native_graph
export HG_BCG_LPO=1 HG_NATIVE_GRAPH_ISLAND=1 HG_PAGED_SPEC=1 HG_MOE_NATIVE=1 HG_DENSE_IQ=1 HG_DENSE_IQ_FMT=13 SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
export HG_DIET_FINDER=1
# export HG_NATIVE_PROF=1 HG_MOE_PROF=1 HG_STEP_PROF=1  # lean: banked run
export HG_IQ_BLOB_DIR=~/hg-store-iqxxs-blobs
export HG_MOE_CACHE_N=24
export HG_NATIVE_TCHUNK=16 HG_HEAD_CHUNK=8192 HG_HEAD_IQ=1
export HG_MOE_HTIER=384
export HG_PREWARM=1
export HG_TIER_SEED=~/hg-store-iqxxs-blobs/TIER_SEED_BATT.json
export HG_MOE_RING_MB=192
export HG_QSA_SDPA=1
export SGLANG_QWEN4_PLE_FILE_DIR=/mnt/docker/hg/ple
if ~/sg-env/bin/python ~/hydragen-experimental/pyhook/hydragen_loader_skip.py --check /mnt/docker/hg/q4e-fp8; then export HG_SKIP_LOAD=1; else echo LOADSKIP_FAILED_FULL; fi
export SGLANG_QWEN4_PLE_FILE_DIR=/mnt/docker/hg/ple
export PYTHONPATH=~/hydragen-experimental/pyhook
nohup ~/sg-env/bin/python -m sglang.launch_server \
  --model-path /mnt/docker/hg/q4e-fp8 --tp 1 --host 0.0.0.0 --port 30099 \
  --json-model-override-args '{"language_model_only": true}' \
  --skip-server-warmup --mem-fraction-static 0.95 \
  --mamba-track-interval 1024 --max-mamba-cache-size 16 --chunked-prefill-size 1024 \
  --context-length 2048 --max-total-tokens 4096 --max-running-requests 4 \
  --mamba-ssm-dtype bfloat16 --trust-remote-code \
  --enable-paged-experts --paged-experts-num-resident 10 \
  --disable-shared-experts-fusion \
  --cuda-graph-backend-decode breakable --cuda-graph-bs-decode 1 4 --cuda-graph-backend-prefill disabled \
  --ple-offload-backend pinned > ~/hydragen-experimental/logs/sg-real.log 2>&1 &
echo LAUNCH_PID=$!
