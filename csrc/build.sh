#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${REPO_ROOT}/build"
LIB_DIR="${REPO_ROOT}/lib"

mkdir -p "${BUILD_DIR}"
mkdir -p "${LIB_DIR}"

NVCC="/usr/local/cuda-13.1/bin/nvcc"
if [ ! -f "${NVCC}" ]; then
    NVCC="nvcc"
fi

CUDA_HOME="$(dirname "$(dirname "$("${NVCC}" --version | grep -o '/.*' | head -n 1 || echo '/usr/local/cuda')")"
if [ ! -d "${CUDA_HOME}/include" ]; then
    CUDA_HOME="/usr/local/cuda-13.1"
fi

echo "====================================================="
echo "Building PagedAttention C++/CUDA & TRT Plugin Kernels"
echo "CUDA Home: ${CUDA_HOME}"
echo "NVCC: ${NVCC}"
echo "====================================================="

echo "[1/4] Compiling paged_attention_kernel.cu with NVCC..."
"${NVCC}" -c -O3 -Xcompiler -fPIC \
    -gencode arch=compute_86,code=sm_86 \
    -I"${CUDA_HOME}/include" \
    -I"${SCRIPT_DIR}/paged_attention" \
    "${SCRIPT_DIR}/paged_attention/paged_attention_kernel.cu" \
    -o "${BUILD_DIR}/paged_attention_kernel.o"

echo "[2/4] Compiling paged_kv_manager.cpp & c_api.cpp with g++..."
g++ -c -O3 -fPIC -std=c++17 \
    -I"${CUDA_HOME}/include" \
    -I"${SCRIPT_DIR}/paged_attention" \
    "${SCRIPT_DIR}/paged_attention/paged_kv_manager.cpp" \
    -o "${BUILD_DIR}/paged_kv_manager.o"

g++ -c -O3 -fPIC -std=c++17 \
    -I"${CUDA_HOME}/include" \
    -I"${SCRIPT_DIR}/paged_attention" \
    "${SCRIPT_DIR}/paged_attention/paged_kv_c_api.cpp" \
    -o "${BUILD_DIR}/paged_kv_c_api.o"

echo "[3/4] Linking libpaged_kv.so..."
g++ -shared -fPIC \
    "${BUILD_DIR}/paged_attention_kernel.o" \
    "${BUILD_DIR}/paged_kv_manager.o" \
    "${BUILD_DIR}/paged_kv_c_api.o" \
    -L"${CUDA_HOME}/lib64" -lcudart \
    -o "${LIB_DIR}/libpaged_kv.so"

echo "[4/4] Compiling and linking libstrata_trtllm_plugin.so..."
g++ -c -O3 -fPIC -std=c++17 \
    -I"${CUDA_HOME}/include" \
    -I"${SCRIPT_DIR}/trtllm_plugin" \
    "${SCRIPT_DIR}/trtllm_plugin/strata_paged_moe_plugin.cpp" \
    -o "${BUILD_DIR}/strata_paged_moe_plugin.o"

g++ -shared -fPIC \
    "${BUILD_DIR}/strata_paged_moe_plugin.o" \
    -L"${CUDA_HOME}/lib64" -lcudart \
    -o "${LIB_DIR}/libstrata_trtllm_plugin.so"

echo "====================================================="
echo "Build complete!"
ls -la "${LIB_DIR}"
echo "====================================================="
