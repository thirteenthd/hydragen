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

CUDA_HOME="/usr/local/cuda-13.1"
if [ ! -d "${CUDA_HOME}/include" ]; then
    CUDA_HOME="/usr/local/cuda"
fi

STRATA_KERNELS_DIR="${REPO_ROOT}/src/hydragen/server"
if [ ! -f "${STRATA_KERNELS_DIR}/libstrata_iq_kernels.so" ]; then
    STRATA_KERNELS_DIR="${REPO_ROOT}/serve"
fi

echo "====================================================="
echo "Building PagedAttention C++/CUDA, PTX Plugin & Fast IPC"
echo "CUDA Home: ${CUDA_HOME}"
echo "NVCC: ${NVCC}"
echo "Strata Kernels: ${STRATA_KERNELS_DIR}"
echo "====================================================="

echo "[1/5] Compiling paged_attention_kernel.cu with NVCC (PTX DP4A SIMD)..."
"${NVCC}" -c -O3 -Xcompiler -fPIC \
    -gencode arch=compute_86,code=sm_86 \
    -I"${CUDA_HOME}/include" \
    -I"${SCRIPT_DIR}/paged_attention" \
    "${SCRIPT_DIR}/paged_attention/paged_attention_kernel.cu" \
    -o "${BUILD_DIR}/paged_attention_kernel.o"

echo "[2/5] Compiling paged_kv_manager.cpp & c_api.cpp with g++..."
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

echo "[3/5] Linking libpaged_kv.so..."
g++ -shared -fPIC \
    "${BUILD_DIR}/paged_attention_kernel.o" \
    "${BUILD_DIR}/paged_kv_manager.o" \
    "${BUILD_DIR}/paged_kv_c_api.o" \
    -L"${CUDA_HOME}/lib64" -lcudart \
    -o "${LIB_DIR}/libpaged_kv.so"

echo "[4/5] Compiling and linking libstrata_trtllm_plugin.so (linked to libstrata_iq_kernels.so)..."
g++ -c -O3 -fPIC -std=c++17 \
    -I"${CUDA_HOME}/include" \
    -I"${SCRIPT_DIR}/trtllm_plugin" \
    "${SCRIPT_DIR}/trtllm_plugin/strata_paged_moe_plugin.cpp" \
    -o "${BUILD_DIR}/strata_paged_moe_plugin.o"

g++ -shared -fPIC \
    "${BUILD_DIR}/strata_paged_moe_plugin.o" \
    -L"${CUDA_HOME}/lib64" -lcudart \
    -L"${STRATA_KERNELS_DIR}" -lstrata_iq_kernels \
    -Wl,-rpath,"\$ORIGIN/../src/hydragen/server:\$ORIGIN/../serve" \
    -o "${LIB_DIR}/libstrata_trtllm_plugin.so"

echo "[5/5] Compiling and linking libstrata_fast_ipc.so (Lock-Free C++ Ring Buffer)..."
g++ -O3 -fPIC -std=c++17 -pthread -shared \
    -I"${SCRIPT_DIR}/ipc" \
    "${SCRIPT_DIR}/ipc/strata_fast_ipc.cpp" \
    -o "${LIB_DIR}/libstrata_fast_ipc.so"

echo "====================================================="
echo "Build complete!"
ls -la "${LIB_DIR}"
echo "====================================================="
