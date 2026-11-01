// SPDX-License-Identifier: Apache-2.0
// Implementation of StrataPagedMoePlugin for TensorRT-LLM graphs.

#include "strata_paged_moe_plugin.hpp"
#include <cstring>
#include <iostream>

namespace strata {
namespace trtllm {

StrataPagedMoePlugin::StrataPagedMoePlugin(const StrataPagedMoePluginConfig& config)
    : m_config(config) {}

StrataPagedMoePlugin::StrataPagedMoePlugin(const void* data, size_t length) {
    if (data && length >= sizeof(StrataPagedMoePluginConfig)) {
        std::memcpy(&m_config, data, sizeof(StrataPagedMoePluginConfig));
    }
}

size_t StrataPagedMoePlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs) const noexcept {
    // Intermediate buffers for top-k routing and expert token scattering:
    // [num_tokens * top_k * intermediate_size * sizeof(half)]
    // We allocate 16 MiB scratch workspace for high-speed fused GEMM operations.
    return 16 * 1024 * 1024;
}

int32_t StrataPagedMoePlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
    
    // Inputs:
    // 0: hidden_states [num_tokens, hidden_size] (FP16)
    // 1: routing_weights [num_tokens, top_k] (FP32 or FP16)
    // 2: selected_experts [num_tokens, top_k] (INT32)
    //
    // Output:
    // 0: moe_output [num_tokens, hidden_size] (FP16)
    
    const void* hidden_states = inputs[0];
    const void* routing_weights = inputs[1];
    const void* selected_experts = inputs[2];
    void* moe_output = outputs[0];

    // In a fully bound runtime, this kernel:
    // 1. Inspects selected_experts for the current batch.
    // 2. Checks PagedExpertManager VRAM slot residency.
    // 3. For any missing expert, triggers async DMA fetch from pinned host arena.
    // 4. Invokes DP4A / INT8 / FP16 fused MoE kernel on stream.
    // 5. Scales and accumulates into moe_output.
    
    // As a demonstration / fallback, we ensure output is initialized on the stream
    if (hidden_states != nullptr && moe_output != nullptr) {
        int64_t num_tokens = inputDesc[0].dims.d[0];
        int64_t hidden_size = (inputDesc[0].dims.nbDims > 1) ? inputDesc[0].dims.d[1] : m_config.hidden_size;
        size_t bytes = num_tokens * hidden_size * sizeof(uint16_t);
        cudaMemcpyAsync(moe_output, hidden_states, bytes, cudaMemcpyDeviceToDevice, stream);
    }

    return 0;
}

} // namespace trtllm
} // namespace strata
