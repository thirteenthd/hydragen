// SPDX-License-Identifier: Apache-2.0
// Implementation of StrataPagedMoePlugin bridging TensorRT-LLM to Strata's native PTX assembly kernels.

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
    // Scratch buffer required by Strata's grouped MoE and Q8_1 quantization:
    // 16 MiB ensures enough headroom for up to batch size 64 x top_k 10 x hidden_size
    int64_t n_ff = m_config.intermediate_size;
    int64_t cap_entries = 64 * m_config.top_k;
    size_t req = strata_vllm_expert_scratch_bytes(cap_entries, n_ff);
    return (req > 16 * 1024 * 1024) ? req : (16 * 1024 * 1024);
}

int32_t StrataPagedMoePlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
    
    // Inputs:
    // 0: hidden_states [num_tokens, hidden_size] (float*)
    // 1: routing_weights [num_tokens, top_k] (float*)
    // 2: selected_experts [num_tokens, top_k] (int32_t*)
    // (Optional 3: expert_grp_ptrs [num_groups] uint64_t*)
    //
    // Output:
    // 0: moe_output [num_tokens, hidden_size] (float*)
    
    const float* hidden_states = reinterpret_cast<const float*>(inputs[0]);
    const float* routing_weights = reinterpret_cast<const float*>(inputs[1]);
    const int32_t* selected_experts = reinterpret_cast<const int32_t*>(inputs[2]);
    float* moe_output = reinterpret_cast<float*>(outputs[0]);

    int64_t num_tokens = inputDesc[0].dims.d[0];
    int64_t hidden_size = (inputDesc[0].dims.nbDims > 1) ? inputDesc[0].dims.d[1] : m_config.hidden_size;
    int64_t n_ff = m_config.intermediate_size;
    int32_t top_k = m_config.top_k;

    // Check if expert pointers are supplied in inputs[3] or registered via config
    if (hidden_states != nullptr && moe_output != nullptr && inputs[3] != nullptr) {
        const unsigned long long* grp_ptr = reinterpret_cast<const unsigned long long*>(inputs[3]);
        const int32_t* grp_start = (inputs[4] != nullptr) ? reinterpret_cast<const int32_t*>(inputs[4]) : nullptr;
        const int32_t* n_groups = (inputs[5] != nullptr) ? reinterpret_cast<const int32_t*>(inputs[5]) : nullptr;
        const int32_t* ent_dst = (inputs[6] != nullptr) ? reinterpret_cast<const int32_t*>(inputs[6]) : nullptr;
        const int32_t* ent_tok = (inputs[7] != nullptr) ? reinterpret_cast<const int32_t*>(inputs[7]) : nullptr;

        // Partition workspace
        uint8_t* ws = reinterpret_cast<uint8_t*>(workspace);
        void* x_q8_1 = ws;
        size_t q8_bytes = num_tokens * hidden_size * sizeof(int8_t) * 2;
        void* scratch = ws + q8_bytes;
        float* parts_buf = reinterpret_cast<float*>(ws + q8_bytes + 2 * 1024 * 1024);

        // 1. Launch Strata's native PTX assembly grouped expert kernel:
        //    Quantizes hidden_states to Q8_1 and executes DP4A / sub-byte IQ dequantized GEMV
        strata_vllm_native_expert_grouped(
            m_config.gu_type,
            m_config.d_type,
            hidden_size,
            n_ff,
            num_tokens,
            grp_ptr,
            grp_start,
            n_groups,
            ent_dst,
            ent_tok,
            top_k * num_tokens,
            top_k * num_tokens,
            hidden_states,
            x_q8_1,
            scratch,
            parts_buf,
            stream
        );

        // 2. Launch Strata's native Top-K warp reduction accumulator kernel:
        strata_vllm_moe_combine_multi(
            parts_buf,
            routing_weights,
            nullptr, // shared expert (null if unshared)
            moe_output,
            hidden_size,
            top_k,
            num_tokens,
            stream
        );
        return 0;
    }

    // Direct stream pass-through fallback if expert pointers are not directly attached
    if (hidden_states != nullptr && moe_output != nullptr) {
        size_t bytes = num_tokens * hidden_size * sizeof(float);
        cudaMemcpyAsync(moe_output, hidden_states, bytes, cudaMemcpyDeviceToDevice, stream);
    }

    return 0;
}

} // namespace trtllm
} // namespace strata
