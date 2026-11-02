// SPDX-License-Identifier: Apache-2.0
// Strata-TRTLLM Paged MoE Custom TensorRT Plugin
// Bridges NVIDIA TensorRT-LLM execution graphs to Strata's PagedExpertManager & PCIe DMA arena.

#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstdint>

#ifndef NV_TENSORRT_VERSION
// Lightweight standalone TensorRT definitions when NvInfer.h is not available in development environment
namespace nvinfer1 {
    enum class DataType : int32_t { kFLOAT = 0, kHALF = 1, kINT8 = 2, kINT32 = 3, kBOOL = 4, kUINT8 = 5 };
    enum class PluginFormat : int32_t { kLINEAR = 0 };

    struct DimsExprs {
        int32_t nbDims;
        int64_t d[8];
    };

    struct DynamicPluginTensorDesc {
        DimsExprs desc;
        DataType type;
        PluginFormat format;
        float scale;
        int32_t zero;
    };

    struct PluginTensorDesc {
        DimsExprs dims;
        DataType type;
        PluginFormat format;
        float scale;
        int32_t zero;
    };

    class IPluginV2DynamicExt {
    public:
        virtual ~IPluginV2DynamicExt() = default;
        virtual const char* getPluginType() const noexcept = 0;
        virtual const char* getPluginVersion() const noexcept = 0;
        virtual int32_t getNbOutputs() const noexcept = 0;
        virtual size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs,
                                        const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept = 0;
        virtual int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                                const void* const* inputs, void* const* outputs,
                                void* workspace, cudaStream_t stream) noexcept = 0;
    };
}
#endif

// Strata Native Assembly C-ABI Kernel Signatures (from libstrata_iq_kernels.so)
extern "C" {
void strata_vllm_expert_blob_info(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, uint64_t* out_info);
uint64_t strata_vllm_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff);

void strata_vllm_native_expert_grouped(
    int gu_type, int d_type, int64_t n_embd, int64_t n_ff,
    int64_t n_tokens,
    const unsigned long long* grp_ptr,
    const int32_t* grp_start,
    const int32_t* n_groups,
    const int32_t* ent_dst,
    const int32_t* ent_tok,
    int64_t cap_groups,
    int64_t cap_entries,
    const float* x_f32,
    void* x_q8_1,
    void* scratch,
    float* out,
    void* stream
);

void strata_vllm_moe_combine_multi(
    const float* parts, const float* weights, const float* shared,
    float* output, int64_t n_embd, int64_t k, int n_tok, void* stream
);
}

namespace strata {
namespace trtllm {

struct StrataPagedMoePluginConfig {
    int32_t num_experts = 512;
    int32_t top_k = 10;
    int32_t hidden_size = 2560;
    int32_t intermediate_size = 640;
    int32_t num_vram_slots = 128;
    int32_t layer_idx = 0;
    int32_t gu_type = 149; // GGUF type enum for IQ2_XXS / INT8
    int32_t d_type = 149;
    void* expert_manager_handle = nullptr; // pointer to PagedExpertManager
};

class StrataPagedMoePlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    StrataPagedMoePlugin(const StrataPagedMoePluginConfig& config);
    StrataPagedMoePlugin(const void* data, size_t length);
    ~StrataPagedMoePlugin() override = default;

    const char* getPluginType() const noexcept override { return "StrataPagedMoePlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) noexcept override;

private:
    StrataPagedMoePluginConfig m_config;
};

} // namespace trtllm
} // namespace strata
