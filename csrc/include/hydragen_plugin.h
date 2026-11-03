// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "hydragen_os.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HYDRAGEN_PLUGIN_ABI_VERSION 1

typedef enum {
    HYDRAGEN_DEVICE_UNKNOWN = 0,
    HYDRAGEN_DEVICE_CPU     = 1,
    HYDRAGEN_DEVICE_CUDA    = 2,
    HYDRAGEN_DEVICE_ROCM    = 3,
    HYDRAGEN_DEVICE_HYBRID  = 4
} hydragen_device_type_t;

// Fixed-size, binary-stable device capabilities structure
typedef struct {
    int32_t device_type;            // hydragen_device_type_t
    char device_name[128];
    int32_t compute_major;
    int32_t compute_minor;
    uint64_t total_vram_bytes;
    uint64_t free_vram_bytes;
    int32_t num_multiprocessors;
    int32_t has_dp4a;               // 1 if supported, 0 otherwise
    int32_t has_cp_async;           // 1 if SM_80+ async copy supported
    int32_t has_wmma;               // 1 if tensor cores supported
    int32_t has_avx512;             // 1 if AVX-512 supported
} hydragen_device_caps_t;

// Paged KV Block Manager & Attention Function Pointer Table
typedef struct {
    // Virtual Page Table Management
    void*   (*create_manager)(int32_t block_size, int32_t num_blocks, int32_t num_layers, int32_t num_kv_heads, int32_t head_dim, int32_t dtype_bytes);
    void    (*destroy_manager)(void* handle);
    int32_t (*allocate_seq)(void* handle, int32_t seq_id, int32_t initial_tokens);
    int32_t (*append_token)(void* handle, int32_t seq_id);
    int32_t (*fork_seq)(void* handle, int32_t parent_seq_id, int32_t child_seq_id);
    int32_t (*free_seq)(void* handle, int32_t seq_id);
    int32_t (*get_num_tokens)(void* handle, int32_t seq_id);
    int32_t (*get_num_blocks)(void* handle, int32_t seq_id);
    int32_t (*export_tables)(void* handle, const int32_t* seq_ids, int32_t num_seqs, int32_t max_blocks, int32_t* out_buffer);

    // Single-token decode attention (PTX DP4A / 128-bit uint4 SIMD)
    void (*paged_attention_decode)(
        float* out, const float* q, const void* k_cache, const void* v_cache,
        const int32_t* block_tables, const int32_t* context_lens,
        int32_t num_seqs, int32_t num_heads, int32_t num_kv_heads, int32_t head_dim,
        int32_t block_size, int32_t max_blocks, float scale, int32_t is_int8_kv, void* stream
    );

    // Causal prefill attention with Reverse Query-Tile Attention Scheduling (LPT)
    void (*paged_attention_prefill)(
        float* out, const float* q, const void* k_cache, const void* v_cache,
        const int32_t* block_tables, const int32_t* context_lens,
        int32_t num_seqs, int32_t query_len, int32_t num_heads, int32_t num_kv_heads, int32_t head_dim,
        int32_t block_size, int32_t max_blocks, float scale, int32_t is_int8_kv, int32_t reverse_lpt, void* stream
    );
} hydragen_attention_ops_t;

// Top-level Dynamic Plugin Descriptor
typedef struct {
    int32_t abi_version;
    char plugin_name[64];
    char target_architecture[32];   // e.g. "cuda_sm86", "cpu_avx2", "rocm_gfx906"
    char build_timestamp[32];
    hydragen_device_caps_t caps;
    const hydragen_attention_ops_t* attention_ops;
} hydragen_plugin_descriptor_t;

// Canonical Plugin Entrypoint Symbol
// Every dynamic plugin (.so / .dll) exports this standard C symbol
HYDRAGEN_C_API const hydragen_plugin_descriptor_t* hydragen_get_descriptor(void);

#ifdef __cplusplus
}
#endif
