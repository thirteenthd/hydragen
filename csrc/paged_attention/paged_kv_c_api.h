// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cstddef>

#ifdef _WIN32
#define PAGED_KV_API __declspec(dllexport)
#else
#define PAGED_KV_API __attribute__((visibility("default")))
#endif

extern "C" {

struct CPagedKVStats {
    int32_t total_blocks;
    int32_t free_blocks;
    int32_t allocated_blocks;
    int32_t shared_cow_blocks;
    int32_t active_sequences;
    uint64_t total_memory_bytes;
    uint64_t used_memory_bytes;
    double fragmentation_ratio;
};

PAGED_KV_API void* paged_kv_create(
    int32_t block_size,
    int32_t num_blocks,
    int32_t num_layers,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t dtype_bytes
);

PAGED_KV_API void paged_kv_destroy(void* handle);

PAGED_KV_API int32_t paged_kv_allocate_seq(void* handle, int32_t seq_id, int32_t initial_tokens);

PAGED_KV_API int32_t paged_kv_append_token(void* handle, int32_t seq_id);

PAGED_KV_API int32_t paged_kv_fork_seq(void* handle, int32_t parent_seq_id, int32_t child_seq_id);

PAGED_KV_API int32_t paged_kv_free_seq(void* handle, int32_t seq_id);

PAGED_KV_API int32_t paged_kv_get_num_tokens(void* handle, int32_t seq_id);

PAGED_KV_API int32_t paged_kv_get_num_blocks(void* handle, int32_t seq_id);

PAGED_KV_API int32_t paged_kv_export_tables(
    void* handle,
    const int32_t* seq_ids,
    int32_t num_seqs,
    int32_t max_blocks,
    int32_t* out_buffer
);

PAGED_KV_API void paged_kv_get_stats(void* handle, CPagedKVStats* out_stats);

PAGED_KV_API void paged_attention_decode(
    float* out,
    const float* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* block_tables,
    const int32_t* context_lens,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t block_size,
    int32_t max_blocks,
    float scale,
    int32_t is_int8_kv,
    void* stream
);

}
