// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace hydragen::paged_attention {

// Single sequence paged attention decode step
void launch_paged_attention_decode_f16(
    float* out,                      // [num_seqs, num_heads, head_dim]
    const float* q,                  // [num_seqs, num_heads, head_dim]
    const void* k_cache,             // [num_blocks, num_kv_heads, block_size, head_dim] (int8 or f16)
    const void* v_cache,             // [num_blocks, num_kv_heads, block_size, head_dim] (int8 or f16)
    const int32_t* block_tables,     // [num_seqs, max_blocks]
    const int32_t* context_lens,     // [num_seqs]
    int32_t num_seqs,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t block_size,
    int32_t max_blocks,
    float scale,
    bool is_int8_kv,
    cudaStream_t stream
);

// Multi-token prefill paged attention with Reverse Query-Tile Attention Scheduling (LPT)
void launch_paged_attention_prefill_f16(
    float* out,                      // [num_seqs, query_len, num_heads, head_dim]
    const float* q,                  // [num_seqs, query_len, num_heads, head_dim]
    const void* k_cache,             // [num_blocks, num_kv_heads, block_size, head_dim]
    const void* v_cache,             // [num_blocks, num_kv_heads, block_size, head_dim]
    const int32_t* block_tables,     // [num_seqs, max_blocks]
    const int32_t* context_lens,     // [num_seqs] past context length before prefill chunk
    int32_t num_seqs,
    int32_t query_len,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t block_size,
    int32_t max_blocks,
    float scale,
    bool is_int8_kv,
    bool reverse_lpt,
    cudaStream_t stream
);

} // namespace hydragen::paged_attention
