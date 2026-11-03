// SPDX-License-Identifier: Apache-2.0
#include "paged_kv_c_api.h"
#include "paged_kv_manager.hpp"
#include "paged_attention_kernel.cuh"
#include "hydragen_plugin.h"

using namespace hydragen::paged_kv;

extern "C" {

void* paged_kv_create(
    int32_t block_size,
    int32_t num_blocks,
    int32_t num_layers,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t dtype_bytes
) {
    PagedKVConfig cfg;
    cfg.block_size = block_size;
    cfg.num_blocks = num_blocks;
    cfg.num_layers = num_layers;
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_dim = head_dim;
    cfg.dtype_bytes = dtype_bytes;

    auto* mgr = new PagedKVManager(cfg);
    return static_cast<void*>(mgr);
}

void paged_kv_destroy(void* handle) {
    if (!handle) return;
    auto* mgr = static_cast<PagedKVManager*>(handle);
    delete mgr;
}

int32_t paged_kv_allocate_seq(void* handle, int32_t seq_id, int32_t initial_tokens) {
    if (!handle) return 0;
    auto* mgr = static_cast<PagedKVManager*>(handle);
    return mgr->allocate_sequence(seq_id, initial_tokens) ? 1 : 0;
}

int32_t paged_kv_append_token(void* handle, int32_t seq_id) {
    if (!handle) return 0;
    auto* mgr = static_cast<PagedKVManager*>(handle);
    return mgr->append_token(seq_id) ? 1 : 0;
}

int32_t paged_kv_fork_seq(void* handle, int32_t parent_seq_id, int32_t child_seq_id) {
    if (!handle) return 0;
    auto* mgr = static_cast<PagedKVManager*>(handle);
    return mgr->fork_sequence(parent_seq_id, child_seq_id) ? 1 : 0;
}

int32_t paged_kv_free_seq(void* handle, int32_t seq_id) {
    if (!handle) return 0;
    auto* mgr = static_cast<PagedKVManager*>(handle);
    return mgr->free_sequence(seq_id) ? 1 : 0;
}

int32_t paged_kv_get_num_tokens(void* handle, int32_t seq_id) {
    if (!handle) return 0;
    auto* mgr = static_cast<PagedKVManager*>(handle);
    return mgr->get_num_tokens(seq_id);
}

int32_t paged_kv_get_num_blocks(void* handle, int32_t seq_id) {
    if (!handle) return 0;
    auto* mgr = static_cast<PagedKVManager*>(handle);
    return mgr->get_num_blocks(seq_id);
}

int32_t paged_kv_export_tables(
    void* handle,
    const int32_t* seq_ids,
    int32_t num_seqs,
    int32_t max_blocks,
    int32_t* out_buffer
) {
    if (!handle || !seq_ids || !out_buffer) return 0;
    auto* mgr = static_cast<PagedKVManager*>(handle);
    std::vector<int32_t> sids(seq_ids, seq_ids + num_seqs);
    return mgr->export_batch_block_tables(sids, max_blocks, out_buffer) ? 1 : 0;
}

void paged_kv_get_stats(void* handle, CPagedKVStats* out_stats) {
    if (!handle || !out_stats) return;
    auto* mgr = static_cast<PagedKVManager*>(handle);
    auto stats = mgr->get_memory_stats();

    out_stats->total_blocks = stats.total_blocks;
    out_stats->free_blocks = stats.free_blocks;
    out_stats->allocated_blocks = stats.allocated_blocks;
    out_stats->shared_cow_blocks = stats.shared_cow_blocks;
    out_stats->active_sequences = stats.active_sequences;
    out_stats->total_memory_bytes = stats.total_memory_bytes;
    out_stats->used_memory_bytes = stats.used_memory_bytes;
    out_stats->fragmentation_ratio = stats.fragmentation_ratio;
}

void paged_attention_decode(
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
) {
    hydragen::paged_attention::launch_paged_attention_decode_f16(
        out, q, k_cache, v_cache, block_tables, context_lens,
        num_seqs, num_heads, num_kv_heads, head_dim, block_size, max_blocks,
        scale, is_int8_kv != 0, static_cast<cudaStream_t>(stream)
    );
}

void paged_attention_prefill(
    float* out,
    const float* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* block_tables,
    const int32_t* context_lens,
    int32_t num_seqs,
    int32_t query_len,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t block_size,
    int32_t max_blocks,
    float scale,
    int32_t is_int8_kv,
    int32_t reverse_lpt,
    void* stream
) {
    hydragen::paged_attention::launch_paged_attention_prefill_f16(
        out, q, k_cache, v_cache, block_tables, context_lens,
        num_seqs, query_len, num_heads, num_kv_heads, head_dim, block_size, max_blocks,
        scale, is_int8_kv != 0, reverse_lpt != 0, static_cast<cudaStream_t>(stream)
    );
}

// ---------------------------------------------------------------------------
// Cross-Platform C-ABI Dynamic Hardware Plugin Implementation
// ---------------------------------------------------------------------------

static const hydragen_attention_ops_t g_cuda_attention_ops = {
    paged_kv_create,
    paged_kv_destroy,
    paged_kv_allocate_seq,
    paged_kv_append_token,
    paged_kv_fork_seq,
    paged_kv_free_seq,
    paged_kv_get_num_tokens,
    paged_kv_get_num_blocks,
    paged_kv_export_tables,
    paged_attention_decode,
    paged_attention_prefill,
};

static hydragen_plugin_descriptor_t g_cuda_plugin_descriptor = {
    HYDRAGEN_PLUGIN_ABI_VERSION,
    "hydragen_cuda_plugin",
    "cuda_sm86",
    __DATE__ " " __TIME__,
    {
        HYDRAGEN_DEVICE_CUDA,
        "NVIDIA GeForce RTX 3080 Ti (Ampere SM_86)",
        8, 6,
        12288ULL * 1024ULL * 1024ULL, // 12 GiB VRAM
        0,                            // Populated at runtime
        80,                           // SM count
        1,                            // has_dp4a (Hardware PTX DP4A SIMD)
        1,                            // has_cp_async (SM_80+ Async Copy)
        1,                            // has_wmma (Tensor Cores)
        0                             // has_avx512
    },
    &g_cuda_attention_ops
};

const hydragen_plugin_descriptor_t* hydragen_get_descriptor(void) {
    return &g_cuda_plugin_descriptor;
}

}
