// SPDX-License-Identifier: Apache-2.0
#include "paged_kv_c_api.h"
#include "paged_kv_manager.hpp"

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

}
