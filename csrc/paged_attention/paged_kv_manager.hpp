// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "paged_kv_types.hpp"
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <chrono>

namespace hydragen::paged_kv {

class PagedKVManager {
public:
    explicit PagedKVManager(const PagedKVConfig& config);
    ~PagedKVManager() = default;

    // Sequence Lifecycle
    bool allocate_sequence(int32_t seq_id, int32_t initial_tokens);
    bool append_token(int32_t seq_id);
    bool fork_sequence(int32_t parent_seq_id, int32_t child_seq_id);
    bool free_sequence(int32_t seq_id);

    // Block Table Queries for GPU Kernels
    const std::vector<int32_t>* get_block_table(int32_t seq_id) const;
    int32_t get_num_tokens(int32_t seq_id) const;
    int32_t get_num_blocks(int32_t seq_id) const;

    // Flatten block tables into a contiguous buffer for batch kernel dispatch: [batch_size, max_blocks]
    bool export_batch_block_tables(
        const std::vector<int32_t>& seq_ids,
        int32_t max_blocks_per_seq,
        int32_t* out_buffer
    ) const;

    // Telemetry & Statistics
    MemoryStats get_memory_stats() const;
    const PagedKVConfig& get_config() const { return config_; }

private:
    int32_t allocate_free_block();
    void release_block(int32_t block_id);

    PagedKVConfig config_;
    mutable std::mutex mutex_;

    // Free block pool (stack for L1 cache locality)
    std::vector<int32_t> free_blocks_;

    // Global physical block metadata [num_blocks]
    std::vector<BlockMetadata> block_metadata_;

    // Active sequences map: seq_id -> SequencePageTable
    std::unordered_map<int32_t, SequencePageTable> sequence_tables_;
};

} // namespace hydragen::paged_kv
