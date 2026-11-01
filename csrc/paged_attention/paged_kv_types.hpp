// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>

namespace hydragen::paged_kv {

constexpr int DEFAULT_BLOCK_SIZE = 16; // 16 tokens per physical KV page

struct BlockMetadata {
    int32_t block_id{-1};
    int32_t ref_count{0};
    int64_t last_accessed_ns{0};
    bool is_cow{false}; // Copy-on-Write flag
};

struct SequencePageTable {
    int32_t seq_id{-1};
    int32_t num_tokens{0};
    std::vector<int32_t> physical_block_ids; // Maps logical block index -> physical block ID
};

struct PagedKVConfig {
    int32_t block_size{DEFAULT_BLOCK_SIZE};
    int32_t num_blocks{4096};         // e.g. 4096 blocks * 16 = 65,536 tokens
    int32_t num_layers{48};
    int32_t num_kv_heads{8};          // Grouped-Query Attention (GQA)
    int32_t head_dim{128};            // 8 * 128 = 1024 KV dim
    int32_t dtype_bytes{1};           // 1 for int8, 2 for fp16
};

struct MemoryStats {
    int32_t total_blocks{0};
    int32_t free_blocks{0};
    int32_t allocated_blocks{0};
    int32_t shared_cow_blocks{0};
    int32_t active_sequences{0};
    size_t total_memory_bytes{0};
    size_t used_memory_bytes{0};
    double fragmentation_ratio{0.0};
};

} // namespace hydragen::paged_kv
