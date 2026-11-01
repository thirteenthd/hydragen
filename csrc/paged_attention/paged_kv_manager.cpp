// SPDX-License-Identifier: Apache-2.0
#include "paged_kv_manager.hpp"
#include <cstring>
#include <algorithm>

namespace hydragen::paged_kv {

static int64_t current_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

PagedKVManager::PagedKVManager(const PagedKVConfig& config)
    : config_(config) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.reserve(config_.num_blocks);
    block_metadata_.resize(config_.num_blocks);

    for (int32_t i = config_.num_blocks - 1; i >= 0; --i) {
        free_blocks_.push_back(i);
        block_metadata_[i].block_id = i;
        block_metadata_[i].ref_count = 0;
        block_metadata_[i].last_accessed_ns = 0;
        block_metadata_[i].is_cow = false;
    }
}

int32_t PagedKVManager::allocate_free_block() {
    if (free_blocks_.empty()) {
        return -1; // Out of physical blocks
    }
    int32_t block_id = free_blocks_.back();
    free_blocks_.pop_back();

    auto& meta = block_metadata_[block_id];
    meta.ref_count = 1;
    meta.last_accessed_ns = current_time_ns();
    meta.is_cow = false;
    return block_id;
}

void PagedKVManager::release_block(int32_t block_id) {
    if (block_id < 0 || block_id >= static_cast<int32_t>(block_metadata_.size())) {
        return;
    }
    auto& meta = block_metadata_[block_id];
    if (meta.ref_count > 0) {
        meta.ref_count--;
    }
    if (meta.ref_count == 0) {
        meta.is_cow = false;
        meta.last_accessed_ns = 0;
        free_blocks_.push_back(block_id);
    }
}

bool PagedKVManager::allocate_sequence(int32_t seq_id, int32_t initial_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (sequence_tables_.find(seq_id) != sequence_tables_.end()) {
        return false; // Already allocated
    }

    int32_t needed_blocks = (initial_tokens + config_.block_size - 1) / config_.block_size;
    if (needed_blocks == 0) needed_blocks = 1; // At least 1 block

    if (static_cast<int32_t>(free_blocks_.size()) < needed_blocks) {
        return false; // Insufficient VRAM pages
    }

    SequencePageTable table;
    table.seq_id = seq_id;
    table.num_tokens = initial_tokens;
    table.physical_block_ids.reserve(needed_blocks);

    for (int32_t i = 0; i < needed_blocks; ++i) {
        int32_t blk = allocate_free_block();
        table.physical_block_ids.push_back(blk);
    }

    sequence_tables_[seq_id] = std::move(table);
    return true;
}

bool PagedKVManager::append_token(int32_t seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sequence_tables_.find(seq_id);
    if (it == sequence_tables_.end()) {
        return false;
    }

    auto& table = it->second;
    int32_t cur_tokens = table.num_tokens;
    int32_t cur_capacity = static_cast<int32_t>(table.physical_block_ids.size()) * config_.block_size;

    if (cur_tokens < cur_capacity) {
        // Appends within the existing block.
        // Check if the current block is shared via Copy-on-Write!
        int32_t last_blk_id = table.physical_block_ids.back();
        auto& meta = block_metadata_[last_blk_id];

        if (meta.ref_count > 1) {
            // Write-fault: This block is shared with other child/parent sequences.
            // Allocate a new private block for this sequence (Copy-on-Write fork).
            int32_t new_blk_id = allocate_free_block();
            if (new_blk_id < 0) return false; // OOM

            release_block(last_blk_id);
            table.physical_block_ids.back() = new_blk_id;
        }
        table.num_tokens++;
        return true;
    }

    // Need a new block
    int32_t new_blk_id = allocate_free_block();
    if (new_blk_id < 0) return false;

    table.physical_block_ids.push_back(new_blk_id);
    table.num_tokens++;
    return true;
}

bool PagedKVManager::fork_sequence(int32_t parent_seq_id, int32_t child_seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sequence_tables_.find(parent_seq_id);
    if (it == sequence_tables_.end()) {
        return false;
    }
    if (sequence_tables_.find(child_seq_id) != sequence_tables_.end()) {
        return false; // Child ID collision
    }

    const auto& parent_table = it->second;
    SequencePageTable child_table;
    child_table.seq_id = child_seq_id;
    child_table.num_tokens = parent_table.num_tokens;
    child_table.physical_block_ids = parent_table.physical_block_ids;

    // Increment ref_count on all shared physical blocks (Zero-Copy Fork)
    for (int32_t blk_id : child_table.physical_block_ids) {
        auto& meta = block_metadata_[blk_id];
        meta.ref_count++;
        meta.is_cow = true;
        meta.last_accessed_ns = current_time_ns();
    }

    sequence_tables_[child_seq_id] = std::move(child_table);
    return true;
}

bool PagedKVManager::free_sequence(int32_t seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sequence_tables_.find(seq_id);
    if (it == sequence_tables_.end()) {
        return false;
    }

    for (int32_t blk_id : it->second.physical_block_ids) {
        release_block(blk_id);
    }
    sequence_tables_.erase(it);
    return true;
}

const std::vector<int32_t>* PagedKVManager::get_block_table(int32_t seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sequence_tables_.find(seq_id);
    if (it != sequence_tables_.end()) {
        return &(it->second.physical_block_ids);
    }
    return nullptr;
}

int32_t PagedKVManager::get_num_tokens(int32_t seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sequence_tables_.find(seq_id);
    if (it != sequence_tables_.end()) {
        return it->second.num_tokens;
    }
    return 0;
}

int32_t PagedKVManager::get_num_blocks(int32_t seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sequence_tables_.find(seq_id);
    if (it != sequence_tables_.end()) {
        return static_cast<int32_t>(it->second.physical_block_ids.size());
    }
    return 0;
}

bool PagedKVManager::export_batch_block_tables(
    const std::vector<int32_t>& seq_ids,
    int32_t max_blocks_per_seq,
    int32_t* out_buffer
) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::memset(out_buffer, -1, seq_ids.size() * max_blocks_per_seq * sizeof(int32_t));

    for (size_t i = 0; i < seq_ids.size(); ++i) {
        int32_t sid = seq_ids[i];
        auto it = sequence_tables_.find(sid);
        if (it != sequence_tables_.end()) {
            const auto& blocks = it->second.physical_block_ids;
            size_t copy_cnt = std::min(blocks.size(), static_cast<size_t>(max_blocks_per_seq));
            std::memcpy(out_buffer + i * max_blocks_per_seq, blocks.data(), copy_cnt * sizeof(int32_t));
        }
    }
    return true;
}

MemoryStats PagedKVManager::get_memory_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    MemoryStats stats;
    stats.total_blocks = config_.num_blocks;
    stats.free_blocks = static_cast<int32_t>(free_blocks_.size());
    stats.allocated_blocks = stats.total_blocks - stats.free_blocks;
    stats.active_sequences = static_cast<int32_t>(sequence_tables_.size());

    // Single block bytes = block_size * num_layers * 2 (K+V) * num_kv_heads * head_dim * dtype_bytes
    size_t block_bytes = static_cast<size_t>(config_.block_size)
                       * config_.num_layers
                       * 2
                       * config_.num_kv_heads
                       * config_.head_dim
                       * config_.dtype_bytes;

    stats.total_memory_bytes = static_cast<size_t>(stats.total_blocks) * block_bytes;
    stats.used_memory_bytes = static_cast<size_t>(stats.allocated_blocks) * block_bytes;

    int32_t cow_count = 0;
    int32_t total_tail_unused_tokens = 0;
    int32_t total_active_tokens = 0;

    for (const auto& meta : block_metadata_) {
        if (meta.ref_count > 1) {
            cow_count++;
        }
    }
    stats.shared_cow_blocks = cow_count;

    for (const auto& kv : sequence_tables_) {
        int32_t cap = static_cast<int32_t>(kv.second.physical_block_ids.size()) * config_.block_size;
        int32_t unused = cap - kv.second.num_tokens;
        total_tail_unused_tokens += unused;
        total_active_tokens += kv.second.num_tokens;
    }

    if (total_active_tokens + total_tail_unused_tokens > 0) {
        stats.fragmentation_ratio = static_cast<double>(total_tail_unused_tokens) /
            static_cast<double>(total_active_tokens + total_tail_unused_tokens);
    } else {
        stats.fragmentation_ratio = 0.0;
    }

    return stats;
}

} // namespace hydragen::paged_kv
