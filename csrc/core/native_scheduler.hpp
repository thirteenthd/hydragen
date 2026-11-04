// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include "../paged_attention/paged_kv_manager.hpp"

namespace hydragen::core {

enum class SlotState {
    IDLE = 0,
    PREFILL = 1,
    DECODE = 2,
    COMPLETED = 3,
    CANCELLED = 4
};

struct SlotInfo {
    int32_t slot_idx;
    SlotState state;
    int32_t seq_id;
    int32_t prompt_tokens;
    int32_t prompt_read;
    int32_t generated_tokens;
    double start_time_s;
    double first_token_time_s;
};

class NativeInFlightScheduler {
public:
    NativeInFlightScheduler(int32_t max_slots, std::shared_ptr<paged_kv::PagedKVManager> kv_manager);
    ~NativeInFlightScheduler() = default;

    // Slot allocation and lifecycle
    int32_t acquire_slot(int32_t prompt_len, int32_t prompt_read, int32_t* out_seq_id);
    bool mark_decoding(int32_t slot_idx);
    bool update_generated(int32_t slot_idx, int32_t count);
    bool release_slot(int32_t slot_idx, int32_t seq_id);

    // Prefix cache matching
    int32_t match_prefix_cache(const int32_t* prompt_ids, int32_t prompt_len);
    void record_prompt_cache(const int32_t* prompt_ids, int32_t prompt_len);

    // Telemetry
    std::vector<SlotInfo> get_slots_view() const;
    int32_t get_active_slots_count() const;
    int32_t get_max_slots() const { return max_slots_; }

private:
    uint64_t compute_hash(const int32_t* ids, int32_t len) const;

    int32_t max_slots_;
    std::shared_ptr<paged_kv::PagedKVManager> kv_manager_;
    mutable std::mutex mutex_;
    std::atomic<int32_t> next_seq_id_{1};

    std::vector<SlotInfo> slots_;
    std::unordered_map<uint64_t, int32_t> prefix_cache_;
};

} // namespace hydragen::core
