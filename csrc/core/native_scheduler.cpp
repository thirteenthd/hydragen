// SPDX-License-Identifier: Apache-2.0
#include "native_scheduler.hpp"

namespace hydragen::core {

static double get_current_time_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

NativeInFlightScheduler::NativeInFlightScheduler(int32_t max_slots, std::shared_ptr<paged_kv::PagedKVManager> kv_manager)
    : max_slots_(max_slots), kv_manager_(std::move(kv_manager)) {
    slots_.resize(max_slots_);
    for (int32_t i = 0; i < max_slots_; ++i) {
        slots_[i] = SlotInfo{i, SlotState::IDLE, -1, 0, 0, 0, 0.0, 0.0};
    }
}

uint64_t NativeInFlightScheduler::compute_hash(const int32_t* ids, int32_t len) const {
    if (!ids || len <= 0) return 0;
    // FNV-1a 64-bit hash
    uint64_t h = 14695981039346656037ULL;
    for (int32_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(ids[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

int32_t NativeInFlightScheduler::match_prefix_cache(const int32_t* prompt_ids, int32_t prompt_len) {
    if (!prompt_ids || prompt_len <= 0) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    // Check block-aligned prefix matches (multiples of 16)
    int32_t matched = 0;
    for (int32_t l = (prompt_len / 16) * 16; l >= 16; l -= 16) {
        uint64_t h = compute_hash(prompt_ids, l);
        auto it = prefix_cache_.find(h);
        if (it != prefix_cache_.end()) {
            matched = it->second;
            break;
        }
    }
    return matched;
}

void NativeInFlightScheduler::record_prompt_cache(const int32_t* prompt_ids, int32_t prompt_len) {
    if (!prompt_ids || prompt_len <= 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (int32_t l = 16; l <= prompt_len; l += 16) {
        uint64_t h = compute_hash(prompt_ids, l);
        prefix_cache_[h] = l;
    }
}

int32_t NativeInFlightScheduler::acquire_slot(int32_t prompt_len, int32_t prompt_read, int32_t* out_seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int32_t i = 0; i < max_slots_; ++i) {
        if (slots_[i].state == SlotState::IDLE) {
            int32_t seq_id = next_seq_id_++;
            slots_[i].state = SlotState::PREFILL;
            slots_[i].seq_id = seq_id;
            slots_[i].prompt_tokens = prompt_len;
            slots_[i].prompt_read = prompt_read;
            slots_[i].generated_tokens = 0;
            slots_[i].start_time_s = get_current_time_s();
            slots_[i].first_token_time_s = 0.0;

            if (out_seq_id) {
                *out_seq_id = seq_id;
            }

            // Allocate virtual paged KV blocks
            if (kv_manager_) {
                kv_manager_->allocate_sequence(seq_id, prompt_len);
            }
            return i;
        }
    }
    return -1; // No free slot available
}

bool NativeInFlightScheduler::mark_decoding(int32_t slot_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot_idx < 0 || slot_idx >= max_slots_) return false;
    if (slots_[slot_idx].state == SlotState::PREFILL) {
        slots_[slot_idx].state = SlotState::DECODE;
        slots_[slot_idx].first_token_time_s = get_current_time_s();
        return true;
    }
    return false;
}

bool NativeInFlightScheduler::update_generated(int32_t slot_idx, int32_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot_idx < 0 || slot_idx >= max_slots_) return false;
    slots_[slot_idx].generated_tokens = count;
    if (kv_manager_ && slots_[slot_idx].seq_id > 0) {
        kv_manager_->append_token(slots_[slot_idx].seq_id);
    }
    return true;
}

bool NativeInFlightScheduler::release_slot(int32_t slot_idx, int32_t seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot_idx < 0 || slot_idx >= max_slots_) return false;
    if (slots_[slot_idx].seq_id == seq_id) {
        slots_[slot_idx].state = SlotState::IDLE;
        slots_[slot_idx].seq_id = -1;
        slots_[slot_idx].prompt_tokens = 0;
        slots_[slot_idx].prompt_read = 0;
        slots_[slot_idx].generated_tokens = 0;

        if (kv_manager_) {
            kv_manager_->free_sequence(seq_id);
        }
        return true;
    }
    return false;
}

std::vector<SlotInfo> NativeInFlightScheduler::get_slots_view() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return slots_;
}

int32_t NativeInFlightScheduler::get_active_slots_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int32_t count = 0;
    for (int32_t i = 0; i < max_slots_; ++i) {
        if (slots_[i].state != SlotState::IDLE) {
            count++;
        }
    }
    return count;
}

} // namespace hydragen::core
