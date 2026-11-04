// SPDX-License-Identifier: Apache-2.0
#include "hydragen_engine.hpp"
#include <sstream>
#include <iostream>

namespace hydragen::core {

NativeHydragenEngine::NativeHydragenEngine(const HydragenEngineConfig& config)
    : config_(config) {
    // 1. Initialize Paged KV Block Manager
    paged_kv::PagedKVConfig kv_cfg;
    kv_cfg.block_size = config_.block_size;
    kv_cfg.num_blocks = config_.max_blocks;
    kv_cfg.num_layers = 32;
    kv_cfg.num_kv_heads = 8;
    kv_cfg.head_dim = 128;
    kv_cfg.dtype_bytes = 1; // INT8 KV

    kv_manager_ = std::make_shared<paged_kv::PagedKVManager>(kv_cfg);

    // 2. Initialize In-Flight Batch Scheduler
    scheduler_ = std::make_shared<NativeInFlightScheduler>(config_.batch_slots, kv_manager_);

    // 3. Initialize Locality Queue
    locality_queue_ = std::make_shared<NativeLocalityQueue>();

    // 4. Initialize Hardware Worker
    if (config_.mock_mode) {
        hardware_worker_ = std::make_shared<MockHardwareWorker>();
    } else {
        // Default to Mock for now until in-process shared library or fast IPC is attached
        hardware_worker_ = std::make_shared<MockHardwareWorker>();
    }
    hardware_worker_->initialize(config_.exe_path, config_.args);
}

NativeHydragenEngine::~NativeHydragenEngine() {
    close();
}

int32_t NativeHydragenEngine::submit_request(
    int32_t req_id,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    int32_t max_new,
    const int32_t* expert_ids,
    int32_t num_experts
) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    if (!is_running_) return -1;

    // Check prefix cache reuse
    int32_t reused = scheduler_->match_prefix_cache(prompt_ids, prompt_len);
    int32_t prompt_read = prompt_len - reused;
    if (prompt_read < 0) prompt_read = 0;

    int32_t seq_id = -1;
    int32_t slot_idx = scheduler_->acquire_slot(prompt_len, prompt_read, &seq_id);

    if (slot_idx < 0) {
        // Enqueue into locality queue
        std::vector<int32_t> experts;
        if (expert_ids && num_experts > 0) {
            experts.assign(expert_ids, expert_ids + num_experts);
        }
        locality_queue_->enqueue(req_id, prompt_len, experts);
        return 0; // Enqueued, waiting for slot
    }

    // Record prefix for subsequent requests
    scheduler_->record_prompt_cache(prompt_ids, prompt_len);

    ActiveRequest req;
    req.req_id = req_id;
    req.slot_idx = slot_idx;
    req.seq_id = seq_id;
    req.max_new = max_new;
    req.generated = 0;
    req.last_token = (prompt_ids && prompt_len > 0) ? prompt_ids[prompt_len - 1] : 100;
    req.is_finished = false;

    active_requests_[req_id] = req;
    return slot_idx + 1; // 1-based active slot index
}

int32_t NativeHydragenEngine::step(int32_t* out_req_id, int32_t* out_token_id, int32_t* out_is_finished) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    if (!is_running_ || active_requests_.empty()) {
        return 0; // Nothing to step
    }

    // Step across active requests in round-robin fashion
    auto it = active_requests_.begin();
    if (round_robin_slot_ > 0 && active_requests_.size() > 1) {
        size_t advance = round_robin_slot_ % active_requests_.size();
        std::advance(it, advance);
    }
    round_robin_slot_++;

    ActiveRequest& req = it->second;

    if (req.generated == 0) {
        scheduler_->mark_decoding(req.slot_idx);
    }

    // Generate token through hardware worker
    int32_t next_token = hardware_worker_->step_token(req.slot_idx, req.last_token);
    req.last_token = next_token;
    req.generated++;

    scheduler_->update_generated(req.slot_idx, req.generated);

    if (req.generated >= req.max_new) {
        req.is_finished = true;
    }

    if (out_req_id) *out_req_id = req.req_id;
    if (out_token_id) *out_token_id = next_token;
    if (out_is_finished) *out_is_finished = req.is_finished ? 1 : 0;

    if (req.is_finished) {
        int32_t finished_req_id = req.req_id;
        int32_t released_slot = req.slot_idx;
        int32_t released_seq = req.seq_id;

        scheduler_->release_slot(released_slot, released_seq);
        active_requests_.erase(finished_req_id);

        // Check if pending request can now be admitted from locality queue
        LocalityCandidate cand;
        if (locality_queue_->pop_best_candidate(&cand)) {
            int32_t new_seq_id = -1;
            int32_t new_slot = scheduler_->acquire_slot(cand.prompt_len, cand.prompt_len, &new_seq_id);
            if (new_slot >= 0) {
                ActiveRequest new_req;
                new_req.req_id = cand.req_id;
                new_req.slot_idx = new_slot;
                new_req.seq_id = new_seq_id;
                new_req.max_new = 32;
                new_req.generated = 0;
                new_req.last_token = 100;
                new_req.is_finished = false;
                active_requests_[cand.req_id] = new_req;
            }
        }
    }

    return 1;
}

bool NativeHydragenEngine::cancel_request(int32_t req_id) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    auto it = active_requests_.find(req_id);
    if (it != active_requests_.end()) {
        scheduler_->release_slot(it->second.slot_idx, it->second.seq_id);
        active_requests_.erase(it);
        return true;
    }
    locality_queue_->remove_request(req_id);
    return true;
}

std::string NativeHydragenEngine::get_telemetry_json() const {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    auto mem_stats = kv_manager_->get_memory_stats();
    auto slots_view = scheduler_->get_slots_view();

    std::stringstream ss;
    ss << "{";
    ss << "\"engine\":\"hydragen_native\",";
    ss << "\"engine_type\":\"cxx_core\",";
    ss << "\"max_slots\":" << config_.batch_slots << ",";
    ss << "\"active_slots\":" << scheduler_->get_active_slots_count() << ",";
    ss << "\"queued_requests\":" << locality_queue_->size() << ",";
    ss << "\"used_blocks\":" << mem_stats.allocated_blocks << ",";
    ss << "\"free_blocks\":" << mem_stats.free_blocks << ",";
    ss << "\"total_blocks\":" << mem_stats.total_blocks << ",";
    ss << "\"hardware_worker\":\"" << hardware_worker_->get_worker_type() << "\"";
    ss << "}";
    return ss.str();
}

void NativeHydragenEngine::close() {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    if (!is_running_) return;
    is_running_ = false;
    if (hardware_worker_) {
        hardware_worker_->close();
    }
    active_requests_.clear();
}

bool NativeHydragenEngine::is_alive() const {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    return is_running_ && hardware_worker_ && hardware_worker_->is_alive();
}

} // namespace hydragen::core
