// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include "hardware_worker.hpp"
#include "native_scheduler.hpp"
#include "native_locality.hpp"
#include "../paged_attention/paged_kv_manager.hpp"

namespace hydragen::core {

struct HydragenEngineConfig {
    int32_t batch_slots = 4;
    int32_t block_size = 16;
    int32_t max_blocks = 4096;
    int32_t max_context = 65536;
    std::string exe_path = "bin/strata";
    std::vector<std::string> args;
    bool mock_mode = false;
};

struct ActiveRequest {
    int32_t req_id;
    int32_t slot_idx;
    int32_t seq_id;
    int32_t max_new;
    int32_t generated;
    int32_t last_token;
    bool is_finished;
};

class NativeHydragenEngine {
public:
    explicit NativeHydragenEngine(const HydragenEngineConfig& config);
    ~NativeHydragenEngine();

    // Request lifecycle
    int32_t submit_request(
        int32_t req_id,
        const int32_t* prompt_ids,
        int32_t prompt_len,
        int32_t max_new,
        const int32_t* expert_ids,
        int32_t num_experts
    );

    int32_t step(int32_t* out_req_id, int32_t* out_token_id, int32_t* out_is_finished);
    bool cancel_request(int32_t req_id);

    // Subsystems & Telemetry
    std::shared_ptr<NativeInFlightScheduler> get_scheduler() const { return scheduler_; }
    std::shared_ptr<paged_kv::PagedKVManager> get_kv_manager() const { return kv_manager_; }
    std::shared_ptr<IHardwareWorker> get_hardware_worker() const { return hardware_worker_; }

    std::string get_telemetry_json() const;
    void close();
    bool is_alive() const;

private:
    HydragenEngineConfig config_;
    std::shared_ptr<paged_kv::PagedKVManager> kv_manager_;
    std::shared_ptr<NativeInFlightScheduler> scheduler_;
    std::shared_ptr<NativeLocalityQueue> locality_queue_;
    std::shared_ptr<IHardwareWorker> hardware_worker_;

    mutable std::mutex engine_mutex_;
    std::unordered_map<int32_t, ActiveRequest> active_requests_; // req_id -> ActiveRequest
    int32_t round_robin_slot_{0};
    bool is_running_{true};
};

} // namespace hydragen::core
