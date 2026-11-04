// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <algorithm>

namespace hydragen::core {

struct LocalityCandidate {
    int32_t req_id;
    int32_t prompt_len;
    std::vector<int32_t> required_experts;
    double locality_score;
};

class NativeLocalityQueue {
public:
    NativeLocalityQueue() = default;
    ~NativeLocalityQueue() = default;

    void update_resident_experts(const std::vector<int32_t>& resident_eids);
    double calculate_affinity_score(const std::vector<int32_t>& req_experts) const;

    void enqueue(int32_t req_id, int32_t prompt_len, const std::vector<int32_t>& req_experts);
    bool pop_best_candidate(LocalityCandidate* out_candidate);
    size_t size() const;
    void remove_request(int32_t req_id);

private:
    mutable std::mutex mutex_;
    std::unordered_set<int32_t> resident_experts_;
    std::vector<LocalityCandidate> queue_;
};

} // namespace hydragen::core
