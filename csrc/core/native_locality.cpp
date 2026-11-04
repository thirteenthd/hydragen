// SPDX-License-Identifier: Apache-2.0
#include "native_locality.hpp"

namespace hydragen::core {

void NativeLocalityQueue::update_resident_experts(const std::vector<int32_t>& resident_eids) {
    std::lock_guard<std::mutex> lock(mutex_);
    resident_experts_.clear();
    for (int32_t eid : resident_eids) {
        resident_experts_.insert(eid);
    }
}

double NativeLocalityQueue::calculate_affinity_score(const std::vector<int32_t>& req_experts) const {
    if (req_experts.empty() || resident_experts_.empty()) {
        return 0.0;
    }
    int32_t hits = 0;
    for (int32_t eid : req_experts) {
        if (resident_experts_.find(eid) != resident_experts_.end()) {
            hits++;
        }
    }
    // Jaccard similarity or hit ratio
    return static_cast<double>(hits) / static_cast<double>(req_experts.size());
}

void NativeLocalityQueue::enqueue(int32_t req_id, int32_t prompt_len, const std::vector<int32_t>& req_experts) {
    std::lock_guard<std::mutex> lock(mutex_);
    LocalityCandidate cand;
    cand.req_id = req_id;
    cand.prompt_len = prompt_len;
    cand.required_experts = req_experts;
    cand.locality_score = calculate_affinity_score(req_experts);
    queue_.push_back(std::move(cand));
}

bool NativeLocalityQueue::pop_best_candidate(LocalityCandidate* out_candidate) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return false;

    // Recalculate affinity scores and find candidate with max score
    auto best_it = queue_.begin();
    double max_score = -1.0;

    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        it->locality_score = calculate_affinity_score(it->required_experts);
        if (it->locality_score > max_score) {
            max_score = it->locality_score;
            best_it = it;
        }
    }

    if (out_candidate) {
        *out_candidate = *best_it;
    }
    queue_.erase(best_it);
    return true;
}

size_t NativeLocalityQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void NativeLocalityQueue::remove_request(int32_t req_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.erase(
        std::remove_if(queue_.begin(), queue_.end(), [req_id](const LocalityCandidate& c) {
            return c.req_id == req_id;
        }),
        queue_.end()
    );
}

} // namespace hydragen::core
