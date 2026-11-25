// include/strata/program/decode_governor.hpp - Adaptive batch decode-width governor (P2-D1).
//
// Separates attached survivorship slots M (the KV pools kept alive across --batch M)
// from the active simultaneous decode width N_active in [1..M].  Maintains a sticky
// ascending-sorted cohort for CUDA batch-graph and MoE expert locality, rotates fairly
// every `cohort_quantum_ms` when runnable slots exceed N_active, and adapts N_active
// via EMA + hysteresis against:
//   1. Hard Physics Brake (`hit_floor_pct`, default 86.0%) to avoid the PCIe thrashing cliff.
//   2. Optional per-stream decode speed floor (`min_stream_tps`, default 0 = throughput mode).
//
// P3-3A adds two things the shipped governor lacked on a 12 GiB card:
//   * a warmup grace window count: a cohort rotation or a width change brings in (or keeps)
//     slots whose adaptive tier has not warmed yet, and their first one or two windows are
//     cold by construction.  Without grace those windows drag the EMA under `hit_floor_pct`
//     and the governor sheds the width it just paid a rotation to get.
//   * optional prefix-affinity cohort packing: slots that share a prompt prefix route to the
//     same experts, so co-scheduling them raises routed-expert VRAM hit for the whole cohort.
//     Aging (`affinity_max_wait`) keeps that from starving a prefix group that never wins.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strata/core/hg_env.hpp"

namespace strata::program {

struct DecodeGovernorConfig {
    bool enabled = false;
    double min_stream_tps = 0.0;        ///< 0.0 = max aggregate throughput mode (physics brake only)
    double hit_floor_pct = 86.0;        ///< MoE resident/VRAM hit-% floor (Hard Physics Brake)
    double cohort_quantum_ms = 2000.0;  ///< Rotation period when active slots > target_n (0 = sticky/FIFO)
    int initial_width = 4;              ///< Lazy-init N_active before probing upward
    int eval_every = 4;                 ///< Decode windows per evaluation step
    int max_width = 64;                 ///< Hard ceiling (also clamped by win_slot_cap)
    double ema_alpha = 0.25;            ///< EMA smoothing factor for hit% and per-stream tok/s
    double step_up_margin_pct = 1.5;    ///< Hysteresis above hit_floor_pct required to widen (+1)
    double step_up_tps_margin = 1.15;   ///< Hysteresis multiplier above min_stream_tps required to widen
    int grace_windows = 3;              ///< P3-3A: windows skipped (not measured) after a resize or a cohort rotation; 0 = off
    bool affinity = false;              ///< P3-3A: prefer waiting slots whose prompt prefix matches the cohort's
    int affinity_max_wait = 2;          ///< P3-3A: picks a waiting slot has been passed over before its prefix no longer matters
};

/// Parse `DecodeGovernorConfig` from `HG_*` / `STRATA_*` env vars and CLI `--min-stream-tps`.
inline DecodeGovernorConfig parse_decode_governor_config(double cli_min_stream_tps = 0.0) {
    DecodeGovernorConfig cfg;

    auto read_double = [](const char* name) -> double {
        const char* v = ::hg_getenv(name);
        if (!v || !*v) return -1.0;
        char* end = nullptr;
        const double d = std::strtod(v, &end);
        if (end == v || !std::isfinite(d)) return -1.0;
        return d;
    };
    auto read_pos_int = [](const char* name) -> int {
        const char* v = ::hg_getenv(name);
        if (!v || !*v) return -1;
        char* end = nullptr;
        const long x = std::strtol(v, &end, 10);
        if (end == v || x <= 0 || x > 4096) return -1;
        return (int) x;
    };
    auto read_int_ge0 = [](const char* name) -> int {   // 0 is a real value here (grace off, aging off)
        const char* v = ::hg_getenv(name);
        if (!v || !*v) return -1;
        char* end = nullptr;
        const long x = std::strtol(v, &end, 10);
        if (end == v || x < 0 || x > 4096) return -1;
        return (int) x;
    };

    const double env_tps = read_double("STRATA_MIN_STREAM_TPS");
    if (cli_min_stream_tps > 0.0 && std::isfinite(cli_min_stream_tps)) {
        cfg.min_stream_tps = cli_min_stream_tps;
    } else if (env_tps >= 0.0) {
        cfg.min_stream_tps = env_tps;
    }

    const double env_hit = read_double("STRATA_HIT_FLOOR_PCT");
    if (env_hit > 0.0 && env_hit <= 100.0) cfg.hit_floor_pct = env_hit;

    const double env_quantum = read_double("STRATA_COHORT_QUANTUM_MS");
    if (env_quantum >= 0.0) cfg.cohort_quantum_ms = env_quantum;

    if (const int init_n = read_pos_int("STRATA_DECODE_INIT_N"); init_n > 0)
        cfg.initial_width = init_n;
    if (const int eval_w = read_pos_int("STRATA_DECODE_EVAL_WINDOWS"); eval_w > 0)
        cfg.eval_every = eval_w;
    if (const int grace = read_int_ge0("STRATA_DECODE_GRACE_WINDOWS"); grace >= 0)
        cfg.grace_windows = grace;
    if (const int amw = read_int_ge0("STRATA_DECODE_AFFINITY_MAX_WAIT"); amw >= 0)
        cfg.affinity_max_wait = amw;
    if (const char* aff = ::hg_getenv("STRATA_DECODE_AFFINITY"); aff != nullptr && aff[0] != '\0' && aff[0] != '0')
        cfg.affinity = true;

    const char* auto_env = ::hg_getenv("STRATA_DECODE_AUTO");
    const bool explicit_on = (auto_env != nullptr && auto_env[0] == '1');
    const bool explicit_off = (auto_env != nullptr && auto_env[0] == '0');
    cfg.enabled = !explicit_off && (explicit_on || cfg.min_stream_tps > 0.0);
    return cfg;
}

/// P3-3A: the cohort-packing key of a slot's prompt — a hash of its first `depth` tokens.
/// Siblings of one fan-out share their whole header, so they hash alike and the governor can
/// put them in the same window (they route to the same experts, which is what lifts the
/// cohort's routed-expert VRAM hit).  The token count is deliberately NOT in the key: the
/// siblings' tails differ and the shared part is the prefix.  0 means "no prefix to pack on".
/// P3-3A: how many leading tokens `prefix_affinity_key` hashes.
constexpr size_t kPrefixAffinityDepth = 512;

template <typename Token>
inline int64_t prefix_affinity_key(const std::vector<Token>& ids, size_t depth = kPrefixAffinityDepth) {
    const size_t n = ids.size() < depth ? ids.size() : depth;
    if (n < 8) return 0;
    uint64_t h = 1469598103934665603ull;   // FNV-1a 64
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) (uint32_t) ids[i];
        h *= 1099511628211ull;
    }
    return (int64_t) (h == 0 ? 1ull : h);
}

class DecodeGovernor {
public:
    explicit DecodeGovernor(DecodeGovernorConfig cfg = {}) : cfg_(cfg) {}

    bool enabled() const { return cfg_.enabled; }
    bool affinity_enabled() const { return cfg_.enabled && cfg_.affinity; }

    /// P3-3A: how many cohort seats were taken by a prefix sibling, and how many waiting slots
    /// were stepped over to do it. Pure instrumentation: the serve log prints both so an
    /// affinity arm can show the mechanism fired instead of only that it was enabled.
    uint64_t affinity_picks() const { return affinity_picks_; }
    uint64_t affinity_deferrals() const { return affinity_deferrals_; }
    const DecodeGovernorConfig& config() const { return cfg_; }
    int target_width() const { return target_n_; }
    double ema_hit_pct() const { return ema_hit_pct_; }
    double ema_stream_tps() const { return ema_stream_tps_; }
    bool has_ema() const { return has_ema_; }

    /// Select up to `target_n_` sticky slots from `runnable_slots`, capped by `win_slot_cap`.
    /// Returned vector is strictly sorted ascending so batch CUDA-graph `rows[]` keys are canonical.
    /// `affinity_keys` (optional, parallel to `runnable_slots`) is each slot's prompt-prefix key
    /// from `prefix_affinity_key`; with `cfg_.affinity` it drives sibling co-scheduling.
    const std::vector<int>& select_cohort(const std::vector<int>& runnable_slots,
                                          int win_slot_cap,
                                          double now_ms,
                                          const std::vector<int64_t>& affinity_keys = {}) {
        if (runnable_slots.empty()) {
            cohort_order_.clear();
            cohort_sorted_.clear();
            has_waiting_ = false;
            quantum_start_ms_ = -1.0;
            grace_left_ = 0;
            waited_.clear();
            aff_.clear();
            return cohort_sorted_;
        }

        const int hard_max = effective_hard_max(win_slot_cap);
        if (target_n_ <= 0) {
            target_n_ = std::max(1, std::min(hard_max, cfg_.initial_width));
        } else {
            target_n_ = std::max(1, std::min(target_n_, hard_max));
        }

        std::vector<int> runnable = runnable_slots;
        std::sort(runnable.begin(), runnable.end());
        runnable.erase(std::unique(runnable.begin(), runnable.end()), runnable.end());

        const int k = std::min((int) runnable.size(), target_n_);

        // Keep only slots still runnable, preserving cohort FIFO tenure order.
        cohort_order_.erase(
            std::remove_if(cohort_order_.begin(), cohort_order_.end(),
                           [&](int id) {
                               return !std::binary_search(runnable.begin(), runnable.end(), id);
                           }),
            cohort_order_.end());

        // If target_n_ shrunk below current cohort size, trim the highest slot ID first
        // so the lowest-indexed prefix stays intact for CUDA graph reuse.
        while ((int) cohort_order_.size() > k) {
            auto max_it = std::max_element(cohort_order_.begin(), cohort_order_.end());
            cohort_order_.erase(max_it);
        }

        auto in_cohort = [&](int id) {
            return std::find(cohort_order_.begin(), cohort_order_.end(), id) != cohort_order_.end();
        };

        // P3-3A: the prefix keys of this call's slots (empty when the caller passes none).
        aff_.clear();
        for (size_t i = 0; i < runnable_slots.size() && i < affinity_keys.size(); ++i) {
            const int id = runnable_slots[i];
            if (affinity_keys[i] != 0 && std::binary_search(runnable.begin(), runnable.end(), id))
                aff_.emplace_back(id, affinity_keys[i]);
        }
        auto aff_of = [&](int id) -> int64_t {
            for (const auto& kv : aff_) {
                if (kv.first == id) return kv.second;
            }
            return 0;
        };
        // The prefix the cohort is packed around: the key its most numerous member shares.
        auto dominant_affinity = [&]() -> int64_t {
            int64_t best = 0;
            int best_n = 0;
            for (const int id : cohort_order_) {
                const int64_t key = aff_of(id);
                if (key == 0) continue;
                int n = 0;
                for (const int other : cohort_order_) {
                    if (aff_of(other) == key) ++n;
                }
                if (n > best_n) { best_n = n; best = key; }
            }
            return best;
        };
        // Aging: a waiting slot passed over `affinity_max_wait` times takes the next seat whatever
        // its prefix, so a prefix group that keeps winning cannot starve the rest of the pool.
        for (auto it = waited_.begin(); it != waited_.end();) {
            if (!std::binary_search(runnable.begin(), runnable.end(), it->first)) it = waited_.erase(it);
            else ++it;
        }
        auto waited_of = [&](int id) -> int {
            for (const auto& kv : waited_) {
                if (kv.first == id) return kv.second;
            }
            return 0;
        };

        // Pick the next non-cohort runnable slot: prefix sibling first when affinity is on and no
        // waiter has aged out, otherwise circular slot-ID order after `last_admitted_`.
        auto pick_next_waiting = [&]() -> int {
            if (runnable.size() <= cohort_order_.size()) return -1;
            std::vector<int> waiting;
            auto it = std::upper_bound(runnable.begin(), runnable.end(), last_admitted_);
            for (size_t step = 0; step < runnable.size(); ++step) {
                if (it == runnable.end()) it = runnable.begin();
                if (!in_cohort(*it)) waiting.push_back(*it);
                ++it;
            }
            if (waiting.empty()) return -1;
            int pick = -1;
            if (cfg_.affinity) {
                for (const int id : waiting) {
                    if (waited_of(id) >= cfg_.affinity_max_wait) { pick = id; break; }
                }
                if (pick < 0) {
                    const int64_t dom = dominant_affinity();
                    if (dom != 0) {
                        for (const int id : waiting) {
                            if (aff_of(id) == dom) { pick = id; ++affinity_picks_; break; }
                        }
                    }
                }
            }
            if (pick < 0) pick = waiting.front();
            if (cfg_.affinity) {
                // Only the slots actually stepped over age: a cold-start fill takes the circular
                // front and passes nobody, so it must not put the whole pool behind a queue.
                for (size_t i = 0; i < waiting.size(); ++i) {
                    if (waiting[i] == pick) break;
                    ++affinity_deferrals_;
                    bool seen = false;
                    for (auto& kv : waited_) {
                        if (kv.first == waiting[i]) { ++kv.second; seen = true; break; }
                    }
                    if (!seen) waited_.emplace_back(waiting[i], 1);
                }
            }
            return pick;
        };

        // Fill empty seats (cold start, finished slots, or step-up).
        const bool was_measuring = !cohort_order_.empty();
        bool admitted = false;
        while ((int) cohort_order_.size() < k) {
            const int next_id = pick_next_waiting();
            if (next_id < 0) break;
            cohort_order_.push_back(next_id);
            last_admitted_ = next_id;
            admitted = true;
        }
        // P3-3A: a slot joining a cohort that was already being measured is cold — its adaptive
        // tier has not seen this window yet, so its first windows are not evidence about width.
        if (was_measuring && admitted) arm_grace();

        // Fair quantum rotation only when runnable slots genuinely exceed active width k.
        if ((int) runnable.size() > k && k > 0 && cfg_.cohort_quantum_ms > 0.0) {
            if (quantum_start_ms_ < 0.0) {
                quantum_start_ms_ = now_ms;
            } else if (now_ms - quantum_start_ms_ >= cfg_.cohort_quantum_ms) {
                if (!cohort_order_.empty()) {
                    cohort_order_.erase(cohort_order_.begin());
                    const int next_id = pick_next_waiting();
                    if (next_id >= 0) {
                        cohort_order_.push_back(next_id);
                        last_admitted_ = next_id;
                    }
                }
                quantum_start_ms_ = now_ms;
                arm_grace();   // P3-3A: the slot just rotated in starts cold
            }
        } else {
            quantum_start_ms_ = (int) runnable.size() > k ? now_ms : -1.0;
        }

        cohort_sorted_ = cohort_order_;
        std::sort(cohort_sorted_.begin(), cohort_sorted_.end());
        has_waiting_ = runnable.size() > cohort_sorted_.size();
        return cohort_sorted_;
    }

    /// Record one completed batch decode window and adjust `target_n_` after `eval_every` windows.
    /// Returns true if `target_n_` changed.
    bool observe_window(double window_ms,
                        int tokens_emitted,
                        int cohort_slots,
                        double hit_pct,
                        int win_slot_cap) {
        if (!cfg_.enabled || tokens_emitted <= 0 || cohort_slots <= 0) return false;
        // P3-3A: a rotation or a resize just changed what this window measures.  The first
        // `grace_windows` windows after that are cold by construction (the incoming slot's
        // adaptive tier has not pulled its experts in yet), so they are neither averaged nor
        // allowed to move the width - that is how one or two cold windows used to undo a widening.
        if (grace_left_ > 0) {
            --grace_left_;
            has_ema_ = false;
            return false;
        }


        const int hard_max = effective_hard_max(win_slot_cap);
        if (target_n_ <= 0) {
            target_n_ = std::max(1, std::min(hard_max, cfg_.initial_width));
        } else {
            target_n_ = std::max(1, std::min(target_n_, hard_max));
        }

        const double dt_s = std::max(window_ms, 0.05) / 1000.0;
        const double stream_tps = ((double) tokens_emitted / (double) cohort_slots) / dt_s;
        const double clamped_hit = std::max(0.0, std::min(100.0, hit_pct));

        if (!has_ema_) {
            ema_hit_pct_ = clamped_hit;
            ema_stream_tps_ = stream_tps;
            has_ema_ = true;
        } else {
            const double a = cfg_.ema_alpha;
            ema_hit_pct_ = a * clamped_hit + (1.0 - a) * ema_hit_pct_;
            ema_stream_tps_ = a * stream_tps + (1.0 - a) * ema_stream_tps_;
        }

        ++windows_since_eval_;
        if (windows_since_eval_ < std::max(1, cfg_.eval_every)) return false;
        windows_since_eval_ = 0;

        int next_n = target_n_;
        if (ema_hit_pct_ < cfg_.hit_floor_pct) {
            // Hard Physics Brake: never thrash host/PCIe tier even in throughput mode.
            next_n = std::max(1, target_n_ - 1);
        } else if (cfg_.min_stream_tps > 0.0 && ema_stream_tps_ < cfg_.min_stream_tps) {
            // User SLA floor violated: shed width to recover per-stream speed.
            next_n = std::max(1, target_n_ - 1);
        } else if (has_waiting_ && target_n_ < hard_max) {
            const bool hit_ok = ema_hit_pct_ >= (cfg_.hit_floor_pct + cfg_.step_up_margin_pct);
            const bool tps_ok = (cfg_.min_stream_tps <= 0.0) ||
                                (ema_stream_tps_ >= cfg_.min_stream_tps * cfg_.step_up_tps_margin);
            if (hit_ok && tps_ok) {
                next_n = target_n_ + 1;
            }
        }

        if (next_n == target_n_) return false;
        target_n_ = next_n;
        has_ema_ = false;  // measure the new width cleanly on subsequent windows
        arm_grace();       // P3-3A: and give the new width that many cold windows before judging it
        return true;
    }

    /// P3-3A: windows still being skipped as warmup (0 once the cohort is being judged).
    int grace_left() const { return grace_left_; }
    /// P3-3A: how many picks this waiting slot has been passed over by prefix affinity.
    int waited_for(int slot) const {
        for (const auto& kv : waited_) {
            if (kv.first == slot) return kv.second;
        }
        return 0;
    }

private:
    int effective_hard_max(int win_slot_cap) const {
        const int cap = win_slot_cap > 0 ? win_slot_cap : cfg_.max_width;
        return std::max(1, std::min(cap, cfg_.max_width));
    }

    /// P3-3A: start a warmup stretch - `grace_windows` windows are observed but not judged.
    void arm_grace() {
        grace_left_ = cfg_.grace_windows > 0 ? cfg_.grace_windows : 0;
        windows_since_eval_ = 0;
        has_ema_ = false;
    }

    DecodeGovernorConfig cfg_;
    int target_n_ = 0;
    bool has_waiting_ = false;
    bool has_ema_ = false;
    double ema_hit_pct_ = 100.0;
    double ema_stream_tps_ = 0.0;
    int windows_since_eval_ = 0;
    double quantum_start_ms_ = -1.0;
    int last_admitted_ = -1;
    int grace_left_ = 0;
    std::vector<int> cohort_order_;   ///< Active cohort in admission FIFO order
    std::vector<int> cohort_sorted_;  ///< Canonical ascending order for batch rows
    std::vector<std::pair<int, int64_t>> aff_;    ///< slot -> prompt-prefix key, this call
    std::vector<std::pair<int, int>> waited_;     ///< slot -> times passed over by affinity
    uint64_t affinity_picks_ = 0;                 ///< cohort seats taken by a prefix sibling
    uint64_t affinity_deferrals_ = 0;             ///< waiting slots stepped over to do it
};

}  // namespace strata::program
