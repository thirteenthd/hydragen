#pragma once
// P5-5C.1: the admission-level auto batch-N controller's pure logic (HydraGen fork, Phase 5 Track 5C).
//
// The unified pool (P5-5B) sizes the expert cache's lending; this decides HOW MANY conversation slots
// (K) admission should keep active between them.  One controller instance sits at the admission
// boundary: every admission/completion event feeds one observation (rows actually running, aggregate
// and per-stream tok/s, routed-expert VRAM hit%, whether the pool has headroom to lend, whether the
// driver refused memory), and the controller moves K by at most one.
//
// Damping, so K never thrashes 1 <-> 2 on noisy samples:
//   - EMAs (ema_alpha) smooth every signal; decisions read the EMAs, never the raw sample.
//   - patience: a bad (or good) streak must reach down_patience (up_patience) consecutive
//     observations before the step applies; a neutral observation resets both streaks.
//   - cooldown_events: after any K change, that many boundaries pass before the next one may fire.
//   - hysteresis: the step-down line sits hit_hysteresis_pct below the step-up knee, and in SLA
//     mode the band [min_stream_tps, sla_up_margin * min_stream_tps) is dead: hold.
//   - mem_refused is a hard clamp, not a vote: a refusal while K exceeds the rows actually running
//     means K is above the real VRAM ceiling, so K drops to those rows at once.
//
// Two modes: min_stream_tps > 0 keeps every stream fluent (step down below the floor, step up only
// with margin AND pool headroom); min_stream_tps == 0 maximizes aggregate throughput (step up while
// the expert hit is at the knee and the aggregate is improving, step down when the hit collapses or
// the aggregate regresses below agg_regress_ratio of the best seen at the previous K).
//
// Everything here is arithmetic on numbers the caller already has: no device calls, no environment
// reads, so the CPU test pins the whole policy.

#include <algorithm>

namespace strata::core {

/// The controller's configuration (the shipped defaults are the 12 GiB mech box's: K in [1, 8],
/// start at 2, auto mode, the 85% hit knee the §240-§243 runs showed).
struct AutoBatchConfig {
    int min_k = 1;                  // the smallest K the controller may hold
    int max_k = 8;                  // the boot --batch carve: the largest K
    int initial_k = 2;              // where K starts (clamped into [min_k, max_k])
    double min_stream_tps = 0.0;    // 0.0 = auto max-aggregate mode; > 0 = per-stream fluency SLA floor
    double knee_hit_pct = 85.0;     // routed-expert VRAM hit% knee: step-up wants the EMAd hit at/above this
    double hit_hysteresis_pct = 3.0;   // step-down threshold = knee_hit_pct - hit_hysteresis_pct
    double sla_up_margin = 1.15;    // SLA mode: step up needs per-stream >= sla_up_margin * min_stream_tps
    double agg_regress_ratio = 0.92;   // auto mode: step down if agg tok/s drops below this * best_agg_at_prev_k
    double ema_alpha = 0.35;        // EMA weight on the newest observation
    int down_patience = 2;          // consecutive bad observations before stepping down
    int up_patience = 3;            // consecutive good observations before stepping up
    int cooldown_events = 2;        // minimum admission/completion boundaries between K changes
};

/// One observation at an admission/completion boundary.
struct AutoBatchObs {
    int active_rows = 0;            // batch-graph rows actually carrying streams at the boundary
    double agg_tps = 0.0;           // aggregate decode tok/s over the interval
    double per_stream_tps = 0.0;    // aggregate / active streams
    double expert_hit_pct = 0.0;    // routed-expert VRAM hit% over the interval
    bool pool_has_headroom = true;  // the unified pool could lend more (or the cache is not the limit)
    bool mem_refused = false;       // the driver refused a session attach this interval
};

class AutoBatchController {
public:
    explicit AutoBatchController(const AutoBatchConfig& cfg)
        : cfg_(cfg), k_active_(std::clamp(cfg.initial_k, cfg.min_k, cfg.max_k)) {}

    /// Feed one boundary observation; returns the K admission should use next.  The EMAs update only
    /// for a valid observation (at least one row running and a positive aggregate rate); a valid
    /// observation then votes: bad streaks step down after down_patience, good streaks step up after
    /// up_patience, neutral ones reset both, and cooldown_events boundaries must pass between changes.
    /// A mem_refused with K above the rows actually running clamps K down at once.
    int observe_boundary(const AutoBatchObs& obs) {
        // the hard clamp first: a refusal above the real ceiling is a fact, not a sample
        if (obs.mem_refused && k_active_ > obs.active_rows) {
            const int to = std::max(cfg_.min_k, obs.active_rows);
            if (to < k_active_) {
                k_active_ = to;
                ++step_downs_;
                change_reset();
                return k_active_;
            }
        }
        if (obs.active_rows < 1 || obs.agg_tps <= 0.0) return k_active_;   // nothing to learn

        if (!ema_init_) {
            ema_agg_ = obs.agg_tps;
            ema_stream_ = obs.per_stream_tps;
            ema_hit_ = obs.expert_hit_pct;
            ema_init_ = true;
        } else {
            const double a = cfg_.ema_alpha;
            ema_agg_ = a * obs.agg_tps + (1.0 - a) * ema_agg_;
            ema_stream_ = a * obs.per_stream_tps + (1.0 - a) * ema_stream_;
            ema_hit_ = a * obs.expert_hit_pct + (1.0 - a) * ema_hit_;
        }
        if (ema_agg_ > best_agg_at_k_) best_agg_at_k_ = ema_agg_;

        if (cooldown_left_ > 0) { --cooldown_left_; return k_active_; }   // damped: no decision this boundary

        bool good = false, bad = false;
        if (cfg_.min_stream_tps > 0.0) {
            // SLA mode: below the floor is bad; the step-up needs margin AND pool headroom; the band
            // between is dead (hold) so K never chases noise around the floor.
            bad = ema_stream_ < cfg_.min_stream_tps;
            good = !bad && ema_stream_ >= cfg_.sla_up_margin * cfg_.min_stream_tps && obs.pool_has_headroom;
        } else {
            // auto mode: the hit knee opens the step-up (with headroom and an aggregate that is not
            // below its best at this K); a collapsed hit, or an aggregate below the ratio of the
            // best the previous K achieved, closes it.
            good = ema_hit_ >= cfg_.knee_hit_pct && obs.pool_has_headroom && ema_agg_ >= best_agg_at_k_;
            bad = ema_hit_ < cfg_.knee_hit_pct - cfg_.hit_hysteresis_pct ||
                  (prev_best_agg_ > 0.0 && ema_agg_ < cfg_.agg_regress_ratio * prev_best_agg_);
        }

        if (bad) {
            good_streak_ = 0;
            if (++bad_streak_ >= cfg_.down_patience && k_active_ > cfg_.min_k) {
                --k_active_;
                ++step_downs_;
                prev_best_agg_ = 0.0;   // below the step the old bar no longer applies
                change_reset();
            }
        } else if (good) {
            bad_streak_ = 0;
            if (++good_streak_ >= cfg_.up_patience && k_active_ < cfg_.max_k) {
                prev_best_agg_ = best_agg_at_k_;   // the bar the new K must clear
                ++k_active_;
                ++step_ups_;
                change_reset();
            }
        } else {
            good_streak_ = bad_streak_ = 0;
        }
        return k_active_;
    }

    int k_active() const { return k_active_; }
    double ema_per_stream_tps() const { return ema_stream_; }
    double ema_agg_tps() const { return ema_agg_; }
    double ema_hit_pct() const { return ema_hit_; }
    int step_ups() const { return step_ups_; }
    int step_downs() const { return step_downs_; }

private:
    void change_reset() {
        good_streak_ = bad_streak_ = 0;
        cooldown_left_ = cfg_.cooldown_events;
        best_agg_at_k_ = 0.0;   // the new K re-reads its own best from the next valid observation
    }

    AutoBatchConfig cfg_;
    int k_active_ = 1;
    bool ema_init_ = false;
    double ema_agg_ = 0.0, ema_stream_ = 0.0, ema_hit_ = 0.0;
    double best_agg_at_k_ = 0.0;    // best EMAd aggregate seen at the current K
    double prev_best_agg_ = 0.0;    // best at the K before the last step up (0: never stepped up)
    int good_streak_ = 0, bad_streak_ = 0;
    int cooldown_left_ = 0;
    int step_ups_ = 0, step_downs_ = 0;
};

}  // namespace strata::core
