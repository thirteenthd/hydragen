#pragma once
// P5-5B.1: the unified pool's pure sizing logic (HydraGen fork, Phase 5 Track 5B).
//
// One VRAM budget, three consumers: the expert cache's slots, the elastic CoW arena's windows and a
// deferred session's private buffers.  The cache may give up cold slots down to a FLOOR - the hottest
// experts' share of the routed mass, read off the profile - and takes them back when the pressure
// leaves.  Everything here is arithmetic on numbers the caller already has: no device calls, no
// environment reads, so the CPU test pins the whole policy.
//
// The runtime wiring (CowExpertPool in generate.cpp) reuses the elastic K/V's proven mechanics:
// quiesce first, relocate hotter residents into colder loaned slots, unmap whole 2 MiB chunks,
// move the residency table, refill from host RAM on the way back.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::core {

/// The pool's configuration, as the caller read it (HG_UNIFIED_POOL / HG_EXPERT_KNEE_PCT /
/// HG_EXPERT_FLOOR_SLOTS / --min-stream-tps live at the call site, not here).
struct UnifiedPoolConfig {
    int64_t full_slots = 0;             // the expert cache's slots at boot (never re-sized)
    int64_t min_slots = 1;              // the smallest floor the policy may pick
    double knee_pct = 86.0;             // the floor covers this % of the routed mass
    int64_t floor_override = 0;         // > 0: the floor, whatever the weights say
    int64_t slot_vram_bytes = 0;        // one deferred conversation slot's VRAM cost: session buffers + CoW arena window
    int64_t free_above_reserve_bytes = 0;  // free VRAM above the serve reserve, at boot
    int64_t flex_bytes = 0;             // bytes the pool can move without evicting (parked chunks, slack)
    int max_batch = 1;                  // the boot --batch carve: how many streams may be active
    double min_stream_tps = 0.0;        // the DecodeGovernor SLA floor (0 = throughput mode)
};

/// The floor: the fewest slots the cache may keep.  `sorted_desc_weights` is the profile's routed
/// mass per pair, hottest first (empty / all-zero: no profile - fall back to 85% of the cache, the
/// share the §240-§243 runs showed the hot set occupies).  Walk the weights until the knee is
/// covered; the result never leaves [min_slots, full_slots].
inline int64_t expert_floor_from_weights(const std::vector<double>& sorted_desc_weights,
                                         int64_t full_slots, int64_t min_slots, double knee_pct,
                                         int64_t floor_override = 0) {
    if (full_slots <= 0) return 0;
    const int64_t min_s = std::clamp<int64_t>(min_slots, 1, full_slots);
    if (floor_override > 0) return std::clamp<int64_t>(floor_override, min_s, full_slots);
    double total = 0.0;
    for (const double w : sorted_desc_weights)
        if (w > 0.0) total += w;
    if (sorted_desc_weights.empty() || total <= 0.0)
        return std::clamp<int64_t>(full_slots * 85 / 100, min_s, full_slots);
    const double knee = std::clamp(knee_pct, 1.0, 100.0) / 100.0;
    const double target = knee * total;
    double acc = 0.0;
    int64_t n = 0;
    const std::size_t lim = std::min<std::size_t>(sorted_desc_weights.size(), (std::size_t) full_slots);
    for (std::size_t i = 0; i < lim; ++i) {
        acc += sorted_desc_weights[i];
        ++n;
        if (acc >= target) break;
    }
    return std::clamp<int64_t>(n, min_s, full_slots);
}

/// How many streams the boot may start active: one always fits (its own buffers were priced), each
/// further one costs a deferred conversation slot's VRAM (`slot_vram_bytes`: its session buffers plus
/// its CoW arena window, paid for by chunks the expert cache lends).  Never above `max_batch`, never
/// below 1.
inline int initial_active_cap(int64_t free_above_reserve_bytes, int64_t flex_bytes,
                              int64_t slot_vram_bytes, int max_batch) {
    if (max_batch < 1) max_batch = 1;
    if (slot_vram_bytes <= 0) return max_batch;
    const int64_t avail = std::max<int64_t>(0, free_above_reserve_bytes) + std::max<int64_t>(0, flex_bytes);
    return std::clamp<int>(1 + (int) (avail / slot_vram_bytes), 1, max_batch);
}

/// Whole chunks the pool must lend to cover a byte deficit (0 when nothing is short, or the
/// granularity is unknown - a caller with no VMM has no chunks to lend).
inline int64_t chunk_deficit(int64_t need_bytes, int64_t avail_bytes, int64_t chunk_granularity) {
    if (need_bytes <= avail_bytes || chunk_granularity <= 0) return 0;
    return (need_bytes - std::max<int64_t>(0, avail_bytes) + chunk_granularity - 1) / chunk_granularity;
}

}  // namespace strata::core
