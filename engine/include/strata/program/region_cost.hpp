// region_cost.hpp — P5-5B shared-region cost accounting: pure logic, no GPU, no I/O.
//
// Track 5B Turn 1 (P5-5B.1) is the paired-probe math behind the sizing law; Turn 2
// wires the chosen size into the arena VMM tail. The mechanism (bench/
// p5_5b_region_cost.sh, §243 boot lines): STRATA_COW_SHARED grows the K/V arena
// reservation (0 -> 1.64, 2048 -> 1.74, 8192 -> 2.03 GiB, all mapped) and the expert
// cache auto-sizes against the remaining free VRAM, so the region is paid for in
// expert-cache slots. This header prices that trade and gates claims on the §243
// swing law: N>=4 aggregate swings +-40 % run to run, so a cost or gain at N>=4 only
// counts when the PAIRED (mirrored-order, two boots per arm) delta leaves that band.
//
// The `--min-stream-tps` flag already exists (generate.cpp:833,4081; server.py:3972);
// this header only consumes its value as an optional per-stream SLA floor. Runtime
// plumbing of the chosen size is Turn 2 — nothing here reads env or argv.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace strata {
namespace program {

// Measured on ms01 12 GiB (§242/§243 boot lines): the region tail grows the arena
// reservation by ~50 KiB per COW_SHARED slot (2048 -> +0.10 GiB, 8192 -> +0.39 GiB).
inline constexpr double kRegionBytesPerSlot = 51200.0;

// §243 swing law: N>=4 aggregate throughput swings +-40 % run to run on this box.
inline constexpr double kSwingLawPct = 40.0;

// One boot's extracted telemetry (one P5BROW line from the 5B harness).
struct RegionProbe {
    int cow_shared = 0;             // STRATA_COW_SHARED slots (the arm id)
    double arena_gib = 0.0;         // K/V arena reserved at boot
    double cache_slots = 0.0;       // expert cache slots (auto-sized vs remaining VRAM)
    double kv_hit_pct = 0.0;        // KV streaming hit % (last serve line)
    double dec_hit_pct = 0.0;       // decode expert cache hit %
    long joins = 0;                 // joins-at-fetch served
    long fetches = 0;               // shared-region lineage fetches
    std::array<double, 4> agg{{0.0, 0.0, 0.0, 0.0}};  // warm2 agg tok/s at N=1,2,4,8
    int refusals = 0;               // engine "session unavailable" lines
    int backoffs = 0;               // wrapper slot backoffs
    int http503 = 0;                // wrapper GpuBusy responses
    int oom = 0;                    // engine out-of-memory lines

    // §243: an arm whose admission path was contaminated (503s / OOM) cannot grade.
    bool clean() const { return oom == 0 && http503 == 0; }
};

// The same arm booted twice in mirrored order (round 1 forward, round 2 reverse) so
// expert-profile drift hits both rounds equally (CONFOUND LAW, 5B harness).
struct ArmPair {
    RegionProbe r1;
    RegionProbe r2;
};

// Arena GiB the region reserves for a slot count (pricing side of the trade).
inline double region_arena_gib(int cow_shared_slots) {
    return static_cast<double>(cow_shared_slots) * kRegionBytesPerSlot /
           (1024.0 * 1024.0 * 1024.0);
}

// Per-N mean of the two mirrored rounds of one arm.
inline std::array<double, 4> mean_pair(const ArmPair& a) {
    std::array<double, 4> out{};
    for (size_t i = 0; i < 4; ++i) {
        out[i] = 0.5 * (a.r1.agg[i] + a.r2.agg[i]);
    }
    return out;
}

// Paired delta of arm vs control: per round (arm - control), then averaged. Pairing
// round-by-round is what makes this a paired probe, not two independent samples.
inline std::array<double, 4> paired_delta(const ArmPair& arm, const ArmPair& control) {
    std::array<double, 4> out{};
    for (size_t i = 0; i < 4; ++i) {
        const double d1 = arm.r1.agg[i] - control.r1.agg[i];
        const double d2 = arm.r2.agg[i] - control.r2.agg[i];
        out[i] = 0.5 * (d1 + d2);
    }
    return out;
}

// Percent delta vs control; 0.0 when the control sample is degenerate.
inline double delta_pct(double arm, double control) {
    if (control <= 0.0 || !std::isfinite(control)) return 0.0;
    return 100.0 * (arm - control) / control;
}

// A claim counts only when it leaves the swing band (§243).
inline bool within_swing_law(double pct) {
    return std::fabs(pct) <= kSwingLawPct;
}

// What the region costs the expert cache: slots the auto-sizer gave up vs control.
inline double expert_cache_slot_loss(const RegionProbe& arm, const RegionProbe& control) {
    return control.cache_slots - arm.cache_slots;
}

// What the region buys: KV streaming hit points vs control (paired means).
inline double kv_hit_gain(const ArmPair& arm, const ArmPair& control) {
    const double a = 0.5 * (arm.r1.kv_hit_pct + arm.r2.kv_hit_pct);
    const double c = 0.5 * (control.r1.kv_hit_pct + control.r2.kv_hit_pct);
    return a - c;
}

struct RegionChoice {
    int slots = 0;            // chosen STRATA_COW_SHARED size (0 = keep default)
    double kv_hit_gain = 0.0; // hit points bought at that size
    bool cost_in_noise = false; // every N>=4 paired delta stayed inside the swing law
};

// Pick the largest candidate arm that (a) is clean in both rounds, (b) keeps every
// N>=4 paired throughput delta inside the swing law or positive, and (c) if
// min_stream_tps > 0, holds the per-stream SLA floor at the largest N
// (mean agg / 8 >= min_stream_tps; the --min-stream-tps flag already exists, this is
// only its arithmetic). The control arm is the one with cow_shared == 0; without it
// nothing can be graded and the default (0) is returned.
inline RegionChoice choose_region_size(const std::vector<ArmPair>& arms,
                                       double min_stream_tps = 0.0) {
    RegionChoice best;
    const ArmPair* control = nullptr;
    for (const auto& a : arms) {
        if (a.r1.cow_shared == 0 && a.r2.cow_shared == 0) { control = &a; break; }
    }
    if (!control || !control->r1.clean() || !control->r2.clean()) return best;

    for (const auto& a : arms) {
        const int slots = a.r1.cow_shared;
        if (slots <= 0 || a.r1.cow_shared != a.r2.cow_shared) continue;
        if (!a.r1.clean() || !a.r2.clean()) continue;

        const std::array<double, 4> d = paired_delta(a, *control);
        bool in_noise = true;
        for (size_t i = 2; i < 4; ++i) {  // N=4 and N=8 only (§243 grades N>=4)
            const double ctrl = 0.5 * (control->r1.agg[i] + control->r2.agg[i]);
            const double pct = delta_pct(ctrl + d[i], ctrl);
            if (pct < 0.0 && !within_swing_law(pct)) { in_noise = false; break; }
        }
        if (!in_noise) continue;

        if (min_stream_tps > 0.0) {
            const double mean_n8 = 0.5 * (a.r1.agg[3] + a.r2.agg[3]);
            if (!(mean_n8 / 8.0 >= min_stream_tps)) continue;
        }

        if (slots > best.slots) {
            best.slots = slots;
            best.kv_hit_gain = kv_hit_gain(a, *control);
            best.cost_in_noise = true;
        }
    }
    return best;
}

}  // namespace program
}  // namespace strata
