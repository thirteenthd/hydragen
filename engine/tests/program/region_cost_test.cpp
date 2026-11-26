// region_cost_test.cpp: unit tests for P5-5B shared-region cost accounting
// (header-only, no GPU). Numbers are the real §243 clean re-run and §242
// contaminated arms from docs/CHANGES.md, so the gates are pinned to measured data.
#include "strata/program/region_cost.hpp"

#include <cmath>
#include <cstdio>

using strata::program::ArmPair;
using strata::program::RegionProbe;
using strata::program::choose_region_size;
using strata::program::delta_pct;
using strata::program::expert_cache_slot_loss;
using strata::program::kv_hit_gain;
using strata::program::mean_pair;
using strata::program::paired_delta;
using strata::program::region_arena_gib;
using strata::program::within_swing_law;

static int fails = 0;
#define CHECK(c)                                                             \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c);          \
            ++fails;                                                         \
        }                                                                    \
    } while (0)

static bool near(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) <= eps;
}

// §243 clean re-run, warm2 agg tok/s N=1/2/4/8 per arm.
static RegionProbe probe_clean(int cow, double arena, double kv,
                               double n1, double n2, double n4, double n8) {
    RegionProbe p;
    p.cow_shared = cow;
    p.arena_gib = arena;
    p.kv_hit_pct = kv;
    p.agg = {{n1, n2, n4, n8}};
    return p;
}

int main() {
    // --- pricing: ~50 KiB per COW_SHARED slot, matching the §242 boot lines
    // (0 -> 1.64, 2048 -> 1.74, 8192 -> 2.03 GiB reserved).
    CHECK(near(region_arena_gib(0), 0.0));
    CHECK(std::fabs(region_arena_gib(2048) - 0.10) < 0.01);
    CHECK(std::fabs(region_arena_gib(8192) - 0.39) < 0.01);

    // --- pairing and paired deltas (round-by-round, then averaged)
    ArmPair ctrl;
    ctrl.r1 = probe_clean(0, 1.64, 82.60, 74.9, 78.9, 66.3, 68.3);
    ctrl.r2 = ctrl.r1;  // §243 quotes one clean pass per arm; duplicate the round
    ctrl.r1.cache_slots = 2343;
    ctrl.r2.cache_slots = 2343;

    ArmPair a2048;
    a2048.r1 = probe_clean(2048, 1.74, 90.69, 70.3, 70.2, 70.4, 66.8);
    a2048.r2 = a2048.r1;
    a2048.r1.cache_slots = 2210;
    a2048.r2.cache_slots = 2210;

    ArmPair a8192;
    a8192.r1 = probe_clean(8192, 2.03, 93.13, 71.8, 70.8, 65.0, 66.2);
    a8192.r2 = a8192.r1;
    a8192.r1.cache_slots = 2020;
    a8192.r2.cache_slots = 2020;

    const auto m = mean_pair(a2048);
    CHECK(near(m[0], 70.3) && near(m[3], 66.8));

    const auto d = paired_delta(a2048, ctrl);
    CHECK(near(d[3], 66.8 - 68.3));  // N=8 paired delta
    CHECK(near(delta_pct(66.8, 68.3), 100.0 * (66.8 - 68.3) / 68.3));
    CHECK(delta_pct(5.0, 0.0) == 0.0);  // degenerate control guard

    // --- swing law gate (§243): +-40 % at N>=4
    CHECK(within_swing_law(-2.2));
    CHECK(within_swing_law(40.0));
    CHECK(!within_swing_law(-40.1));
    CHECK(!within_swing_law(41.0));

    // The §243 clean arms: every N>=4 delta is inside the band -> the claimed
    // region cost at N>=4 is noise, exactly what §243 measured.
    const auto d8192 = paired_delta(a8192, ctrl);
    CHECK(within_swing_law(delta_pct(ctrl.r1.agg[2] + d8192[2], ctrl.r1.agg[2])));   // N=4: -1.96 %
    CHECK(within_swing_law(delta_pct(ctrl.r1.agg[3] + d8192[3], ctrl.r1.agg[3])));  // N=8: -3.07 %

    // --- contamination gate: the §242 arms carried wrapper 503s (errors 2/6,
    // N=8 collapse 51.0/45.9/43.7); they must never grade.
    RegionProbe dirty = probe_clean(8192, 2.03, 93.13, 69.2, 74.3, 58.8, 43.7);
    dirty.http503 = 2;
    CHECK(!dirty.clean());
    RegionProbe dirty_oom = probe_clean(2048, 1.74, 90.69, 72.7, 66.4, 60.6, 45.9);
    dirty_oom.oom = 1;
    CHECK(!dirty_oom.clean());

    // --- expert cache pricing: the auto-sizer shrinks the cache by the region's
    // arena growth (measured arms lose 133 and 323 slots).
    CHECK(near(expert_cache_slot_loss(a2048.r1, ctrl.r1), 133.0));
    CHECK(near(expert_cache_slot_loss(a8192.r1, ctrl.r1), 323.0));

    // --- gain: KV hit points bought (82.60 -> 90.69 -> 93.13, §243 telemetry)
    CHECK(near(kv_hit_gain(a2048, ctrl), 90.69 - 82.60, 1e-9));
    CHECK(near(kv_hit_gain(a8192, ctrl), 93.13 - 82.60, 1e-9));

    // --- sizing decision: largest clean arm with N>=4 cost inside the band wins.
    const std::vector<ArmPair> arms{{ctrl, a2048, a8192}};
    auto choice = choose_region_size(arms);
    CHECK(choice.slots == 8192);
    CHECK(choice.cost_in_noise);
    CHECK(near(choice.kv_hit_gain, 93.13 - 82.60, 1e-9));

    // A swing-law-violating big arm (N=8 -45 %) is rejected; the next arm wins.
    ArmPair a8192_bad = a8192;
    a8192_bad.r1.agg[3] = 68.3 * 0.55;  // -45 % vs control
    a8192_bad.r2.agg[3] = 68.3 * 0.55;
    const std::vector<ArmPair> arms_bad{{ctrl, a2048, a8192_bad}};
    CHECK(choose_region_size(arms_bad).slots == 2048);

    // A contaminated arm is skipped even when its numbers look fine.
    ArmPair a8192_dirty = a8192;
    a8192_dirty.r1.http503 = 3;
    const std::vector<ArmPair> arms_dirty{{ctrl, a2048, a8192_dirty}};
    CHECK(choose_region_size(arms_dirty).slots == 2048);

    // --- per-stream SLA floor (the --min-stream-tps arithmetic; flag exists at
    // generate.cpp:833,4081 / server.py:3972). mean agg at N=8: 2048 -> 8.35,
    // 8192 -> 8.275 per stream: a 8.3 floor keeps 2048, drops 8192.
    CHECK(choose_region_size(arms, 8.3).slots == 2048);
    CHECK(choose_region_size(arms, 8.0).slots == 8192);
    CHECK(choose_region_size(arms, 9.0).slots == 0);

    // --- no clean control -> nothing grades, default stays.
    ArmPair ctrl_dirty = ctrl;
    ctrl_dirty.r1.oom = 1;
    CHECK(choose_region_size({ctrl_dirty, a2048}).slots == 0);
    CHECK(choose_region_size({a2048}).slots == 0);

    if (fails == 0) {
        std::printf("region_cost_test: all checks passed\n");
    }
    return fails;
}
