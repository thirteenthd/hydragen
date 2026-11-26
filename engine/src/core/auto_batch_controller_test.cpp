// src/core/auto_batch_controller_test.cpp - the damped admission-level auto batch-N controller
// (P5-5C.1, pure logic, no GPU).  Every scenario feeds boundary observations and asserts K after
// each one, so the EMAs, patience, cooldown, hysteresis and the mem_refused clamp are all pinned.
#include "strata/core/auto_batch_controller.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static bool near(double a, double b) { return a - b > -1e-9 && a - b < 1e-9; }

using strata::core::AutoBatchConfig;
using strata::core::AutoBatchController;
using strata::core::AutoBatchObs;

static AutoBatchObs mk(int rows, double agg, double stream, double hit,
                       bool headroom = true, bool refused = false) {
    AutoBatchObs o;
    o.active_rows = rows; o.agg_tps = agg; o.per_stream_tps = stream;
    o.expert_hit_pct = hit; o.pool_has_headroom = headroom; o.mem_refused = refused;
    return o;
}

int main() {
    // ---- shipped defaults ------------------------------------------------------------------------
    {
        AutoBatchConfig c;
        CHECK(c.min_k == 1 && c.max_k == 8 && c.initial_k == 2);
        CHECK(near(c.min_stream_tps, 0.0) && near(c.knee_hit_pct, 85.0) && near(c.hit_hysteresis_pct, 3.0));
        CHECK(near(c.sla_up_margin, 1.15) && near(c.agg_regress_ratio, 0.92) && near(c.ema_alpha, 0.35));
        CHECK(c.down_patience == 2 && c.up_patience == 3 && c.cooldown_events == 2);
    }

    // ---- construction clamps initial_k into [min_k, max_k] ----------------------------------------
    {
        AutoBatchConfig c; c.min_k = 2; c.max_k = 4; c.initial_k = 9;
        CHECK(AutoBatchController(c).k_active() == 4);
        c.initial_k = 0;
        CHECK(AutoBatchController(c).k_active() == 2);
    }

    // ---- SLA mode: per-stream below the floor steps down after down_patience ----------------------
    {
        AutoBatchConfig c; c.min_stream_tps = 25.0; c.initial_k = 2;
        AutoBatchController k(c);
        CHECK(k.observe_boundary(mk(2, 36.0, 18.0, 90.0)) == 2);   // bad 1
        CHECK(k.observe_boundary(mk(2, 36.0, 18.0, 90.0)) == 1);   // bad 2 -> step down
        CHECK(k.step_downs() == 1 && k.step_ups() == 0);
        CHECK(near(k.ema_per_stream_tps(), 18.0));
    }

    // ---- SLA mode: the deadband [25.0, 28.75) holds (no step either way) --------------------------
    {
        AutoBatchConfig c; c.min_stream_tps = 25.0; c.initial_k = 2;
        AutoBatchController k(c);
        CHECK(k.observe_boundary(mk(2, 50.0, 25.0, 90.0)) == 2);   // exactly the floor: not bad
        CHECK(k.observe_boundary(mk(2, 56.0, 28.74, 90.0)) == 2);  // just under the up line: not good
        CHECK(k.observe_boundary(mk(2, 50.0, 25.0, 90.0)) == 2);   // EMAd 25.85: still deadband
        CHECK(k.step_ups() == 0 && k.step_downs() == 0);
    }

    // ---- SLA mode: step up needs margin AND headroom AND up_patience, then the cooldown ------------
    {
        AutoBatchConfig c; c.min_stream_tps = 25.0; c.initial_k = 2;
        AutoBatchController k(c);
        CHECK(k.observe_boundary(mk(2, 58.0, 29.0, 90.0)) == 2);   // good 1
        CHECK(k.observe_boundary(mk(2, 58.0, 29.0, 90.0)) == 2);   // good 2
        CHECK(k.observe_boundary(mk(2, 58.0, 29.0, 90.0)) == 3);   // good 3 -> step up
        CHECK(k.observe_boundary(mk(3, 87.0, 29.0, 90.0)) == 3);   // cooldown boundary 1
        CHECK(k.observe_boundary(mk(3, 87.0, 29.0, 90.0)) == 3);   // cooldown boundary 2
        CHECK(k.observe_boundary(mk(3, 87.0, 29.0, 90.0)) == 3);   // good 1 (streak restarted)
        CHECK(k.observe_boundary(mk(3, 87.0, 29.0, 90.0)) == 3);   // good 2
        CHECK(k.observe_boundary(mk(3, 87.0, 29.0, 90.0)) == 4);   // good 3 -> step up again
        CHECK(k.step_ups() == 2);
    }

    // ---- SLA mode: no pool headroom blocks the step up (deadband hold, not a bad vote) ------------
    {
        AutoBatchConfig c; c.min_stream_tps = 25.0; c.initial_k = 2;
        AutoBatchController k(c);
        for (int i = 0; i < 5; ++i)
            CHECK(k.observe_boundary(mk(2, 60.0, 30.0, 90.0, false)) == 2);
        CHECK(k.step_ups() == 0 && k.step_downs() == 0);
    }

    // ---- auto mode: hit at the knee + improving aggregate steps up ---------------------------------
    {
        AutoBatchConfig c;   // min_stream_tps = 0: auto max-aggregate
        c.initial_k = 2;
        AutoBatchController k(c);
        CHECK(k.observe_boundary(mk(2, 40.0, 20.0, 90.0)) == 2);   // good 1 (agg 40 is the new best)
        CHECK(k.observe_boundary(mk(2, 42.0, 21.0, 90.0)) == 2);   // good 2 (EMAd agg improving)
        CHECK(k.observe_boundary(mk(2, 44.0, 22.0, 90.0)) == 3);   // good 3 -> step up
        CHECK(k.step_ups() == 1);
        CHECK(near(k.ema_agg_tps(), 0.35 * 44.0 + 0.65 * (0.35 * 42.0 + 0.65 * 40.0)));
        CHECK(near(k.ema_hit_pct(), 90.0));
    }

    // ---- auto mode: a collapsed hit% steps down (hysteresis line 82, cooldown respected) -----------
    {
        AutoBatchConfig c; c.initial_k = 2;
        AutoBatchController k(c);
        for (int i = 0; i < 3; ++i) k.observe_boundary(mk(2, 40.0 + 2.0 * i, 20.0 + i, 90.0));
        CHECK(k.k_active() == 3);                                  // stepped up at boundary 3
        CHECK(k.observe_boundary(mk(3, 44.0, 14.7, 70.0)) == 3);   // cooldown 1 (EMAd hit 83: not yet bad)
        CHECK(k.observe_boundary(mk(3, 44.0, 14.7, 70.0)) == 3);   // cooldown 2 (EMAd hit 78.5: bad, held)
        CHECK(k.observe_boundary(mk(3, 44.0, 14.7, 70.0)) == 3);   // bad 1
        CHECK(k.observe_boundary(mk(3, 44.0, 14.7, 70.0)) == 2);   // bad 2 -> step down (k 3 -> 2)
        CHECK(k.step_downs() == 1);
    }

    // ---- auto mode: aggregate regressing below 0.92 * best-at-previous-K steps down ----------------
    {
        AutoBatchConfig c; c.initial_k = 2;
        AutoBatchController k(c);
        for (int i = 0; i < 3; ++i) k.observe_boundary(mk(2, 40.0 + 2.0 * i, 20.0 + i, 90.0));
        CHECK(k.k_active() == 3);                                  // prev best 41.85 becomes the bar
        CHECK(k.observe_boundary(mk(3, 30.0, 10.0, 90.0)) == 3);   // cooldown 1 (EMAd 37.4 < 38.5: bad, held)
        CHECK(k.observe_boundary(mk(3, 30.0, 10.0, 90.0)) == 3);   // cooldown 2 (bad, held)
        CHECK(k.observe_boundary(mk(3, 30.0, 10.0, 90.0)) == 3);   // bad 1
        CHECK(k.observe_boundary(mk(3, 30.0, 10.0, 90.0)) == 2);   // bad 2 -> back down
        CHECK(k.step_downs() == 1);
    }

    // ---- auto mode: alternating noisy samples never oscillate 1 <-> 2 ------------------------------
    {
        AutoBatchConfig c; c.initial_k = 2; c.min_k = 1;
        AutoBatchController k(c);
        double agg = 40.0;
        for (int i = 0; i < 10; ++i) {
            if (i % 2 == 0) { agg += 2.0; k.observe_boundary(mk(2, agg, agg / 2.0, 90.0)); }   // good-looking
            else            k.observe_boundary(mk(2, agg, agg / 2.0, 70.0));                    // bad-looking
        }
        CHECK(k.k_active() == 2 && k.step_ups() == 0 && k.step_downs() == 0);
    }

    // ---- mem_refused clamps K to the rows actually running (no patience, no cooldown wait) ---------
    {
        AutoBatchConfig c; c.initial_k = 4;
        AutoBatchController k(c);
        CHECK(k.observe_boundary(mk(2, 30.0, 15.0, 90.0, true, true)) == 2);   // 4 -> 2 at once
        CHECK(k.step_downs() == 1);
        // no clamp when K already sits at/below the real ceiling
        CHECK(k.observe_boundary(mk(3, 45.0, 15.0, 90.0, true, true)) == 2);
        CHECK(k.step_downs() == 1);
        // the clamp never goes below min_k
        AutoBatchConfig c2; c2.initial_k = 4; c2.min_k = 3;
        AutoBatchController k2(c2);
        CHECK(k2.observe_boundary(mk(1, 10.0, 10.0, 90.0, true, true)) == 3);
    }

    // ---- [min_k, max_k] bounds hold under sustained pressure in both directions --------------------
    {
        AutoBatchConfig c; c.min_k = 2; c.initial_k = 2; c.min_stream_tps = 25.0;
        AutoBatchController k(c);
        for (int i = 0; i < 6; ++i) CHECK(k.observe_boundary(mk(2, 20.0, 10.0, 90.0)) == 2);
        CHECK(k.step_downs() == 0);
        AutoBatchConfig d; d.max_k = 3; d.initial_k = 3; d.min_stream_tps = 25.0;
        AutoBatchController m(d);
        for (int i = 0; i < 8; ++i) CHECK(m.observe_boundary(mk(3, 120.0, 40.0, 90.0)) == 3);
        CHECK(m.step_ups() == 0);
    }

    // ---- invalid observations teach nothing (no rows / no rate: no EMA, no vote) --------------------
    {
        AutoBatchConfig c; c.min_stream_tps = 25.0; c.initial_k = 2;
        AutoBatchController k(c);
        CHECK(k.observe_boundary(mk(0, 0.0, 0.0, 0.0)) == 2);
        CHECK(k.observe_boundary(mk(2, 0.0, 0.0, 0.0)) == 2);
        CHECK(near(k.ema_agg_tps(), 0.0) && near(k.ema_hit_pct(), 0.0));
        CHECK(k.observe_boundary(mk(2, 50.0, 25.0, 88.0)) == 2);   // first valid: EMAs init to it
        CHECK(near(k.ema_agg_tps(), 50.0) && near(k.ema_hit_pct(), 88.0));
    }

    if (fails) { std::printf("auto_batch_controller_test: %d FAILURES\n", fails); return 1; }
    std::printf("auto_batch_controller_test: all checks pass\n");
    return 0;
}
