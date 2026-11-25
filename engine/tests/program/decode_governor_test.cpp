// decode_governor_test.cpp: unit tests for P2-D1 DecodeGovernor (header-only, no GPU).
#include "strata/program/decode_governor.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::program::DecodeGovernor;
using strata::program::DecodeGovernorConfig;
using strata::program::parse_decode_governor_config;

static int fails = 0;
#define CHECK(c)                                                             \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c);          \
            ++fails;                                                         \
        }                                                                    \
    } while (0)

static void set_env_var(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value && *value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static bool is_strictly_sorted(const std::vector<int>& v) {
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] <= v[i - 1]) return false;
    }
    return true;
}

int main() {
    // 1. Default-OFF behavior & env parsing (HG_* precedes STRATA_*, CLI overrides env)
    {
        set_env_var("HG_DECODE_AUTO", nullptr);
        set_env_var("STRATA_DECODE_AUTO", nullptr);
        set_env_var("HG_MIN_STREAM_TPS", nullptr);
        set_env_var("STRATA_MIN_STREAM_TPS", nullptr);
        set_env_var("HG_HIT_FLOOR_PCT", nullptr);
        set_env_var("HG_COHORT_QUANTUM_MS", nullptr);

        DecodeGovernorConfig dflt = parse_decode_governor_config(0.0);
        CHECK(!dflt.enabled);
        CHECK(dflt.min_stream_tps == 0.0);
        CHECK(dflt.hit_floor_pct == 86.0);
        CHECK(dflt.cohort_quantum_ms == 2000.0);
        CHECK(dflt.initial_width == 4);

        // Passing --min-stream-tps > 0 enables the governor automatically
        DecodeGovernorConfig cli_sla = parse_decode_governor_config(30.0);
        CHECK(cli_sla.enabled);
        CHECK(cli_sla.min_stream_tps == 30.0);

        // Explicit HG_DECODE_AUTO=0 acts as kill switch even with SLA set
        set_env_var("HG_DECODE_AUTO", "0");
        DecodeGovernorConfig off = parse_decode_governor_config(30.0);
        CHECK(!off.enabled);

        // HG_DECODE_AUTO=1 + custom HG_HIT_FLOOR_PCT / HG_COHORT_QUANTUM_MS
        set_env_var("STRATA_DECODE_AUTO", "0");
        set_env_var("HG_DECODE_AUTO", "1");
        set_env_var("HG_HIT_FLOOR_PCT", "88.5");
        set_env_var("HG_COHORT_QUANTUM_MS", "500");
        DecodeGovernorConfig hg = parse_decode_governor_config(0.0);
        CHECK(hg.enabled);
        CHECK(hg.hit_floor_pct == 88.5);
        CHECK(hg.cohort_quantum_ms == 500.0);

        // Clean up env vars
        set_env_var("HG_DECODE_AUTO", nullptr);
        set_env_var("STRATA_DECODE_AUTO", nullptr);
        set_env_var("HG_HIT_FLOOR_PCT", nullptr);
        set_env_var("HG_COHORT_QUANTUM_MS", nullptr);
    }

    // 2. Sticky cohort, lazy init min(hard_max, 4), and canonical ascending sort
    {
        DecodeGovernorConfig cfg;
        cfg.enabled = true;
        cfg.cohort_quantum_ms = 2000.0;
        DecodeGovernor gov(cfg);

        // Unsorted input -> lazy init target_n=4, sorted ascending [0,1,2,3]
        const std::vector<int> runnable8 = {7, 0, 5, 2, 6, 1, 4, 3};
        const auto c1 = gov.select_cohort(runnable8, /*win_slot_cap=*/8, /*now_ms=*/0.0);
        CHECK(gov.target_width() == 4);
        CHECK((c1 == std::vector<int>{0, 1, 2, 3}));
        CHECK(is_strictly_sorted(c1));

        // Sticky within quantum
        const auto c2 = gov.select_cohort(runnable8, 8, 500.0);
        CHECK(c2 == c1);

        // Small win_slot_cap clamps initial width to min(hard_max, 4)
        DecodeGovernor gov_narrow(cfg);
        const auto cn = gov_narrow.select_cohort({0, 1, 2, 3}, /*win_slot_cap=*/2, 0.0);
        CHECK(gov_narrow.target_width() == 2);
        CHECK((cn == std::vector<int>{0, 1}));

        // Start with 2 runnable slots, then grow to 8 -> retains sticky 2,5 and expands to 4
        DecodeGovernor gov_grow(cfg);
        const auto cg1 = gov_grow.select_cohort({5, 2}, 8, 0.0);
        CHECK((cg1 == std::vector<int>{2, 5}));
        CHECK(gov_grow.target_width() == 4);
        const auto cg2 = gov_grow.select_cohort({0, 1, 2, 3, 4, 5, 6, 7}, 8, 10.0);
        CHECK(cg2.size() == 4);
        CHECK(is_strictly_sorted(cg2));
        CHECK(std::find(cg2.begin(), cg2.end(), 2) != cg2.end());
        CHECK(std::find(cg2.begin(), cg2.end(), 5) != cg2.end());
    }

    // 3. Hard Physics Brake (min_stream_tps == 0 throughput mode)
    {
        DecodeGovernorConfig cfg;
        cfg.enabled = true;
        cfg.min_stream_tps = 0.0;
        cfg.hit_floor_pct = 86.0;
        cfg.initial_width = 4;
        cfg.eval_every = 4;
        DecodeGovernor gov(cfg);

        const std::vector<int> r8 = {0, 1, 2, 3, 4, 5, 6, 7};
        CHECK(gov.select_cohort(r8, 8, 0.0).size() == 4);

        // Healthy cache (95%) -> widens 4 -> 5 after 4 windows, then 5 -> 6 after 4 more
        for (int i = 0; i < 4; ++i) gov.observe_window(25.0, 4, 4, 95.0, 8);
        CHECK(gov.target_width() == 5);
        CHECK(gov.select_cohort(r8, 8, 100.0).size() == 5);

        for (int i = 0; i < 4; ++i) gov.observe_window(28.0, 5, 5, 93.0, 8);
        CHECK(gov.target_width() == 6);
        CHECK(gov.select_cohort(r8, 8, 200.0).size() == 6);

        // PCIe thrash cliff: hit_pct = 82% < 86% -> sheds width 6 -> 5 even at 0 min_stream_tps
        for (int i = 0; i < 4; ++i) gov.observe_window(35.0, 6, 6, 82.0, 8);
        CHECK(gov.target_width() == 5);
        CHECK((gov.select_cohort(r8, 8, 300.0) == std::vector<int>{0, 1, 2, 3, 4}));
    }

    // 4. User SLA floor (min_stream_tps == 30.0)
    {
        DecodeGovernorConfig cfg;
        cfg.enabled = true;
        cfg.min_stream_tps = 30.0;
        cfg.hit_floor_pct = 86.0;
        cfg.initial_width = 4;
        cfg.eval_every = 4;
        DecodeGovernor gov(cfg);

        const std::vector<int> r6 = {0, 1, 2, 3, 4, 5};
        CHECK(gov.select_cohort(r6, 8, 0.0).size() == 4);

        // 4 tokens across 4 slots in 40ms = 25.0 tok/s per stream (< 30.0) despite 98% hit rate
        for (int i = 0; i < 4; ++i) gov.observe_window(40.0, 4, 4, 98.0, 8);
        CHECK(gov.target_width() == 3);
        CHECK((gov.select_cohort(r6, 8, 160.0) == std::vector<int>{0, 1, 2}));
    }

    // 5. Hysteresis deadband stability (no oscillation inside deadband)
    {
        // 5a. Cache deadband [86.0%, 87.5%)
        DecodeGovernorConfig cfg_hit;
        cfg_hit.enabled = true;
        cfg_hit.min_stream_tps = 0.0;
        cfg_hit.hit_floor_pct = 86.0;
        cfg_hit.initial_width = 4;
        cfg_hit.eval_every = 4;
        DecodeGovernor gov_hit(cfg_hit);
        const std::vector<int> r8 = {0, 1, 2, 3, 4, 5, 6, 7};
        gov_hit.select_cohort(r8, 8, 0.0);
        for (int i = 0; i < 20; ++i) gov_hit.observe_window(25.0, 4, 4, 86.8, 8);
        CHECK(gov_hit.target_width() == 4);

        // 5b. Stream-TPS deadband [30.0, 34.5) tok/s
        DecodeGovernorConfig cfg_tps;
        cfg_tps.enabled = true;
        cfg_tps.min_stream_tps = 30.0;
        cfg_tps.hit_floor_pct = 86.0;
        cfg_tps.initial_width = 3;
        cfg_tps.eval_every = 4;
        DecodeGovernor gov_tps(cfg_tps);
        gov_tps.select_cohort(r8, 8, 0.0);
        // 1 token/slot in 31.25ms = 32.0 tok/s per stream -> in deadband [30.0, 34.5)
        for (int i = 0; i < 20; ++i) gov_tps.observe_window(31.25, 3, 3, 98.0, 8);
        CHECK(gov_tps.target_width() == 3);
    }

    // 6. Step-up gated on waiting slots + hysteresis thresholds
    {
        DecodeGovernorConfig cfg;
        cfg.enabled = true;
        cfg.min_stream_tps = 30.0;
        cfg.hit_floor_pct = 86.0;
        cfg.initial_width = 4;
        cfg.eval_every = 4;
        DecodeGovernor gov(cfg);

        // Only 4 runnable slots == target_n (no waiting slots): do not widen even at 60 tok/s & 99% hit
        const std::vector<int> r4 = {0, 1, 2, 3};
        gov.select_cohort(r4, 8, 0.0);
        for (int i = 0; i < 12; ++i) gov.observe_window(16.67, 4, 4, 99.0, 8);
        CHECK(gov.target_width() == 4);

        // Now 6 slots become runnable -> has_waiting_ = true -> 4 healthy windows widen 4 -> 5
        const std::vector<int> r6 = {0, 1, 2, 3, 4, 5};
        gov.select_cohort(r6, 8, 200.0);
        for (int i = 0; i < 4; ++i) gov.observe_window(16.67, 4, 4, 99.0, 8);
        CHECK(gov.target_width() == 5);
    }

    // 7. cohort_quantum_ms rotation vs 0 (sticky run-to-completion)
    {
        // 7a. cohort_quantum_ms == 0 -> never rotates while slots remain active
        DecodeGovernorConfig cfg0;
        cfg0.enabled = true;
        cfg0.cohort_quantum_ms = 0.0;
        cfg0.initial_width = 4;
        DecodeGovernor gov0(cfg0);
        const std::vector<int> r8 = {0, 1, 2, 3, 4, 5, 6, 7};
        CHECK((gov0.select_cohort(r8, 8, 0.0) == std::vector<int>{0, 1, 2, 3}));
        CHECK((gov0.select_cohort(r8, 8, 100000.0) == std::vector<int>{0, 1, 2, 3}));

        // 7b. cohort_quantum_ms == 50.0 -> rotates fairly and stays sorted ascending
        DecodeGovernorConfig cfg50;
        cfg50.enabled = true;
        cfg50.cohort_quantum_ms = 50.0;
        cfg50.initial_width = 3;
        DecodeGovernor gov50(cfg50);
        const std::vector<int> r6 = {0, 1, 2, 3, 4, 5};

        CHECK((gov50.select_cohort(r6, 8, 0.0) == std::vector<int>{0, 1, 2}));
        CHECK((gov50.select_cohort(r6, 8, 30.0) == std::vector<int>{0, 1, 2}));
        CHECK((gov50.select_cohort(r6, 8, 55.0) == std::vector<int>{1, 2, 3}));
        CHECK((gov50.select_cohort(r6, 8, 110.0) == std::vector<int>{2, 3, 4}));
        CHECK((gov50.select_cohort(r6, 8, 165.0) == std::vector<int>{3, 4, 5}));
        CHECK((gov50.select_cohort(r6, 8, 220.0) == std::vector<int>{0, 4, 5}));
    }

    // 8. Finished-slot replacement preserves remaining cohort sticky and pulls next waiter
    {
        DecodeGovernorConfig cfg;
        cfg.enabled = true;
        cfg.cohort_quantum_ms = 2000.0;
        cfg.initial_width = 3;
        DecodeGovernor gov(cfg);

        CHECK((gov.select_cohort({0, 1, 2, 3, 4}, 8, 0.0) == std::vector<int>{0, 1, 2}));
        // Slot 1 finishes; remaining cohort [0, 2] stays sticky and admits slot 3
        CHECK((gov.select_cohort({0, 2, 3, 4}, 8, 10.0) == std::vector<int>{0, 2, 3}));
    }

    if (fails == 0) std::printf("decode_governor_test: all passed\n");
    return fails != 0;
}
