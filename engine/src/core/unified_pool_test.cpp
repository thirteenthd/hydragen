// src/core/unified_pool_test.cpp - the unified pool's floor, boot cap and chunk math (P5-5B.1, no GPU needed).
#include "strata/core/unified_pool.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using namespace strata::core;

    // ---- expert_floor_from_weights -------------------------------------------------------------
    // no profile at all, and an all-zero one: the 85% fallback (the hot set's measured share),
    // integer arithmetic: 2048 * 85 / 100 = 1740, 1000 -> 850, 100 -> 85
    CHECK(expert_floor_from_weights({}, 2048, 1, 86.0) == 1740);
    CHECK(expert_floor_from_weights({}, 1000, 1, 86.0) == 850);
    CHECK(expert_floor_from_weights({0.0, 0.0, 0.0}, 100, 1, 86.0) == 85);
    CHECK(expert_floor_from_weights({-1.0, 0.0}, 100, 1, 86.0) == 85);   // no positive mass: the fallback
    CHECK(expert_floor_from_weights({}, 0, 1, 86.0) == 0);               // no cache: no floor
    CHECK(expert_floor_from_weights({}, -5, 1, 86.0) == 0);

    // a skewed profile at the shipped 86% knee: 50+30 = 80 < 86, +10 = 90 >= 86 -> 3 slots
    CHECK(expert_floor_from_weights({50.0, 30.0, 10.0, 10.0}, 1000, 1, 86.0) == 3);
    // the knee is a fraction of the total, not a count: a flat profile needs 86 of 100
    std::vector<double> flat(100, 1.0);
    CHECK(expert_floor_from_weights(flat, 1000, 1, 86.0) == 86);
    // the knee clamps both ways: 200% -> 100% (every positive pair), 0.5% -> 1% (the hottest alone)
    CHECK(expert_floor_from_weights({50.0, 30.0, 10.0, 10.0}, 1000, 1, 200.0) == 4);
    CHECK(expert_floor_from_weights({50.0, 30.0, 10.0, 10.0}, 1000, 1, 0.5) == 1);
    // more pairs than slots: the walk stops at full_slots and the floor is the whole cache
    std::vector<double> many(2000, 1.0);
    CHECK(expert_floor_from_weights(many, 100, 1, 86.0) == 100);

    // floor_override wins over the weights, and still sits inside [min_slots, full_slots]
    CHECK(expert_floor_from_weights({50.0, 30.0, 10.0, 10.0}, 1000, 1, 86.0, 100) == 100);
    CHECK(expert_floor_from_weights({}, 1000, 1, 86.0, 5000) == 1000);   // above full: clamped down
    CHECK(expert_floor_from_weights({}, 1000, 64, 86.0, 7) == 64);       // below min: clamped up
    CHECK(expert_floor_from_weights({}, 1000, 1, 86.0, 0) == 850);       // 0 = auto

    // the [min_slots, full_slots] clamp on the walked result: the knee wants 1 slot, min_slots says 4
    CHECK(expert_floor_from_weights({90.0, 10.0}, 1000, 4, 86.0) == 4);
    // min_slots itself clamps: 0 -> 1, above full -> full
    CHECK(expert_floor_from_weights({90.0, 10.0}, 1000, 0, 86.0) == 1);
    CHECK(expert_floor_from_weights({90.0, 10.0}, 1000, 5000, 86.0) == 1000);

    // ---- initial_active_cap --------------------------------------------------------------------
    // the same 128 KiB of free-above-reserve VRAM: a 131072-byte slot buys one more active stream,
    // a 16384-byte slot buys eight (the 16k envelope's small slots start more streams from the same slack)
    CHECK(initial_active_cap(131072, 0, 131072, 8) == 2);
    CHECK(initial_active_cap(131072, 0, 16384, 8) == 8);
    CHECK(initial_active_cap(131072, 0, 16384, 4) == 4);    // never above the boot --batch carve
    // flex counts like free: 16 KiB free + 112 KiB parked chunks = 8 slots' worth at 16 KiB, plus the one that always fits
    CHECK(initial_active_cap(16384, 114688, 16384, 16) == 9);
    // one stream always fits; nothing negative leaks in
    CHECK(initial_active_cap(0, 0, 131072, 8) == 1);
    CHECK(initial_active_cap(-1000000, -1000000, 131072, 8) == 1);
    // no slot cost known (no cache): the whole carve; a bad max_batch still allows one stream
    CHECK(initial_active_cap(1 << 20, 0, 0, 6) == 6);
    CHECK(initial_active_cap(1 << 20, 0, 131072, 0) == 1);

    // ---- chunk_deficit -------------------------------------------------------------------------
    CHECK(chunk_deficit(0, 0, 2097152) == 0);
    CHECK(chunk_deficit(1024, 1024, 2097152) == 0);         // not short
    CHECK(chunk_deficit(1024, 2048, 2097152) == 0);         // over-covered
    CHECK(chunk_deficit(1024, 0, 0) == 0);                  // no VMM: no chunks
    CHECK(chunk_deficit(1024, 0, -4) == 0);
    // rounding up to whole 2 MiB chunks: 1 MiB short -> 1 chunk, 2 MiB -> 1, 2 MiB + 1 -> 2
    CHECK(chunk_deficit(3 * 1048576, 1 * 1048576, 2097152) == 1);
    CHECK(chunk_deficit(2 * 1048576, 0, 2097152) == 1);
    CHECK(chunk_deficit(2 * 1048576 + 1, 0, 2097152) == 2);
    CHECK(chunk_deficit(10, 0, 4) == 3);                    // (10 + 3) / 4
    CHECK(chunk_deficit(8, 0, 4) == 2);                     // exact
    CHECK(chunk_deficit(8, -5, 4) == 2);                    // a negative avail reads as 0

    // ---- the config's shipped defaults ----------------------------------------------------------
    UnifiedPoolConfig c;
    CHECK(c.full_slots == 0 && c.min_slots == 1 && c.knee_pct == 86.0 && c.floor_override == 0);
    CHECK(c.slot_vram_bytes == 0 && c.free_above_reserve_bytes == 0 && c.flex_bytes == 0);
    CHECK(c.max_batch == 1 && c.min_stream_tps == 0.0);

    if (fails) { std::printf("unified_pool_test: %d FAILURES\n", fails); return 1; }
    std::printf("unified_pool_test: all checks pass\n");
    return 0;
}
