// SPDX-License-Identifier: Apache-2.0
#include "hydragen_engine_c_api.h"
#include "hydragen_engine.hpp"
#include <cstring>
#include <new>

extern "C" {

hydragen_engine_t* hydragen_engine_create(
    int32_t batch_slots,
    int32_t block_size,
    int32_t max_blocks,
    int32_t mock_worker
) {
    try {
        hydragen::core::HydragenEngineConfig cfg;
        cfg.batch_slots = batch_slots > 0 ? batch_slots : 4;
        cfg.block_size = block_size > 0 ? block_size : 16;
        cfg.max_blocks = max_blocks > 0 ? max_blocks : 4096;
        cfg.mock_mode = (mock_worker != 0);

        auto* engine = new hydragen::core::NativeHydragenEngine(cfg);
        return reinterpret_cast<hydragen_engine_t*>(engine);
    } catch (...) {
        return nullptr;
    }
}

int32_t hydragen_engine_submit(
    hydragen_engine_t* engine,
    int32_t req_id,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    int32_t max_new,
    const int32_t* expert_ids,
    int32_t num_experts
) {
    if (!engine) return -1;
    auto* eng = reinterpret_cast<hydragen::core::NativeHydragenEngine*>(engine);
    return eng->submit_request(req_id, prompt_ids, prompt_len, max_new, expert_ids, num_experts);
}

int32_t hydragen_engine_step(
    hydragen_engine_t* engine,
    int32_t* out_req_id,
    int32_t* out_token_id,
    int32_t* out_is_finished
) {
    if (!engine) return -1;
    auto* eng = reinterpret_cast<hydragen::core::NativeHydragenEngine*>(engine);
    return eng->step(out_req_id, out_token_id, out_is_finished);
}

int32_t hydragen_engine_cancel(
    hydragen_engine_t* engine,
    int32_t req_id
) {
    if (!engine) return 0;
    auto* eng = reinterpret_cast<hydragen::core::NativeHydragenEngine*>(engine);
    return eng->cancel_request(req_id) ? 1 : 0;
}

int32_t hydragen_engine_get_telemetry(
    hydragen_engine_t* engine,
    char* out_buf,
    int32_t buf_size
) {
    if (!engine || !out_buf || buf_size <= 0) return -1;
    auto* eng = reinterpret_cast<hydragen::core::NativeHydragenEngine*>(engine);
    std::string telem = eng->get_telemetry_json();
    if (static_cast<int32_t>(telem.size()) >= buf_size) {
        return -2; // Buffer too small
    }
    std::strncpy(out_buf, telem.c_str(), buf_size);
    out_buf[buf_size - 1] = '\0';
    return static_cast<int32_t>(telem.size());
}

int32_t hydragen_engine_is_alive(
    hydragen_engine_t* engine
) {
    if (!engine) return 0;
    auto* eng = reinterpret_cast<hydragen::core::NativeHydragenEngine*>(engine);
    return eng->is_alive() ? 1 : 0;
}

void hydragen_engine_destroy(
    hydragen_engine_t* engine
) {
    if (!engine) return;
    auto* eng = reinterpret_cast<hydragen::core::NativeHydragenEngine*>(engine);
    delete eng;
}

} // extern "C"
