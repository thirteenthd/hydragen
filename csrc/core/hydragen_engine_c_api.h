// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hydragen_engine_t hydragen_engine_t;

/**
 * @brief Initialize a native HydragenEngine instance.
 * @param batch_slots Number of parallel continuous execution slots (e.g. 2, 4, 8)
 * @param block_size Size of virtual paged KV blocks (e.g. 16 tokens)
 * @param max_blocks Maximum number of paged memory blocks allocated
 * @param mock_worker 1 for deterministic MockHardwareWorker, 0 for hardware worker
 * @return Handle to initialized engine, or NULL on error
 */
hydragen_engine_t* hydragen_engine_create(
    int32_t batch_slots,
    int32_t block_size,
    int32_t max_blocks,
    int32_t mock_worker
);

/**
 * @brief Submit an inference request to the native In-Flight Scheduler.
 * @return > 0 (active slot assigned), 0 (queued in locality queue), < 0 (error)
 */
int32_t hydragen_engine_submit(
    hydragen_engine_t* engine,
    int32_t req_id,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    int32_t max_new,
    const int32_t* expert_ids,
    int32_t num_experts
);

/**
 * @brief Step one decode iteration across active batch slots.
 * @return 1 on token generated, 0 on idle/no active requests, -1 on error
 */
int32_t hydragen_engine_step(
    hydragen_engine_t* engine,
    int32_t* out_req_id,
    int32_t* out_token_id,
    int32_t* out_is_finished
);

/**
 * @brief Cancel a running or queued request.
 */
int32_t hydragen_engine_cancel(
    hydragen_engine_t* engine,
    int32_t req_id
);

/**
 * @brief Query live engine telemetry JSON.
 */
int32_t hydragen_engine_get_telemetry(
    hydragen_engine_t* engine,
    char* out_buf,
    int32_t buf_size
);

/**
 * @brief Check if the engine is alive and operational.
 */
int32_t hydragen_engine_is_alive(
    hydragen_engine_t* engine
);

/**
 * @brief Destroy the engine instance and free all resources.
 */
void hydragen_engine_destroy(
    hydragen_engine_t* engine
);

#ifdef __cplusplus
}
#endif
