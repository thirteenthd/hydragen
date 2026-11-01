// SPDX-License-Identifier: Apache-2.0
// High-performance C++ Lock-Free SPSC Ring Buffer IPC Reader for Strata Engines.
// Resolves Bottleneck 5: Eliminates Python Text-Pipe & GIL contention.

#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for Fast IPC instance
typedef struct strata_fast_ipc_t strata_fast_ipc_t;

// Callback signature for forwarding control lines (BSTAT, PSTAT, INFO, etc.)
typedef void (*strata_control_callback_fn)(const char* line, void* user_data);

/**
 * Creates and starts a high-performance C++ IPC reader thread over the engine's stdout file descriptor.
 * @param stdout_fd File descriptor of child process's stdout (e.g. proc.stdout.fileno())
 * @param num_slots Maximum number of concurrent batch slots (e.g. 4 or 8)
 * @return Pointer to initialized strata_fast_ipc_t, or nullptr on error.
 */
strata_fast_ipc_t* strata_fast_ipc_create(int stdout_fd, int num_slots);

/**
 * Register a callback for non-token control lines (BSTAT, PSTAT, INFO, DONE, ERR, etc.)
 * Allows the C++ reader to take exclusive ownership of stdout while preserving server telemetry.
 */
void strata_fast_ipc_set_control_callback(strata_fast_ipc_t* ipc, strata_control_callback_fn cb, void* user_data);

/**
 * Reads the next token for a given batch slot directly from the lock-free ring buffer.
 * Zero Python string allocations, zero Python Queue lock contention.
 * 
 * @param ipc Pointer to strata_fast_ipc_t
 * @param slot Batch slot index (0 .. num_slots-1)
 * @param out_token Output pointer for the decoded integer token ID
 * @param timeout_ms Maximum time to wait in milliseconds (0 for non-blocking poll)
 * @return 1 on new token, 0 on timeout, -1 on end-of-sequence (DONE), -2 on error/EOF.
 */
int32_t strata_fast_ipc_read_token(strata_fast_ipc_t* ipc, int32_t slot, int32_t* out_token, int32_t timeout_ms);

/**
 * Clears and resets the ring buffer for a given slot.
 * Call before starting a new batch request in this slot to guarantee no stale tokens or EOF markers.
 */
void strata_fast_ipc_clear_slot(strata_fast_ipc_t* ipc, int32_t slot);

/**
 * Stop the background reader thread.
 */
void strata_fast_ipc_stop(strata_fast_ipc_t* ipc);

/**
 * Destroy the IPC instance and free all resources.
 */
void strata_fast_ipc_destroy(strata_fast_ipc_t* ipc);

#ifdef __cplusplus
}
#endif
