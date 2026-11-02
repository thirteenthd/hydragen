# SPDX-License-Identifier: Apache-2.0
"""Test module forwarding shims between serve and hydragen."""
import serve.server
import hydragen.server.server
assert serve.server is hydragen.server.server
print("serve.server is hydragen.server.server: OK")

import serve.frontend
import hydragen.server.frontend
assert serve.frontend is hydragen.server.frontend
print("serve.frontend is hydragen.server.frontend: OK")

import serve.trtllm_engine
import hydragen.trtllm_engine
assert serve.trtllm_engine is hydragen.trtllm_engine
print("serve.trtllm_engine is hydragen.trtllm_engine: OK")

import serve.fast_ipc
import hydragen.fast_ipc
assert serve.fast_ipc is hydragen.fast_ipc
print("serve.fast_ipc is hydragen.fast_ipc: OK")

import serve.telemetry
import hydragen.server.telemetry
assert serve.telemetry is hydragen.server.telemetry
print("serve.telemetry is hydragen.server.telemetry: OK")

print("ALL SHIM INTEGRATION TESTS PASSED PERFECTLY!")
