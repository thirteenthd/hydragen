# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim redirecting to hydragen.server.vllm_engine."""
import sys
from hydragen.server import vllm_engine as _mod
from hydragen.server.vllm_engine import *  # noqa: F401, F403

if hasattr(_mod, "__all__"):
    __all__ = _mod.__all__

sys.modules[__name__] = _mod
