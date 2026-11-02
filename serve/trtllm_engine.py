# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim redirecting to hydragen.trtllm_engine."""
import sys
from hydragen import trtllm_engine as _mod
from hydragen.trtllm_engine import *  # noqa: F401, F403

if hasattr(_mod, "__all__"):
    __all__ = _mod.__all__

sys.modules[__name__] = _mod
