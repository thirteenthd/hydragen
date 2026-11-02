# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/server/fast_ipc.py - Re-export fast_ipc for Strata Server."""
import sys
from hydragen import fast_ipc as _mod
from hydragen.fast_ipc import *  # noqa: F401, F403

if hasattr(_mod, "__all__"):
    __all__ = _mod.__all__

sys.modules[__name__] = _mod
