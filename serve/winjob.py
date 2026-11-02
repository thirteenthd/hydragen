# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim redirecting to hydragen.server.winjob."""
import sys
from hydragen.server import winjob as _mod
from hydragen.server.winjob import *  # noqa: F401, F403

if hasattr(_mod, "__all__"):
    __all__ = _mod.__all__

sys.modules[__name__] = _mod
