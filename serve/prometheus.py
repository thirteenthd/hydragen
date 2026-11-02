# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim redirecting to hydragen.server.prometheus."""
import sys
from hydragen.server import prometheus as _mod
from hydragen.server.prometheus import *  # noqa: F401, F403

if hasattr(_mod, "__all__"):
    __all__ = _mod.__all__

sys.modules[__name__] = _mod
