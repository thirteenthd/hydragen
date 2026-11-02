# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim redirecting to hydragen.server.server."""
import sys
from hydragen.server import server as _mod
from hydragen.server.server import *  # noqa: F401, F403

if hasattr(_mod, "__all__"):
    __all__ = _mod.__all__

sys.modules[__name__] = _mod
