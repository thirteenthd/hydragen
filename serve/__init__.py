# SPDX-License-Identifier: Apache-2.0
"""Legacy serve package compatibility shim forwarding to hydragen.server."""
import sys
from hydragen import server as _mod
from hydragen.server import *  # noqa: F401, F403

if hasattr(_mod, "__all__"):
    __all__ = _mod.__all__

sys.modules[__name__] = _mod
