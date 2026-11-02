# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim redirecting to hydragen.server.launch_identity."""
import sys
from hydragen.server import launch_identity as _mod
from hydragen.server.launch_identity import *  # noqa: F401, F403

if hasattr(_mod, "__all__"):
    __all__ = _mod.__all__

sys.modules[__name__] = _mod
