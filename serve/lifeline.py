# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim redirecting to hydragen.server.lifeline."""
import sys
from hydragen.server import lifeline as _mod
from hydragen.server.lifeline import *  # noqa: F401, F403

if hasattr(_mod, "__all__"):
    __all__ = _mod.__all__

sys.modules[__name__] = _mod
