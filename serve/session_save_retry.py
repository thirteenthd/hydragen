# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim redirecting to hydragen.server.session_save_retry."""
import sys
from hydragen.server import session_save_retry as _mod
from hydragen.server.session_save_retry import *  # noqa: F401, F403

if hasattr(_mod, "__all__"):
    __all__ = _mod.__all__

sys.modules[__name__] = _mod
