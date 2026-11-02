# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim redirecting to hydragen.server.logit_bias."""
import sys
from hydragen.server import logit_bias as _mod
from hydragen.server.logit_bias import *  # noqa: F401, F403

if hasattr(_mod, "__all__"):
    __all__ = _mod.__all__

sys.modules[__name__] = _mod
