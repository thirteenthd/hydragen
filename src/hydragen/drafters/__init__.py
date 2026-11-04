# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/drafters/__init__.py - Pluggable Speculative Drafter Architecture."""

from hydragen.drafters.base import (
    DraftCandidate,
    DraftingMode,
    SpeculativeDrafter,
    WorkerDelegatedDrafter,
)
from hydragen.drafters.dflash2 import DFlash2Drafter
from hydragen.drafters.factory import (
    DisabledDrafter,
    get_speculative_drafter,
    resolve_drafting_mode,
)
from hydragen.drafters.mtp import MtpDrafter

__all__ = [
    "DraftingMode",
    "DraftCandidate",
    "SpeculativeDrafter",
    "WorkerDelegatedDrafter",
    "MtpDrafter",
    "DFlash2Drafter",
    "DisabledDrafter",
    "resolve_drafting_mode",
    "get_speculative_drafter",
]
