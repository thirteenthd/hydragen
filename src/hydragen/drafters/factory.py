# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/drafters/factory.py - Architecture-Aware Speculative Drafter Factory.

Selects and instantiates the optimal speculative drafting engine (MTP vs DFlash2 vs Off)
based on host GPU architecture, batch concurrency requirements, and configuration overrides.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from hydragen.backends.base import DeviceCapability, DeviceType, HardwareBackend
from hydragen.backends.registry import BackendRegistry
from hydragen.drafters.base import DraftCandidate, DraftingMode, SpeculativeDrafter
from hydragen.drafters.dflash2 import DFlash2Drafter
from hydragen.drafters.mtp import MtpDrafter


class DisabledDrafter(SpeculativeDrafter):
    """Null drafter used when speculative decoding is explicitly disabled."""

    @property
    def name(self) -> str:
        return "speculative_drafting_disabled"

    @property
    def mode(self) -> DraftingMode:
        return DraftingMode.OFF

    def is_available(self) -> bool:
        return False

    def prefill_step(self, prompt_tokens: torch.Tensor, hidden_states: torch.Tensor) -> None:
        pass

    def draft(self, anchor_token: int, hidden_state: torch.Tensor, n_tokens: int = 4, **kwargs: Any) -> DraftCandidate:
        return DraftCandidate(tokens=[], confidence=0.0, source="none")

    def accept_feedback(self, n_accepted: int) -> None:
        pass


def resolve_drafting_mode(
    requested_mode: Optional[DraftingMode | HardwareBackend] = None,
    hardware_backend: Optional[HardwareBackend] = None,
) -> DraftingMode:
    """Determines the optimal speculative drafting mode based on hardware and env."""
    if isinstance(requested_mode, HardwareBackend):
        hardware_backend = requested_mode
        requested_mode = None

    forced = os.environ.get("HYDRAGEN_DRAFTER")
    if forced:
        forced_lower = forced.lower().strip()
        if forced_lower in ("mtp", "1"):
            return DraftingMode.MTP
        elif forced_lower in ("dflash2", "dflash"):
            return DraftingMode.DFLASH2
        elif forced_lower in ("off", "none", "0", "false"):
            return DraftingMode.OFF
        elif forced_lower == "auto":
            requested_mode = DraftingMode.AUTO

    mode = requested_mode or DraftingMode.AUTO
    if mode != DraftingMode.AUTO:
        return mode

    # Architecture-aware resolution:
    # On NVIDIA Ampere (sm_86) and concurrent serving (--batch 2), MTP is faster.
    # On NVIDIA Blackwell (sm_120) and AMD Strix Halo (gfx1151), DFlash2 is faster and saves VRAM.
    hw = hardware_backend or BackendRegistry.get_hardware_backend()
    cap = hw.capability

    dev_name = cap.device_name.lower()
    details = cap.details or {}
    arch = str(details.get("architecture", "")).lower()

    # Check for Blackwell or Strix Halo
    if "sm_120" in dev_name or "blackwell" in dev_name or "5070" in dev_name:
        return DraftingMode.DFLASH2
    if "gfx1151" in dev_name or "strix" in dev_name or "strix" in arch:
        return DraftingMode.DFLASH2

    # Default to MTP for Ampere, Ada, Hopper, Intel, and CPU
    return DraftingMode.MTP


def get_speculative_drafter(
    mode: Optional[DraftingMode] = None,
    hardware_backend: Optional[HardwareBackend] = None,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> SpeculativeDrafter:
    """Instantiates the optimal SpeculativeDrafter instance."""
    resolved = resolve_drafting_mode(mode, hardware_backend)

    if resolved == DraftingMode.MTP:
        return MtpDrafter(device=device, **kwargs)
    elif resolved == DraftingMode.DFLASH2:
        return DFlash2Drafter(device=device, **kwargs)
    else:
        return DisabledDrafter()
