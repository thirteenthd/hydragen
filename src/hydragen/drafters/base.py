# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/drafters/base.py - Abstract Interfaces for Speculative Drafting Backends.

Defines the pluggable speculative decoding contract:
- DraftingMode: Auto, MTP, DFlash2, Off.
- DraftCandidate: Container holding speculative tokens, confidence scores, and source metadata.
- SpeculativeDrafter: Abstract base class for multi-token speculative guessers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch


class DraftingMode(str, Enum):
    AUTO = "auto"
    MTP = "mtp"
    DFLASH2 = "dflash2"
    OFF = "off"


@dataclass
class DraftCandidate:
    """Represents candidate tokens produced by a speculative drafter."""
    tokens: List[int]
    logits: Optional[torch.Tensor] = None
    confidence: float = 1.0
    source: str = "speculative"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpeculativeDrafter(ABC):
    """Abstract base class for speculative drafting heads (MTP, DFLASH2, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the speculative drafter."""
        ...

    @property
    @abstractmethod
    def mode(self) -> DraftingMode:
        """Active drafting mode enumeration."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the drafter has model weights loaded and is ready to speculate."""
        ...

    @abstractmethod
    def prefill_step(self, prompt_tokens: torch.Tensor, hidden_states: torch.Tensor) -> None:
        """Pre-warms internal draft KV caches during prefill to eliminate cold-start penalties."""
        ...

    @abstractmethod
    def draft(
        self,
        anchor_token: int,
        hidden_state: torch.Tensor,
        n_tokens: int = 4,
        **kwargs: Any,
    ) -> DraftCandidate:
        """Generates n_tokens speculative candidate tokens given the current anchor state."""
        ...

    @abstractmethod
    def accept_feedback(self, n_accepted: int) -> None:
        """Updates internal acceptance telemetry and adaptive thresholds."""
        ...
