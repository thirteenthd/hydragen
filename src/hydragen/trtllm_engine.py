# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/trtllm_engine.py - Re-export facade for Hydragen Core TRTLLMEngine.

Maintains backward compatibility with all existing imports.
"""

from hydragen.core.block_manager import TRTPagedBlockManager
from hydragen.core.engine import TRTLLMEngine
from hydragen.core.scheduler import InFlightScheduler

__all__ = ["TRTLLMEngine", "TRTPagedBlockManager", "InFlightScheduler"]
