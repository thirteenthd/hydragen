# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/server/exceptions.py - Exception classes for Hydragen server and engine."""

from __future__ import annotations


class EngineDied(RuntimeError):
    """The engine process ended in the middle of a request (on Linux, often the out-of-memory killer or SIGABRT)."""


class EngineStarting(RuntimeError):
    """The engine is (re)starting and has not said READY yet: no context size to plan a request with - 503."""


class ClientGone(Exception):
    """The client hung up while its request was being prepared."""


class ModelBusy(RuntimeError):
    """Explicit model controls must not interrupt active or queued requests."""


class QueueWaitExpired(RuntimeError):
    """An inference call waited too long for its first engine turn."""


class QueueCancelled(RuntimeError):
    """The client left before its first serial engine turn."""


class EngineSilent(EngineDied):
    """The engine said nothing for too long during a request (or never acknowledged a STOP)."""


class EngineNonFinite(ValueError):
    """The engine's DONE (or slot's BDONE) said `nonfinite`: logits held NaN or infinity."""


class SessionRefused(ValueError):
    """Disk sessions: the engine refused or failed a SAVE / RESTORE and is still in step."""

    STATUS = {"invalid": 400, "storage": 507, "memory": 503, "io": 500}

    def __init__(self, kind: str, message: str, published: bool = False):
        super().__init__(message)
        self.kind = kind if kind in self.STATUS else "io"
        self.published = published

    @property
    def status(self) -> int:
        return self.STATUS[self.kind]


class EngineStuck(RuntimeError):
    """The engine process did not end after QUIT, terminate and kill."""


class GpuBusy(RuntimeError):
    """The model is unloaded and the GPU has less free VRAM than min_free_vram_mib."""

    def __init__(self, message: str = "", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


__all__ = [
    "EngineDied",
    "EngineStarting",
    "ClientGone",
    "ModelBusy",
    "QueueWaitExpired",
    "QueueCancelled",
    "EngineSilent",
    "EngineNonFinite",
    "SessionRefused",
    "EngineStuck",
    "GpuBusy",
]
