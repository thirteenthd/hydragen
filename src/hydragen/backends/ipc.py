# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/backends/ipc.py - Dynamic IPC Backend Plugins.

Provides:
- NativeIPCBackend: High-speed zero-copy C++ ring buffer (libstrata_fast_ipc.so).
- FallbackIPCBackend: Pure Python fallback channel for tests and standalone mode.
"""

from __future__ import annotations

import ctypes
import queue
from typing import Any, Callable, Dict, Optional

from hydragen.backends.base import IPCBackend

CONTROL_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


class NativeIPCBackend(IPCBackend):
    """C++ fast lock-free ring buffer IPC using libstrata_fast_ipc.so."""

    def __init__(self, native_lib: Optional[ctypes.CDLL] = None):
        self._lib = native_lib
        if self._lib is None:
            from hydragen.lib_loader import load_native_library
            self._lib = load_native_library("libstrata_fast_ipc.so", env_var="LIBSTRATA_FAST_IPC_SO")

        if self._lib is not None:
            self._setup_c_api(self._lib)

    @property
    def name(self) -> str:
        return "native_fast_ipc"

    def is_available(self) -> bool:
        return self._lib is not None and hasattr(self._lib, "strata_fast_ipc_create")

    def _setup_c_api(self, lib: ctypes.CDLL) -> None:
        try:
            lib.strata_fast_ipc_create.argtypes = [ctypes.c_int, ctypes.c_int]
            lib.strata_fast_ipc_create.restype = ctypes.c_void_p

            lib.strata_fast_ipc_set_control_callback.argtypes = [
                ctypes.c_void_p,
                CONTROL_CB_TYPE,
                ctypes.c_void_p,
            ]
            lib.strata_fast_ipc_set_control_callback.restype = None

            lib.strata_fast_ipc_read_token.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int32,
            ]
            lib.strata_fast_ipc_read_token.restype = ctypes.c_int32

            if hasattr(lib, "strata_fast_ipc_clear_slot"):
                lib.strata_fast_ipc_clear_slot.argtypes = [ctypes.c_void_p, ctypes.c_int32]
                lib.strata_fast_ipc_clear_slot.restype = None

            lib.strata_fast_ipc_stop.argtypes = [ctypes.c_void_p]
            lib.strata_fast_ipc_stop.restype = None

            lib.strata_fast_ipc_destroy.argtypes = [ctypes.c_void_p]
            lib.strata_fast_ipc_destroy.restype = None
        except Exception:
            pass

    def create(self, stdout_fd: int, num_slots: int = 4) -> Any:
        if not self.is_available() or stdout_fd < 0:
            return None
        return self._lib.strata_fast_ipc_create(int(stdout_fd), int(num_slots))

    def set_control_callback(self, handle: Any, cb: Callable[[bytes], None]) -> None:
        if not self.is_available() or not handle:
            return

        def _wrapper(msg_ptr: bytes, _user: Any) -> None:
            cb(msg_ptr)

        c_cb = CONTROL_CB_TYPE(_wrapper)
        self._lib.strata_fast_ipc_set_control_callback(handle, c_cb, None)
        # Store callback reference to prevent garbage collection
        setattr(handle, "_cb_ref", c_cb)

    def read_token(self, handle: Any, slot: int, timeout_ms: int = 100) -> Optional[int]:
        if not self.is_available() or not handle:
            return None
        token_out = ctypes.c_int32(-1)
        res = self._lib.strata_fast_ipc_read_token(
            handle,
            int(slot),
            ctypes.byref(token_out),
            int(timeout_ms),
        )
        if res == 1:
            return int(token_out.value)
        return None

    def clear_slot(self, handle: Any, slot: int) -> None:
        if self.is_available() and handle and hasattr(self._lib, "strata_fast_ipc_clear_slot"):
            self._lib.strata_fast_ipc_clear_slot(handle, int(slot))

    def stop(self, handle: Any) -> None:
        if self.is_available() and handle:
            self._lib.strata_fast_ipc_stop(handle)

    def destroy(self, handle: Any) -> None:
        if self.is_available() and handle:
            self._lib.strata_fast_ipc_destroy(handle)


class FallbackIPCBackend(IPCBackend):
    """Pure Python fallback IPC using thread-safe queues."""

    def __init__(self):
        self._queues: Dict[int, queue.Queue] = {}

    @property
    def name(self) -> str:
        return "fallback_queue_ipc"

    def is_available(self) -> bool:
        return True

    def create(self, stdout_fd: int, num_slots: int = 4) -> Any:
        return {"stdout_fd": stdout_fd, "slots": {i: queue.Queue() for i in range(num_slots)}}

    def set_control_callback(self, handle: Any, cb: Callable[[bytes], None]) -> None:
        pass

    def read_token(self, handle: Any, slot: int, timeout_ms: int = 100) -> Optional[int]:
        if not handle or "slots" not in handle or slot not in handle["slots"]:
            return None
        q = handle["slots"][slot]
        try:
            return q.get(timeout=timeout_ms / 1000.0)
        except queue.Empty:
            return None

    def clear_slot(self, handle: Any, slot: int) -> None:
        if handle and "slots" in handle and slot in handle["slots"]:
            q = handle["slots"][slot]
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def stop(self, handle: Any) -> None:
        pass

    def destroy(self, handle: Any) -> None:
        pass
