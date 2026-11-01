# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/fast_ipc.py - Python ctypes wrapper for C++ Fast Lock-Free IPC."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Callable, Optional

CONTROL_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


class FastEngineIPC:
    """High-speed zero-copy C++ IPC bridge for reading engine tokens."""

    _lib: Optional[ctypes.CDLL] = None

    @classmethod
    def _get_lib(cls) -> Optional[ctypes.CDLL]:
        if cls._lib is not None:
            return cls._lib

        search_paths = [
            Path(__file__).resolve().parent / "libstrata_fast_ipc.so",
            Path(__file__).resolve().parent / "server" / "libstrata_fast_ipc.so",
            Path(__file__).resolve().parent.parent.parent / "lib" / "libstrata_fast_ipc.so",
            Path.cwd() / "lib" / "libstrata_fast_ipc.so",
            Path("/home/thirteenthd/hydragen/lib/libstrata_fast_ipc.so"),
            Path("/home/thirteenthd/hydragen/serve/libstrata_fast_ipc.so"),
        ]

        for p in search_paths:
            if p.exists():
                try:
                    lib = ctypes.CDLL(str(p))
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

                    cls._lib = lib
                    return cls._lib
                except Exception as e:
                    print(f"[fast_ipc] Warning: Failed to load {p}: {e}")

        return None

    def __init__(self, stdout_fd: int, num_slots: int = 4):
        self._lib = self._get_lib()
        self._handle = None
        self._cb_ref = None
        if self._lib is not None and stdout_fd >= 0:
            self._handle = self._lib.strata_fast_ipc_create(int(stdout_fd), int(num_slots))

    @property
    def is_available(self) -> bool:
        return self._handle is not None

    def set_control_callback(self, cb_func: Callable[[str], None]) -> None:
        """Register a callback for non-token control lines (BSTAT, PSTAT, INFO, etc.)."""
        if self._handle is not None and self._lib is not None:
            def _thunk(c_line, udata):
                try:
                    s = c_line.decode("utf-8", errors="replace") if c_line else ""
                    cb_func(s)
                except Exception:
                    pass

            self._cb_ref = CONTROL_CB_TYPE(_thunk)
            self._lib.strata_fast_ipc_set_control_callback(self._handle, self._cb_ref, None)

    def clear_slot(self, slot: int) -> None:
        """Clear and reset ring buffer for a given slot before starting a new batch request."""
        if self._handle is not None and self._lib is not None:
            if hasattr(self._lib, "strata_fast_ipc_clear_slot"):
                self._lib.strata_fast_ipc_clear_slot(self._handle, int(slot))

    def read_token(self, slot: int, timeout_ms: int = 500) -> Optional[int]:
        """Read next token for a slot directly from C++ lock-free SPSC ring buffer.
        Returns:
            int: token id if token received
            None: if timed out or still generating
            Raise StopIteration on EOF / DONE
            Raise RuntimeError on error
        """
        if self._handle is None:
            return None

        out_val = ctypes.c_int32(0)
        res = self._lib.strata_fast_ipc_read_token(
            self._handle, int(slot), ctypes.byref(out_val), int(timeout_ms)
        )
        if res == 1:
            return int(out_val.value)
        elif res == 0:
            return None
        elif res == -1:
            raise StopIteration
        else:
            raise RuntimeError(f"Engine IPC reported error or terminated (status {res})")

    def close(self):
        if self._handle is not None and self._lib is not None:
            self._lib.strata_fast_ipc_stop(self._handle)
            self._lib.strata_fast_ipc_destroy(self._handle)
            self._handle = None
            self._cb_ref = None
