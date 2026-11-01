# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/fast_ipc.py - Python ctypes wrapper for C++ Fast Lock-Free IPC."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional


class FastEngineIPC:
    """High-speed zero-copy C++ IPC bridge for reading engine tokens."""

    _lib: Optional[ctypes.CDLL] = None

    @classmethod
    def _get_lib(cls) -> Optional[ctypes.CDLL]:
        if cls._lib is not None:
            return cls._lib

        search_paths = [
            Path(__file__).resolve().parent.parent.parent / "lib" / "libstrata_fast_ipc.so",
            Path.cwd() / "lib" / "libstrata_fast_ipc.so",
            Path("/home/thirteenthd/hydragen/lib/libstrata_fast_ipc.so"),
        ]

        for p in search_paths:
            if p.exists():
                try:
                    lib = ctypes.CDLL(str(p))
                    lib.strata_fast_ipc_create.argtypes = [ctypes.c_int, ctypes.c_int]
                    lib.strata_fast_ipc_create.restype = ctypes.c_void_p

                    lib.strata_fast_ipc_read_token.argtypes = [
                        ctypes.c_void_p,
                        ctypes.c_int32,
                        ctypes.POINTER(ctypes.c_int32),
                        ctypes.c_int32,
                    ]
                    lib.strata_fast_ipc_read_token.restype = ctypes.c_int32

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
        if self._lib is not None and stdout_fd >= 0:
            self._handle = self._lib.strata_fast_ipc_create(int(stdout_fd), int(num_slots))

    @property
    def is_available(self) -> bool:
        return self._handle is not None

    def read_token(self, slot: int, timeout_ms: int = 500) -> Optional[int]:
        """Read next token for a slot.
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
