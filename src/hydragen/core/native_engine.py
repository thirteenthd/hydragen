# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/core/native_engine.py - C-ABI Python bindings for NativeHydragenEngine (libhydragen_core.so)."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _find_libhydragen_core() -> Optional[str]:
    search_dirs = [
        Path(__file__).parent.parent / "lib",
        Path(__file__).parent.parent.parent.parent / "lib",
        Path(__file__).parent.parent.parent.parent / "build",
        Path("/usr/local/lib"),
    ]
    lib_names = ["libhydragen_core.so", "libhydragen_core.dll", "libhydragen_core.dylib"]
    for d in search_dirs:
        for name in lib_names:
            p = d / name
            if p.exists():
                return str(p.resolve())
    return None


class NativeHydragenEngineWrapper:
    """Python interface wrapping the high-performance native C++ Hydragen Engine."""

    _lib = None
    _lib_path = None

    @classmethod
    def load_library(cls, custom_path: Optional[str] = None) -> bool:
        if cls._lib is not None:
            return True
        path = custom_path or _find_libhydragen_core()
        if not path or not os.path.exists(path):
            return False
        try:
            lib = ctypes.CDLL(path)
            # Define function signatures
            lib.hydragen_engine_create.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
            lib.hydragen_engine_create.restype = ctypes.c_void_p

            lib.hydragen_engine_submit.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int32,
            ]
            lib.hydragen_engine_submit.restype = ctypes.c_int32

            lib.hydragen_engine_step.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
            ]
            lib.hydragen_engine_step.restype = ctypes.c_int32

            lib.hydragen_engine_cancel.argtypes = [ctypes.c_void_p, ctypes.c_int32]
            lib.hydragen_engine_cancel.restype = ctypes.c_int32

            lib.hydragen_engine_get_telemetry.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
            lib.hydragen_engine_get_telemetry.restype = ctypes.c_int32

            lib.hydragen_engine_is_alive.argtypes = [ctypes.c_void_p]
            lib.hydragen_engine_is_alive.restype = ctypes.c_int32

            lib.hydragen_engine_destroy.argtypes = [ctypes.c_void_p]
            lib.hydragen_engine_destroy.restype = None

            cls._lib = lib
            cls._lib_path = path
            return True
        except Exception as e:
            print(f"[hydragen] Notice: could not load native C++ engine library: {e}")
            return False

    def __init__(
        self,
        batch_slots: int = 4,
        block_size: int = 16,
        max_blocks: int = 4096,
        mock_worker: bool = False,
    ):
        if not self.load_library():
            raise RuntimeError("Native C++ engine library (libhydragen_core.so) not found or failed to load")
        self._handle = self._lib.hydragen_engine_create(
            ctypes.c_int32(batch_slots),
            ctypes.c_int32(block_size),
            ctypes.c_int32(max_blocks),
            ctypes.c_int32(1 if mock_worker else 0),
        )
        if not self._handle:
            raise RuntimeError("Failed to allocate native NativeHydragenEngine instance")

    def submit(
        self,
        req_id: int,
        prompt_ids: List[int],
        max_new: int = 32,
        expert_ids: Optional[List[int]] = None,
    ) -> int:
        c_prompt = (ctypes.c_int32 * len(prompt_ids))(*prompt_ids)
        c_exp = None
        exp_len = 0
        if expert_ids:
            exp_len = len(expert_ids)
            c_exp = (ctypes.c_int32 * exp_len)(*expert_ids)
        return int(
            self._lib.hydragen_engine_submit(
                self._handle,
                ctypes.c_int32(req_id),
                c_prompt,
                ctypes.c_int32(len(prompt_ids)),
                ctypes.c_int32(max_new),
                c_exp,
                ctypes.c_int32(exp_len),
            )
        )

    def step(self) -> Tuple[int, int, int, bool]:
        out_req = ctypes.c_int32(-1)
        out_tok = ctypes.c_int32(-1)
        out_fin = ctypes.c_int32(0)
        ret = int(
            self._lib.hydragen_engine_step(
                self._handle,
                ctypes.byref(out_req),
                ctypes.byref(out_tok),
                ctypes.byref(out_fin),
            )
        )
        return ret, int(out_req.value), int(out_tok.value), bool(out_fin.value)

    def cancel(self, req_id: int) -> bool:
        return bool(self._lib.hydragen_engine_cancel(self._handle, ctypes.c_int32(req_id)))

    def get_telemetry(self) -> Dict[str, Any]:
        buf = ctypes.create_string_buffer(4096)
        sz = self._lib.hydragen_engine_get_telemetry(self._handle, buf, 4096)
        if sz > 0:
            try:
                return json.loads(buf.value.decode("utf-8"))
            except Exception:
                return {}
        return {}

    def is_alive(self) -> bool:
        return bool(self._lib.hydragen_engine_is_alive(self._handle))

    def close(self) -> None:
        if self._handle and self._lib:
            self._lib.hydragen_engine_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()
