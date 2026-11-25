"""Service wrapper for HydraGen on Windows.

Applies a GPU core V/F curve offset (undervolt via nvapi64.dll), power limit,
and clock clamp via nvidia-smi before starting ``terminal/serve/server.py``, and
restores stock settings on exit.
"""

from __future__ import annotations

import argparse
import ctypes
import signal
import subprocess
import sys
from pathlib import Path

DEFAULT_POWER_LIMIT_W = 240
DEFAULT_MIN_CLOCK_MHZ = 210
DEFAULT_MAX_CLOCK_MHZ = 1680
DEFAULT_VF_OFFSET_MHZ = 105
STOCK_POWER_LIMIT_W = 350

NVAPI_INITIALIZE_ID = 0x0150E828
NVAPI_ENUM_PHYSICAL_GPUS_ID = 0xE5AC921F
NVAPI_SET_PSTATES20_ID = 0x0F4DAE6B


class _NvValueRange(ctypes.Structure):
    _fields_ = [("value", ctypes.c_int32), ("min", ctypes.c_int32), ("max", ctypes.c_int32)]


class _NvClockEntry(ctypes.Structure):
    _fields_ = [
        ("domainId", ctypes.c_uint32),
        ("typeId", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("freqDelta_kHz", _NvValueRange),
        ("data", ctypes.c_uint32 * 5),
    ]


class _NvBaseVoltageEntry(ctypes.Structure):
    _fields_ = [
        ("domainId", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("volt_uV", ctypes.c_uint32),
        ("voltDelta_uV", _NvValueRange),
    ]


class _NvPState(ctypes.Structure):
    _fields_ = [
        ("pstateId", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("clocks", _NvClockEntry * 8),
        ("baseVoltages", _NvBaseVoltageEntry * 4),
    ]


class _NvPStates20V1(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("numPstates", ctypes.c_uint32),
        ("numClocks", ctypes.c_uint32),
        ("numBaseVoltages", ctypes.c_uint32),
        ("pstates", _NvPState * 16),
    ]


def _set_nvapi_core_offset(offset_mhz: int) -> bool:
    """Apply a P0 graphics core clock V/F offset in MHz via nvapi64.dll."""
    try:
        nvapi = ctypes.WinDLL("nvapi64.dll")
        qi = nvapi.nvapi_QueryInterface
        qi.argtypes = [ctypes.c_uint32]
        qi.restype = ctypes.c_void_p

        init_ptr = qi(NVAPI_INITIALIZE_ID)
        enum_ptr = qi(NVAPI_ENUM_PHYSICAL_GPUS_ID)
        set_ps20_ptr = qi(NVAPI_SET_PSTATES20_ID)
        if not (init_ptr and enum_ptr and set_ps20_ptr):
            print("[gpu-clamp] Required NVAPI entrypoints not found", flush=True)
            return False

        nvapi_init = ctypes.CFUNCTYPE(ctypes.c_int)(init_ptr)
        nvapi_enum = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(ctypes.c_void_p * 64), ctypes.POINTER(ctypes.c_uint32)
        )(enum_ptr)
        nvapi_set_ps20 = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_NvPStates20V1))(
            set_ps20_ptr
        )

        if nvapi_init() != 0:
            print("[gpu-clamp] NvAPI_Initialize failed", flush=True)
            return False

        gpus = (ctypes.c_void_p * 64)()
        count = ctypes.c_uint32()
        if nvapi_enum(ctypes.byref(gpus), ctypes.byref(count)) != 0 or count.value == 0:
            print("[gpu-clamp] NvAPI_EnumPhysicalGPUs found no GPUs", flush=True)
            return False

        info = _NvPStates20V1()
        info.version = (1 << 16) | ctypes.sizeof(_NvPStates20V1)
        info.numPstates = 1
        info.numClocks = 1
        info.pstates[0].pstateId = 0  # P0
        info.pstates[0].clocks[0].domainId = 0  # Graphics / Core clock
        info.pstates[0].clocks[0].freqDelta_kHz.value = int(offset_mhz) * 1000

        res = nvapi_set_ps20(gpus[0], ctypes.byref(info))
        if res == 0:
            print(f"[gpu-clamp] Set core V/F offset to {offset_mhz:+d} MHz via NVAPI", flush=True)
            return True

        print(f"[gpu-clamp] NvAPI_GPU_SetPstates20 returned {res}", flush=True)
        return False
    except Exception as exc:
        print(f"[gpu-clamp] NVAPI V/F offset failed ({exc})", flush=True)
        return False


def _run_nvidia_smi(args: list[str]) -> bool:
    """Run an nvidia-smi command and log failures without aborting service startup."""
    try:
        subprocess.run(["nvidia-smi", *args], check=True)
        return True
    except Exception as exc:
        print(f"[gpu-clamp] nvidia-smi {' '.join(args)} failed: {exc}", flush=True)
        return False


def apply_gpu_clamp(power_limit: int, min_clock: int, max_clock: int, vf_offset: int) -> None:
    """Apply V/F undervolt offset first, then power limit and clock clamp."""
    _set_nvapi_core_offset(vf_offset)
    _run_nvidia_smi(["-pl", str(power_limit)])
    _run_nvidia_smi(["-lgc", f"{min_clock},{max_clock}"])


def reset_gpu_clamp() -> None:
    """Restore stock GPU clocks, V/F offset, and power limit."""
    _run_nvidia_smi(["-rgc"])
    _set_nvapi_core_offset(0)
    _run_nvidia_smi(["-pl", str(STOCK_POWER_LIMIT_W)])


def main() -> int:
    parser = argparse.ArgumentParser(description="HydraGen Windows service wrapper with GPU clamp")
    parser.add_argument("--power-limit", type=int, default=DEFAULT_POWER_LIMIT_W)
    parser.add_argument("--min-clock", type=int, default=DEFAULT_MIN_CLOCK_MHZ)
    parser.add_argument("--max-clock", type=int, default=DEFAULT_MAX_CLOCK_MHZ)
    parser.add_argument("--vf-offset", type=int, default=DEFAULT_VF_OFFSET_MHZ)
    parser.add_argument("--no-reset-on-exit", action="store_true")
    args, server_args = parser.parse_known_args()

    if not server_args:
        server_args = [
            "terminal/serve/server.py",
            "--engine",
            "hydragen",
            "--config",
            "hydragen-local.json",
            "--port",
            "8080",
        ]

    apply_gpu_clamp(args.power_limit, args.min_clock, args.max_clock, args.vf_offset)

    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.Popen([sys.executable, *server_args], cwd=str(repo_root))

    def _forward_signal(signum: int, _frame: object) -> None:
        if proc.poll() is None:
            proc.terminate()

    for sig_name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _forward_signal)

    try:
        return proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)
        if not args.no_reset_on_exit:
            reset_gpu_clamp()


if __name__ == "__main__":
    raise SystemExit(main())
