# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/lib_loader.py - Canonical Native Library Loader for Hydragen.

Provides robust cross-platform discovery and loading of compiled C++/CUDA binaries:
- libpaged_kv.so (PTX SIMD PagedAttention + Virtual Block Tables)
- libstrata_fast_ipc.so (Lock-Free Shared Memory Ring Buffer)
- libstrata_trtllm_plugin.so (Custom TensorRT-LLM Plugin)
- libstrata_iq_kernels.so (IQ3_XXS Dequantization Kernels)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import List, Optional


def get_search_paths(lib_name: str, env_var: Optional[str] = None) -> List[Path]:
    """Return prioritized candidate paths for a native shared library."""
    paths: List[Path] = []

    # 1. Environment variable override
    if env_var:
        env_val = os.environ.get(env_var)
        if env_val:
            paths.append(Path(env_val))

    # 2. Inside hydragen package (installed / editable)
    pkg_dir = Path(__file__).resolve().parent
    paths.append(pkg_dir / "lib" / lib_name)
    paths.append(pkg_dir / "server" / lib_name)
    paths.append(pkg_dir / lib_name)

    # 3. Workspace repository root lib/ and build/
    repo_root = pkg_dir.parent.parent
    paths.append(repo_root / "lib" / lib_name)
    paths.append(repo_root / "build" / lib_name)

    # 4. Current working directory
    paths.append(Path.cwd() / "lib" / lib_name)
    paths.append(Path.cwd() / lib_name)

    # 5. Fallback paths for legacy setups
    paths.append(Path("/home/thirteenthd/hydragen/lib") / lib_name)
    paths.append(Path("/home/thirteenthd/hydragen/serve") / lib_name)
    paths.append(Path("/home/thirteenthd") / lib_name)

    return paths


def load_native_library(
    lib_name: str,
    env_var: Optional[str] = None,
    mode: int = ctypes.RTLD_GLOBAL if hasattr(ctypes, "RTLD_GLOBAL") else 0,
) -> Optional[ctypes.CDLL]:
    """Search for and load a compiled native shared library.

    Returns the loaded ctypes.CDLL or None if not found or failed to load.
    """
    for candidate in get_search_paths(lib_name, env_var):
        if candidate.exists() and candidate.is_file():
            try:
                if mode:
                    return ctypes.CDLL(str(candidate), mode=mode)
                return ctypes.CDLL(str(candidate))
            except Exception as e:
                # File exists but failed to dlopen (e.g. missing transitive deps or architecture mismatch)
                continue

    # 6. System loader fallback
    sys_path = ctypes.util.find_library(lib_name.replace(".so", "").replace(".dll", "").replace("lib", ""))
    if sys_path:
        try:
            return ctypes.CDLL(sys_path)
        except Exception:
            pass

    return None


__all__ = ["get_search_paths", "load_native_library"]
