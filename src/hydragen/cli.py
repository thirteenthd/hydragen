# SPDX-License-Identifier: Apache-2.0
"""CLI Launcher for Strata-vLLM Hybrid Engine."""

from __future__ import annotations

import os
import sys


def main() -> int:
    """Launch the Hydragen server with the Triton + PagedExpertManager backend."""
    cur_dir = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    src_parent = os.path.dirname(os.path.dirname(file_dir))
    server_dir = os.path.join(file_dir, "server")
    strata_dir = os.environ.get("STRATA_DIR")
    vllm_dir = os.environ.get("VLLM_DIR")

    for p in (cur_dir, src_parent, server_dir, strata_dir, vllm_dir):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    if "serve" not in sys.modules:
        try:
            import hydragen.server as server_pkg
            sys.modules["serve"] = server_pkg
        except Exception:
            pass

    # Register vLLM quantization plugin
    from hydragen.vllm_plugin import register_strata_vllm_plugin

    register_strata_vllm_plugin()

    try:
        from hydragen.server.server import main as server_main
    except ImportError:
        from serve.server import main as server_main

    argv = list(sys.argv[1:])
    if "--engine" not in argv:
        default_engine = os.environ.get("HYDRAGEN_ENGINE", "strata")
        argv = ["--engine", default_engine] + argv
    if "--config" not in argv:
        for candidate in ["hydragen.json", "strata-iq3_xxs.json", "config.json"]:
            if os.path.exists(candidate):
                argv = ["--config", candidate] + argv
                break

    return int(server_main(argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
