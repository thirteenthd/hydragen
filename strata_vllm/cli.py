# SPDX-License-Identifier: Apache-2.0
"""CLI Launcher for Strata-vLLM Hybrid Engine."""

from __future__ import annotations

import os
import sys


def main() -> int:
    """Launch the Strata-vLLM server with the Triton + PagedExpertManager backend."""
    strata_dir = os.environ.get("STRATA_DIR", "/home/thirteenthd/Strata")
    vllm_dir = os.environ.get("VLLM_DIR", "/home/thirteenthd/vllm")

    for p in (strata_dir, vllm_dir):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    # Register vLLM quantization plugin
    from strata_vllm.vllm_plugin import register_strata_vllm_plugin

    register_strata_vllm_plugin()

    from serve.server import main as strata_server_main

    argv = list(sys.argv[1:])
    if "--engine" not in argv:
        argv = ["--engine", "vllm"] + argv
    if "--config" not in argv:
        default_cfg = os.path.join(strata_dir, "strata-iq3_xxs.json")
        if os.path.exists(default_cfg):
            argv = ["--config", default_cfg] + argv

    return int(strata_server_main(argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
