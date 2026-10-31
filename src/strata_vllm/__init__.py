# SPDX-License-Identifier: Apache-2.0
"""strata_vllm: Compatibility shim re-exporting from hydragen."""

import sys
import hydragen
from hydragen import *  # noqa: F401, F403
from hydragen import __all__, __version__  # noqa: F401

# Alias sys.modules for any submodule imports
for mod in ["cpu_fallback", "engine", "expert_locality_policy", "expert_manager", "gguf_loader", "triton_moe", "vllm_plugin"]:
    sys.modules[f"strata_vllm.{mod}"] = getattr(__import__(f"hydragen.{mod}", fromlist=[mod]), "")
