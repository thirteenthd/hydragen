import os
import traceback
import torch
from hydragen.backends.registry import BackendRegistry
from hydragen.drafters import get_speculative_drafter

hw = BackendRegistry.get_hardware_backend()
mtp_path = "models/mtp/rt"
print(f"Checking mtp_path: {mtp_path}, exists={os.path.exists(mtp_path)}")
try:
    drafter = get_speculative_drafter(
        hardware_backend=hw,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        n_embd=2560,
        hc=4,
        n_head_kv=2,
        head_dim=128,
        max_context=65536,
        mtp_dir=mtp_path if os.path.exists(mtp_path) else None,
    )
    print("Drafter successfully created:", drafter.name, drafter.mode)
except Exception as e:
    traceback.print_exc()
