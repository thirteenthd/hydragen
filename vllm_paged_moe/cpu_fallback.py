# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project.

"""CPUExpertFallbackExecutor: Offloads cold expert MLP computations to host CPU threads,
preventing PCIe weight transfer thrashing and preserving VRAM cache stability.
"""

from typing import List, Tuple
import torch
import torch.nn.functional as F
import concurrent.futures


class CPUExpertFallbackExecutor:
    """Executes isolated cold MoE expert forward passes on host CPU cores using
    pinned host memory weights, avoiding PCIe weight transfers and VRAM slot evictions.
    """

    def __init__(self, host_weights: torch.Tensor, num_threads: int = 11):
        """Args:
            host_weights: Pinned host memory tensor of shape [total_experts, *expert_shape].
            num_threads: Number of CPU worker threads (defaults to 11 on Ryzen 5900X).
        """
        self.host_weights = host_weights
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    def compute_single_expert(self, expert_id: int, x_cpu: torch.Tensor) -> torch.Tensor:
        """Executes a single expert MLP on host CPU."""
        w = self.host_weights[expert_id]
        h = F.silu(torch.matmul(x_cpu, w))
        out = torch.matmul(h, w.t())
        return out

    def dispatch_fallback(self, cold_requests: List[Tuple[int, torch.Tensor]]) -> List[torch.Tensor]:
        """Dispatches cold activation vectors to CPU and streams results back to GPU.

        Args:
            cold_requests: List of (expert_id, x_gpu_tensor) tuples.
        Returns:
            List of resulting output activation tensors on GPU.
        """
        results = []
        for expert_id, x_gpu in cold_requests:
            x_cpu = x_gpu.cpu()
            out_cpu = self.compute_single_expert(expert_id, x_cpu)
            out_gpu = out_cpu.cuda(non_blocking=True)
            results.append(out_gpu)
        return results
