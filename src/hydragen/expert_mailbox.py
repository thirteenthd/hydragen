import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class HybridExpertMailbox:
    """Concurrent CPU+GPU Hybrid Expert Mailbox (RustingStrata-inspired LPT-MoE).

    When tokens route to non-resident experts (VRAM cache misses), instead of
    blocking the GPU compute stream while serializing PCIe DMA transfers and evicting
    cache victims, the Hybrid Expert Mailbox dispatches missed expert slices to an
    asynchronous host CPU worker pool.

    The GPU immediately executes the resident expert slices via Triton kernels,
    while the CPU worker pool computes the missing expert projections in host memory
    concurrently. The results are gathered and fused with exact numerical parity.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        if max_workers is None:
            # Use up to 8 threads or available host CPU cores
            cpu_cnt = os.cpu_count() or 4
            max_workers = min(8, max(2, cpu_cnt // 2))
        self.max_workers = int(max_workers)
        self.device = device
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="hydragen_expert_mailbox",
        )

        # Telemetry metrics
        self.total_dispatches: int = 0
        self.total_gpu_experts: int = 0
        self.total_cpu_experts: int = 0
        self.total_fusions: int = 0
        self.last_cpu_time_ms: float = 0.0

    def compute_cpu_expert(
        self,
        x_cpu: torch.Tensor,
        host_expert_weights: torch.Tensor,
        has_up: bool,
        N_ff: int,
        K: int,
    ) -> torch.Tensor:
        """Compute gate/up/down MLP projection for an expert directly in host memory.

        Args:
            x_cpu: Input tokens [M, K] in float16/float32 on CPU.
            host_expert_weights: Pinned host weights [3, ...] or [2, ...] for this expert.
            has_up: Whether SwiGLU up-projection is present.
            N_ff: Intermediate dimension.
            K: Hidden dimension.

        Returns:
            down_out: Expert projection [M, K] on CPU.
        """
        # Ensure float32 for high CPU precision and stability
        x = x_cpu.float()
        w_gate = host_expert_weights[0].float()
        if w_gate.shape[0] == N_ff and w_gate.shape[1] == K:
            gate_out = torch.matmul(x, w_gate.t())
        elif w_gate.shape[0] == K and w_gate.shape[1] == N_ff:
            gate_out = torch.matmul(x, w_gate)
        else:
            gate_out = torch.matmul(x, w_gate.reshape(-1, K).t())

        if has_up:
            w_up = host_expert_weights[1].float()
            if w_up.shape[0] == N_ff and w_up.shape[1] == K:
                up_out = torch.matmul(x, w_up.t())
            elif w_up.shape[0] == K and w_up.shape[1] == N_ff:
                up_out = torch.matmul(x, w_up)
            else:
                up_out = torch.matmul(x, w_up.reshape(-1, K).t())
            inter = F.silu(gate_out) * up_out
            down_idx = 2
        else:
            inter = F.silu(gate_out)
            down_idx = 1

        w_down = host_expert_weights[down_idx].float()
        if w_down.shape[0] == K and w_down.shape[1] == N_ff:
            down_out = torch.matmul(inter, w_down.t())
        elif w_down.shape[0] == N_ff and w_down.shape[1] == K:
            down_out = torch.matmul(inter, w_down)
        else:
            down_out = torch.matmul(inter, w_down.reshape(N_ff, K))

        return down_out.to(x_cpu.dtype)

    def dispatch_missed_experts(
        self,
        x_cpu: torch.Tensor,
        missed_items: List[Tuple[int, int]],
        host_pool: torch.Tensor,
        has_up: bool,
        N_ff: int,
        K: int,
    ) -> List[Tuple[int, Future]]:
        """Dispatch missed expert evaluations to the CPU worker thread pool.

        Args:
            x_cpu: Input tokens [M, K] cloned to CPU.
            missed_items: List of (compact_idx, expert_id).
            host_pool: Host tensor holding full weights [total_experts, *shape].
            has_up: Whether SwiGLU up-projection is present.
            N_ff: Intermediate dimension.
            K: Hidden dimension.

        Returns:
            List of (compact_idx, Future[torch.Tensor]).
        """
        futures: List[Tuple[int, Future]] = []
        for compact_idx, eid in missed_items:
            expert_weights = host_pool[eid]
            fut = self.executor.submit(
                self.compute_cpu_expert,
                x_cpu,
                expert_weights,
                has_up,
                N_ff,
                K,
            )
            futures.append((compact_idx, fut))
        return futures

    def gather_and_merge(
        self,
        down_out_gpu: torch.Tensor,
        cpu_futures: List[Tuple[int, Future]],
        cuda_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Collect completed CPU expert projections and copy them into the GPU output tensor."""
        t0 = time.perf_counter()
        for compact_idx, fut in cpu_futures:
            res_cpu = fut.result()
            if down_out_gpu.device.type == "cuda":
                # Asynchronously copy to GPU slot in down_out
                if cuda_stream is not None:
                    with torch.cuda.stream(cuda_stream):
                        down_out_gpu[compact_idx].copy_(res_cpu, non_blocking=True)
                else:
                    down_out_gpu[compact_idx].copy_(res_cpu, non_blocking=True)
            else:
                down_out_gpu[compact_idx].copy_(res_cpu)

        if down_out_gpu.device.type == "cuda" and cuda_stream is not None:
            cuda_stream.synchronize()

        self.last_cpu_time_ms = (time.perf_counter() - t0) * 1000.0
        self.total_fusions += 1

    def close(self):
        """Shut down the background thread pool."""
        self.executor.shutdown(wait=False)
