# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project.

"""ExpertLocalityPolicy: Prioritizes and clusters requests in continuous batching
that share VRAM-resident experts to maximize cache hit rates and eliminate thrashing.
"""

from typing import Any, List, Set, Tuple


class ExpertLocalityPolicy:
    """Scores candidate requests based on the overlap between their active/predicted
    expert assignments and the currently hot experts in the GPU VRAM arena.
    """

    def __init__(self, resident_experts: Set[int]):
        self.resident_experts = set(resident_experts)

    def update_resident_experts(self, resident_experts: Set[int]):
        """Updates the set of currently hot experts in VRAM."""
        self.resident_experts = set(resident_experts)

    def calculate_affinity(self, request_experts: Set[int]) -> float:
        """Calculates affinity score in [0.0, 1.0].
        A score of 1.0 means all required experts already reside in VRAM.
        """
        if not request_experts:
            return 1.0
        hits = len(request_experts.intersection(self.resident_experts))
        return hits / len(request_experts)

    def rank_candidates(self, candidate_requests: List[Tuple[Any, Set[int]]]) -> List[Tuple[Any, float]]:
        """Ranks requests in descending order of cache affinity.

        Args:
            candidate_requests: List of (request_object, set_of_expert_ids) tuples.
        Returns:
            List of (request_object, affinity_score) sorted by score descending.
        """
        scored = [
            (req, self.calculate_affinity(exp_set))
            for req, exp_set in candidate_requests
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
