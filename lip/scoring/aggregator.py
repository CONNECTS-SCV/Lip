"""Score aggregation — weighted sum and Pareto ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from lip.constraints.base import ConstraintResult

log = logging.getLogger(__name__)


@dataclass
class AggregatedScore:
    total_score: float
    component_scores: dict[str, float]


class ScoreAggregator:
    """Aggregate multiple constraint/scoring results into a single score."""

    def weighted_sum(
        self,
        results: dict[str, ConstraintResult],
        weights: dict[str, float],
        idx: int,
    ) -> AggregatedScore:
        """Compute weighted sum of scores for molecule at index idx."""
        total = 0.0
        weight_sum = 0.0
        components = {}

        for name, result in results.items():
            w = weights.get(name, 1.0)
            score = result.scores[idx] if idx < len(result.scores) else 0.0
            components[name] = score
            total += w * score
            weight_sum += w

        if weight_sum > 0:
            total /= weight_sum

        return AggregatedScore(total_score=total, component_scores=components)

    def weighted_sum_batch(
        self,
        results: dict[str, ConstraintResult],
        weights: dict[str, float],
        n_molecules: int,
    ) -> list[AggregatedScore]:
        """Compute weighted sum for a batch of molecules."""
        return [
            self.weighted_sum(results, weights, i)
            for i in range(n_molecules)
        ]

    def pareto_rank(
        self,
        results: dict[str, ConstraintResult],
        n_molecules: int,
    ) -> list[float]:
        """Compute Pareto-based scores (normalized rank).

        Non-dominated molecules get higher scores.
        Uses fast non-dominated sort: O(M * N^2) where M = objectives, N = molecules.
        """
        if not results or n_molecules == 0:
            return []

        # Build score matrix: (n_molecules, n_objectives)
        names = list(results.keys())
        matrix = np.zeros((n_molecules, len(names)))

        for j, name in enumerate(names):
            for i in range(n_molecules):
                if i < len(results[name].scores):
                    matrix[i, j] = results[name].scores[i]

        # Fast non-dominated sort (NSGA-II style)
        n = n_molecules
        domination_count = np.zeros(n, dtype=int)
        dominated_sets: list[list[int]] = [[] for _ in range(n)]
        ranks = np.zeros(n, dtype=int)

        # For each pair, check domination using vectorized comparison
        for i in range(n):
            for j in range(i + 1, n):
                i_ge_j = np.all(matrix[i] >= matrix[j])
                i_gt_j = np.any(matrix[i] > matrix[j])
                j_ge_i = np.all(matrix[j] >= matrix[i])
                j_gt_i = np.any(matrix[j] > matrix[i])

                if i_ge_j and i_gt_j:
                    # i dominates j
                    dominated_sets[i].append(j)
                    domination_count[j] += 1
                elif j_ge_i and j_gt_i:
                    # j dominates i
                    dominated_sets[j].append(i)
                    domination_count[i] += 1

        # Build fronts iteratively
        current_front = [i for i in range(n) if domination_count[i] == 0]
        front_rank = 1

        while current_front:
            for i in current_front:
                ranks[i] = front_rank
            next_front = []
            for i in current_front:
                for j in dominated_sets[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_rank += 1

        max_rank = front_rank - 1

        # Normalize: front 1 → highest score
        if max_rank > 0:
            scores = 1.0 - (ranks - 1) / max(max_rank, 1)
        else:
            scores = np.ones(n_molecules)

        return scores.tolist()
