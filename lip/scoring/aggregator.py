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

        # Compute Pareto fronts
        ranks = np.zeros(n_molecules)
        remaining = set(range(n_molecules))
        front_rank = 0

        while remaining:
            front_rank += 1
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    # j dominates i if j >= i in all objectives and j > i in at least one
                    if (
                        np.all(matrix[j] >= matrix[i])
                        and np.any(matrix[j] > matrix[i])
                    ):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)

            for i in front:
                ranks[i] = front_rank
                remaining.discard(i)

        # Normalize: front 1 → highest score
        if front_rank > 0:
            scores = 1.0 - (ranks - 1) / max(front_rank, 1)
        else:
            scores = np.ones(n_molecules)

        return scores.tolist()
