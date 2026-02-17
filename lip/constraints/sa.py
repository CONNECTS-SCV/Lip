"""Synthetic Accessibility (SA) score constraint."""

from __future__ import annotations

import os
from typing import Any

from rdkit.Chem import RDConfig

from lip.constraints.base import BaseConstraint
from lip.constraints.registry import register
from lip.utils.math import sigmoid

# Load SA scorer from RDKit contrib
_sa_scorer = None


def _get_sa_scorer():
    global _sa_scorer
    if _sa_scorer is None:
        import sys
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.insert(0, sa_path)
        import sascorer
        _sa_scorer = sascorer
    return _sa_scorer


@register("sa")
class SAConstraint(BaseConstraint):
    """SA score constraint.

    SA ranges from 1 (easy) to 10 (hard). Sigmoid normalizes to [0, 1].

    Params:
        threshold (float): Sigmoid center (default: 5.0). SA <= threshold -> high score.
    """

    def __init__(self, weight: float = 1.0, params: dict | None = None):
        super().__init__(weight, params)
        self.threshold = self.params.get("threshold", 5.0)

    def _score_mol(self, mol: Any, smi: str) -> tuple[float, bool, Any]:
        scorer = _get_sa_scorer()
        sa_val = scorer.calculateScore(mol)
        normalized = sigmoid(sa_val, self.threshold)
        return normalized, sa_val <= self.threshold, sa_val
