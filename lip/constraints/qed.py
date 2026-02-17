"""QED (Quantitative Estimate of Drug-likeness) constraint."""

from __future__ import annotations

from typing import Any

from rdkit.Chem import QED as RDKitQED

from lip.constraints.base import BaseConstraint
from lip.constraints.registry import register


@register("qed")
class QEDConstraint(BaseConstraint):
    """QED score - already in [0, 1], used directly.

    Params:
        threshold (float): Pass threshold (default: 0.5).
    """

    def __init__(self, weight: float = 1.0, params: dict | None = None):
        super().__init__(weight, params)
        self.threshold = self.params.get("threshold", 0.5)

    def _score_mol(self, mol: Any, smi: str) -> tuple[float, bool, Any]:
        qed_val = RDKitQED.qed(mol)
        return qed_val, qed_val >= self.threshold, qed_val
