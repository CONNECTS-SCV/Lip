"""Property range constraint - MW, LogP, TPSA, HBD, HBA, etc."""

from __future__ import annotations

import math
from typing import Any

from lip.constraints.base import BaseConstraint, ConstraintResult
from lip.constraints.registry import register
from lip.utils.chem import PROPERTY_FUNCTIONS


@register("property_range")
class PropertyRangeConstraint(BaseConstraint):
    """Score molecules based on whether a property falls within [min, max].

    Params:
        property (str): Property name (e.g. "molecular_weight", "logp").
        min (float): Minimum value (default: -inf).
        max (float): Maximum value (default: +inf).
        margin (float): Fractional margin for soft scoring (default: 0.1).
    """

    def __init__(self, weight: float = 1.0, params: dict | None = None):
        super().__init__(weight, params)
        prop_name = self.params.get("property", "molecular_weight")
        self._func = PROPERTY_FUNCTIONS.get(prop_name)
        self._lo = self.params.get("min", -math.inf)
        self._hi = self.params.get("max", math.inf)
        self._margin = self.params.get("margin", 0.1)

    def score(self, smiles_list: list[str]) -> ConstraintResult:
        if self._func is None:
            return ConstraintResult(
                scores=[0.0] * len(smiles_list),
                passed=[False] * len(smiles_list),
            )
        return super().score(smiles_list)

    def _score_mol(self, mol: Any, smi: str) -> tuple[float, bool, Any]:
        value = float(self._func(mol))
        if self._lo <= value <= self._hi:
            return 1.0, True, value

        if value < self._lo:
            dist = (self._lo - value) / max(abs(self._lo) * self._margin, 1e-6)
        else:
            dist = (value - self._hi) / max(abs(self._hi) * self._margin, 1e-6)
        return max(0.0, 1.0 - dist), False, value

    @classmethod
    def get_ui_schema(cls) -> dict:
        return {
            "property": {"type": "select", "options": list(PROPERTY_FUNCTIONS.keys())},
            "min": {"type": "float", "default": 0},
            "max": {"type": "float", "default": 500},
            "margin": {"type": "float", "default": 0.1},
        }
