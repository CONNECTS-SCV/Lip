"""SMARTS pattern matching constraint."""

from __future__ import annotations

from typing import Any

from rdkit import Chem

from lip.constraints.base import BaseConstraint
from lip.constraints.registry import register


@register("smarts")
class SMARTSConstraint(BaseConstraint):
    """Match (or exclude) a SMARTS pattern.

    Params:
        pattern (str): SMARTS pattern (required).
        must_match (bool): True = molecule must contain pattern,
                           False = molecule must NOT contain pattern (default: True).
    """

    def __init__(self, weight: float = 1.0, params: dict | None = None):
        super().__init__(weight, params)
        self._query = None

    def _get_query(self) -> Chem.Mol | None:
        if self._query is None:
            pattern = self.params.get("pattern", "")
            if pattern:
                self._query = Chem.MolFromSmarts(pattern)
        return self._query

    def _score_mol(self, mol: Any, smi: str) -> tuple[float, bool, Any]:
        query = self._get_query()
        if query is None:
            return 0.0, False, None

        must_match = self.params.get("must_match", True)
        has_match = mol.HasSubstructMatch(query)

        if must_match:
            return (1.0 if has_match else 0.0), has_match, has_match
        else:
            return (0.0 if has_match else 1.0), not has_match, has_match

    def validate_params(self) -> list[str]:
        errors = []
        pattern = self.params.get("pattern", "")
        if not pattern:
            errors.append("pattern is required")
        elif Chem.MolFromSmarts(pattern) is None:
            errors.append(f"Invalid SMARTS pattern: {pattern}")
        return errors
