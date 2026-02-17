"""Base constraint interface."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConstraintResult:
    """Result from evaluating a constraint on a batch of SMILES."""
    scores: list[float]
    passed: list[bool]
    raw_values: list[Any] = field(default_factory=list)


class BaseConstraint(ABC):
    """Abstract base class for all constraints.

    Subclasses should implement _score_mol() for standard SMILES->mol->score
    constraints. For constraints needing custom preprocessing (e.g. 3D conformers,
    subprocess calls), override score() directly.
    """

    def __init__(self, weight: float = 1.0, params: dict | None = None):
        self.weight = weight
        self.params = params or {}

    def score(self, smiles_list: list[str]) -> ConstraintResult:
        """Template: parse SMILES, call _score_mol for each valid mol.

        Override this entirely for constraints that don't follow the standard
        SMILES -> mol -> score pattern.
        """
        from rdkit import Chem

        scores, passed, raw_values = [], [], []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scores.append(0.0)
                passed.append(False)
                raw_values.append(None)
                continue
            s, p, r = self._score_mol(mol, smi)
            scores.append(s)
            passed.append(p)
            raw_values.append(r)
        return ConstraintResult(scores=scores, passed=passed, raw_values=raw_values)

    def _score_mol(self, mol: Any, smi: str) -> tuple[float, bool, Any]:
        """Score a single valid molecule.

        Returns:
            (score, passed, raw_value) tuple.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _score_mol() or override score()"
        )

    @classmethod
    def get_name(cls) -> str:
        """Return the registered name of this constraint."""
        return getattr(cls, "_registered_name", cls.__name__)

    @classmethod
    def get_ui_schema(cls) -> dict:
        """Return parameter schema for UI/documentation."""
        return {}

    def validate_params(self) -> list[str]:
        """Validate parameters, return list of error messages."""
        return []
