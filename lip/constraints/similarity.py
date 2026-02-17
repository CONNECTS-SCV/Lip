"""Tanimoto similarity constraint."""

from __future__ import annotations

from typing import Any

from rdkit import DataStructs
from rdkit.Chem import AllChem

from lip.constraints.base import BaseConstraint
from lip.constraints.registry import register
from lip.utils.chem import get_morgan_fp


@register("similarity")
class SimilarityConstraint(BaseConstraint):
    """Tanimoto similarity to a reference molecule.

    Params:
        reference_smiles (str): Reference SMILES (required).
        fp_radius (int): Morgan FP radius (default: 2).
        fp_bits (int): FP bit count (default: 2048).
        mode (str): "maximize" | "minimize" | "range" (default: "maximize").
        threshold (float): Pass threshold for maximize/minimize (default: 0.5).
        min_similarity (float): For range mode, minimum (default: 0.0).
        max_similarity (float): For range mode, maximum (default: 1.0).
    """

    def __init__(self, weight: float = 1.0, params: dict | None = None):
        super().__init__(weight, params)
        self._ref_fp = None
        self.threshold = self.params.get("threshold", 0.5)

    def _get_ref_fp(self):
        if self._ref_fp is None:
            ref_smi = self.params.get("reference_smiles", "")
            radius = self.params.get("fp_radius", 2)
            n_bits = self.params.get("fp_bits", 2048)
            self._ref_fp = get_morgan_fp(ref_smi, radius=radius, n_bits=n_bits)
        return self._ref_fp

    def _score_mol(self, mol: Any, smi: str) -> tuple[float, bool, Any]:
        ref_fp = self._get_ref_fp()
        if ref_fp is None:
            return 0.0, False, None

        mode = self.params.get("mode", "maximize")
        radius = self.params.get("fp_radius", 2)
        n_bits = self.params.get("fp_bits", 2048)
        min_sim = self.params.get("min_similarity", 0.0)
        max_sim = self.params.get("max_similarity", 1.0)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        sim = DataStructs.TanimotoSimilarity(ref_fp, fp)

        if mode == "maximize":
            return sim, sim >= self.threshold, sim
        elif mode == "minimize":
            return 1.0 - sim, sim <= self.threshold, sim
        elif mode == "range":
            if min_sim <= sim <= max_sim:
                return 1.0, True, sim
            dist = min(abs(sim - min_sim), abs(sim - max_sim))
            return max(0.0, 1.0 - dist * 5), False, sim
        else:
            return sim, True, sim

    def validate_params(self) -> list[str]:
        errors = []
        if not self.params.get("reference_smiles"):
            errors.append("reference_smiles is required")
        return errors
