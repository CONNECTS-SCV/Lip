"""3D shape similarity constraint (CrippenO3A alignment)."""

from __future__ import annotations

import logging

from rdkit import Chem
from rdkit.Chem import AllChem

from lip.constraints.base import BaseConstraint, ConstraintResult
from lip.constraints.registry import register
from lip.utils.conformer import generate_conformer

log = logging.getLogger(__name__)


def _compute_shape_similarity(query_mol: Chem.Mol, ref_mol: Chem.Mol) -> float:
    """Compute shape similarity using CrippenO3A alignment.

    Iterates over all conformers in ref_mol to find best alignment.
    Returns value in [0, 1] where 1 = identical shape.
    """
    best_sim = 0.0
    query_conf_id = 0

    for ref_conf_id in range(ref_mol.GetNumConformers()):
        try:
            o3a = AllChem.GetCrippenO3A(
                query_mol, ref_mol,
                prbCid=query_conf_id,
                refCid=ref_conf_id,
            )
            o3a.Align()

            dist = AllChem.ShapeTanimotoDist(
                query_mol, ref_mol,
                confId1=query_conf_id,
                confId2=ref_conf_id,
            )
            sim = 1.0 - dist
            best_sim = max(best_sim, sim)
        except Exception:
            continue

    return best_sim


@register("shape")
class ShapeSimilarityConstraint(BaseConstraint):
    """3D shape similarity to reference molecule(s).

    Overrides score() directly because it needs 3D conformer generation
    instead of simple SMILES->mol parsing.

    Params:
        reference_smiles (str): Single reference SMILES.
        reference_sdf (str): Path to multi-mol SDF (alternative to SMILES).
        mode (str): "maximize" | "minimize" (default: "maximize").
        threshold (float): Pass threshold (default: 0.3).
        n_ref_conformers (int): Conformers for SMILES mode (default: 10).
    """

    def __init__(self, weight: float = 1.0, params: dict | None = None):
        super().__init__(weight, params)
        self._ref_mols: list[Chem.Mol] | None = None
        self.threshold = self.params.get("threshold", 0.3)

    def _get_ref_mols(self) -> list[Chem.Mol]:
        if self._ref_mols is not None:
            return self._ref_mols

        self._ref_mols = []

        sdf_path = self.params.get("reference_sdf", "")
        if sdf_path:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            self._ref_mols = [m for m in suppl if m is not None]
            if self._ref_mols:
                return self._ref_mols

        ref_smi = self.params.get("reference_smiles", "")
        if ref_smi:
            n_conf = self.params.get("n_ref_conformers", 10)
            mol = generate_conformer(ref_smi, n_conformers=n_conf)
            if mol is not None:
                self._ref_mols = [mol]

        return self._ref_mols

    def score(self, smiles_list: list[str]) -> ConstraintResult:
        ref_mols = self._get_ref_mols()
        mode = self.params.get("mode", "maximize")

        scores = []
        passed = []
        raw_values = []

        for smi in smiles_list:
            query_mol = generate_conformer(smi, n_conformers=1)
            if query_mol is None or not ref_mols:
                scores.append(0.0)
                passed.append(False)
                raw_values.append(None)
                continue

            best_sim = max(
                _compute_shape_similarity(query_mol, ref_mol)
                for ref_mol in ref_mols
            )
            raw_values.append(best_sim)

            if mode == "maximize":
                scores.append(best_sim)
                passed.append(best_sim >= self.threshold)
            else:
                scores.append(1.0 - best_sim)
                passed.append(best_sim <= self.threshold)

        return ConstraintResult(scores=scores, passed=passed, raw_values=raw_values)
