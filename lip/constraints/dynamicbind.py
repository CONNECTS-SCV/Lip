"""DynamicBind docking constraint (subprocess-based)."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem

from lip.constraints.base import BaseConstraint, ConstraintResult
from lip.constraints.registry import register
from lip.utils.conformer import smiles_to_sdf

log = logging.getLogger(__name__)


@register("dynamicbind")
class DynamicBindConstraint(BaseConstraint):
    """DynamicBind docking score.

    Runs DynamicBind as a subprocess for each molecule.

    Params:
        protein_pdb (str): Path to protein PDB (required).
        dynamicbind_dir (str): DynamicBind installation directory.
        work_dir (str): Working directory (default: temp).
        best_affinity (float): Normalization best (default: -12.0 kcal/mol).
        worst_affinity (float): Normalization worst (default: 0.0).
        pocket_center (list[float]): [x, y, z] pocket center.
        pocket_radius (float): Pocket radius in Angstroms (default: 15.0).
        timeout (int): Per-molecule timeout in seconds (default: 600).
    """

    def score(self, smiles_list: list[str]) -> ConstraintResult:
        protein_pdb = self.params.get("protein_pdb", "")
        dynamicbind_dir = self.params.get("dynamicbind_dir", "")
        best = self.params.get("best_affinity", -12.0)
        worst = self.params.get("worst_affinity", 0.0)
        timeout = self.params.get("timeout", 600)

        if not protein_pdb or not Path(protein_pdb).exists():
            log.error("protein_pdb not provided or not found")
            return ConstraintResult(
                scores=[0.0] * len(smiles_list),
                passed=[False] * len(smiles_list),
            )

        scores = []
        passed = []
        raw_values = []

        for smi in smiles_list:
            affinity = self._dock_single(
                smi, protein_pdb, dynamicbind_dir, timeout
            )
            raw_values.append(affinity)

            if affinity is None:
                scores.append(0.0)
                passed.append(False)
            else:
                # Normalize: best_affinity → 1.0, worst_affinity → 0.0
                normalized = (worst - affinity) / (worst - best)
                normalized = max(0.0, min(1.0, normalized))
                scores.append(normalized)
                passed.append(affinity <= best * 0.5)

        return ConstraintResult(scores=scores, passed=passed, raw_values=raw_values)

    def _dock_single(
        self,
        smiles: str,
        protein_pdb: str,
        dynamicbind_dir: str,
        timeout: int,
    ) -> float | None:
        """Run DynamicBind for a single molecule."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ligand_sdf = Path(tmpdir) / "ligand.sdf"
            if not smiles_to_sdf(smiles, ligand_sdf):
                return None

            # Build CSV input
            csv_path = Path(tmpdir) / "input.csv"
            csv_path.write_text(
                "protein_path,ligand\n"
                f"{protein_pdb},{ligand_sdf}\n"
            )

            result_dir = Path(tmpdir) / "results"
            cmd = [
                "python", f"{dynamicbind_dir}/run_single_protein_inference.py",
                csv_path, result_dir,
                "--samples_per_complex", "1",
                "--savings_per_complex", "1",
                "--inference_steps", "20",
                "--header", "protein_path,ligand",
            ]

            pocket_center = self.params.get("pocket_center")
            pocket_radius = self.params.get("pocket_radius", 15.0)
            if pocket_center:
                cmd.extend([
                    "--pocket_center_x", str(pocket_center[0]),
                    "--pocket_center_y", str(pocket_center[1]),
                    "--pocket_center_z", str(pocket_center[2]),
                    "--pocket_radius", str(pocket_radius),
                ])

            try:
                subprocess.run(
                    cmd, capture_output=True, timeout=timeout, check=True,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                log.debug(f"DynamicBind failed for {smiles}: {e}")
                return None

            return self._parse_affinity(result_dir)

    def _parse_affinity(self, result_dir: Path) -> float | None:
        """Parse DynamicBind output for affinity score."""
        try:
            import pandas as pd
            csv_files = list(result_dir.rglob("affinity_prediction.csv"))
            if not csv_files:
                return None
            df = pd.read_csv(csv_files[0])
            if "affinity" in df.columns and len(df) > 0:
                return float(df["affinity"].iloc[0])
        except Exception as e:
            log.debug(f"Failed to parse DynamicBind output: {e}")
        return None

    def validate_params(self) -> list[str]:
        errors = []
        if not self.params.get("protein_pdb"):
            errors.append("protein_pdb is required")
        if not self.params.get("dynamicbind_dir"):
            errors.append("dynamicbind_dir is required")
        return errors
