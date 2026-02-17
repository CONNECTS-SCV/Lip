"""Molecular docking scorers — AutoDock Vina and GNINA.

Includes:
- BaseDockingScorer: Abstract base with PDBQT preparation and receptor caching
- VinaDockingScorer: CPU-based AutoDock Vina
- GninaDockingScorer: GPU-accelerated CNN docking
- ExternalProcess CLI entry points for REINVENT4 integration
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DockingResult:
    smiles: str
    score: float
    pose_pdbqt: str = ""
    interaction_count: int = 0
    success: bool = True


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseDockingScorer(ABC):
    """Abstract base for docking scorers."""

    def __init__(
        self,
        receptor_pdb: str,
        pocket_center: tuple[float, float, float],
        box_size: tuple[float, float, float] = (25.0, 25.0, 25.0),
        exhaustiveness: int = 8,
    ):
        self.receptor_pdb = receptor_pdb
        self.pocket_center = pocket_center
        self.box_size = box_size
        self.exhaustiveness = exhaustiveness
        self._receptor_pdbqt_cache: dict[str, str] = {}
        self._cache_dir = Path(receptor_pdb).parent / ".cache"

    @abstractmethod
    def dock_smiles(self, smiles: str) -> DockingResult:
        """Dock a single SMILES and return result."""

    def dock_batch(self, smiles_list: list[str]) -> list[DockingResult]:
        """Dock a batch of SMILES."""
        return [self.dock_smiles(smi) for smi in smiles_list]

    def prepare_ligand_pdbqt(self, smiles: str) -> str | None:
        """Convert SMILES → 3D Mol → PDBQT string via RDKit + Meeko."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from meeko import MoleculePreparation, PDBQTWriterLegacy

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol, params) < 0:
                return None
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            if not mol_setups:
                return None

            pdbqt_string, is_ok, _ = PDBQTWriterLegacy.write_string(
                mol_setups[0]
            )
            return pdbqt_string if is_ok else None
        except Exception as e:
            log.debug(f"Ligand preparation failed for {smiles}: {e}")
            return None

    def get_receptor_pdbqt(self) -> str | None:
        """Get receptor PDBQT (cached by MD5 hash of PDB content).

        Cache lookup order:
        1. In-memory dict (fastest, same process)
        2. File-based cache in .cache/ (persistent across runs)
        3. Fresh conversion via obabel (slowest, then cached)
        """
        pdb_path = self.receptor_pdb
        if not Path(pdb_path).exists():
            return None

        content = Path(pdb_path).read_bytes()
        md5 = hashlib.md5(content).hexdigest()

        # 1. In-memory cache
        if md5 in self._receptor_pdbqt_cache:
            return self._receptor_pdbqt_cache[md5]

        # 2. File-based persistent cache
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cached_pdbqt = self._cache_dir / f"receptor_{md5}.pdbqt"
        if cached_pdbqt.exists():
            pdbqt_path = str(cached_pdbqt)
            self._receptor_pdbqt_cache[md5] = pdbqt_path
            log.debug(f"Loaded receptor PDBQT from file cache: {cached_pdbqt}")
            return pdbqt_path

        # 3. Fresh conversion
        pdbqt_path = self._convert_receptor(pdb_path)
        if pdbqt_path:
            try:
                cached_pdbqt.write_bytes(Path(pdbqt_path).read_bytes())
                self._receptor_pdbqt_cache[md5] = str(cached_pdbqt)
                log.debug(f"Cached receptor PDBQT to {cached_pdbqt}")
                return str(cached_pdbqt)
            except OSError:
                self._receptor_pdbqt_cache[md5] = pdbqt_path
                return pdbqt_path

        return None

    def _convert_receptor(self, pdb_path: str) -> str | None:
        """Convert receptor PDB → PDBQT via obabel."""
        try:
            out_f = tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False)
            out_path = out_f.name
            out_f.close()
            subprocess.run(
                ["obabel", pdb_path, "-O", out_path, "-xr"],
                capture_output=True,
                timeout=60,
                check=True,
            )
            return out_path
        except Exception as e:
            log.error(f"Receptor conversion failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Vina
# ---------------------------------------------------------------------------

class VinaDockingScorer(BaseDockingScorer):
    """AutoDock Vina docking scorer."""

    def dock_smiles(self, smiles: str) -> DockingResult:
        ligand_pdbqt = self.prepare_ligand_pdbqt(smiles)
        receptor_pdbqt = self.get_receptor_pdbqt()

        if ligand_pdbqt is None or receptor_pdbqt is None:
            return DockingResult(smiles=smiles, score=0.0, success=False)

        try:
            from vina import Vina

            v = Vina(sf_name="vina", verbosity=0)
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_string(ligand_pdbqt)
            v.compute_vina_maps(
                center=list(self.pocket_center),
                box_size=list(self.box_size),
            )
            v.dock(exhaustiveness=self.exhaustiveness, n_poses=1)

            energy = v.energies(n_poses=1)
            score = float(energy[0][0]) if energy is not None else 0.0
            pose = v.poses(n_poses=1)

            return DockingResult(
                smiles=smiles,
                score=score,
                pose_pdbqt=pose,
            )
        except Exception as e:
            log.debug(f"Vina docking failed for {smiles}: {e}")
            return DockingResult(smiles=smiles, score=0.0, success=False)


# ---------------------------------------------------------------------------
# GNINA
# ---------------------------------------------------------------------------

class GninaDockingScorer(BaseDockingScorer):
    """GNINA GPU-accelerated docking scorer."""

    def __init__(
        self,
        receptor_pdb: str,
        pocket_center: tuple[float, float, float],
        box_size: tuple[float, float, float] = (25.0, 25.0, 25.0),
        exhaustiveness: int = 8,
        cnn_scoring: str = "rescore",
        score_mode: str = "vina",
    ):
        super().__init__(receptor_pdb, pocket_center, box_size, exhaustiveness)
        self.cnn_scoring = cnn_scoring
        self.score_mode = score_mode

    def dock_smiles(self, smiles: str) -> DockingResult:
        ligand_pdbqt = self.prepare_ligand_pdbqt(smiles)
        receptor_pdbqt = self.get_receptor_pdbqt()

        if ligand_pdbqt is None or receptor_pdbqt is None:
            return DockingResult(smiles=smiles, score=0.0, success=False)

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pdbqt", mode="w", delete=False
            ) as lig_f:
                lig_f.write(ligand_pdbqt)
                lig_path = lig_f.name

            out_f = tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False)
            out_path = out_f.name
            out_f.close()

            cx, cy, cz = self.pocket_center
            sx, sy, sz = self.box_size

            cmd = [
                "gnina",
                "-r", receptor_pdbqt,
                "-l", lig_path,
                "-o", out_path,
                "--center_x", str(cx),
                "--center_y", str(cy),
                "--center_z", str(cz),
                "--size_x", str(sx),
                "--size_y", str(sy),
                "--size_z", str(sz),
                "--exhaustiveness", str(self.exhaustiveness),
                "--cnn_scoring", self.cnn_scoring,
                "--num_modes", "1",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )

            score = self._parse_gnina_output(result.stdout)
            pose = Path(out_path).read_text() if Path(out_path).exists() else ""

            return DockingResult(smiles=smiles, score=score, pose_pdbqt=pose)

        except Exception as e:
            log.debug(f"GNINA docking failed for {smiles}: {e}")
            return DockingResult(smiles=smiles, score=0.0, success=False)
        finally:
            for p in [lig_path, out_path]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _parse_gnina_output(self, stdout: str) -> float:
        """Parse GNINA stdout for best docking score."""
        for line in stdout.split("\n"):
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "1":
                try:
                    if self.score_mode == "cnn_affinity":
                        return float(parts[2])  # CNN affinity
                    return float(parts[1])  # Vina score
                except (ValueError, IndexError):
                    pass
        return 0.0


# ---------------------------------------------------------------------------
# ExternalProcess CLI — for REINVENT4 integration
# ---------------------------------------------------------------------------

def vina_external_process_main():
    """CLI entry point for Vina as REINVENT4 ExternalProcess."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--center", required=True, help="x,y,z")
    parser.add_argument("--box-size", default="25,25,25", help="sx,sy,sz")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--analyze-interactions", action="store_true")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    center = tuple(float(x) for x in args.center.split(","))
    box_size = tuple(float(x) for x in args.box_size.split(","))

    scorer = VinaDockingScorer(
        receptor_pdb=args.receptor,
        pocket_center=center,
        box_size=box_size,
        exhaustiveness=args.exhaustiveness,
    )

    # REINVENT4 ExternalProcess protocol: read JSON from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = json.loads(line)
        smiles_list = request.get("smiles", [])

        docking_scores = []
        interaction_counts = []

        for smi in smiles_list:
            result = scorer.dock_smiles(smi)
            docking_scores.append(result.score)
            interaction_counts.append(result.interaction_count)

        response = {
            "version": 1,
            "payload": {
                "docking_score": docking_scores,
            },
        }

        if args.analyze_interactions:
            response["payload"]["interaction_count"] = interaction_counts

        print(json.dumps(response), flush=True)


# ---------------------------------------------------------------------------
# Docking scorer registry
# ---------------------------------------------------------------------------

DOCKING_REGISTRY: dict[str, type[BaseDockingScorer]] = {
    "vina": VinaDockingScorer,
    "gnina": GninaDockingScorer,
}


def create_docking_scorer(method: str, **kwargs) -> BaseDockingScorer:
    """Create a docking scorer by method name.

    Args:
        method: Scorer name ("vina" or "gnina").
        **kwargs: Arguments passed to the scorer constructor.

    Raises:
        KeyError: If method is not registered.
    """
    cls = DOCKING_REGISTRY[method]
    return cls(**kwargs)


if __name__ == "__main__":
    vina_external_process_main()
