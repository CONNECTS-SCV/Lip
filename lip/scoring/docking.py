"""Molecular docking scorer — AutoDock Vina.

Includes:
- BaseDockingScorer: Abstract base with PDBQT preparation and receptor caching
- VinaDockingScorer: CPU-based AutoDock Vina (pre-computes grid maps once)
- ExternalProcess CLI entry points for REINVENT4 integration
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys

# Ensure project root is in sys.path (needed when run as ExternalProcess by REINVENT4)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import tempfile
from abc import ABC, abstractmethod
import multiprocessing
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
# Shared utilities
# ---------------------------------------------------------------------------

def prepare_ligand_pdbqt(smiles: str) -> tuple[str, bool]:
    """Convert SMILES to PDBQT string via RDKit + Meeko.

    Returns:
        (pdbqt_string, success)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", False

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1:
        if AllChem.EmbedMolecule(mol, AllChem.EmbedParameters()) == -1:
            return "", False
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    pdbqt_str, is_ok, _ = PDBQTWriterLegacy.write_string(mol_setups[0])
    return pdbqt_str, is_ok


def prepare_receptor_pdbqt(pdb_path: str, output_path: str):
    """Convert PDB to PDBQT via OpenBabel (Gasteiger charges, no sqm)."""
    result = subprocess.run(
        ["obabel", pdb_path, "-O", output_path, "-xrh",
         "--partialcharge", "gasteiger"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Receptor preparation failed: {result.stderr}")
    log.info(f"Receptor PDBQT prepared: {output_path}")


_RECEPTOR_CACHE_DIR = os.path.join(tempfile.gettempdir(), "dock_receptor_cache")


def get_cached_receptor_pdbqt(pdb_path: str) -> str:
    """Get or create a cached receptor PDBQT file.

    Caches based on file content hash so the obabel conversion
    only runs once per unique receptor, persisting across calls.
    """
    abs_path = os.path.abspath(pdb_path)
    with open(abs_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:12]

    os.makedirs(_RECEPTOR_CACHE_DIR, exist_ok=True)
    cached_pdbqt = os.path.join(_RECEPTOR_CACHE_DIR, f"receptor_{file_hash}.pdbqt")

    if os.path.exists(cached_pdbqt) and os.path.getsize(cached_pdbqt) > 0:
        log.info(f"Using cached receptor PDBQT: {cached_pdbqt}")
        return cached_pdbqt

    prepare_receptor_pdbqt(abs_path, cached_pdbqt)
    return cached_pdbqt


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
        self.pocket_center = pocket_center
        self.box_size = box_size
        self.exhaustiveness = exhaustiveness
        self._rec_pdbqt = get_cached_receptor_pdbqt(receptor_pdb)

    @abstractmethod
    def dock_smiles(self, smiles: str) -> DockingResult:
        """Dock a single SMILES and return result."""

    def dock_batch(self, smiles_list: list[str]) -> list[DockingResult]:
        """Dock a batch of SMILES."""
        return [self.dock_smiles(smi) for smi in smiles_list]


# ---------------------------------------------------------------------------
# Vina
# ---------------------------------------------------------------------------

class VinaDockingScorer(BaseDockingScorer):
    """AutoDock Vina docking scorer.

    Pre-computes Vina grid maps once on init, then reuses them for
    every molecule (maps depend on receptor + pocket, not the ligand).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from vina import Vina
        self._vina = Vina(sf_name="vina", verbosity=0)
        self._vina.set_receptor(rigid_pdbqt_filename=self._rec_pdbqt)
        self._vina.compute_vina_maps(
            center=list(self.pocket_center),
            box_size=list(self.box_size),
        )
        log.info("Vina grid maps pre-computed (reused for all molecules)")

    def dock_smiles(self, smiles: str) -> DockingResult:
        try:
            pdbqt_str, is_ok = prepare_ligand_pdbqt(smiles)
            if not is_ok:
                log.debug(f"Ligand PDBQT preparation failed for {smiles}")
                return DockingResult(smiles=smiles, score=0.0, success=False)

            with tempfile.NamedTemporaryFile(
                suffix=".pdbqt", mode="w", delete=False
            ) as f:
                f.write(pdbqt_str)
                lig_path = f.name

            try:
                self._vina.set_ligand_from_file(lig_path)
                self._vina.dock(exhaustiveness=self.exhaustiveness, n_poses=1)
                energy = self._vina.energies()[0][0]
                pose = self._vina.poses(n_poses=1)
                return DockingResult(
                    smiles=smiles, score=energy, success=True, pose_pdbqt=pose,
                )
            finally:
                os.unlink(lig_path)

        except Exception as e:
            log.warning(f"Vina docking failed for {smiles}: {e}")
            return DockingResult(smiles=smiles, score=0.0, success=False)


# ---------------------------------------------------------------------------
# Multiprocessing worker functions (module-level for pickling)
# ---------------------------------------------------------------------------

_worker_vina = None
_worker_exhaustiveness = None


def _init_worker(rec_pdbqt: str, center: list, box_size: list, exhaustiveness: int):
    """Initialize a Vina instance per worker process (maps computed once)."""
    global _worker_vina, _worker_exhaustiveness
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _devnull = open(os.devnull, "w")
    sys.stdout = _devnull
    from vina import Vina
    _worker_vina = Vina(sf_name="vina", verbosity=0)
    _worker_vina.set_receptor(rigid_pdbqt_filename=rec_pdbqt)
    _worker_vina.compute_vina_maps(center=center, box_size=box_size)
    _worker_exhaustiveness = exhaustiveness


def _dock_one(smiles: str) -> tuple:
    """Dock a single SMILES in a worker process. Returns (smiles, energy, success, pose)."""
    global _worker_vina, _worker_exhaustiveness
    try:
        pdbqt_str, is_ok = prepare_ligand_pdbqt(smiles)
        if not is_ok:
            return (smiles, 0.0, False, "")

        with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as f:
            f.write(pdbqt_str)
            lig_path = f.name

        try:
            _worker_vina.set_ligand_from_file(lig_path)
            _worker_vina.dock(exhaustiveness=_worker_exhaustiveness, n_poses=1)
            energy = _worker_vina.energies()[0][0]
            pose = _worker_vina.poses(n_poses=1)
            return (smiles, energy, True, pose)
        finally:
            os.unlink(lig_path)
    except Exception:
        return (smiles, 0.0, False, "")


# ---------------------------------------------------------------------------
# ExternalProcess CLI — for REINVENT4 integration
# ---------------------------------------------------------------------------

def vina_external_process_main():
    """CLI entry point for Vina as REINVENT4 ExternalProcess."""
    # MUST set before ANY library imports
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    import io
    import argparse

    # Save real stdout, redirect to buffer to prevent stray output
    _real_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Configure stderr logging so failures are visible
    logging.basicConfig(
        level=logging.DEBUG, stream=sys.stderr,
        format="[LIP-DOCK] %(levelname)s %(message)s",
    )

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--receptor", required=True)
        parser.add_argument("--center", required=True, help="x,y,z")
        parser.add_argument("--box-size", default="25,25,25", help="sx,sy,sz")
        parser.add_argument("--exhaustiveness", type=int, default=8)
        parser.add_argument("--analyze-interactions", action="store_true")
        parser.add_argument("--n-workers", type=int, default=0)
        parser.add_argument("--method", default="vina", choices=["vina", "unidock", "auto"])
        args = parser.parse_args()

        center = tuple(float(x) for x in args.center.split(","))
        box_size = tuple(float(x) for x in args.box_size.split(","))

        # Resolve docking method
        method = args.method
        if method == "auto":
            method = "unidock" if _unidock_available() else "vina"
        use_unidock = (method == "unidock")

        log.info(f"Receptor: {args.receptor}, Method: {method}")
        log.info(f"Center: {center}, Box: {box_size}, Exhaustiveness: {args.exhaustiveness}")

        # Read SMILES from stdin (plain text, one per line)
        smiles_list = [line.strip() for line in sys.stdin if line.strip()]

        if not smiles_list:
            sys.stdout = _real_stdout
            payload = {"docking_score": []}
            if args.analyze_interactions:
                payload["interaction_count"] = []
            print(json.dumps({"version": 1, "payload": payload}))
            return

        log.info(f"Received {len(smiles_list)} SMILES for docking")

        # Determine worker count
        n_workers = args.n_workers
        if n_workers <= 0:
            n_workers = min(max(os.cpu_count() // 4, 1), 4)
        n_workers = max(1, min(n_workers, len(smiles_list)))

        # Ensure cached receptor PDBQT exists
        try:
            rec_pdbqt = get_cached_receptor_pdbqt(args.receptor)
            log.info(f"Receptor PDBQT ready: {rec_pdbqt}")
        except Exception as e:
            log.error(f"Receptor PDBQT preparation failed: {e}")
            sys.stdout = _real_stdout
            payload = {"docking_score": [0.0] * len(smiles_list)}
            if args.analyze_interactions:
                payload["interaction_count"] = [0.0] * len(smiles_list)
            print(json.dumps({"version": 1, "payload": payload}))
            return

        # Dock molecules
        results = None

        if use_unidock:
            # GPU batch docking via Uni-Dock
            log.info("GPU docking with Uni-Dock")
            try:
                scorer = UniDockScorer(
                    receptor_pdb=args.receptor,
                    pocket_center=center,
                    box_size=box_size,
                    exhaustiveness=args.exhaustiveness,
                )
                dock_results = scorer.dock_batch(smiles_list)
                results = [
                    (r.smiles, r.score if r.success else 0.0, r.success, r.pose_pdbqt)
                    for r in dock_results
                ]
            except Exception as e:
                log.error(f"Uni-Dock failed: {e}, falling back to Vina")
                results = None

        if results is None and n_workers > 1:
            log.info(f"Parallel Vina docking with {n_workers} workers")
            try:
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(rec_pdbqt, list(center), list(box_size), args.exhaustiveness),
                ) as pool:
                    results = list(pool.map(_dock_one, smiles_list))
            except Exception as e:
                log.error(f"Parallel docking failed: {e}, falling back to sequential")
                results = None

        if results is None:
            log.info("Sequential Vina docking (single worker)")
            scorer = VinaDockingScorer(
                receptor_pdb=args.receptor,
                pocket_center=center,
                box_size=box_size,
                exhaustiveness=args.exhaustiveness,
            )
            results = []
            for smi in smiles_list:
                r = scorer.dock_smiles(smi)
                results.append((smi, r.score if r.success else 0.0, r.success, r.pose_pdbqt))

        # Collect docking scores
        docking_scores = [r[1] for r in results]
        n_success = sum(1 for r in results if r[2])
        log.info(f"Docking done: {n_success}/{len(results)} succeeded")
        if n_success > 0:
            valid_scores = [r[1] for r in results if r[2]]
            log.info(f"Score range: [{min(valid_scores):.2f}, {max(valid_scores):.2f}]")

        # Interaction analysis (sequential, uses docked poses)
        interaction_counts = []
        if args.analyze_interactions:
            protein_mol = None
            try:
                from rdkit import Chem
                protein_mol = Chem.MolFromPDBFile(
                    args.receptor, removeHs=False, sanitize=False,
                )
            except Exception as e:
                log.warning(f"Failed to load protein for interaction analysis: {e}")

            for smi, energy, success, pose in results:
                if success and pose and protein_mol is not None:
                    try:
                        from lip.scoring.interactions import analyze_pose
                        analysis = analyze_pose(
                            protein_pdb=args.receptor,
                            pose_pdbqt=pose,
                            smiles=smi,
                            protein_mol=protein_mol,
                        )
                        interaction_counts.append(float(analysis.total_count))
                    except Exception:
                        interaction_counts.append(0.0)
                else:
                    interaction_counts.append(0.0)

        payload = {"docking_score": docking_scores}
        if args.analyze_interactions:
            payload["interaction_count"] = interaction_counts

        output = {"version": 1, "payload": payload}

        sys.stdout = _real_stdout
        print(json.dumps(output))
    finally:
        sys.stdout = _real_stdout


# ---------------------------------------------------------------------------
# Uni-Dock (GPU)
# ---------------------------------------------------------------------------

def _unidock_available() -> bool:
    """Check if Uni-Dock CLI is available."""
    import shutil
    return shutil.which("unidock") is not None


class UniDockScorer(BaseDockingScorer):
    """GPU-accelerated docking via Uni-Dock.

    Uni-Dock is Vina-compatible (same scoring function, same PDBQT format)
    but runs on NVIDIA GPU. Falls back to VinaDockingScorer if unavailable.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not _unidock_available():
            raise RuntimeError(
                "Uni-Dock not found. Install with: "
                "pip install unidock  (or ensure 'unidock' is on PATH)"
            )

    def dock_smiles(self, smiles: str) -> DockingResult:
        """Dock single SMILES via Uni-Dock CLI."""
        try:
            pdbqt_str, is_ok = prepare_ligand_pdbqt(smiles)
            if not is_ok:
                return DockingResult(smiles=smiles, score=0.0, success=False)

            with tempfile.TemporaryDirectory() as tmpdir:
                lig_path = os.path.join(tmpdir, "ligand.pdbqt")
                out_path = os.path.join(tmpdir, "out.pdbqt")
                with open(lig_path, "w") as f:
                    f.write(pdbqt_str)

                cx, cy, cz = self.pocket_center
                sx, sy, sz = self.box_size
                cmd = [
                    "unidock",
                    "--receptor", self._rec_pdbqt,
                    "--ligand", lig_path,
                    "--center_x", str(cx),
                    "--center_y", str(cy),
                    "--center_z", str(cz),
                    "--size_x", str(sx),
                    "--size_y", str(sy),
                    "--size_z", str(sz),
                    "--exhaustiveness", str(self.exhaustiveness),
                    "--num_modes", "1",
                    "--dir", tmpdir,
                    "--out", out_path,
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    log.warning(f"Uni-Dock failed for {smiles}: {result.stderr}")
                    return DockingResult(smiles=smiles, score=0.0, success=False)

                with open(out_path) as f:
                    pose = f.read()

                energy = self._parse_energy(pose)
                return DockingResult(
                    smiles=smiles, score=energy, success=True, pose_pdbqt=pose,
                )
        except Exception as e:
            log.warning(f"Uni-Dock docking failed for {smiles}: {e}")
            return DockingResult(smiles=smiles, score=0.0, success=False)

    def dock_batch(self, smiles_list: list[str]) -> list[DockingResult]:
        """Batch dock via Uni-Dock --gpu_batch (true GPU parallel docking)."""
        if not smiles_list:
            return []

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Prepare all ligand PDBQTs
                lig_paths = []
                valid_indices = []
                results = [
                    DockingResult(smiles=smi, score=0.0, success=False)
                    for smi in smiles_list
                ]

                for i, smi in enumerate(smiles_list):
                    pdbqt_str, is_ok = prepare_ligand_pdbqt(smi)
                    if not is_ok:
                        continue
                    lig_path = os.path.join(tmpdir, f"lig_{i}.pdbqt")
                    with open(lig_path, "w") as f:
                        f.write(pdbqt_str)
                    lig_paths.append(lig_path)
                    valid_indices.append(i)

                if not lig_paths:
                    return results

                out_dir = os.path.join(tmpdir, "out")
                os.makedirs(out_dir)

                cx, cy, cz = self.pocket_center
                sx, sy, sz = self.box_size
                cmd = [
                    "unidock",
                    "--receptor", self._rec_pdbqt,
                    "--gpu_batch", *lig_paths,
                    "--center_x", str(cx),
                    "--center_y", str(cy),
                    "--center_z", str(cz),
                    "--size_x", str(sx),
                    "--size_y", str(sy),
                    "--size_z", str(sz),
                    "--exhaustiveness", str(self.exhaustiveness),
                    "--num_modes", "1",
                    "--dir", out_dir,
                    "--verbosity", "0",
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=10800,
                )
                if proc.returncode != 0:
                    log.error(f"Uni-Dock gpu_batch failed: {proc.stderr}")
                    return results

                # Parse output files
                for lig_path, idx in zip(lig_paths, valid_indices):
                    stem = Path(lig_path).stem
                    out_file = os.path.join(out_dir, f"{stem}_out.pdbqt")
                    if not os.path.exists(out_file):
                        continue
                    with open(out_file) as f:
                        pose = f.read()
                    energy = self._parse_energy(pose)
                    results[idx] = DockingResult(
                        smiles=smiles_list[idx],
                        score=energy,
                        success=True,
                        pose_pdbqt=pose,
                    )

                n_ok = sum(1 for r in results if r.success)
                log.info(f"Uni-Dock gpu_batch: {n_ok}/{len(smiles_list)} succeeded")
                return results

        except Exception as e:
            log.error(f"Uni-Dock batch docking failed: {e}")
            return [
                DockingResult(smiles=smi, score=0.0, success=False)
                for smi in smiles_list
            ]

    @staticmethod
    def _parse_energy(pdbqt_str: str) -> float:
        """Extract best binding energy from PDBQT REMARK line."""
        for line in pdbqt_str.splitlines():
            if line.startswith("REMARK VINA RESULT"):
                parts = line.split()
                if len(parts) >= 4:
                    return float(parts[3])
        return 0.0


# ---------------------------------------------------------------------------
# Docking scorer registry
# ---------------------------------------------------------------------------

DOCKING_REGISTRY: dict[str, type[BaseDockingScorer]] = {
    "vina": VinaDockingScorer,
    "unidock": UniDockScorer,
}


def create_docking_scorer(method: str, **kwargs) -> BaseDockingScorer:
    """Create a docking scorer by method name."""
    cls = DOCKING_REGISTRY[method]
    return cls(**kwargs)


if __name__ == "__main__":
    vina_external_process_main()
