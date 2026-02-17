"""Pocket2Mol reference molecule generation (subprocess-based)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.SimDivFilters import rdSimDivPickers

from lip.utils.chem import get_morgan_fp
from lip.utils.conformer import generate_conformer

log = logging.getLogger(__name__)


def run_pocket2mol(
    pdb_path: str,
    pocket_center: list[float],
    bbox_size: float = 23.0,
    n_samples: int = 3,
    output_dir: str | None = None,
    pocket2mol_dir: str = "",
    conda_env: str = "pocket2mol",
    timeout: int = 600,
) -> list[str]:
    """Run Pocket2Mol to generate reference molecules.

    Args:
        pdb_path: Path to receptor PDB.
        pocket_center: [x, y, z] pocket center coordinates.
        bbox_size: Bounding box size in Angstroms.
        n_samples: Number of molecules to generate.
        output_dir: Output directory (default: temp).
        pocket2mol_dir: Pocket2Mol installation directory.
        conda_env: Conda environment name for Pocket2Mol.
        timeout: Max execution time in seconds.

    Returns:
        List of SDF file paths for generated molecules.
    """
    if not pocket2mol_dir:
        log.error("pocket2mol_dir not specified")
        return []

    output_dir = output_dir or tempfile.mkdtemp(prefix="lip_p2m_")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    python_bin = _find_conda_python(conda_env)
    if not python_bin:
        log.error(f"Could not find Python in conda env '{conda_env}'")
        return []

    # Set env to avoid threading issues
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cx, cy, cz = pocket_center
    beam_size = n_samples * 2

    cmd = [
        python_bin,
        os.path.join(pocket2mol_dir, "sample.py"),
        "--pdb_path", pdb_path,
        "--center", str(cx), str(cy), str(cz),
        "--bbox_size", str(bbox_size),
        "--n_samples", str(n_samples),
        "--beam_size", str(beam_size),
        "--result_path", output_dir,
    ]

    try:
        log.info(f"Running Pocket2Mol: {' '.join(cmd[:5])}...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
        if result.returncode != 0:
            log.error(f"Pocket2Mol failed: {result.stderr[:500]}")
            return []
    except subprocess.TimeoutExpired:
        log.error(f"Pocket2Mol timed out after {timeout}s")
        return []

    # Collect output SDFs
    sdf_files = sorted(Path(output_dir).glob("*.sdf"))
    return [str(f) for f in sdf_files]


def _find_conda_python(env_name: str) -> str | None:
    """Find Python binary in a conda environment."""
    home = Path.home()
    candidates = [
        home / "miniforge3" / "envs" / env_name,
        home / "miniconda3" / "envs" / env_name,
        home / "anaconda3" / "envs" / env_name,
    ]

    for base in candidates:
        # Linux/Mac
        py = base / "bin" / "python"
        if py.exists():
            return str(py)
        # Windows
        py = base / "python.exe"
        if py.exists():
            return str(py)

    return None


def select_diverse_representatives(
    sdf_paths: list[str], n_picks: int = 3
) -> list[str]:
    """Select diverse representatives from SDF files using MaxMin picking.

    Args:
        sdf_paths: List of SDF file paths.
        n_picks: Number of representatives to select.

    Returns:
        List of selected SDF file paths.
    """
    fps = []
    valid_paths = []

    for path in sdf_paths:
        suppl = Chem.SDMolSupplier(path, removeHs=True)
        for mol in suppl:
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
            fp = get_morgan_fp(smi)
            if fp is not None:
                fps.append(fp)
                valid_paths.append(path)
            break  # One mol per SDF

    if len(fps) <= n_picks:
        return valid_paths

    # Distance function for MaxMin
    def dist_func(i, j):
        return 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])

    picker = rdSimDivPickers.MaxMinPicker()
    picks = picker.LazyPick(dist_func, len(fps), n_picks)

    return [valid_paths[i] for i in picks]


def combine_sdfs(sdf_paths: list[str], output_path: str) -> str:
    """Combine multiple SDF files into a single multi-molecule SDF.

    Returns output path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(output_path)

    for path in sdf_paths:
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        for mol in suppl:
            if mol is not None:
                writer.write(mol)

    writer.close()
    return output_path
