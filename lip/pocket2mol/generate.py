"""Pocket2Mol reference molecule generation (subprocess-based)."""

from __future__ import annotations

import logging
import os
import re
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
    conda_env: str = "Pocket2Mol",
    timeout: int = 600,
    device: str = "cuda",
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

    Raises:
        FileNotFoundError: PDB file or Pocket2Mol directory not found.
        RuntimeError: Pocket2Mol execution failed.
    """
    if not pocket2mol_dir:
        log.error("pocket2mol_dir not specified")
        return []

    if not os.path.isdir(pocket2mol_dir):
        raise FileNotFoundError(
            f"Pocket2Mol directory not found: {pocket2mol_dir}\n"
            f"Run: bash scripts/setup_pocket2mol.sh"
        )

    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    ckpt_path = os.path.join(pocket2mol_dir, "ckpt", "pretrained_Pocket2Mol.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Pretrained model not found: {ckpt_path}\n"
            f"Download from: https://drive.google.com/drive/folders/1KfdOczjUPITPhIvCuBmnj4xFTV-iI2xB"
        )

    base_config_path = os.path.join(pocket2mol_dir, "configs", "sample_for_pdb.yml")

    output_dir = output_dir or tempfile.mkdtemp(prefix="lip_p2m_")
    os.makedirs(output_dir, exist_ok=True)

    abs_pdb = os.path.abspath(pdb_path)
    abs_output = os.path.abspath(output_dir)

    # Patch n_samples/beam_size into config
    with open(base_config_path) as f:
        config_text = f.read()
    config_text = re.sub(r"num_samples:\s*\d+", f"num_samples: {n_samples}", config_text)
    config_text = re.sub(r"beam_size:\s*\d+", f"beam_size: {n_samples * 2}", config_text)
    config_path = os.path.join(abs_output, "sample_config.yml")
    with open(config_path, "w") as f:
        f.write(config_text)

    python_bin = _find_conda_python(conda_env)
    if not python_bin:
        log.error(f"Could not find Python in conda env '{conda_env}'")
        return []

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cmd = [
        python_bin,
        os.path.join(pocket2mol_dir, "sample_for_pdb.py"),
        "--pdb_path", abs_pdb,
        f"--center={','.join(str(c) for c in pocket_center)}",
        "--bbox_size", str(bbox_size),
        "--config", config_path,
        "--device", device,
        "--outdir", abs_output,
    ]

    try:
        log.info(f"Running Pocket2Mol: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=pocket2mol_dir,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Pocket2Mol timed out after {timeout}s. "
            f"Try reducing n_samples or increasing timeout."
        )

    if result.stderr:
        log.error(f"Pocket2Mol stderr: {result.stderr}")

    # Collect SDF files from output (even if exit code != 0)
    sdf_files = _collect_sdf_files(abs_output)

    if not sdf_files:
        # Fallback: Pocket2Mol outputs SMILES.txt — convert to SDF
        smiles_path = os.path.join(abs_output, "SMILES.txt")
        if os.path.isfile(smiles_path):
            sdf_files = _smiles_to_sdfs(smiles_path, abs_output, n_samples)

    if result.returncode != 0:
        if sdf_files:
            log.warning(
                f"Pocket2Mol exited with code {result.returncode} "
                f"but produced {len(sdf_files)} molecules — continuing"
            )
        else:
            raise RuntimeError(
                f"Pocket2Mol failed (exit {result.returncode}) "
                f"with no output: {result.stderr[-500:]}"
            )

    log.info(f"Pocket2Mol generated {len(sdf_files)} molecules")
    return sdf_files


def _collect_sdf_files(output_dir: str) -> list[str]:
    """Collect valid SDF files from output directory."""
    sdf_paths = []
    for root, _dirs, files in os.walk(output_dir):
        for f in sorted(files):
            if f.endswith(".sdf"):
                path = os.path.join(root, f)
                try:
                    supplier = Chem.SDMolSupplier(path, removeHs=False)
                    valid = any(m is not None for m in supplier)
                    if valid:
                        sdf_paths.append(path)
                except Exception:
                    continue
    return sdf_paths


def _smiles_to_sdfs(smiles_path: str, output_dir: str, max_n: int) -> list[str]:
    """Convert SMILES.txt to individual SDF files (fallback)."""
    sdf_paths = []
    with open(smiles_path) as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            smi = line.strip()
            if not smi:
                continue
            sdf_path = os.path.join(output_dir, f"mol_{i:03d}.sdf")
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                conf_mol = generate_conformer(smi)
                if conf_mol is not None:
                    writer = Chem.SDWriter(sdf_path)
                    writer.write(conf_mol)
                    writer.close()
                    sdf_paths.append(sdf_path)
    return sdf_paths


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
