"""3D conformer generation using RDKit ETKDGv3."""

from __future__ import annotations

import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

log = logging.getLogger(__name__)


def generate_conformer(
    smiles: str,
    n_conformers: int = 1,
    optimize: bool = True,
    random_seed: int = 42,
) -> Chem.Mol | None:
    """Generate 3D conformer(s) for a SMILES string.

    Returns an RDKit Mol with embedded conformer(s), or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0  # use all available

    result = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if len(result) == 0:
        params.useRandomCoords = True
        result = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        if len(result) == 0:
            log.debug(f"Conformer embedding failed for {smiles}")
            return None

    if optimize:
        for conf_id in result:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)

    return mol


def smiles_to_sdf(
    smiles: str, output_path: str | Path, n_conformers: int = 1
) -> bool:
    """Generate conformer and write to SDF file."""
    mol = generate_conformer(smiles, n_conformers=n_conformers)
    if mol is None:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = Chem.SDWriter(str(output_path))
    for conf_id in range(mol.GetNumConformers()):
        writer.write(mol, confId=conf_id)
    writer.close()
    return True


def smiles_to_pdb(smiles: str, output_path: str | Path) -> bool:
    """Generate single conformer and write to PDB."""
    mol = generate_conformer(smiles, n_conformers=1)
    if mol is None:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Chem.MolToPDBFile(mol, str(output_path))
    return True


def batch_generate_conformers(
    smiles_list: list[str],
    output_dir: str | Path,
    fmt: str = "sdf",
) -> list[str | None]:
    """Generate conformers for a list of SMILES, save to individual files.

    Returns list of output file paths (None for failures).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, smi in enumerate(smiles_list):
        out_path = output_dir / f"mol_{i:04d}.{fmt}"
        if fmt == "sdf":
            ok = smiles_to_sdf(smi, out_path)
        elif fmt == "pdb":
            ok = smiles_to_pdb(smi, out_path)
        else:
            log.warning(f"Unsupported format: {fmt}")
            ok = False
        results.append(str(out_path) if ok else None)

    return results
