"""Shape similarity scoring — ExternalProcess CLI for REINVENT4 integration."""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeHelpers, rdMolAlign

from lip.utils.conformer import generate_conformer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def prepare_reference_mol(smiles: str, n_conformers: int = 10) -> Chem.Mol | None:
    """Generate reference molecule with conformer(s)."""
    return generate_conformer(smiles, n_conformers=n_conformers)


def load_reference_mols_from_sdf(sdf_path: str) -> list[Chem.Mol]:
    """Load reference molecules from SDF file."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    return [mol for mol in suppl if mol is not None]


def score_smiles_shape(smiles: str, ref_mol: Chem.Mol) -> float:
    """Score shape similarity of a SMILES against a single reference.

    Returns value in [0, 1] where 1 = identical shape.
    """
    query = generate_conformer(smiles, n_conformers=1)
    if query is None:
        return 0.0

    try:
        o3a = rdMolAlign.GetCrippenO3A(query, ref_mol)
        if o3a is not None:
            o3a.Align()
        dist = rdShapeHelpers.ShapeTanimotoDist(query, ref_mol)
        return max(0.0, 1.0 - dist)
    except Exception:
        return 0.0


def score_smiles_multi_ref(smiles: str, ref_mols: list[Chem.Mol]) -> float:
    """Score shape similarity against multiple references, return max."""
    if not ref_mols:
        return 0.0

    best = 0.0
    query = generate_conformer(smiles, n_conformers=1)
    if query is None:
        return 0.0

    for ref_mol in ref_mols:
        try:
            o3a = rdMolAlign.GetCrippenO3A(query, ref_mol)
            if o3a is not None:
                o3a.Align()
            dist = rdShapeHelpers.ShapeTanimotoDist(query, ref_mol)
            sim = max(0.0, 1.0 - dist)
            best = max(best, sim)
        except Exception:
            continue

    return best


# ---------------------------------------------------------------------------
# Batch scoring with multiprocessing
# ---------------------------------------------------------------------------

def _score_one(args: tuple) -> float:
    """Worker function for parallel scoring."""
    smiles, ref_smi, ref_sdf, n_ref_conf = args

    if ref_sdf:
        ref_mols = load_reference_mols_from_sdf(ref_sdf)
        return score_smiles_multi_ref(smiles, ref_mols)
    elif ref_smi:
        ref_mol = prepare_reference_mol(ref_smi, n_ref_conf)
        if ref_mol is None:
            return 0.0
        return score_smiles_shape(smiles, ref_mol)
    return 0.0


def score_batch(
    smiles_list: list[str],
    reference_smiles: str = "",
    reference_sdf: str = "",
    n_ref_conformers: int = 10,
    n_workers: int = 1,
) -> list[float]:
    """Score a batch of SMILES for shape similarity."""
    tasks = [
        (smi, reference_smiles, reference_sdf, n_ref_conformers)
        for smi in smiles_list
    ]

    if n_workers <= 1:
        return [_score_one(t) for t in tasks]

    scores = [0.0] * len(smiles_list)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_score_one, t): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                scores[idx] = future.result()
            except Exception:
                scores[idx] = 0.0
    return scores


# ---------------------------------------------------------------------------
# ExternalProcess CLI — for REINVENT4
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for shape similarity as REINVENT4 ExternalProcess."""
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-smiles", default="")
    parser.add_argument("--reference-sdf", default="")
    parser.add_argument("--n-ref-conformers", type=int, default=10)
    parser.add_argument(
        "--n-workers", type=int,
        default=max(1, min(os.cpu_count() // 4, 4)),
    )
    args = parser.parse_args()

    # Pre-load references
    ref_mols = None
    ref_mol = None

    if args.reference_sdf:
        ref_mols = load_reference_mols_from_sdf(args.reference_sdf)
    elif args.reference_smiles:
        ref_mol = prepare_reference_mol(
            args.reference_smiles, args.n_ref_conformers
        )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = json.loads(line)
        smiles_list = request.get("smiles", [])

        results = []
        for smi in smiles_list:
            if ref_mols:
                results.append(score_smiles_multi_ref(smi, ref_mols))
            elif ref_mol:
                results.append(score_smiles_shape(smi, ref_mol))
            else:
                results.append(0.0)

        response = {
            "version": 1,
            "payload": {"shape_similarity": results},
        }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
