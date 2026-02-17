"""I/O utilities — CSV, SDF, JSON read/write."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def load_smiles_from_csv(
    filepath: str | Path, column: str = "smiles"
) -> list[str]:
    """Load SMILES strings from a CSV file."""
    filepath = Path(filepath)
    smiles = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if column in row and row[column].strip():
                smiles.append(row[column].strip())
    return smiles


def save_results_csv(
    results: list[dict[str, Any]], filepath: str | Path
) -> None:
    """Save a list of result dicts to CSV (overwrite)."""
    _write_csv(results, filepath, mode="w", write_header=True)


def append_results_csv(
    results: list[dict[str, Any]], filepath: str | Path
) -> None:
    """Append results to an existing CSV (or create if new)."""
    filepath = Path(filepath)
    exists = filepath.exists()
    _write_csv(results, filepath, mode="a", write_header=not exists)


def _write_csv(
    results: list[dict[str, Any]],
    filepath: str | Path,
    mode: str = "w",
    write_header: bool = True,
) -> None:
    """Internal: write result dicts to CSV with given mode."""
    if not results:
        return
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    keys = list(results[0].keys())
    with open(filepath, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# SDF
# ---------------------------------------------------------------------------

def save_results_sdf(
    smiles_list: list[str], filepath: str | Path
) -> None:
    """Save SMILES as 2D molecules to SDF."""
    from rdkit import Chem

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    writer = Chem.SDWriter(str(filepath))
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol.SetProp("SMILES", smi)
            writer.write(mol)
    writer.close()


def load_mols_from_sdf(filepath: str | Path) -> list:
    """Load RDKit Mol objects from an SDF file."""
    from rdkit import Chem

    suppl = Chem.SDMolSupplier(str(filepath), removeHs=False)
    return [mol for mol in suppl if mol is not None]


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_json(data: Any, filepath: str | Path) -> None:
    """Save data to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str | Path) -> Any:
    """Load data from JSON file."""
    with open(filepath) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Round results
# ---------------------------------------------------------------------------

def save_round_results(
    round_num: int,
    molecules: list[dict[str, Any]],
    output_dir: str | Path,
) -> str:
    """Save round results to CSV, return file path."""
    output_dir = Path(output_dir) / "molecules"
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"round_{round_num:03d}.csv"
    save_results_csv(molecules, filepath)
    return str(filepath)
