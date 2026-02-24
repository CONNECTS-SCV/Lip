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


# ---------------------------------------------------------------------------
# Progress plot
# ---------------------------------------------------------------------------

def save_progress_plot(
    results: list[Any],
    output_dir: str | Path,
) -> None:
    """Save a best/mean score line chart per step as progress.png.

    Args:
        results: List of RoundResult (must have .best_score, .avg_score, .round_num).
        output_dir: Directory to save progress.png.
    """
    if not results:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping progress plot")
        return

    steps = [r.round_num for r in results]
    bests = [r.best_score for r in results]
    means = [r.avg_score for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, bests, "o-", color="#2563eb", label="Best", linewidth=2, markersize=5)
    ax.plot(steps, means, "s-", color="#f97316", label="Mean", linewidth=2, markersize=5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_title("Optimization Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.5)
    ax.set_ylim(bottom=0.0, top=1.0)

    filepath = Path(output_dir) / "progress.png"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Progress plot saved: {filepath}")


def save_top_molecules(
    results: list[Any],
    output_dir: str | Path,
    top_n: int = 20,
) -> None:
    """Save top N molecules across all rounds to top_molecules.csv.

    Args:
        results: List of RoundResult (must have .molecules list of dicts with "score").
        output_dir: Directory to save top_molecules.csv.
        top_n: Number of top molecules to save.
    """
    if not results:
        return

    all_mols = []
    for r in results:
        for m in r.molecules:
            mol = {
                "smiles": m.get("smiles", ""),
                "score": m.get("score", 0.0),
                "round": r.round_num,
            }
            for k, v in m.get("scores", {}).items():
                mol[k] = v
            for k, v in m.get("raw_values", {}).items():
                mol[f"raw_{k}"] = v
            all_mols.append(mol)

    if not all_mols:
        return

    all_mols.sort(key=lambda m: m.get("score", 0.0), reverse=True)
    top = all_mols[:top_n]

    filepath = Path(output_dir) / "top_molecules.csv"
    save_results_csv(top, filepath)
    log.info(f"Top {len(top)} molecules saved: {filepath}")
