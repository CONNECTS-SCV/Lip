"""I/O utilities — CSV, SDF, JSON read/write."""

from __future__ import annotations

import csv
import json
import logging
import math
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
    """Save score trend lines per step as progress.png.

    Args:
        results: List of RoundResult with best, mean, and optional diagnostic scores.
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
    all_means = [getattr(r, "all_avg_score", None) for r in results]
    top10_means = [getattr(r, "top10_avg_score", None) for r in results]

    def _series_values(values: list[Any]) -> list[float | None]:
        return [float(v) if v is not None else None for v in values]

    def _plot_values(values: list[float | None]) -> list[float]:
        return [v if v is not None else float("nan") for v in values]

    all_means = _series_values(all_means)
    top10_means = _series_values(top10_means)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if any(v is not None for v in all_means):
        ax.plot(
            steps,
            _plot_values(all_means),
            "^-",
            color="#94a3b8",
            label="All Mean",
            linewidth=1.4,
            markersize=4,
            alpha=0.75,
            zorder=1,
        )
    ax.plot(
        steps, means, "s-", color="#f97316",
        label="Valid Mean", linewidth=2.2, markersize=5,
        zorder=2,
    )
    if any(v is not None for v in top10_means):
        ax.plot(
            steps,
            _plot_values(top10_means),
            "D--",
            color="#16a34a",
            label="Top 10 Mean",
            linewidth=2.0,
            markersize=5,
            alpha=0.95,
            zorder=4,
        )
    ax.plot(
        steps, bests, "o-",
        color="#2563eb",
        label="Best",
        linewidth=2.3,
        markersize=5,
        zorder=5,
    )

    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_title("Optimization Progress")
    ax.legend(loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1 - 0.3, right=max(steps) + 0.3)

    y_values = [
        float(v)
        for series in (bests, means, all_means, top10_means)
        for v in series
        if v is not None and math.isfinite(float(v))
    ]
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        if y_min >= 0.6:
            span = max(y_max - y_min, 0.08)
            pad = max(span * 0.18, 0.025)
            ax.set_ylim(bottom=max(0.0, y_min - pad), top=min(1.0, y_max + pad))
        else:
            ax.set_ylim(bottom=0.0, top=1.0)
    else:
        ax.set_ylim(bottom=0.0, top=1.0)

    filepath = Path(output_dir) / "progress.png"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Progress plot saved: {filepath}")


def collect_top_molecules(
    results: list[Any],
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Collect top N unique molecules across all rounds, sorted by score descending."""
    if not results:
        return []

    all_mols = []
    for r in results:
        for m in r.molecules:
            mol = {"smiles": m.get("smiles", ""), "score": m.get("score", 0.0), "round": r.round_num}
            for k, v in m.items():
                if k not in ("smiles", "score"):
                    mol[k] = v
            all_mols.append(mol)

    if not all_mols:
        return []

    all_mols.sort(key=lambda m: m.get("score", 0.0), reverse=True)

    seen: set[str] = set()
    unique: list[dict] = []
    for mol in all_mols:
        smi = mol["smiles"]
        if smi and smi not in seen:
            seen.add(smi)
            unique.append(mol)
            if len(unique) >= top_n:
                break

    return unique


def save_top_molecules(
    results: list[Any],
    output_dir: str | Path,
    top_n: int = 20,
) -> None:
    """Save top N molecules across all rounds to top_molecules.csv."""
    top = collect_top_molecules(results, top_n)
    if not top:
        return
    filepath = Path(output_dir) / "top_molecules.csv"
    save_results_csv(top, filepath)
    log.info(f"Top {len(top)} molecules saved: {filepath}")


def _parse_int_field(text: str, default: int = 0) -> int:
    try:
        return int(text.strip())
    except ValueError:
        return default


def _parse_float_field(text: str, default: float) -> float:
    try:
        return float(text.strip())
    except ValueError:
        return default


def _load_protein_only_receptor(
    receptor_pdb_path: str | Path,
) -> tuple[list[str], set[str], int]:
    """Load receptor PDB lines while dropping docked ligands and other HETATM records."""
    receptor_pdb_path = Path(receptor_pdb_path)

    receptor_lines: list[str] = []
    chain_ids: set[str] = set()
    max_serial = 0

    with open(receptor_pdb_path) as f:
        for raw_line in f:
            line = raw_line.rstrip()
            record = line[:6].strip()

            if record in ("END", "ENDMDL"):
                break
            if record in ("MODEL", "HETATM", "ANISOU", "CONECT", "MASTER"):
                continue

            if record == "ATOM":
                chain_id = line[21].strip()
                if chain_id:
                    chain_ids.add(chain_id)
                max_serial = max(max_serial, _parse_int_field(line[6:11]))

            receptor_lines.append(line)

    return receptor_lines, chain_ids, max_serial


def save_protein_only_receptor_pdb(
    receptor_pdb_path: str | Path,
    output_path: str | Path,
) -> None:
    """Save a receptor PDB with all HETATM records removed."""
    receptor_lines, _, _ = _load_protein_only_receptor(receptor_pdb_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        if receptor_lines:
            f.write("\n".join(receptor_lines) + "\n")
        f.write("END\n")


def choose_ligand_chain_id(
    receptor_pdb_path: str | Path,
    preferred: str = "L",
) -> str:
    """Choose a ligand chain ID that does not collide with protein chains."""
    _, chain_ids, _ = _load_protein_only_receptor(receptor_pdb_path)

    candidates = [preferred] + list("LMNOPQRSTUVWXYZABCDEFGHIJK")
    for candidate in candidates:
        if candidate and candidate not in chain_ids:
            return candidate
    return preferred or "L"


def _infer_element_from_pdbqt(line: str) -> str:
    atom_type = ""
    fields = line.split()
    if fields:
        atom_type = fields[-1]

    atom_type_map = {
        "A": "C",
        "BR": "Br",
        "C": "C",
        "CA": "Ca",
        "CL": "Cl",
        "CU": "Cu",
        "F": "F",
        "FE": "Fe",
        "HD": "H",
        "HS": "H",
        "I": "I",
        "K": "K",
        "MG": "Mg",
        "MN": "Mn",
        "N": "N",
        "NA": "N",
        "OA": "O",
        "P": "P",
        "S": "S",
        "SA": "S",
        "ZN": "Zn",
    }

    normalized = atom_type.strip().upper()
    if normalized in atom_type_map:
        return atom_type_map[normalized]

    atom_name = "".join(ch for ch in line[12:16] if ch.isalpha()).upper()
    if atom_name:
        two = atom_name[:2]
        if len(atom_name) >= 2 and two in {"BR", "CL", "FE", "MG", "MN", "ZN"}:
            return two[0] + two[1].lower()
        return atom_name[0]

    return "C"


def _generate_conect_records(
    ligand_pdbqt: str,
    serial_start: int,
) -> list[str]:
    """Generate PDB CONECT records from a PDBQT ligand via obabel."""
    import os
    import subprocess
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdbqt_path = os.path.join(tmpdir, "ligand.pdbqt")
            pdb_path = os.path.join(tmpdir, "ligand.pdb")
            with open(pdbqt_path, "w") as f:
                f.write(ligand_pdbqt)

            result = subprocess.run(
                ["obabel", pdbqt_path, "-O", pdb_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return []

            with open(pdb_path) as f:
                pdb_lines = f.readlines()

            obabel_atom_count = sum(
                1 for l in pdb_lines if l[:6].strip() in ("ATOM", "HETATM")
            )
            pdbqt_atom_count = sum(
                1 for l in ligand_pdbqt.splitlines()
                if l[:6].strip() in ("ATOM", "HETATM")
            )
            if obabel_atom_count != pdbqt_atom_count:
                log.warning(
                    f"Atom count mismatch (obabel={obabel_atom_count}, "
                    f"pdbqt={pdbqt_atom_count}), skipping CONECT records"
                )
                return []

            offset = serial_start - 1
            conect_lines = []
            for line in pdb_lines:
                if not line.startswith("CONECT"):
                    continue
                serials = []
                field = line[6:].rstrip()
                for i in range(0, len(field), 5):
                    chunk = field[i : i + 5].strip()
                    if chunk.isdigit():
                        serials.append(int(chunk) + offset)
                if len(serials) >= 2:
                    conect_lines.append(
                        "CONECT" + "".join(f"{s:5d}" for s in serials)
                    )

            return conect_lines

    except Exception as e:
        log.warning(f"Failed to generate CONECT records: {e}")
        return []


def _pdbqt_atom_to_pdb_line(
    line: str,
    serial: int,
    chain_id: str,
    resseq: int = 1,
    resname: str = "LIG",
) -> str:
    """Convert a PDBQT atom line into a PDB HETATM line with explicit chain ID."""
    atom_name = (line[12:16] if len(line) >= 16 else "").ljust(4)[:4]
    alt_loc = line[16:17] if len(line) >= 17 else " "
    x = _parse_float_field(line[30:38], 0.0)
    y = _parse_float_field(line[38:46], 0.0)
    z = _parse_float_field(line[46:54], 0.0)
    occupancy = _parse_float_field(line[54:60], 1.0)
    temp_factor = _parse_float_field(line[60:66], 0.0)
    element = _infer_element_from_pdbqt(line)

    return (
        f"HETATM{serial:5d} {atom_name}{alt_loc}{resname:>3} {chain_id[:1]}"
        f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{temp_factor:6.2f}          {element:>2}"
    )


def save_complex_pdb(
    receptor_pdb_path: str,
    ligand_pdbqt: str,
    output_path: str | Path,
    ligand_chain_id: str | None = None,
) -> None:
    """Combine receptor PDB and docked ligand PDBQT into a complex PDB file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    receptor_lines, _, max_serial = _load_protein_only_receptor(receptor_pdb_path)
    if ligand_chain_id is None:
        ligand_chain_id = choose_ligand_chain_id(receptor_pdb_path)

    ligand_lines = []
    next_serial = max_serial + 1
    for line in ligand_pdbqt.splitlines():
        record = line[:6].strip()
        if record in ("ATOM", "HETATM"):
            pdb_line = _pdbqt_atom_to_pdb_line(
                line=line,
                serial=next_serial,
                chain_id=ligand_chain_id,
            )
            ligand_lines.append(pdb_line)
            next_serial += 1

    all_lines = []
    chain_marker = "APPLY THE FOLLOWING TO CHAINS:"
    for line in receptor_lines:
        if chain_marker in line and ligand_chain_id not in line.split(chain_marker)[1]:
            line = line.rstrip() + f", {ligand_chain_id}"
        all_lines.append(line)
    if ligand_lines and (not all_lines or all_lines[-1][:6].strip() != "TER"):
        all_lines.append("TER")
    all_lines.extend(ligand_lines)
    if ligand_lines:
        all_lines.append("TER")

    conect_lines = _generate_conect_records(ligand_pdbqt, max_serial + 1)
    all_lines.extend(conect_lines)

    all_lines.append("END")

    with open(output_path, "w") as f:
        f.write("\n".join(all_lines) + "\n")
