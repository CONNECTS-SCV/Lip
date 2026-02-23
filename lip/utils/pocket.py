"""Ligand extraction from PDB files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ligand extraction from PDB
# ---------------------------------------------------------------------------

@dataclass
class ExtractedLigand:
    resname: str
    chain: str
    resnum: int
    num_atoms: int
    center: tuple[float, float, float]


def extract_ligands(
    pdb_path: str | Path, min_atoms: int = 5
) -> list[ExtractedLigand]:
    """Extract non-water HETATM ligands from a PDB file.

    Args:
        pdb_path: Path to PDB file.
        min_atoms: Minimum heavy atoms to count as a ligand.
    """
    pdb_path = Path(pdb_path)
    excluded = {"HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "ACE", "NH2"}

    ligands: dict[tuple[str, str, int], list[tuple[float, float, float]]] = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in excluded:
                continue
            chain = line[21].strip()
            try:
                resnum = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except (ValueError, IndexError):
                continue

            key = (resname, chain, resnum)
            ligands.setdefault(key, []).append((x, y, z))

    results = []
    for (resname, chain, resnum), coords in ligands.items():
        if len(coords) < min_atoms:
            continue
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)
        results.append(ExtractedLigand(
            resname=resname,
            chain=chain,
            resnum=resnum,
            num_atoms=len(coords),
            center=(cx, cy, cz),
        ))

    results.sort(key=lambda lig: lig.num_atoms, reverse=True)
    return results
