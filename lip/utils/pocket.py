"""Ligand extraction from PDB files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_EXCLUDED_RESNAMES = {"HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "ACE", "NH2"}


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

    ligands: dict[tuple[str, str, int], list[tuple[float, float, float]]] = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in _EXCLUDED_RESNAMES:
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


def find_ligand_by_id(
    ligands: list[ExtractedLigand], ligand_id: str
) -> ExtractedLigand | None:
    """Find a ligand by 'chain:resnum' identifier (e.g., 'A:300')."""
    parts = ligand_id.split(":")
    if len(parts) != 2:
        return None
    chain, resnum_str = parts
    try:
        resnum = int(resnum_str)
    except ValueError:
        return None
    return next(
        (lig for lig in ligands if lig.chain == chain and lig.resnum == resnum),
        None,
    )


def find_residue_by_id(
    pdb_path: str | Path, ligand_id: str, min_atoms: int = 1
) -> ExtractedLigand | None:
    """Find any non-water ATOM/HETATM residue by 'chain:resnum'.

    This is a fallback for modified peptide residues such as M3L that may be
    selected as the pocket anchor even when strict HETATM ligand extraction
    fails on a normalized PDB file.
    """
    parts = ligand_id.split(":")
    if len(parts) != 2:
        return None
    target_chain, resnum_str = parts
    try:
        target_resnum = int(resnum_str)
    except ValueError:
        return None

    coords: list[tuple[float, float, float]] = []
    resname = ""
    pdb_path = Path(pdb_path)

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            current_resname = line[17:20].strip()
            if current_resname in _EXCLUDED_RESNAMES:
                continue
            chain = line[21].strip()
            try:
                resnum = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except (ValueError, IndexError):
                continue
            if chain != target_chain or resnum != target_resnum:
                continue
            resname = current_resname
            coords.append((x, y, z))

    if len(coords) < min_atoms:
        return None

    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)
    return ExtractedLigand(
        resname=resname,
        chain=target_chain,
        resnum=target_resnum,
        num_atoms=len(coords),
        center=(cx, cy, cz),
    )


def extract_declared_residues(
    pdb_path: str | Path, min_atoms: int = 1
) -> list[ExtractedLigand]:
    """Extract residues declared by HET/MODRES records using ATOM/HETATM coords.

    Some upload pipelines normalize modified peptide residues from HETATM to
    ATOM while keeping the HET/MODRES metadata. This recovers those residues as
    possible pocket anchors without treating arbitrary protein residues as
    ligands.
    """
    pdb_path = Path(pdb_path)
    candidates: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    with open(pdb_path) as f:
        for line in f:
            record = line[:6]
            if record == "HET   ":
                resname = line[7:10].strip()
                chain = line[12].strip()
                resnum_text = line[13:17].strip()
            elif record == "MODRES":
                resname = line[12:15].strip()
                chain = line[16].strip()
                resnum_text = line[18:22].strip()
            else:
                continue
            if resname in _EXCLUDED_RESNAMES:
                continue
            try:
                resnum = int(resnum_text)
            except ValueError:
                continue
            key = (chain, resnum)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(key)

    residues = []
    for chain, resnum in candidates:
        residue = find_residue_by_id(pdb_path, f"{chain}:{resnum}", min_atoms=min_atoms)
        if residue is not None:
            residues.append(residue)
    return residues
