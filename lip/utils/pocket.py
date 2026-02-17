"""Pocket detection (fpocket) and ligand extraction from PDB files."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pocket detection via fpocket
# ---------------------------------------------------------------------------

@dataclass
class Pocket:
    rank: int
    score: float
    center: tuple[float, float, float]
    volume: float
    druggability: float = 0.0
    n_alpha_spheres: int = 0
    polar_sasa: float = 0.0
    apolar_sasa: float = 0.0
    residue_ids: list[int] = field(default_factory=list)
    residue_names: list[str] = field(default_factory=list)


def detect_pockets(
    pdb_path: str | Path,
    fpocket_bin: str = "",
    max_pockets: int = 10,
) -> list[Pocket]:
    """Run fpocket on a PDB file and return detected pockets.

    Args:
        pdb_path: Path to receptor PDB file.
        fpocket_bin: Path to fpocket binary. If empty, search PATH.
        max_pockets: Maximum number of pockets to return.
    """
    fpocket = _find_fpocket(fpocket_bin)
    if fpocket is None:
        log.error("fpocket not found")
        return []

    pdb_path = Path(pdb_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pdb = Path(tmpdir) / pdb_path.name
        shutil.copy2(pdb_path, tmp_pdb)

        try:
            subprocess.run(
                [fpocket, "-f", str(tmp_pdb)],
                cwd=tmpdir,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            log.error(f"fpocket failed: {e}")
            return []

        out_dir = Path(tmpdir) / f"{tmp_pdb.stem}_out"
        if not out_dir.exists():
            return []

        return _parse_fpocket_output(out_dir, max_pockets)


def _find_fpocket(fpocket_bin: str) -> str | None:
    """Find fpocket binary."""
    if fpocket_bin and Path(fpocket_bin).exists():
        return fpocket_bin
    found = shutil.which("fpocket")
    if found:
        return found
    return None


def _parse_fpocket_output(out_dir: Path, max_pockets: int) -> list[Pocket]:
    """Parse fpocket output directory."""
    info_file = out_dir / f"{out_dir.stem.replace('_out', '')}_info.txt"
    if not info_file.exists():
        # Try to find any info file
        info_files = list(out_dir.glob("*_info.txt"))
        if not info_files:
            return []
        info_file = info_files[0]

    pockets_dir = out_dir / "pockets"

    pockets = []
    try:
        text = info_file.read_text()
        blocks = re.split(r"Pocket\s+(\d+)\s*:", text)
        for i in range(1, len(blocks), 2):
            rank = int(blocks[i])
            block = blocks[i + 1]

            score = _extract_float(block, r"Score\s*:\s*([\d.]+)")
            volume = _extract_float(block, r"Volume\s*:\s*([\d.]+)")
            cx = _extract_float(block, r"Center of mass.*?x\s*:\s*([-\d.]+)")
            cy = _extract_float(block, r"Center of mass.*?y\s*:\s*([-\d.]+)")
            cz = _extract_float(block, r"Center of mass.*?z\s*:\s*([-\d.]+)")

            druggability = _extract_float(
                block, r"Druggability Score\s*:\s*([\d.]+)"
            )
            n_alpha_spheres = int(
                _extract_float(block, r"Number of Alpha Spheres\s*:\s*(\d+)")
            )
            polar_sasa = _extract_float(block, r"Polar SASA\s*:\s*([\d.]+)")
            apolar_sasa = _extract_float(block, r"Apolar SASA\s*:\s*([\d.]+)")

            res_ids, res_names = _parse_pocket_residues(pockets_dir, rank)

            pockets.append(Pocket(
                rank=rank,
                score=score,
                center=(cx, cy, cz),
                volume=volume,
                druggability=druggability,
                n_alpha_spheres=n_alpha_spheres,
                polar_sasa=polar_sasa,
                apolar_sasa=apolar_sasa,
                residue_ids=res_ids,
                residue_names=res_names,
            ))

            if len(pockets) >= max_pockets:
                break
    except Exception as e:
        log.error(f"Failed to parse fpocket output: {e}")

    return pockets


def _extract_float(text: str, pattern: str) -> float:
    match = re.search(pattern, text, re.DOTALL)
    return float(match.group(1)) if match else 0.0


def _parse_pocket_residues(
    pockets_dir: Path, rank: int,
) -> tuple[list[int], list[str]]:
    """Extract unique residue IDs and names from pocket*_atm.pdb."""
    atm_path = pockets_dir / f"pocket{rank}_atm.pdb"
    if not atm_path.exists():
        return [], []

    seen: set[tuple[str, int]] = set()
    res_ids: list[int] = []
    res_names: list[str] = []

    with open(atm_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                resname = line[17:20].strip()
                resid = int(line[22:26].strip())
                key = (resname, resid)
                if key not in seen:
                    seen.add(key)
                    res_ids.append(resid)
                    res_names.append(resname)
            except (ValueError, IndexError):
                continue

    return res_ids, res_names


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
