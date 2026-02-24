"""Protein-ligand interaction analysis — geometry-based detection."""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interaction types and thresholds
# ---------------------------------------------------------------------------

@dataclass
class InteractionType:
    name: str
    distance: float  # Angstroms
    description: str


INTERACTION_TYPES = [
    InteractionType("HBond", 3.5, "Hydrogen bond (N/O donor-acceptor)"),
    InteractionType("Hydrophobic", 4.5, "Hydrophobic contact (C-C)"),
    InteractionType("PiStacking", 5.5, "Aromatic ring stacking"),
    InteractionType("SaltBridge", 4.0, "Opposite charge interaction"),
    InteractionType("HalogenBond", 3.5, "C-X...N/O halogen bond"),
    InteractionType("CationPi", 6.0, "Cation-aromatic ring"),
]


@dataclass
class Interaction:
    type: str
    protein_atom: str
    ligand_atom: str
    distance: float
    residue: str = ""


@dataclass
class InteractionReport:
    total_count: int
    interactions: list[Interaction]
    by_type: dict[str, int]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_pose(
    protein_pdb: str,
    pose_pdbqt: str,
    smiles: str = "",
    protein_mol: Chem.Mol | None = None,
    interaction_types: list[InteractionType] | None = None,
) -> InteractionReport:
    """Analyze interactions between protein and docked ligand pose.

    Args:
        protein_pdb: Path to protein PDB file.
        pose_pdbqt: PDBQT string of docked pose.
        smiles: Original SMILES (for logging).
        protein_mol: Pre-loaded RDKit Mol (for reuse).
        interaction_types: Custom interaction types (default: INTERACTION_TYPES).
    """
    if protein_mol is None:
        protein_mol = Chem.MolFromPDBFile(protein_pdb, removeHs=False)

    ligand_mol = _pdbqt_to_mol(pose_pdbqt)

    if protein_mol is None or ligand_mol is None:
        return InteractionReport(total_count=0, interactions=[], by_type={})

    types = interaction_types if interaction_types is not None else INTERACTION_TYPES
    interactions = _detect_interactions(protein_mol, ligand_mol, types)
    by_type = {}
    for inter in interactions:
        by_type[inter.type] = by_type.get(inter.type, 0) + 1

    return InteractionReport(
        total_count=len(interactions),
        interactions=interactions,
        by_type=by_type,
    )


def _pdbqt_to_mol(pdbqt_string: str) -> Chem.Mol | None:
    """Convert PDBQT string to RDKit Mol via obabel."""
    import os

    pdbqt_path = None
    pdb_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdbqt", mode="w", delete=False
        ) as f:
            f.write(pdbqt_string)
            pdbqt_path = f.name

        pdb_f = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        pdb_path = pdb_f.name
        pdb_f.close()

        subprocess.run(
            ["obabel", pdbqt_path, "-O", pdb_path],
            capture_output=True, timeout=30,
        )

        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        return mol
    except Exception as e:
        log.debug(f"PDBQT conversion failed: {e}")
        return None
    finally:
        for p in [pdbqt_path, pdb_path]:
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Ring-based interaction helpers
# ---------------------------------------------------------------------------

# Protein charged residue atoms (CHEM-identical)
_POS_RESIDUES: dict[str, set[str]] = {
    "ARG": {"NH1", "NH2", "NE"},
    "LYS": {"NZ"},
    "HIS": {"ND1", "NE2"},
}
_NEG_RESIDUES: dict[str, set[str]] = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}


def _get_residue_label(atom) -> str:
    """Get residue label like 'MET793' from a protein atom."""
    info = atom.GetPDBResidueInfo()
    if info:
        name = info.GetResidueName().strip()
        num = info.GetResidueNumber()
        return f"{name}{num}"
    return "UNK"


def _get_ring_centroid(
    conf: Chem.Conformer, ring_atoms: list[int],
) -> tuple[float, float, float]:
    """Compute centroid of a ring from atom positions."""
    xs, ys, zs = [], [], []
    for idx in ring_atoms:
        p = conf.GetAtomPosition(idx)
        xs.append(p.x)
        ys.append(p.y)
        zs.append(p.z)
    n = len(ring_atoms)
    return (sum(xs) / n, sum(ys) / n, sum(zs) / n)


def _centroid_dist(
    c1: tuple[float, float, float], c2: tuple[float, float, float],
) -> float:
    """Euclidean distance between two centroids."""
    return math.sqrt(
        (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2
    )


def _get_aromatic_rings(mol: Chem.Mol) -> list[list[int]]:
    """Get aromatic ring atom index lists from a molecule."""
    return [
        list(r) for r in mol.GetRingInfo().AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
    ]


# ---------------------------------------------------------------------------
# Interaction detection
# ---------------------------------------------------------------------------

def _dist(conf1, idx1, conf2, idx2) -> float:
    """Euclidean distance between two atoms."""
    p1 = conf1.GetAtomPosition(idx1)
    p2 = conf2.GetAtomPosition(idx2)
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


def _detect_interactions(
    protein: Chem.Mol,
    ligand: Chem.Mol,
    types: list[InteractionType] | None = None,
) -> list[Interaction]:
    """Detect interactions with CHEM-identical residue-level deduplication.

    Same residue is counted only once per interaction type.
    """
    if types is None:
        types = INTERACTION_TYPES

    type_map = {t.name: t.distance for t in types}
    interactions: list[Interaction] = []
    prot_conf = protein.GetConformer()
    lig_conf = ligand.GetConformer()

    # --- H-bonds: N/O ... N/O < 3.5A ---
    hbond_d = type_map.get("HBond")
    if hbond_d:
        seen: set[str] = set()
        lig_polar = [(a, a.GetIdx()) for a in ligand.GetAtoms() if a.GetAtomicNum() in (7, 8)]
        prot_polar = [(a, a.GetIdx()) for a in protein.GetAtoms() if a.GetAtomicNum() in (7, 8)]
        for la, li in lig_polar:
            for pa, pi in prot_polar:
                d = _dist(lig_conf, li, prot_conf, pi)
                if d < hbond_d:
                    res = _get_residue_label(pa)
                    if res not in seen:
                        seen.add(res)
                        interactions.append(Interaction("HBond", res, f"{la.GetSymbol()}{li}", d, res))

    # --- Hydrophobic: C ... C < 4.5A ---
    hydro_d = type_map.get("Hydrophobic")
    if hydro_d:
        seen = set()
        lig_carbons = [a for a in ligand.GetAtoms() if a.GetAtomicNum() == 6]
        prot_carbons = [a for a in protein.GetAtoms() if a.GetAtomicNum() == 6]
        for la in lig_carbons:
            for pa in prot_carbons:
                d = _dist(lig_conf, la.GetIdx(), prot_conf, pa.GetIdx())
                if d < hydro_d:
                    res = _get_residue_label(pa)
                    if res not in seen:
                        seen.add(res)
                        interactions.append(Interaction("Hydrophobic", res, f"C{la.GetIdx()}", d, res))

    # --- Salt bridges: opposite charges < 4.0A ---
    salt_d = type_map.get("SaltBridge")
    if salt_d:
        seen = set()
        lig_pos = [a for a in ligand.GetAtoms() if a.GetFormalCharge() > 0]
        lig_neg = [a for a in ligand.GetAtoms() if a.GetFormalCharge() < 0]
        prot_pos_atoms, prot_neg_atoms = [], []
        for a in protein.GetAtoms():
            info = a.GetPDBResidueInfo()
            if not info:
                continue
            resname = info.GetResidueName().strip()
            atomname = info.GetName().strip()
            if resname in _POS_RESIDUES and atomname in _POS_RESIDUES[resname]:
                prot_pos_atoms.append(a)
            elif resname in _NEG_RESIDUES and atomname in _NEG_RESIDUES[resname]:
                prot_neg_atoms.append(a)
        for lig_atoms, prot_atoms in [(lig_pos, prot_neg_atoms), (lig_neg, prot_pos_atoms)]:
            for la in lig_atoms:
                for pa in prot_atoms:
                    d = _dist(lig_conf, la.GetIdx(), prot_conf, pa.GetIdx())
                    if d < salt_d:
                        res = _get_residue_label(pa)
                        if res not in seen:
                            seen.add(res)
                            interactions.append(Interaction("SaltBridge", res, f"{la.GetSymbol()}{la.GetIdx()}", d, res))

    # --- Halogen bonds: X ... N/O/S < 3.5A ---
    hal_d = type_map.get("HalogenBond")
    if hal_d:
        seen = set()
        lig_halogens = [a for a in ligand.GetAtoms() if a.GetAtomicNum() in (17, 35, 53)]
        prot_acceptors = [a for a in protein.GetAtoms() if a.GetAtomicNum() in (7, 8, 16)]
        for lx in lig_halogens:
            for pa in prot_acceptors:
                d = _dist(lig_conf, lx.GetIdx(), prot_conf, pa.GetIdx())
                if d < hal_d:
                    res = _get_residue_label(pa)
                    if res not in seen:
                        seen.add(res)
                        interactions.append(Interaction("HalogenBond", res, f"{lx.GetSymbol()}{lx.GetIdx()}", d, res))

    # --- PiStacking: aromatic ring centroid distance < 5.5A ---
    pi_dist = type_map.get("PiStacking")
    if pi_dist:
        seen = set()
        lig_rings = _get_aromatic_rings(ligand)
        prot_rings = _get_aromatic_rings(protein)
        for lr in lig_rings:
            lc = _get_ring_centroid(lig_conf, lr)
            for pr in prot_rings:
                pc = _get_ring_centroid(prot_conf, pr)
                d = _centroid_dist(lc, pc)
                if d <= pi_dist:
                    res = _get_residue_label(protein.GetAtomWithIdx(pr[0]))
                    if res not in seen:
                        seen.add(res)
                        interactions.append(Interaction("PiStacking", res, f"Ring{lr[0]}", d, res))

    # --- CationPi: cation-aromatic < 6.0A (bidirectional) ---
    cat_dist = type_map.get("CationPi")
    if cat_dist:
        seen = set()
        lig_rings = _get_aromatic_rings(ligand)
        prot_rings = _get_aromatic_rings(protein)

        # Ligand cation -> Protein aromatic ring
        lig_cations = [a for a in ligand.GetAtoms() if a.GetFormalCharge() > 0]
        for lc_atom in lig_cations:
            lp = lig_conf.GetAtomPosition(lc_atom.GetIdx())
            lp_t = (lp.x, lp.y, lp.z)
            for pr in prot_rings:
                pc = _get_ring_centroid(prot_conf, pr)
                d = _centroid_dist(lp_t, pc)
                if d < cat_dist:
                    res = _get_residue_label(protein.GetAtomWithIdx(pr[0]))
                    key = f"CatPi_{res}"
                    if key not in seen:
                        seen.add(key)
                        interactions.append(Interaction("CationPi", res, f"{lc_atom.GetSymbol()}{lc_atom.GetIdx()}", d, res))

        # Protein cation (ARG/LYS/HIS) -> Ligand aromatic ring
        for prot_idx in range(protein.GetNumAtoms()):
            pa = protein.GetAtomWithIdx(prot_idx)
            info = pa.GetPDBResidueInfo()
            if not info:
                continue
            resname = info.GetResidueName().strip()
            atomname = info.GetName().strip()
            if resname not in _POS_RESIDUES or atomname not in _POS_RESIDUES[resname]:
                continue
            pp = prot_conf.GetAtomPosition(prot_idx)
            pp_t = (pp.x, pp.y, pp.z)
            for lr in lig_rings:
                lc = _get_ring_centroid(lig_conf, lr)
                d = _centroid_dist(pp_t, lc)
                if d < cat_dist:
                    res = _get_residue_label(pa)
                    key = f"CatPi_{res}"
                    if key not in seen:
                        seen.add(key)
                        interactions.append(Interaction("CationPi", res, f"Ring{lr[0]}", d, res))

    return interactions


def format_report(report: InteractionReport) -> str:
    """Format interaction report as readable string."""
    lines = [f"Total interactions: {report.total_count}"]
    for itype, count in sorted(report.by_type.items()):
        lines.append(f"  {itype}: {count}")
    return "\n".join(lines)
