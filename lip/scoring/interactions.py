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
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdbqt", mode="w", delete=False
        ) as f:
            f.write(pdbqt_string)
            pdbqt_path = f.name

        pdb_path = tempfile.mktemp(suffix=".pdb")
        subprocess.run(
            ["obabel", pdbqt_path, "-O", pdb_path],
            capture_output=True, timeout=30,
        )

        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        return mol
    except Exception as e:
        log.debug(f"PDBQT conversion failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Ring-based interaction helpers
# ---------------------------------------------------------------------------

# Protein cation residues for CationPi detection
_POS_RESIDUES: dict[str, set[str]] = {
    "ARG": {"NH1", "NH2", "NE"},
    "LYS": {"NZ"},
    "HIS": {"ND1", "NE2"},
}


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

def _detect_interactions(
    protein: Chem.Mol,
    ligand: Chem.Mol,
    types: list[InteractionType] | None = None,
) -> list[Interaction]:
    """Detect all interactions based on atom distances and types.

    Args:
        types: Interaction types to detect (default: INTERACTION_TYPES).
    """
    if types is None:
        types = INTERACTION_TYPES

    # Pre-build lookup: name -> (distance, detector_func)
    max_dist = max(t.distance for t in types)
    type_map = {t.name: t.distance for t in types}

    interactions = []
    prot_conf = protein.GetConformer()
    lig_conf = ligand.GetConformer()

    for lig_idx in range(ligand.GetNumAtoms()):
        lig_atom = ligand.GetAtomWithIdx(lig_idx)
        lig_pos = lig_conf.GetAtomPosition(lig_idx)
        lig_symbol = lig_atom.GetSymbol()

        for prot_idx in range(protein.GetNumAtoms()):
            prot_atom = protein.GetAtomWithIdx(prot_idx)
            prot_pos = prot_conf.GetAtomPosition(prot_idx)
            prot_symbol = prot_atom.GetSymbol()

            dist = lig_pos.Distance(prot_pos)
            if dist > max_dist:
                continue

            prot_label = f"{prot_symbol}{prot_idx}"
            lig_label = f"{lig_symbol}{lig_idx}"

            # H-bond
            hbond_d = type_map.get("HBond")
            if hbond_d and dist <= hbond_d and lig_symbol in ("N", "O") and prot_symbol in ("N", "O"):
                interactions.append(Interaction("HBond", prot_label, lig_label, dist))

            # Hydrophobic
            hydro_d = type_map.get("Hydrophobic")
            if hydro_d and dist <= hydro_d and lig_symbol == "C" and prot_symbol == "C":
                if lig_atom.GetDegree() > 1 and prot_atom.GetDegree() > 1:
                    interactions.append(Interaction("Hydrophobic", prot_label, lig_label, dist))

            # Salt bridge
            salt_d = type_map.get("SaltBridge")
            if salt_d and dist <= salt_d:
                lig_charge = lig_atom.GetFormalCharge()
                prot_charge = prot_atom.GetFormalCharge()
                if lig_charge != 0 and prot_charge != 0 and lig_charge * prot_charge < 0:
                    interactions.append(Interaction("SaltBridge", prot_label, lig_label, dist))

            # Halogen bond
            hal_d = type_map.get("HalogenBond")
            if hal_d and dist <= hal_d and lig_symbol in ("F", "Cl", "Br", "I") and prot_symbol in ("N", "O"):
                interactions.append(Interaction("HalogenBond", prot_label, lig_label, dist))

    # -------------------------------------------------------------------
    # Ring-based interactions (PiStacking, CationPi)
    # -------------------------------------------------------------------

    # PiStacking: aromatic ring centroid distance
    pi_dist = type_map.get("PiStacking")
    if pi_dist:
        lig_rings = _get_aromatic_rings(ligand)
        prot_rings = _get_aromatic_rings(protein)

        for lr in lig_rings:
            lc = _get_ring_centroid(lig_conf, lr)
            for pr in prot_rings:
                pc = _get_ring_centroid(prot_conf, pr)
                d = _centroid_dist(lc, pc)
                if d <= pi_dist:
                    interactions.append(
                        Interaction("PiStacking", f"Ring{pr[0]}", f"Ring{lr[0]}", d)
                    )

    # CationPi: cation-aromatic ring centroid distance (bidirectional)
    cat_dist = type_map.get("CationPi")
    if cat_dist:
        lig_rings = _get_aromatic_rings(ligand)
        prot_rings = _get_aromatic_rings(protein)

        # Direction 1: Ligand cation -> Protein aromatic ring
        for lig_idx in range(ligand.GetNumAtoms()):
            lig_atom = ligand.GetAtomWithIdx(lig_idx)
            if lig_atom.GetFormalCharge() <= 0:
                continue
            lp = lig_conf.GetAtomPosition(lig_idx)
            lp_tuple = (lp.x, lp.y, lp.z)
            for pr in prot_rings:
                pc = _get_ring_centroid(prot_conf, pr)
                d = _centroid_dist(lp_tuple, pc)
                if d <= cat_dist:
                    interactions.append(Interaction(
                        "CationPi", f"Ring{pr[0]}",
                        f"{lig_atom.GetSymbol()}{lig_idx}", d,
                    ))

        # Direction 2: Protein cation (ARG/LYS/HIS) -> Ligand aromatic ring
        for prot_idx in range(protein.GetNumAtoms()):
            prot_atom = protein.GetAtomWithIdx(prot_idx)
            info = prot_atom.GetPDBResidueInfo()
            if info is None:
                continue
            resname = info.GetResidueName().strip()
            atomname = info.GetName().strip()
            if resname not in _POS_RESIDUES:
                continue
            if atomname not in _POS_RESIDUES[resname]:
                continue

            pp = prot_conf.GetAtomPosition(prot_idx)
            pp_tuple = (pp.x, pp.y, pp.z)
            for lr in lig_rings:
                lc = _get_ring_centroid(lig_conf, lr)
                d = _centroid_dist(pp_tuple, lc)
                if d <= cat_dist:
                    interactions.append(Interaction(
                        "CationPi", f"{prot_atom.GetSymbol()}{prot_idx}",
                        f"Ring{lr[0]}", d,
                    ))

    return interactions


def format_report(report: InteractionReport) -> str:
    """Format interaction report as readable string."""
    lines = [f"Total interactions: {report.total_count}"]
    for itype, count in sorted(report.by_type.items()):
        lines.append(f"  {itype}: {count}")
    return "\n".join(lines)
