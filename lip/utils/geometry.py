"""Geometry utilities — centroid calculations for molecular coordinates."""

from __future__ import annotations


def centroid(coords: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Compute centroid from a list of (x, y, z) coordinates."""
    n = len(coords)
    if n == 0:
        raise ValueError("Empty coordinate list")
    return (
        sum(c[0] for c in coords) / n,
        sum(c[1] for c in coords) / n,
        sum(c[2] for c in coords) / n,
    )


def sdf_centroid(sdf_path: str) -> tuple[float, float, float]:
    """Compute centroid of the first valid molecule in an SDF file."""
    from rdkit import Chem

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None or mol.GetNumConformers() == 0:
        raise RuntimeError(f"SDF에서 3D 좌표를 읽을 수 없습니다: {sdf_path}")

    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append((pos.x, pos.y, pos.z))
    return centroid(coords)


def pdbqt_centroid(pdbqt_string: str) -> tuple[float, float, float]:
    """Compute centroid of ATOM/HETATM coordinates in a PDBQT string."""
    coords = []
    for line in pdbqt_string.split("\n"):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
            except (ValueError, IndexError):
                continue
    if not coords:
        raise RuntimeError("PDBQT에서 원자 좌표를 찾을 수 없습니다")
    return centroid(coords)
