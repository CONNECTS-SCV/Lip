"""Chemistry utilities — SMILES validation, molecular properties, fingerprints."""

from __future__ import annotations

import logging
import re
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, FilterCatalog, QED, Draw, DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SMILES helpers
# ---------------------------------------------------------------------------

def is_valid(smiles: str) -> bool:
    """Return True if SMILES parses to a valid molecule."""
    return Chem.MolFromSmiles(smiles) is not None


def canonicalize(smiles: str) -> str | None:
    """Return canonical SMILES, or None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def to_mol(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


# ---------------------------------------------------------------------------
# Molecular properties
# ---------------------------------------------------------------------------

def compute_properties(smiles: str) -> dict[str, Any] | None:
    """Compute standard molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "molecular_weight": Descriptors.ExactMolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "ring_count": Descriptors.RingCount(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "fraction_csp3": Descriptors.FractionCSP3(mol),
        "qed": QED.qed(mol),
    }


PROPERTY_FUNCTIONS: dict[str, callable] = {
    "molecular_weight": Descriptors.ExactMolWt,
    "logp": Descriptors.MolLogP,
    "tpsa": Descriptors.TPSA,
    "hbd": Descriptors.NumHDonors,
    "hba": Descriptors.NumHAcceptors,
    "rotatable_bonds": Descriptors.NumRotatableBonds,
    "heavy_atom_count": lambda mol: mol.GetNumHeavyAtoms(),
    "ring_count": Descriptors.RingCount,
    "aromatic_rings": Descriptors.NumAromaticRings,
    "fraction_csp3": Descriptors.FractionCSP3,
}


def get_property(mol: Chem.Mol, name: str) -> float | None:
    """Get a single property value by name."""
    func = PROPERTY_FUNCTIONS.get(name)
    if func is None:
        log.warning(f"Unknown property: {name}")
        return None
    return float(func(mol))


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

_PAINS_CATALOG = None


def _get_pains_catalog() -> FilterCatalog.FilterCatalog:
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
    return _PAINS_CATALOG


def check_lipinski(smiles: str) -> bool:
    """Return True if molecule passes Lipinski's Rule of Five."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return (
        Descriptors.ExactMolWt(mol) <= 500
        and Descriptors.MolLogP(mol) <= 5
        and Descriptors.NumHDonors(mol) <= 5
        and Descriptors.NumHAcceptors(mol) <= 10
    )


def check_pains(smiles: str) -> bool:
    """Return True if molecule has NO PAINS alerts (i.e. passes)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    catalog = _get_pains_catalog()
    return not catalog.HasMatch(mol)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def get_morgan_fp(
    smiles: str, radius: int = 2, n_bits: int = 2048
) -> DataStructs.ExplicitBitVect | None:
    """Compute Morgan fingerprint as bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity(smiles_a: str, smiles_b: str, radius: int = 2) -> float:
    """Compute Tanimoto similarity between two SMILES."""
    fp_a = get_morgan_fp(smiles_a, radius)
    fp_b = get_morgan_fp(smiles_b, radius)
    if fp_a is None or fp_b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def mol_to_svg(smiles: str, size: tuple[int, int] = (300, 200)) -> str:
    """Render molecule as SVG string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = Draw.MolDraw2DSVG(*size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_to_png_base64(smiles: str, size: tuple[int, int] = (300, 200)) -> str:
    """Render molecule as base64-encoded PNG."""
    import base64
    from io import BytesIO

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# REINVENT4 compatibility filter
# ---------------------------------------------------------------------------

_REINVENT_ALLOWED_BRACKETS = {"[S+]", "[n+]", "[N-]", "[O-]", "[N+]", "[nH]"}
_BRACKET_RE = re.compile(r"\[[^\]]+\]")


def is_reinvent_compatible(smiles: str) -> bool:
    """Check if all bracket tokens in a SMILES are in the REINVENT4 vocabulary.

    REINVENT4's tokenizer rejects SMILES with bracket tokens outside its
    vocabulary (e.g. [SH], [Si], [P], [B]).
    """
    bracket_tokens = _BRACKET_RE.findall(smiles)
    return all(t in _REINVENT_ALLOWED_BRACKETS for t in bracket_tokens)
