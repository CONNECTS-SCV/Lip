"""RL bias metrics — scaffold diversity, novelty, KL divergence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MurckoScaffold

log = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    step: int
    n_valid: int
    n_unique: int
    n_novel: int
    validity: float
    uniqueness: float
    novelty: float
    scaffold_diversity: float
    avg_score: float
    top_score: float


@dataclass
class KLDivergences:
    """KL divergence between agent and prior distributions per property."""
    mw: float = 0.0
    logp: float = 0.0
    qed: float = 0.0
    tpsa: float = 0.0
    hbd: float = 0.0
    hba: float = 0.0
    rotbonds: float = 0.0
    num_rings: float = 0.0


# ---------------------------------------------------------------------------
# Step metrics
# ---------------------------------------------------------------------------

def compute_step_metrics(
    smiles_list: list[str],
    scores: list[float],
    step: int,
    seen_smiles: set[str] | None = None,
) -> StepMetrics:
    """Compute per-step RL metrics."""
    valid = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    unique = list(set(valid))

    novel = unique
    if seen_smiles is not None:
        novel = [s for s in unique if s not in seen_smiles]

    n_total = len(smiles_list)
    validity = len(valid) / n_total if n_total > 0 else 0.0
    uniqueness = len(unique) / len(valid) if valid else 0.0
    novelty_ratio = len(novel) / len(unique) if unique else 0.0

    # Scaffold diversity
    scaffolds = set()
    for smi in unique:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.add(Chem.MolToSmiles(scaffold))
            except Exception:
                pass

    scaffold_diversity = len(scaffolds) / len(unique) if unique else 0.0

    valid_scores = [s for s in scores if s > 0]
    avg_score = np.mean(valid_scores) if valid_scores else 0.0
    top_score = max(valid_scores) if valid_scores else 0.0

    return StepMetrics(
        step=step,
        n_valid=len(valid),
        n_unique=len(unique),
        n_novel=len(novel),
        validity=validity,
        uniqueness=uniqueness,
        novelty=novelty_ratio,
        scaffold_diversity=scaffold_diversity,
        avg_score=float(avg_score),
        top_score=float(top_score),
    )


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

_PROPERTY_FUNCS = {
    "mw": Descriptors.ExactMolWt,
    "logp": Descriptors.MolLogP,
    "qed": lambda mol: __import__("rdkit.Chem.QED", fromlist=["qed"]).qed(mol),
    "tpsa": Descriptors.TPSA,
    "hbd": Descriptors.NumHDonors,
    "hba": Descriptors.NumHAcceptors,
    "rotbonds": Descriptors.NumRotatableBonds,
    "num_rings": Descriptors.RingCount,
}


def _smiles_to_values(smiles_list: list[str], prop: str) -> list[float]:
    func = _PROPERTY_FUNCS[prop]
    values = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            values.append(float(func(mol)))
    return values


def _kl_divergence(p_values: list[float], q_values: list[float], n_bins: int = 30) -> float:
    """Compute KL(P||Q) using histogram approximation."""
    if not p_values or not q_values:
        return 0.0

    all_vals = p_values + q_values
    lo, hi = min(all_vals), max(all_vals)
    if lo == hi:
        return 0.0

    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p_values, bins=bins, density=True)
    q_hist, _ = np.histogram(q_values, bins=bins, density=True)

    # Add small epsilon to avoid division by zero
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    # Normalize
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    kl = np.sum(p_hist * np.log(p_hist / q_hist))
    return float(kl)


def compute_kl_divergences(
    agent_smiles: list[str],
    prior_smiles: list[str],
    n_bins: int = 30,
) -> KLDivergences:
    """Compute KL divergences between agent and prior distributions.

    Measures how much the agent's molecular property distributions
    have diverged from the prior's (RL bias detection).
    """
    result = KLDivergences()

    for prop in _PROPERTY_FUNCS:
        agent_vals = _smiles_to_values(agent_smiles, prop)
        prior_vals = _smiles_to_values(prior_smiles, prop)
        kl = _kl_divergence(agent_vals, prior_vals, n_bins)
        setattr(result, prop, kl)

    return result
