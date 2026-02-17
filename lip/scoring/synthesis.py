"""Retrosynthetic analysis via AiZynthFinder."""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

try:
    from aizynthfinder.aizynthfinder import AiZynthFinder
    _AIZYNTHFINDER_AVAILABLE = True
except ImportError:
    _AIZYNTHFINDER_AVAILABLE = False


def is_available() -> bool:
    return _AIZYNTHFINDER_AVAILABLE


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SynthesisRoute:
    score: float
    n_steps: int
    reactions: list[str]
    starting_materials: list[str]


@dataclass
class SynthesisResult:
    smiles: str
    is_solved: bool
    n_routes: int
    best_score: float
    routes: list[SynthesisRoute]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_synthesis(
    smiles: str,
    config_path: str | None = None,
    time_limit: int = 120,
    iteration_limit: int = 100,
    max_routes: int = 5,
) -> SynthesisResult:
    """Analyze retrosynthetic routes for a target molecule.

    Args:
        smiles: Target SMILES.
        config_path: AiZynthFinder config YAML path.
        time_limit: MCTS max time in seconds.
        iteration_limit: MCTS max iterations.
        max_routes: Max routes to return.
    """
    if not _AIZYNTHFINDER_AVAILABLE:
        log.warning("AiZynthFinder not installed")
        return SynthesisResult(
            smiles=smiles, is_solved=False, n_routes=0,
            best_score=0.0, routes=[],
        )

    try:
        finder = AiZynthFinder(configfile=config_path)
        finder.target_smiles = smiles
        finder.config.search.time_limit = time_limit
        finder.config.search.iteration_limit = iteration_limit

        finder.tree_search()
        finder.build_routes()

        stats = finder.routes.compute_scores()
        routes = []

        for i, (tree, score_dict) in enumerate(
            zip(finder.routes, stats)
        ):
            if i >= max_routes:
                break
            routes.append(SynthesisRoute(
                score=score_dict.get("state score", 0.0),
                n_steps=len(tree.reactions()),
                reactions=[str(r) for r in tree.reactions()],
                starting_materials=[str(m) for m in tree.leafs()],
            ))

        is_solved = any(r.score > 0 for r in routes)
        best_score = max((r.score for r in routes), default=0.0)

        return SynthesisResult(
            smiles=smiles,
            is_solved=is_solved,
            n_routes=len(routes),
            best_score=best_score,
            routes=routes,
        )

    except Exception as e:
        log.error(f"Synthesis analysis failed for {smiles}: {e}")
        return SynthesisResult(
            smiles=smiles, is_solved=False, n_routes=0,
            best_score=0.0, routes=[],
        )


def analyze_batch(
    smiles_list: list[str],
    config_path: str | None = None,
    time_limit: int = 60,
) -> list[SynthesisResult]:
    """Analyze synthesis for a batch of molecules."""
    return [
        analyze_synthesis(smi, config_path=config_path, time_limit=time_limit)
        for smi in smiles_list
    ]
