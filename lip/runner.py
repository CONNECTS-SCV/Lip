"""Main runner — orchestrates the full optimization pipeline."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lip.config import LipConfig, ConstraintConfig
from lip.loop import OptimizationLoop, RoundResult, RunState
from lip.utils.io import save_json

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    output_dir: str
    rounds: list[RoundResult]
    best_score: float
    total_molecules: int
    resumed: bool = False


def run(config: LipConfig, resume_dir: str | None = None) -> RunResult:
    """Execute the full Lip optimization pipeline.

    Args:
        config: Full configuration.
        resume_dir: If provided, resume from this run directory.

    Returns:
        RunResult with output paths and summary.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resumed = False

    # Resume handling
    if resume_dir:
        resume_path = Path(resume_dir)
        state_file = resume_path / "run_state.json"
        if state_file.exists():
            state = RunState.load(state_file)
            if state.completed:
                log.info("Previous run already completed")
                return RunResult(
                    output_dir=str(resume_path),
                    rounds=[],
                    best_score=state.best_score,
                    total_molecules=state.total_molecules,
                    resumed=True,
                )
            # Use resume dir as output
            output_dir = resume_path
            config.output_dir = str(output_dir)
            resumed = True
            log.info(
                f"Resuming from chunk {state.current_chunk}, "
                f"round {state.current_round}"
            )

    # Save config snapshot
    config.save_yaml(output_dir / "config.yaml")

    # Step 0: Auto-detect pocket if needed
    _pocket_is_default = (config.pocket_center == [0.0, 0.0, 0.0])
    if _pocket_is_default and config.auto_pocket and config.receptor_pdb:
        _auto_detect_pocket(config)
        config.save_yaml(output_dir / "config.yaml")
    elif _pocket_is_default and config.docking.enabled:
        log.warning(
            "pocket_center is [0,0,0] and auto_pocket is disabled. "
            "Docking will use origin as center -- this is likely incorrect."
        )

    # Step 1: Pocket2Mol (optional)
    if config.pocket2mol.enabled and config.receptor_pdb:
        _run_pocket2mol(config)

    # Step 2: Build and run optimization loop
    loop = OptimizationLoop.from_config(config)

    # Restore state if resuming
    if resumed and (output_dir / "run_state.json").exists():
        loop.state = RunState.load(output_dir / "run_state.json")

    # Register progress callback
    def on_round(result: RoundResult):
        log.info(
            f"Round {result.round_num}: "
            f"best={result.best_score:.4f} avg={result.avg_score:.4f} "
            f"({result.n_valid}/{result.n_total} valid)"
        )

    loop.on_round_complete(on_round)

    # Run
    log.info(f"Starting optimization (mode={config.optimization.mode})")
    rounds = loop.run()

    # Step 3: Post-optimization synthesis analysis
    if config.synthesis.enabled and rounds:
        _run_synthesis_analysis(config, rounds, output_dir)

    return RunResult(
        output_dir=str(output_dir),
        rounds=rounds,
        best_score=loop.state.best_score,
        total_molecules=loop.state.total_molecules,
        resumed=resumed,
    )


def _run_pocket2mol(config: LipConfig) -> None:
    """Run Pocket2Mol and inject shape_similarity constraint."""
    from lip.pocket2mol.generate import (
        run_pocket2mol,
        select_diverse_representatives,
        combine_sdfs,
    )

    log.info("Running Pocket2Mol for reference molecule generation...")

    sdf_paths = run_pocket2mol(
        pdb_path=config.receptor_pdb,
        pocket_center=config.pocket_center,
        bbox_size=config.pocket2mol.bbox_size,
        n_samples=config.pocket2mol.n_samples,
        output_dir=str(Path(config.output_dir) / "pocket2mol"),
        pocket2mol_dir=config.paths.pocket2mol_dir,
        conda_env=config.pocket2mol.conda_env,
        timeout=config.pocket2mol.timeout,
    )

    if not sdf_paths:
        log.warning("Pocket2Mol produced no molecules, skipping shape constraint")
        return

    # Select diverse representatives
    selected = select_diverse_representatives(sdf_paths, n_picks=3)

    # Combine into single SDF
    combined_sdf = str(Path(config.output_dir) / "pocket2mol" / "reference.sdf")
    combine_sdfs(selected, combined_sdf)

    # Auto-inject shape_similarity constraint
    config.constraints.append(ConstraintConfig(
        type="shape",
        weight=config.pocket2mol.shape_weight,
        params={"reference_sdf": combined_sdf, "mode": "maximize"},
    ))

    log.info(
        f"Pocket2Mol: {len(sdf_paths)} generated, {len(selected)} selected, "
        f"shape constraint injected (weight={config.pocket2mol.shape_weight})"
    )


def _auto_detect_pocket(config: LipConfig) -> None:
    """Auto-detect pocket center from receptor PDB if not explicitly set.

    Strategy:
    1. Try extract_ligands() — co-crystallized ligand center is most reliable
    2. Try detect_pockets() via fpocket — use highest-druggability pocket
    3. If both fail, raise with clear message
    """
    from lip.utils.pocket import extract_ligands, detect_pockets

    receptor = config.receptor_pdb
    if not receptor:
        raise RuntimeError(
            "Cannot auto-detect pocket: no --receptor provided. "
            "Please provide --receptor and either --pocket-center "
            "or a PDB with a co-crystallized ligand."
        )

    log.info(f"Auto-detecting pocket from {receptor}...")

    # Strategy 1: co-crystallized ligand
    try:
        ligands = extract_ligands(receptor)
        if ligands:
            best = ligands[0]  # sorted by num_atoms descending
            config.pocket_center = list(best.center)
            log.info(
                f"Pocket auto-detected from ligand "
                f"{best.resname} (chain {best.chain}, {best.num_atoms} atoms): "
                f"center={config.pocket_center}"
            )
            return
    except Exception as e:
        log.debug(f"Ligand extraction failed: {e}")

    # Strategy 2: fpocket
    try:
        pockets = detect_pockets(receptor, fpocket_bin=config.paths.fpocket)
        if pockets:
            pockets_sorted = sorted(
                pockets, key=lambda p: p.druggability, reverse=True,
            )
            best = pockets_sorted[0]
            config.pocket_center = list(best.center)
            log.info(
                f"Pocket auto-detected via fpocket "
                f"(rank={best.rank}, druggability={best.druggability:.3f}, "
                f"volume={best.volume:.1f}): center={config.pocket_center}"
            )
            return
    except Exception as e:
        log.debug(f"fpocket detection failed: {e}")

    raise RuntimeError(
        "Pocket auto-detection failed: no co-crystallized ligands found "
        "and fpocket detection returned no results. "
        "Please provide --pocket-center explicitly or ensure fpocket is installed."
    )


def _run_synthesis_analysis(
    config: LipConfig, rounds: list[RoundResult], output_dir: Path,
) -> None:
    """Post-optimization: analyze synthesis routes for top molecules."""
    from lip.scoring.synthesis import analyze_batch, is_available

    if not is_available():
        log.info(
            "Skipping synthesis analysis: AiZynthFinder not installed. "
            "Install with: pip install aizynthfinder"
        )
        return

    # Collect all molecules across all rounds, sort by score
    all_molecules = []
    for rr in rounds:
        for mol in rr.molecules:
            if "smiles" in mol and "score" in mol:
                all_molecules.append(mol)

    if not all_molecules:
        log.info("No molecules to analyze for synthesis")
        return

    all_molecules.sort(key=lambda m: m["score"], reverse=True)

    # Deduplicate by SMILES, keeping highest score
    seen: set[str] = set()
    unique_top: list[dict] = []
    for mol in all_molecules:
        smi = mol["smiles"]
        if smi not in seen:
            seen.add(smi)
            unique_top.append(mol)
            if len(unique_top) >= config.synthesis.top_n:
                break

    smiles_list = [m["smiles"] for m in unique_top]
    log.info(f"Running synthesis analysis on top {len(smiles_list)} molecules...")

    aizyn_config = config.paths.aizynthfinder_config or None
    results = analyze_batch(
        smiles_list,
        config_path=aizyn_config,
        time_limit=config.synthesis.time_limit,
    )

    # Save results
    from dataclasses import asdict
    synthesis_output = output_dir / "synthesis_analysis.json"
    save_json([asdict(r) for r in results], synthesis_output)

    # Log summary
    n_solved = sum(1 for r in results if r.is_solved)
    log.info(
        f"Synthesis analysis complete: {n_solved}/{len(results)} solved, "
        f"saved to {synthesis_output}"
    )
    for r in results:
        status = "solved" if r.is_solved else "unsolved"
        log.info(
            f"  {r.smiles[:50]}: {status}, "
            f"{r.n_routes} routes, best_score={r.best_score:.3f}"
        )
