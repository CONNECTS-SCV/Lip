"""Main runner — orchestrates the full optimization pipeline."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lip.config import LipConfig
from lip.loop import OptimizationLoop, RoundResult, RunState
from lip.utils.geometry import sdf_centroid, pdbqt_centroid
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

    # Step 0: Determine pocket center
    if config.pocket_center == [0.0, 0.0, 0.0] and config.receptor_pdb:
        if config.is_docked:
            _extract_pocket_from_pdb(config)
        else:
            _dock_and_extract_pocket(config)
        config.save_yaml(output_dir / "config.yaml")

    # Step 1: Pocket2Mol (3D Shape 제약조건이 있고 reference가 없을 때만)
    if _find_shape_constraint_without_reference(config) and config.receptor_pdb:
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

    # Step 3: Re-dock top molecules and save complex PDBs
    if rounds and config.receptor_pdb:
        _redock_top_molecules(config, rounds, output_dir)

    # Step 4: Post-optimization synthesis analysis
    if config.synthesis.enabled and rounds:
        _run_synthesis_analysis(config, rounds, output_dir)

    return RunResult(
        output_dir=str(output_dir),
        rounds=rounds,
        best_score=loop.state.best_score,
        total_molecules=loop.state.total_molecules,
        resumed=resumed,
    )


def _find_shape_constraint_without_reference(config: LipConfig):
    """constraints에서 type='shape'이고 reference가 아직 없는 것을 찾는다."""
    for cc in config.constraints:
        if cc.type == "shape":
            has_ref = cc.params.get("reference_sdf") or cc.params.get("reference_smiles")
            if not has_ref:
                return cc
    return None


def _run_pocket2mol(config: LipConfig) -> None:
    """Run Pocket2Mol and inject shape_similarity constraint."""
    from lip.pocket2mol.generate import (
        run_pocket2mol,
        select_diverse_representatives,
        combine_sdfs,
    )

    log.info("Running Pocket2Mol for reference molecule generation...")

    try:
        sdf_paths = run_pocket2mol(
            pdb_path=config.receptor_pdb,
            pocket_center=config.pocket_center,
            bbox_size=config.pocket2mol.bbox_size,
            n_samples=config.pocket2mol.n_samples,
            output_dir=str(Path(config.output_dir) / "pocket2mol"),
            pocket2mol_dir=config.paths.pocket2mol_dir,
            conda_env=config.pocket2mol.conda_env,
            timeout=config.pocket2mol.timeout,
            device=config.generator.device,
        )
    except RuntimeError as e:
        log.warning(f"Pocket2Mol failed, skipping shape constraint: {e}")
        return

    if not sdf_paths:
        log.warning("Pocket2Mol produced no molecules, skipping shape constraint")
        return

    # Select diverse representatives
    selected = select_diverse_representatives(sdf_paths, n_picks=3)

    # Combine into single SDF
    combined_sdf = str(Path(config.output_dir) / "pocket2mol" / "reference.sdf")
    combine_sdfs(selected, combined_sdf)

    # 기존 shape constraint에 reference_sdf 주입 (새로 append하지 않음)
    shape_cc = _find_shape_constraint_without_reference(config)
    shape_cc.params["reference_sdf"] = combined_sdf
    shape_cc.params.setdefault("mode", "maximize")

    log.info(
        f"Pocket2Mol: {len(sdf_paths)} generated, {len(selected)} selected, "
        f"shape reference injected (weight={shape_cc.weight})"
    )


def _extract_pocket_from_pdb(config: LipConfig) -> None:
    """PDB 내 공결정 리간드에서 포켓 중심 추출."""
    from lip.utils.pocket import extract_ligands

    ligands = extract_ligands(config.receptor_pdb)
    if not ligands:
        raise RuntimeError(
            "is_docked=True이지만 PDB에서 리간드를 찾을 수 없습니다. "
            "PDB에 HETATM 리간드가 포함되어 있는지 확인하세요."
        )
    best = ligands[0]  # sorted by num_atoms descending
    config.pocket_center = list(best.center)
    log.info(
        f"Pocket from co-crystal ligand {best.resname} "
        f"(chain {best.chain}, {best.num_atoms} atoms): "
        f"center={config.pocket_center}"
    )


def _dock_and_extract_pocket(config: LipConfig) -> None:
    """SDF 리간드를 PDB에 도킹한 후, best pose에서 포켓 중심 추출."""
    from rdkit import Chem
    from lip.scoring.docking import VinaDockingScorer

    if not config.ligand_sdf:
        raise RuntimeError(
            "is_docked=False이지만 --ligand-sdf가 제공되지 않았습니다."
        )

    log.info(f"Docking ligand {config.ligand_sdf} into {config.receptor_pdb}...")

    # 1. SDF에서 리간드 centroid 계산 → 초기 docking center
    sdf_center = sdf_centroid(config.ligand_sdf)
    log.info(f"SDF ligand centroid: {sdf_center}")

    # 2. SDF → SMILES 변환
    suppl = Chem.SDMolSupplier(config.ligand_sdf, removeHs=True)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise RuntimeError(
            f"SDF 파일에서 분자를 읽을 수 없습니다: {config.ligand_sdf}"
        )
    smiles = Chem.MolToSmiles(mol)

    # 3. Vina로 도킹
    box_sz = config.docking.box_size
    scorer = VinaDockingScorer(
        receptor_pdb=config.receptor_pdb,
        pocket_center=tuple(sdf_center),
        box_size=(box_sz, box_sz, box_sz),
        exhaustiveness=config.docking.exhaustiveness,
    )
    result = scorer.dock_smiles(smiles)
    if not result.success:
        raise RuntimeError(f"초기 도킹 실패: {smiles}")

    # 4. Best pose에서 centroid 추출 → pocket_center
    pose_center = pdbqt_centroid(result.pose_pdbqt)
    config.pocket_center = list(pose_center)
    log.info(
        f"Pocket from docked pose (score={result.score:.2f} kcal/mol): "
        f"center={config.pocket_center}"
    )



def _redock_top_molecules(
    config: LipConfig, rounds: list[RoundResult], output_dir: Path,
    top_n: int = 20,
) -> None:
    """Re-dock top N molecules and save protein-ligand complex PDB files."""
    from lip.scoring.docking import create_docking_scorer
    from lip.utils.io import collect_top_molecules, save_complex_pdb, save_results_csv

    top_mols = collect_top_molecules(rounds, top_n)
    if not top_mols:
        return

    center = tuple(config.pocket_center)
    box_sz = config.docking.box_size

    log.info(f"Re-docking top {len(top_mols)} molecules...")

    try:
        scorer = create_docking_scorer(
            method=config.docking.method,
            receptor_pdb=config.receptor_pdb,
            pocket_center=center,
            box_size=(box_sz, box_sz, box_sz),
            exhaustiveness=config.docking.exhaustiveness,
        )
    except Exception as e:
        log.warning(f"Failed to create docking scorer for re-docking: {e}")
        return

    poses_dir = output_dir / "docked_poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    docked_records = []
    for i, mol in enumerate(top_mols):
        smiles = mol["smiles"]
        result = scorer.dock_smiles(smiles)

        if not result.success:
            log.warning(f"Re-docking failed: {smiles[:50]}")
            continue

        rank = i + 1
        filename = f"rank{rank:02d}_{result.score:.1f}.pdb"

        save_complex_pdb(
            receptor_pdb_path=config.receptor_pdb,
            ligand_pdbqt=result.pose_pdbqt,
            output_path=poses_dir / filename,
        )

        docked_records.append({
            "rank": rank,
            "smiles": smiles,
            "docking_score": result.score,
            "optimization_score": mol.get("score", 0.0),
            "complex_pdb": filename,
        })

        log.info(f"  Rank {rank}: {result.score:.2f} kcal/mol — {smiles[:50]}")

    if docked_records:
        save_results_csv(docked_records, poses_dir / "docking_summary.csv")

    log.info(
        f"Re-docking complete: {len(docked_records)}/{len(top_mols)} succeeded, "
        f"saved to {poses_dir}"
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
