"""RL optimization loop - multi-cycle managed and manual modes."""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lip.config import LipConfig
from lip.component_builder import ComponentBuilder
from lip.inception import InceptionBuffer
from lip.constraints.base import ConstraintResult
from lip.constraints.registry import create_constraint
from lip.generator.reinvent import ReinventWrapper, StageConfig
from lip.scoring.aggregator import ScoreAggregator
from lip.scoring.docking import BaseDockingScorer, VinaDockingScorer
from lip.utils.chem import (
    is_valid, canonicalize, check_lipinski, check_pains, is_reinvent_compatible,
)
from lip.utils.io import save_round_results, save_json, load_json, save_progress_plot, save_top_molecules
from lip.utils.math import normalize_score

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Round result
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    round_num: int
    molecules: list[dict[str, Any]]
    best_score: float
    avg_score: float
    n_valid: int
    n_total: int


# ---------------------------------------------------------------------------
# Run state (for resume)
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    current_chunk: int = 0
    current_round: int = 0
    best_score: float = 0.0
    total_molecules: int = 0
    completed: bool = False

    def save(self, path: str | Path):
        save_json(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> RunState:
        data = load_json(path)
        return cls(**data)


# ---------------------------------------------------------------------------
# Optimization Loop
# ---------------------------------------------------------------------------

class OptimizationLoop:
    """Core RL optimization loop.

    Orchestrates generate -> filter -> score -> aggregate cycle.
    Component building is delegated to ComponentBuilder.
    Docking scorer creation is delegated to the docking registry.
    """

    def __init__(self, config: LipConfig):
        self.config = config
        self.generator: ReinventWrapper | None = None
        self.constraints: list[tuple[str, float, Any]] = []
        self.docking_scorer: BaseDockingScorer | None = None
        self.aggregator = ScoreAggregator()
        self.component_builder = ComponentBuilder(config)
        self.state = RunState()

        self._on_round_complete: Callable[[RoundResult], None] | None = None

    @classmethod
    def from_config(cls, config: LipConfig) -> OptimizationLoop:
        """Build loop from LipConfig."""
        loop = cls(config)

        # Generator
        loop.generator = ReinventWrapper({
            "prior_model": config.generator.prior_model,
            "agent_model": config.generator.agent_model,
            "device": config.generator.device,
            "batch_size": config.generator.batch_size,
            "sigma": config.generator.sigma,
            "learning_rate": config.generator.learning_rate,
            "diversity_filter": config.generator.diversity_filter,
            "inception_memory_size": config.optimization.inception.memory_size,
            "inception_sample_size": config.optimization.inception.sample_size,
            "inception_retention_mode": config.optimization.inception.retention,
            "work_dir": config.generator.work_dir,
        })

        # Constraints
        for cc in config.constraints:
            constraint = create_constraint(cc.type, cc.weight, cc.params)
            loop.constraints.append((cc.type, cc.weight, constraint))

        # Docking scorer
        if config.docking.enabled and config.receptor_pdb:
            center = tuple(config.pocket_center)
            box = (config.docking.box_size,) * 3
            loop.docking_scorer = VinaDockingScorer(
                receptor_pdb=config.receptor_pdb,
                pocket_center=center,
                box_size=box,
                exhaustiveness=config.docking.exhaustiveness,
            )

        return loop

    def on_round_complete(self, callback: Callable[[RoundResult], None]):
        """Register callback for round completion."""
        self._on_round_complete = callback

    # -----------------------------------------------------------------------
    # Main entry
    # -----------------------------------------------------------------------

    def run(self) -> list[RoundResult]:
        """Run the optimization loop."""
        if self.config.optimization.mode == "managed":
            return self._run_managed_mode()
        else:
            return self._run_manual_mode()

    # -----------------------------------------------------------------------
    # Managed mode: REINVENT4 handles RL internally
    # -----------------------------------------------------------------------

    def _run_managed_mode(self) -> list[RoundResult]:
        """Run RL via REINVENT4 staged_learning, chunk by chunk.

        Manages:
        - Per-step RoundResult creation and callbacks
        - Cross-chunk inception buffer with dedup and retention policy
        - REINVENT4 bracket compatibility filtering for inception seeds
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_steps = self.config.optimization.n_steps
        chunk_size = 1  # 1 step per chunk for per-step granularity
        n_chunks = n_steps

        all_results: list[RoundResult] = []
        memory_size = self.config.optimization.inception.memory_size
        inception_enabled = memory_size > 0
        inception = InceptionBuffer(
            memory_size=memory_size,
            retention=self.config.optimization.inception.retention,
        )
        steps_completed = self.state.current_chunk

        for chunk_idx in range(self.state.current_chunk, n_chunks):
            log.info(f"Step {chunk_idx + 1}/{n_steps}")

            chunk_output_dir = output_dir / f"chunk_{chunk_idx:03d}"
            chunk_output_dir.mkdir(parents=True, exist_ok=True)

            # Write inception buffer as seed CSV for REINVENT4
            inception_file = None
            if inception_enabled and inception.size > 0:
                entries = inception.get_entries()
                compatible = [
                    e for e in entries
                    if is_reinvent_compatible(e["smiles"])
                ]
                n_filtered = len(entries) - len(compatible)
                if n_filtered > 0:
                    log.info(
                        f"Inception filter: removed {n_filtered} "
                        f"incompatible SMILES"
                    )
                if compatible:
                    inception_csv = chunk_output_dir / "inception_seed.csv"
                    with open(inception_csv, "w") as f:
                        for entry in compatible:
                            f.write(f"{entry['smiles']}\n")
                    inception_file = str(inception_csv)

            # Build components and stage config
            components = self.component_builder.build_all(self.constraints)
            chkpt_path = str(
                output_dir / "checkpoints" / f"chunk_{chunk_idx:03d}.chkpt"
            )
            Path(chkpt_path).parent.mkdir(parents=True, exist_ok=True)

            stage = StageConfig(
                max_steps=chunk_size,
                min_steps=1,
                max_score=self.config.optimization.max_score,
                scoring_components=components,
                chkpt_file=chkpt_path,
            )

            process = self.generator.run_staged_learning(
                stages=[stage],
                output_dir=str(chunk_output_dir),
                inception_smiles_file=inception_file,
            )
            process.wait()

            if process.returncode != 0:
                stderr = process.stderr.read().decode() if process.stderr else ""
                stdout = process.stdout.read().decode() if process.stdout else ""
                log.error(
                    f"REINVENT4 exited with code {process.returncode}\n"
                    f"  stderr: {stderr[-1000:]}\n"
                    f"  stdout: {stdout[-500:]}"
                )
            else:
                log.info(f"REINVENT4 chunk {chunk_idx} completed successfully")

            # Parse RL CSV for per-step scores and molecules
            chunk_molecules: list[dict[str, Any]] = []
            step_scores: list[float] = []

            rl_csv = chunk_output_dir / "staged_learning_1.csv"
            if rl_csv.exists():
                with open(rl_csv) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            score = float(row.get("Score", 0))
                            smiles = row.get("SMILES", "")
                            smiles_state = row.get("SMILES_state", "0")

                            step_scores.append(score)

                            if smiles_state == "1" and smiles and score > 0:
                                chunk_molecules.append({
                                    "smiles": smiles,
                                    "score": score,
                                })
                        except (ValueError, KeyError):
                            continue

            # Save per-round CSV (same format as manual mode)
            if chunk_molecules:
                save_round_results(chunk_idx, chunk_molecules, str(output_dir))

            # Create RoundResult for this step
            if step_scores:
                rr = RoundResult(
                    round_num=chunk_idx,
                    molecules=chunk_molecules,
                    best_score=max(step_scores),
                    avg_score=sum(step_scores) / len(step_scores),
                    n_valid=len(chunk_molecules),
                    n_total=self.config.generator.batch_size,
                )
                all_results.append(rr)
                if self._on_round_complete:
                    self._on_round_complete(rr)

            # Update inception buffer
            if inception_enabled and chunk_molecules:
                n_added = inception.update(chunk_molecules)
                log.info(
                    f"Inception buffer: {inception.size}/{memory_size} "
                    f"(+{n_added} new, mode={inception.retention})"
                )

            # Load checkpoint for next chunk
            if Path(chkpt_path).exists():
                self.generator.load_checkpoint(chkpt_path)

            self.state.current_chunk = chunk_idx + 1
            self.state.best_score = max(
                self.state.best_score,
                max((r.best_score for r in all_results), default=0.0),
            )
            self.state.save(output_dir / "run_state.json")

            if self._should_early_stop(all_results):
                log.info("Early stopping triggered")
                break

        self.state.completed = True
        self.state.save(output_dir / "run_state.json")
        save_progress_plot(all_results, output_dir)
        save_top_molecules(all_results, output_dir)
        return all_results

    # -----------------------------------------------------------------------
    # Manual mode: generate -> filter -> score -> select
    # -----------------------------------------------------------------------

    def _run_manual_mode(self) -> list[RoundResult]:
        """Manual optimization: generate -> filter -> score -> select."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_rounds = self.config.optimization.n_steps
        n_mols = self.config.optimization.batch_size

        all_results = []

        for round_num in range(self.state.current_round, n_rounds):
            log.info(f"Round {round_num + 1}/{n_rounds}")

            smiles = self._generate_batch(n_mols)

            # Filter
            if self.config.filter.lipinski:
                smiles = [s for s in smiles if check_lipinski(s)]
            if self.config.filter.pains:
                smiles = [s for s in smiles if check_pains(s)]

            if not smiles:
                log.warning(f"Round {round_num}: no molecules passed filters")
                continue

            # Score constraints
            constraint_results, constraint_weights = self._score_constraints(smiles)

            # Aggregate
            final_scores = self._aggregate_scores(
                constraint_results, constraint_weights, len(smiles)
            )

            # Build records and save
            molecules = self._build_records(
                smiles, final_scores, constraint_results, round_num
            )
            molecules.sort(key=lambda m: m["score"], reverse=True)
            molecules = self._select_diverse(molecules, n_mols)
            save_round_results(round_num, molecules, str(output_dir))

            best = molecules[0]["score"] if molecules else 0.0
            avg = sum(m["score"] for m in molecules) / len(molecules) if molecules else 0.0

            round_result = RoundResult(
                round_num=round_num,
                molecules=molecules,
                best_score=best,
                avg_score=avg,
                n_valid=len(molecules),
                n_total=n_mols,
            )
            all_results.append(round_result)

            if self._on_round_complete:
                self._on_round_complete(round_result)

            self.state.current_round = round_num + 1
            self.state.best_score = max(self.state.best_score, best)
            self.state.total_molecules += len(molecules)
            self.state.save(output_dir / "run_state.json")

            if (round_num + 1) % self.config.optimization.checkpoint_every == 0:
                chkpt = output_dir / "checkpoints" / f"round_{round_num:03d}.chkpt"
                chkpt.parent.mkdir(parents=True, exist_ok=True)
                self.generator.save_checkpoint(str(chkpt))

            if self._should_early_stop(all_results):
                log.info("Early stopping triggered")
                break

        self.state.completed = True
        self.state.save(output_dir / "run_state.json")
        save_progress_plot(all_results, output_dir)
        save_top_molecules(all_results, output_dir)

        return all_results

    # -----------------------------------------------------------------------
    # Scoring helpers
    # -----------------------------------------------------------------------

    def _score_constraints(
        self, smiles: list[str],
    ) -> tuple[dict[str, ConstraintResult], dict[str, float]]:
        """Score SMILES against all constraints + docking."""
        results = {}
        weights = {}

        for name, weight, constraint in self.constraints:
            results[name] = constraint.score(smiles)
            weights[name] = weight

        if self.docking_scorer is not None:
            docking_results = self.docking_scorer.dock_batch(smiles)

            t_high = self.config.docking.transform.high
            t_low = self.config.docking.transform.low
            norm_scores = []
            passed_list = []
            raw_scores = []

            for r in docking_results:
                raw_scores.append(r.score)
                if not r.success:
                    norm_scores.append(0.0)
                    passed_list.append(False)
                else:
                    norm_scores.append(
                        max(0.0, min(1.0, normalize_score(r.score, low=t_high, high=t_low)))
                    )
                    passed_list.append(r.score <= t_high)

            results["docking"] = ConstraintResult(
                scores=norm_scores,
                passed=passed_list,
                raw_values=raw_scores,
            )
            weights["docking"] = self.config.docking.weight

            # Interaction analysis
            if self.config.docking.analyze_interactions:
                int_scores, int_passed, int_raw = (
                    self._analyze_interactions_batch(docking_results)
                )
                results["interactions"] = ConstraintResult(
                    scores=int_scores,
                    passed=int_passed,
                    raw_values=int_raw,
                )
                weights["interactions"] = self.config.docking.interaction_weight

        return results, weights

    def _get_protein_mol(self):
        """Load and cache protein RDKit Mol for interaction analysis."""
        if not hasattr(self, "_protein_mol_cache"):
            self._protein_mol_cache = None

        if self._protein_mol_cache is not None:
            return self._protein_mol_cache

        try:
            from rdkit import Chem
            mol = Chem.MolFromPDBFile(self.config.receptor_pdb, removeHs=False)
            self._protein_mol_cache = mol
            if mol is not None:
                log.debug(f"Loaded protein mol: {mol.GetNumAtoms()} atoms")
            else:
                log.warning("Failed to parse protein PDB for interaction analysis")
            return mol
        except Exception as e:
            log.warning(f"Could not load protein for interaction analysis: {e}")
            return None

    def _analyze_interactions_batch(
        self, docking_results: list,
    ) -> tuple[list[float], list[bool], list[int]]:
        """Run interaction analysis on successful docking poses.

        Returns:
            (normalized_scores, passed_list, raw_interaction_counts)
        """
        from lip.scoring.interactions import analyze_pose

        scores = []
        passed = []
        raw_counts = []

        protein_mol = self._get_protein_mol()

        for r in docking_results:
            if not r.success or not r.pose_pdbqt:
                scores.append(0.0)
                passed.append(False)
                raw_counts.append(0)
                continue

            try:
                report = analyze_pose(
                    protein_pdb=self.config.receptor_pdb,
                    pose_pdbqt=r.pose_pdbqt,
                    smiles=r.smiles,
                    protein_mol=protein_mol,
                )
                count = report.total_count
                r.interaction_count = count

                # Sigmoid normalization (CHEM-identical: high=8, low=0, k=0.5)
                midpoint = 4.0  # (8.0 + 0.0) / 2
                norm = 1.0 / (1.0 + math.exp(-0.5 * (count - midpoint)))
                scores.append(norm)
                passed.append(count > 0)
                raw_counts.append(count)
            except Exception as e:
                log.debug(f"Interaction analysis failed for {r.smiles}: {e}")
                scores.append(0.0)
                passed.append(False)
                raw_counts.append(0)

        return scores, passed, raw_counts

    def _aggregate_scores(
        self,
        results: dict[str, ConstraintResult],
        weights: dict[str, float],
        n: int,
    ) -> list[float]:
        """Aggregate constraint scores into final scores."""
        if self.config.scoring_method == "pareto":
            return self.aggregator.pareto_rank(results, n)
        agg = self.aggregator.weighted_sum_batch(results, weights, n)
        return [a.total_score for a in agg]

    def _build_records(
        self,
        smiles: list[str],
        final_scores: list[float],
        constraint_results: dict[str, ConstraintResult],
        round_num: int,
    ) -> list[dict[str, Any]]:
        """Build molecule record dicts from scoring results."""
        molecules = []
        for i, (smi, score) in enumerate(zip(smiles, final_scores)):
            rec = {"smiles": smi, "score": score, "round": round_num}
            for name, result in constraint_results.items():
                if i < len(result.scores):
                    rec[f"{name}_score"] = result.scores[i]
                if i < len(result.raw_values):
                    rec[f"{name}_raw"] = result.raw_values[i]
            molecules.append(rec)
        return molecules

    # -----------------------------------------------------------------------
    # Misc helpers
    # -----------------------------------------------------------------------

    def _generate_batch(self, n: int) -> list[str]:
        """Generate a batch of valid, canonical SMILES via REINVENT4 sampling."""
        gen_result = self.generator.sample(n)

        seen: set[str] = set()
        valid_smiles: list[str] = []
        for smi in gen_result.smiles:
            if not is_valid(smi):
                continue
            canonical = canonicalize(smi)
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            valid_smiles.append(canonical)

        log.info(
            f"Generated {len(gen_result.smiles)} raw, "
            f"{len(valid_smiles)} valid unique SMILES"
        )
        return valid_smiles

    def _select_diverse(
        self, molecules: list[dict[str, Any]], top_k: int,
    ) -> list[dict[str, Any]]:
        """Greedy diversity selection from scored molecules.

        Molecules must already be sorted by score (desc).
        Accepts a candidate only if its maximum Tanimoto similarity to all
        previously selected molecules is <= (1.0 - diversity_threshold).
        """
        from lip.utils.chem import get_morgan_fp
        from rdkit.Chem import DataStructs

        threshold = self.config.optimization.diversity_threshold
        if threshold <= 0:
            return molecules[:top_k]

        selected: list[dict[str, Any]] = []
        selected_fps: list[Any] = []

        for mol_data in molecules:
            if len(selected) >= top_k:
                break

            fp = get_morgan_fp(mol_data["smiles"])
            if fp is None:
                continue

            if selected_fps:
                max_sim = max(
                    DataStructs.TanimotoSimilarity(fp, sfp) for sfp in selected_fps
                )
                if max_sim > (1.0 - threshold):
                    continue

            selected.append(mol_data)
            selected_fps.append(fp)

        return selected

    def _should_early_stop(self, results: list[RoundResult]) -> bool:
        """Check if early stopping criteria are met."""
        patience = self.config.optimization.early_stop_patience
        max_score = self.config.optimization.max_score

        if not results:
            return False

        if results[-1].best_score >= max_score:
            return True

        if len(results) >= patience:
            recent = results[-patience:]
            scores = [r.best_score for r in recent]
            if max(scores) - min(scores) < self.config.optimization.early_stop_threshold:
                return True

        return False
