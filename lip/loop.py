"""RL optimization loop - multi-cycle managed and manual modes."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lip.config import LipConfig
from lip.component_builder import ComponentBuilder
from lip.constraints.base import ConstraintResult
from lip.constraints.registry import create_constraint
from lip.generator.reinvent import ReinventWrapper, StageConfig
from lip.scoring.aggregator import ScoreAggregator
from lip.scoring.docking import BaseDockingScorer, VinaDockingScorer, create_docking_scorer
from lip.utils.chem import (
    is_valid, canonicalize, check_lipinski, check_pains,
)
from lip.utils.io import (
    save_round_results,
    save_json,
    load_json,
    save_progress_plot,
    save_results_csv,
    save_top_molecules,
)
from lip.utils.math import normalize_score

log = logging.getLogger(__name__)

_PROTEIN_NOT_LOADED = object()


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
    all_avg_score: float | None = None
    top10_avg_score: float | None = None


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
            "timeout": config.optimization.reinvent_timeout,
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

        # Docking scorer (manual mode only — managed uses REINVENT4 ExternalProcess)
        if config.optimization.mode != "managed":
            if config.docking.enabled and config.receptor_pdb:
                center = tuple(config.pocket_center)
                box = (config.docking.box_size,) * 3
                method = config.docking.method
                if method == "auto":
                    from lip.scoring.docking import _unidock_available
                    method = "unidock" if _unidock_available() else "vina"
                    log.info(f"Auto-detected docking method: {method}")
                loop.docking_scorer = create_docking_scorer(
                    method=method,
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
        """Run managed RL as ONE continuous in-process REINVENT4 training loop.

        이전에는 step마다 REINVENT를 별도 subprocess로 실행해 .chkpt를 체이닝했는데,
        .chkpt는 가중치만 담아 매 step optimizer/diversity-filter/inception이 리셋되어
        학습이 망가졌다. 이제는 lip 엔진이 REINVENT4 RL 컴포넌트를 in-process로
        조립해 한 프로세스에서 n_steps를 연속 학습하며(상태 유지), best step에서
        전체 학습상태를 스냅샷해 단일 번들 checkpoints/agent_step{best}.chkpt로 저장한다.
        early-stop patience도 엔진이 처리한다.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_steps = self.config.optimization.n_steps
        components = self.component_builder.build_all(self.constraints)
        chkpt_dir = output_dir / "checkpoints"
        chkpt_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Starting in-process continuous REINVENT4 RL ({n_steps} steps)")

        # 진행 로그: 엔진 step 진행을 실시간으로 보여준다(콜백/RoundResult는 최종
        # _build_managed_round_results가 담당하므로 여기선 로깅만 — 이중 호출 방지).
        def _log_step(sm: dict) -> None:
            log.info(
                "Step %d: mean=%.4f best=%.4f valid=%d/%d",
                sm["step"], sm["mean_score"], sm["max_score"],
                sm["n_valid"], sm["n"],
            )

        self.generator._on_engine_step = _log_step
        self.generator._patience = self.config.optimization.early_stop_patience

        # 단일 stage(max_steps=n_steps)를 1회 호출 → 엔진이 연속 학습 수행
        stage = StageConfig(
            max_steps=n_steps,
            min_steps=1,
            max_score=self.config.optimization.max_score,
            scoring_components=components,
            chkpt_file=str(chkpt_dir / "agent_step.chkpt"),
        )
        result = self.generator.run_staged_learning(
            stages=[stage],
            output_dir=str(output_dir),
        )

        if not result.get("success"):
            raise RuntimeError(
                "In-process managed learning failed: "
                f"{result.get('error', 'unknown')}"
            )

        all_results = self._build_managed_round_results(
            result.get("scores_by_step", {}).get(1, []),
            result.get("molecules_by_stage", {}).get(1, []),
            round_start=1,
            reinvent_stage=1,
        )
        if not all_results:
            raise RuntimeError(
                "In-process managed learning produced no metrics; "
                "refusing to write a misleading progress plot."
            )

        metrics_rows = []
        for rr in all_results:
            if rr.molecules:
                save_round_results(rr.round_num, rr.molecules, str(output_dir))
            metrics_rows.append(
                {
                    "step": rr.round_num,
                    "best_score": rr.best_score,
                    "valid_mean_score": rr.avg_score,
                    "all_mean_score": rr.all_avg_score,
                    "top10_mean_score": rr.top10_avg_score,
                    "n_valid": rr.n_valid,
                    "n_total": rr.n_total,
                    "valid_ratio": rr.n_valid / rr.n_total if rr.n_total else 0.0,
                }
            )
            log.info(
                f"Step {rr.round_num} done: best={rr.best_score:.3f}, "
                f"valid_avg={rr.avg_score:.3f}, all_avg={rr.all_avg_score or 0.0:.3f}, "
                f"valid={rr.n_valid}/{rr.n_total}"
            )

        if metrics_rows:
            save_results_csv(metrics_rows, output_dir / "optimization_metrics.csv")

        self.state.best_score = max(
            (r.best_score for r in all_results), default=0.0,
        )
        self.state.total_molecules = sum(r.n_valid for r in all_results)
        self.state.completed = True
        self.state.save(output_dir / "run_state.json")
        save_progress_plot(all_results, output_dir)
        save_top_molecules(all_results, output_dir)
        return all_results

    def _build_managed_round_results(
        self,
        steps_data: list[dict[str, Any]],
        stage_mols: list[dict[str, Any]],
        round_start: int = 1,
        reinvent_stage: int | None = None,
    ) -> list[RoundResult]:
        """Build per-step results from one REINVENT staged-learning CSV parse."""
        mols_by_step: dict[int, list[dict[str, Any]]] = {}
        for mol in stage_mols:
            step = int(mol.get("step", 0))
            rec = {
                "smiles": mol["smiles"],
                "score": mol["total_score"],
                "reinvent_step": step,
            }
            if reinvent_stage is not None:
                rec["reinvent_stage"] = reinvent_stage
            scores = mol.get("scores", {})
            raw_values = mol.get("raw_values", {})
            for k in scores:
                rec[k] = scores[k]
                if k in raw_values:
                    rec[f"{k}_raw"] = raw_values[k]
            for k in raw_values:
                if k not in scores:
                    rec[f"{k}_raw"] = raw_values[k]
            mols_by_step.setdefault(step, []).append(rec)

        all_results: list[RoundResult] = []
        for offset, step_data in enumerate(steps_data):
            round_num = round_start + offset
            reinvent_step = int(step_data.get("step", offset + 1))
            step_mols = mols_by_step.get(reinvent_step, [])
            valid_avg = step_data.get("valid_mean_score")
            if valid_avg is None:
                valid_avg = (
                    sum(m["score"] for m in step_mols) / len(step_mols)
                    if step_mols else 0.0
                )
            rr = RoundResult(
                round_num=round_num,
                molecules=step_mols,
                best_score=step_data.get("max_score", 0.0),
                avg_score=valid_avg,
                n_valid=step_data.get("n_valid", len(step_mols)),
                n_total=step_data.get("n", self.config.generator.batch_size),
                all_avg_score=step_data.get("mean_score", valid_avg),
                top10_avg_score=step_data.get("top10_mean_score"),
            )
            all_results.append(rr)

            if self._on_round_complete:
                self._on_round_complete(rr)

        if not all_results and stage_mols:
            step_mols = [m for mols in mols_by_step.values() for m in mols]
            best = max((m["score"] for m in step_mols), default=0.0)
            avg = sum(m["score"] for m in step_mols) / len(step_mols)
            all_results.append(
                RoundResult(
                    round_num=round_start,
                    molecules=step_mols,
                    best_score=best,
                    avg_score=avg,
                    n_valid=len(step_mols),
                    n_total=len(step_mols),
                    all_avg_score=avg,
                    top10_avg_score=avg,
                )
            )

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
            self._protein_mol_cache = _PROTEIN_NOT_LOADED

        if self._protein_mol_cache is not _PROTEIN_NOT_LOADED:
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
            self._protein_mol_cache = None
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
        if protein_mol is None:
            # 로드 실패면 interaction 성분 전체가 0이 되어 목적함수가 조용히
            # 비활성화된다. _get_protein_mol이 이미 warning을 남기지만, 여기서도
            # 이 배치 전체가 0점 처리됨을 명시한다.
            log.warning(
                "Protein unavailable; interaction_count is 0 for all %d molecules "
                "in this batch.", len(docking_results),
            )

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

                # Sigmoid normalization
                norm_max = self.config.docking.interaction_norm_max
                midpoint = norm_max / 2.0
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
