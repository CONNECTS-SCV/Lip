"""BestTrackingReinventLearning — step 루프를 lip이 직접 소유한다.

REINVENT4 `Learning.optimize()`는 내부에서 `range(max_steps)` 루프를 끝까지 돌아버려
step 사이에 끼어들 수 없다(그리고 재진입 시 timing/reporter를 리셋). 그래서 그
per-step 본체(learning.py:133-188: sample→score→DF.update_score→update→report)를
여기서 1 step씩 직접 실행하고, step 사이에 best 추적·스냅샷·patience를 건다.

핵심: optimizer/diversity_filter/inception가 모두 동일한 live 객체로 유지되므로
모멘텀·스캐폴드 메모리·replay buffer가 step을 넘어 누적된다(= 학습 정상화).
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from reinvent.runmodes.RL.reinvent import ReinventLearning
from reinvent.models.model_factory.sample_batch import SmilesState

log = logging.getLogger(__name__)

_TOP_K = 10  # top-N 평균 계산용

# 학습 진단 CSV의 컬럼(순서 고정)
_DIAG_FIELDS = (
    "step", "agent_nll", "prior_nll", "agent_prior_gap", "loss",
    "mean_score", "valid_mean_score", "n_valid", "n",
)


class BestTrackingReinventLearning(ReinventLearning):
    """자체 step 루프 + best 추적 + patience early-stop + per-step 메트릭 수집."""

    def run(
        self,
        patience: int,
        on_step: Callable[[dict], None],
        snapshot_best: Callable[[int, float], None],
        diagnostics_path: str | None = None,
    ) -> tuple[list[dict], list[dict], int]:
        """step 루프를 돈다.

        Args:
            patience: best 유효분자 평균 점수가 이 step 수만큼 갱신 안 되면 종료.
            on_step: 각 step의 step-metric dict를 받는 콜백(진행 표시).
            snapshot_best: 새 best 발견 시 (step, score)로 호출(스냅샷 훅).
            diagnostics_path: 학습 진단 CSV 저장 경로(선택). agent_nll/loss/gap 등 기록.

        Returns:
            (step_metrics, molecules, best_step)
            step_metrics: lip _parse_rl_csv "steps"와 동일 형태 dict 리스트.
            molecules: lip _parse_rl_csv "molecules"와 동일 형태 dict 리스트.
            best_step: best 유효분자 평균을 낸 1-기반 step 번호(없으면 0).
        """
        self.start_time = time.time()
        best_valid_mean = -float("inf")
        best_step = 0
        no_improve = 0
        step_metrics: list[dict] = []
        molecules: list[dict] = []
        diagnostics: list[dict] = []  # step별 학습 진단 지표

        for step in range(self.max_steps):
            # --- learning.py:133-188 per-step 본체 미러링 ---
            self.sampled = self.sampling_model.sample(self.input_smilies)
            self.smiles_memory.update(self.sampled.smilies)
            self.invalid_mask = np.where(
                self.sampled.states == SmilesState.INVALID, False, True
            )
            self.duplicate_mask = np.where(
                self.sampled.states == SmilesState.DUPLICATE, False, True
            )

            results = self.score()  # -> Scorer.__call__ -> ScoreResults

            scaffolds = None
            if self._state.diversity_filter:
                df_mask = np.where(self.invalid_mask, True, False)
                scaffolds = self._state.diversity_filter.update_score(
                    results.total_scores, results.smilies, df_mask
                )

            # update(): 내부에서 optimizer.zero_grad/backward/step 수행.
            # orig_smilies는 inception 재정렬용으로 Reinvent 분기의 sampled.output.
            agent_lls, prior_lls, augmented_nll, loss = self.update(
                results, self.sampled.output
            )

            self._state_info.update(self._state.as_dict())

            step_no = step + 1
            mean_score, sm, mols = self._collect_step_metrics(step_no, results)
            step_metrics.append(sm)
            molecules.extend(mols)

            # reporter 재사용(CSV filename=None이라 파일은 안 쓰고 tb만 옵션 반영)
            self.report(
                step,
                mean_score,
                scaffolds,
                score_results=results,
                agent_lls=agent_lls,
                prior_lls=prior_lls,
                augmented_nll=augmented_nll,
                loss=loss.item(),
            )

            # --- 학습 진단 지표 (agent NLL이 내려가고 gap이 벌어지면 학습 진행) ---
            agent_nll = float((-agent_lls).mean().item()) if agent_lls.numel() else 0.0
            prior_nll = float((-prior_lls).mean().item()) if prior_lls.numel() else 0.0
            diag = {
                "step": step_no,
                "agent_nll": round(agent_nll, 4),
                "prior_nll": round(prior_nll, 4),
                "agent_prior_gap": round(agent_nll - prior_nll, 4),
                "loss": round(float(loss.item()), 4),
                "mean_score": round(mean_score, 4),
                "valid_mean_score": round(sm["valid_mean_score"], 4),
                "n_valid": sm["n_valid"],
                "n": sm["n"],
            }
            diagnostics.append(diag)
            log.info(
                "Step %d train: agent_nll=%.3f gap=%.3f loss=%.4f valid_mean=%.4f",
                step_no, agent_nll, agent_nll - prior_nll,
                float(loss.item()), sm["valid_mean_score"],
            )

            self._log_continuity(step_no)
            on_step(sm)

            # --- best 추적 + patience (유효분자 평균 기준) ---
            # 전체 평균(mean_score)은 invalid=0에 눌려 둔하므로, 학습 신호가 더
            # 잘 드러나는 valid_mean_score로 best/patience를 판단한다. 유효분자가
            # 하나도 없는 step(valid_mean=0)은 개선으로 오판하지 않도록 가드.
            valid_mean = sm["valid_mean_score"]
            if sm["n_valid"] > 0 and valid_mean > best_valid_mean + 1e-9:
                best_valid_mean = valid_mean
                best_step = step_no
                no_improve = 0
                snapshot_best(step_no, valid_mean)
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info(
                        "Patience %d reached at step %d (best=%d, valid_mean=%.4f); stopping.",
                        patience, step_no, best_step, best_valid_mean,
                    )
                    break

        if self.tb_reporter:
            self.tb_reporter.flush()
            self.tb_reporter.close()

        if diagnostics_path and diagnostics:
            self._save_diagnostics(diagnostics, diagnostics_path)

        return step_metrics, molecules, best_step

    @staticmethod
    def _save_diagnostics(rows: list[dict], path: str) -> None:
        """step별 학습 진단 지표를 CSV로 저장(학습이 진짜 진행되는지 확인용)."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(_DIAG_FIELDS))
                writer.writeheader()
                writer.writerows(rows)
            log.info("Training diagnostics saved: %s", path)
        except OSError as e:
            # 진단 저장 실패가 학습 결과를 막지 않도록 로그만 남기고 넘어간다
            log.warning("Failed to save training diagnostics to %s: %s", path, e)

    def _collect_step_metrics(
        self, step_no: int, results
    ) -> tuple[float, dict, list[dict]]:
        """ScoreResults → (mean_score, step-metric dict, molecule dict 리스트).

        lip/generator/reinvent.py `_parse_rl_csv`가 만들던 형태를 객체에서 직접 재현.
        """
        total = results.total_scores
        nan_mask = np.isnan(total)
        finite = total[~nan_mask]
        mean_score = float(finite.mean()) if finite.size else 0.0
        max_score = float(finite.max()) if finite.size else 0.0
        n = int(total.shape[0])

        valid_state_mask = self.sampled.states == SmilesState.VALID

        # 컴포넌트별 transformed/raw 값을 SMILES 인덱스로 정리
        comp_scores_by_idx, raw_by_idx = self._extract_component_values(results, n)

        valid_scores: list[float] = []
        unique_smiles: set[str] = set()
        molecules: list[dict] = []
        for i, smiles in enumerate(results.smilies):
            score = float(total[i]) if not nan_mask[i] else 0.0
            # 유효(valid) & 양수 점수만 — 기존 _parse_rl_csv 기준과 동일
            if valid_state_mask[i] and smiles and score > 0:
                valid_scores.append(score)
                unique_smiles.add(smiles)
                molecules.append(
                    {
                        "smiles": smiles,
                        "total_score": score,
                        "scores": comp_scores_by_idx.get(i, {}),
                        "raw_values": raw_by_idx.get(i, {}),
                        "step": step_no,
                    }
                )

        top_scores = sorted(valid_scores, reverse=True)[:_TOP_K]
        step_metric = {
            "step": step_no,
            "mean_score": mean_score,
            "max_score": max_score,
            "n": n,
            "valid_mean_score": (
                sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            ),
            "top10_mean_score": (
                sum(top_scores) / len(top_scores) if top_scores else 0.0
            ),
            "n_valid": len(valid_scores),
            "n_unique": len(unique_smiles),
        }
        return mean_score, step_metric, molecules

    @staticmethod
    def _extract_component_values(
        results, n: int
    ) -> tuple[dict[int, dict], dict[int, dict]]:
        """TransformResults에서 SMILES 인덱스별 컴포넌트 점수/raw 값을 추출.

        TransformResults: component_names[list], transformed_scores[list of ndarray],
        component_result.scores[list of ndarray](raw). 각 endpoint가 한 슬롯.
        """
        comp_scores: dict[int, dict] = {}
        raw_values: dict[int, dict] = {}
        for tr in results.completed_components:
            names = tr.component_names
            transformed = tr.transformed_scores
            raw = getattr(tr.component_result, "scores", None)
            for slot, name in enumerate(names):
                if slot < len(transformed):
                    arr = transformed[slot]
                    for i in range(min(n, len(arr))):
                        val = arr[i]
                        if val is not None and not _isnan(val):
                            comp_scores.setdefault(i, {})[name] = float(val)
                if raw is not None and slot < len(raw):
                    rarr = raw[slot]
                    for i in range(min(n, len(rarr))):
                        val = rarr[i]
                        if val is not None and not _isnan(val):
                            raw_values.setdefault(i, {})[name] = float(val)
        return comp_scores, raw_values

    def _log_continuity(self, step_no: int) -> None:
        """학습 연속성 추적용 debug 로그(기본 레벨에선 조용).

        optimizer id 불변 / DF 메모리 단조 증가 / inception 성장은 이 로그로 확인.
        """
        if not log.isEnabledFor(logging.DEBUG):
            return
        df_mem = (
            len(self._state.diversity_filter.smiles_memory)
            if self._state.diversity_filter
            else 0
        )
        inc_len = len(self.inception) if self.inception is not None else 0
        opt_state = self.reward_nlls._optimizer.state
        log.debug(
            "[continuity] step=%d opt_id=%s df_mem=%d inception=%d adam_state=%d",
            step_no,
            id(self.reward_nlls._optimizer),
            df_mem,
            inc_len,
            len(opt_state),
        )


def _isnan(val) -> bool:
    try:
        return bool(np.isnan(val))
    except (TypeError, ValueError):
        return False
