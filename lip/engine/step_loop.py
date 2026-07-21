"""BestTrackingReinventLearning — step 루프를 lip이 직접 소유한다.

REINVENT4 `Learning.optimize()`는 내부에서 `range(max_steps)` 루프를 끝까지 돌아버려
step 사이에 끼어들 수 없다(그리고 재진입 시 timing/reporter를 리셋). 그래서 그
per-step 본체(learning.py:133-188: sample→score→DF.update_score→update→report)를
여기서 1 step씩 직접 실행하고, step 사이에 best 추적·스냅샷·patience를 건다.

핵심: optimizer/diversity_filter/inception가 모두 동일한 live 객체로 유지되므로
모멘텀·스캐폴드 메모리·replay buffer가 step을 넘어 누적된다(= 학습 정상화).
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from reinvent.runmodes.RL.reinvent import ReinventLearning
from reinvent.models.model_factory.sample_batch import SmilesState

log = logging.getLogger(__name__)

_TOP_K = 10  # top-N 평균 계산용


class BestTrackingReinventLearning(ReinventLearning):
    """자체 step 루프 + best 추적 + patience early-stop + per-step 메트릭 수집."""

    def run(
        self,
        patience: int,
        on_step: Callable[[dict], None],
        snapshot_best: Callable[[int, float], None],
    ) -> tuple[list[dict], list[dict], int]:
        """step 루프를 돈다.

        Args:
            patience: best mean-score가 이 step 수만큼 갱신 안 되면 종료.
            on_step: 각 step의 step-metric dict를 받는 콜백(진행 표시).
            snapshot_best: 새 best 발견 시 (step, score)로 호출(스냅샷 훅).

        Returns:
            (step_metrics, molecules, best_step)
            step_metrics: lip _parse_rl_csv "steps"와 동일 형태 dict 리스트.
            molecules: lip _parse_rl_csv "molecules"와 동일 형태 dict 리스트.
            best_step: best mean_score를 낸 1-기반 step 번호(없으면 0).
        """
        self.start_time = time.time()
        best_mean = -float("inf")
        best_step = 0
        no_improve = 0
        step_metrics: list[dict] = []
        molecules: list[dict] = []

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

            self._log_continuity(step_no)
            on_step(sm)

            # --- best 추적 + patience ---
            if mean_score > best_mean + 1e-9:
                best_mean = mean_score
                best_step = step_no
                no_improve = 0
                snapshot_best(step_no, mean_score)
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info(
                        "Patience %d reached at step %d (best=%d, %.4f); stopping.",
                        patience, step_no, best_step, best_mean,
                    )
                    break

        if self.tb_reporter:
            self.tb_reporter.flush()
            self.tb_reporter.close()

        return step_metrics, molecules, best_step

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
