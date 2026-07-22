"""lip in-process RL 학습 엔진.

기존 subprocess-per-step 방식(학습 상태가 매 step 리셋되어 학습이 망가짐)을
대체한다. 한 프로세스에서 REINVENT4 RL building block을 조립해 연속 학습하며,
best step에서 전체 학습 상태(가중치+optimizer+DF+inception)를 스냅샷해 하나의
번들로 저장한다. 이어학습은 그 번들을 로드해 모멘텀까지 그대로 이어간다.

공개 API:
    run_managed_training(...) -> dict   # ReinventWrapper.run_staged_learning 대체
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from lip.engine.config import EngineConfig
from lip.engine.assembly import build_context, build_learner
from lip.engine import snapshot

log = logging.getLogger(__name__)

__all__ = ["EngineConfig", "run_managed_training"]


def run_managed_training(
    engine_cfg: EngineConfig,
    scorer_config: dict,
    checkpoint_path: str,
    on_step: Callable[[dict], None] | None = None,
    tb_logdir: str | None = None,
) -> dict:
    """in-process managed 학습을 실행하고 best 번들을 저장한다.

    Args:
        engine_cfg: EngineConfig(모델 경로·학습 파라미터·patience 등).
        scorer_config: build_scorer_config()가 만든 Scorer 입력 dict.
        checkpoint_path: best 번들을 저장할 경로(예: checkpoints/agent_step{best}.chkpt).
                         실제 파일명은 best_step으로 확정되므로 디렉토리만 쓰인다.
        on_step: 각 step 진행 콜백(선택).
        tb_logdir: TensorBoard 로그 디렉토리(선택).

    Returns:
        ReinventWrapper.run_staged_learning과 동일한 계약의 dict:
        {success, checkpoint_path, output_dir, scores_by_step, molecules_by_stage}
    """
    on_step = on_step or (lambda _sm: None)

    # 1) REINVENT4 객체 조립(resume면 완전 상태 복원 포함)
    ctx = build_context(engine_cfg, scorer_config)
    learner = build_learner(ctx, engine_cfg, tb_logdir=tb_logdir)

    # 2) best 스냅샷 훅 — 새 best마다 in-memory deepcopy 보관
    best = {"snapshot": None, "step": 0, "score": -float("inf")}

    def snapshot_best(step_no: int, score: float) -> None:
        best["snapshot"] = snapshot.take_snapshot(ctx)
        best["step"] = step_no
        best["score"] = score
        log.debug("New best at step %d: %.4f (snapshot taken)", step_no, score)

    # 3) step 루프 실행. 학습 진단 CSV는 output_dir(=checkpoints의 상위)에 저장.
    diagnostics_path = str(Path(checkpoint_path).parent.parent / "training_diagnostics.csv")
    step_metrics, molecules, best_step = learner.run(
        patience=engine_cfg.patience,
        on_step=on_step,
        snapshot_best=snapshot_best,
        diagnostics_path=diagnostics_path,
    )

    # 4) best가 없으면(유효 step 0) 오해를 부르는 산출 대신 실패 보고
    if best["snapshot"] is None or best_step == 0:
        return {
            "success": False,
            "error": "No valid training step produced a best snapshot.",
            "checkpoint_path": None,
            "output_dir": str(Path(checkpoint_path).parent),
            "scores_by_step": {1: step_metrics},
            "molecules_by_stage": {1: molecules},
        }

    # 5) best 번들 저장 — 파일명은 best_step으로 확정(기존 glob 호환)
    ckpt_dir = Path(checkpoint_path).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path = ckpt_dir / f"agent_step{best_step}.chkpt"
    snapshot.save_bundle(best["snapshot"], str(final_path))

    log.info(
        "Managed training done: %d steps, best step=%d (%.4f), checkpoint=%s",
        len(step_metrics), best_step, best["score"], final_path,
    )

    return {
        "success": True,
        "checkpoint_path": str(final_path),
        "output_dir": str(ckpt_dir.parent),
        "scores_by_step": {1: step_metrics},
        "molecules_by_stage": {1: molecules},
    }
