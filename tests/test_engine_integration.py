"""in-process 엔진 통합 검증: 학습 연속성 + best 스냅샷 + 이어학습.

REINVENT4가 설치돼 있고 `.reinvent` prior를 resolve할 수 있는 환경에서만 실행된다.
그 외 환경에서는 자동 스킵. 실제 학습 스모크런이므로 CPU + 작은 n_steps로 제한.
"""

import numpy as np
import pytest

pytest.importorskip("reinvent")
pytest.importorskip("torch")
pytest.importorskip("rdkit")

from lip.engine.config import EngineConfig
from lip.engine.scorer_config import build_scorer_config
from lip.engine import run_managed_training, snapshot
from lip.engine.assembly import build_context, build_learner
from lip.generator.reinvent import qed_component, mw_component


def _engine_cfg(tmp_path, n_steps=6, patience=5, agent_file=".reinvent"):
    return EngineConfig(
        prior_file=".reinvent",
        agent_file=agent_file,
        device="cpu",
        n_steps=n_steps,
        batch_size=16,
        sigma=128,
        learning_rate=0.0001,
        patience=patience,
        diversity_filter="IdenticalMurckoScaffold",
        inception_memory_size=50,
        inception_sample_size=10,
        inception_smiles_file=None,
        work_dir=str(tmp_path),
    )


def _scorer_cfg():
    # docking 없이 QED+MW만 — 빠르고 오프라인
    return build_scorer_config([qed_component(0.5), mw_component(250, 550, 0.5)])


def test_smoke_learning_and_best_checkpoint(tmp_path):
    """학습이 돌고, best 번들이 정확히 1개 저장되며, is_lip_bundle True."""
    ckpt_dir = tmp_path / "checkpoints"
    result = run_managed_training(
        _engine_cfg(tmp_path, n_steps=6),
        _scorer_cfg(),
        checkpoint_path=str(ckpt_dir / "agent_step.chkpt"),
    )

    assert result["success"] is True
    steps = result["scores_by_step"][1]
    assert len(steps) >= 1

    # best 체크포인트 1개만, lip 번들
    bundles = list(ckpt_dir.glob("agent_step*.chkpt"))
    assert len(bundles) == 1
    assert snapshot.is_lip_bundle(str(bundles[0]))

    # 파일명 step이 per-step mean_score argmax와 일치
    best_step = int(bundles[0].stem.replace("agent_step", ""))
    means = [s["mean_score"] for s in steps]
    assert best_step == (int(np.argmax(means)) + 1)


def test_continuity_optimizer_df_inception(tmp_path):
    """step을 넘어 optimizer id 불변 / DF·inception 성장(상태 유지 증거)."""
    ctx = build_context(_engine_cfg(tmp_path, n_steps=5), _scorer_cfg())
    learner = build_learner(ctx, _engine_cfg(tmp_path, n_steps=5))

    opt_ids, df_sizes, inc_sizes = [], [], []

    def _probe(_sm):
        opt_ids.append(id(learner.reward_nlls._optimizer))
        df_sizes.append(len(learner._state.diversity_filter.smiles_memory))
        inc_sizes.append(len(learner.inception) if learner.inception else 0)

    learner.run(patience=99, on_step=_probe, snapshot_best=lambda *_: None)

    assert len(set(opt_ids)) == 1, "optimizer must be the same object across steps"
    assert df_sizes == sorted(df_sizes), "diversity filter memory must be monotonic"
    # Adam 모멘텀이 step1 후 채워짐(연속 학습 증거)
    assert len(learner.reward_nlls._optimizer.state) > 0


def test_resume_restores_full_state(tmp_path):
    """best 번들로 이어학습 시 Adam 상태가 복원된다(가중치만이 아님)."""
    ckpt_dir = tmp_path / "checkpoints"
    first = run_managed_training(
        _engine_cfg(tmp_path, n_steps=4),
        _scorer_cfg(),
        checkpoint_path=str(ckpt_dir / "agent_step.chkpt"),
    )
    bundle = first["checkpoint_path"]
    assert snapshot.is_lip_bundle(bundle)

    # 번들을 agent_file로 지정해 context 재조립 → Adam state가 비어있지 않아야 함
    ctx = build_context(
        _engine_cfg(tmp_path / "resume", n_steps=3, agent_file=bundle),
        _scorer_cfg(),
    )
    assert len(ctx.reward_strategy._optimizer.state) > 0, \
        "resumed optimizer must carry momentum (full-state restore)"
