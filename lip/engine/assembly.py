"""REINVENT4 RL building block을 in-process로 조립한다.

run_staged_learning.py:56-134의 조립 순서를 미러링하되, subprocess를 띄우지 않고
모든 상태 객체(agent/optimizer/diversity_filter/inception)를 한 프로세스 안에
상주시킨다. 이것이 학습 연속성(모멘텀/DF/inception 유지)을 보장하는 핵심이다.

REINVENT4 소스는 수정하지 않고 import만 한다.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from reinvent.runmodes.create_adapter import create_adapter
from reinvent.runmodes import RL
from reinvent.runmodes.setup_sampler import setup_sampler
from reinvent.runmodes.RL.setup.reward_strategy import setup_reward_strategy
from reinvent.runmodes.RL.setup.diversity_filter import setup_diversity_filter
from reinvent.runmodes.RL.setup.inception import setup_inception
from reinvent.runmodes.RL.data_classes import ModelState
from reinvent.runmodes.utils.helpers import disable_gradients
from reinvent.scoring import Scorer

from lip.engine.config import EngineConfig
from lip.engine import snapshot

log = logging.getLogger(__name__)

# diversity filter 기본값 — 기존 TOML의 bucket_size=25, minscore=0.4와 동일
_DF_BUCKET_SIZE = 25
_DF_MINSCORE = 0.4
_DF_MINSIMILARITY = 0.4
_DF_PENALTY_MULTIPLIER = 0.5


@dataclass
class EngineContext:
    """조립된 REINVENT4 live 객체 묶음. step_loop / snapshot이 공유한다."""

    prior: Any
    agent: Any
    sampler: Any
    reward_strategy: Any
    diversity_filter: Any
    inception: Any
    scorer: Any
    state: ModelState
    model_type: str
    device: torch.device
    rdkit_smiles_flags: dict


def build_context(cfg: EngineConfig, scorer_config: dict) -> EngineContext:
    """EngineConfig + scorer dict → 조립된 EngineContext.

    agent_file이 lip composite 번들이면 완전 상태 resume 경로를 탄다:
    번들에서 가중치만 떼어 임시 .model로 create_adapter에 넘겨 정상 조립한 뒤,
    snapshot.restore_into로 Adam/DF/inception을 덮어쓴다.
    """
    device = torch.device(cfg.device)

    # resume 감지: lip 번들이면 가중치만 임시 파일로 추출해 사용
    resume_bundle_path: str | None = None
    agent_path = cfg.agent_file
    if os.path.exists(cfg.agent_file) and snapshot.is_lip_bundle(cfg.agent_file):
        resume_bundle_path = cfg.agent_file
        agent_path = _extract_bundle_weights_to_tmp(cfg.agent_file, cfg.work_dir)
        log.info("Resume from lip bundle detected; full-state restore will follow.")

    # 1) prior — inference, gradient off (run_staged_learning.py:65-67)
    prior_adapter, _, model_type = create_adapter(cfg.prior_file, "inference", device)
    disable_gradients(prior_adapter)

    # 2) agent — training (Adam이 학습 파라미터를 잡아야 하므로 training 모드)
    agent_adapter, _, agent_model_type = create_adapter(agent_path, "training", device)
    if model_type != agent_model_type:
        raise RuntimeError(
            f"Inconsistent model types: prior is {model_type}, agent is {agent_model_type}"
        )

    # Reinvent(비-transformer) 기준 rdkit flags (run_staged_learning.py:69-77)
    rdkit_smiles_flags = dict(allowTautomers=True)
    rdkit_smiles_flags2: dict = dict()

    # 3) sampler — config는 평범한 dict (setup_sampler가 .get으로 읽음)
    sampler, _ = setup_sampler(
        model_type,
        {"batch_size": cfg.batch_size, "randomize_smiles": True},
        agent_adapter,
    )

    # 4) reward strategy — 내부에서 torch.optim.Adam 생성, RLReward가 보유
    ls_config = SimpleNamespace(type="dap", rate=cfg.learning_rate, sigma=cfg.sigma)
    reward_strategy = setup_reward_strategy(ls_config, agent_adapter)

    # 5) diversity filter (SectionDiversityFilter 형태의 attribute bag)
    df_config = SimpleNamespace(
        type=cfg.diversity_filter,
        bucket_size=_DF_BUCKET_SIZE,
        minscore=_DF_MINSCORE,
        minsimilarity=_DF_MINSIMILARITY,
        penalty_multiplier=_DF_PENALTY_MULTIPLIER,
    )
    diversity_filter = setup_diversity_filter(df_config, rdkit_smiles_flags2)

    # 6) inception (SectionInception 형태)
    inc_config = SimpleNamespace(
        memory_size=cfg.inception_memory_size,
        sample_size=cfg.inception_sample_size,
        smiles_file=cfg.inception_smiles_file or "",
    )
    inception = setup_inception(inc_config, prior_adapter)

    # 7) scorer — TOML 없이 dict 직접 (fail-fast: 여기서 구성 검증)
    scorer = Scorer(scorer_config)

    # 8) inception은 첫 사용 전 scoring_function 바인딩 필요
    #    (run_staged_learning.py:192-193)
    if inception is not None:
        inception.update(scorer)

    state = ModelState(agent_adapter, diversity_filter)

    ctx = EngineContext(
        prior=prior_adapter,
        agent=agent_adapter,
        sampler=sampler,
        reward_strategy=reward_strategy,
        diversity_filter=diversity_filter,
        inception=inception,
        scorer=scorer,
        state=state,
        model_type=model_type,
        device=device,
        rdkit_smiles_flags=rdkit_smiles_flags,
    )

    # resume면 Adam/DF/inception을 번들 상태로 덮어쓴다(완전 상태 이어학습)
    if resume_bundle_path is not None:
        snapshot.restore_into(ctx, resume_bundle_path)

    return ctx


def build_learner(ctx: EngineContext, cfg: EngineConfig, tb_logdir: str | None = None):
    """EngineContext로 BestTrackingReinventLearning 인스턴스를 만든다."""
    # 순환 import 회피 — step_loop가 assembly의 EngineContext를 참조하므로 지연 import
    from lip.engine.step_loop import BestTrackingReinventLearning

    return BestTrackingReinventLearning(
        max_steps=cfg.n_steps,
        stage_no=1,
        prior=ctx.prior,
        state=ctx.state,
        scoring_function=ctx.scorer,
        reward_strategy=ctx.reward_strategy,
        sampling_model=ctx.sampler,
        smilies=None,
        distance_threshold=0,
        rdkit_smiles_flags=ctx.rdkit_smiles_flags,
        inception=ctx.inception,
        responder_config=None,
        tb_logdir=tb_logdir,
        tb_isim=False,
        intrinsic_penalty=None,
    )


def _extract_bundle_weights_to_tmp(bundle_path: str, work_dir: str) -> str:
    """번들에서 agent_save_dict만 떼어 임시 .model 파일로 저장, 경로 반환.

    create_adapter는 REINVENT model save-dict만 로드할 수 있으므로(check_metadata),
    composite 번들을 직접 못 넘긴다. 가중치 부분만 표준 save-dict로 떼어낸다.
    """
    save_dict = snapshot.extract_agent_save_dict(bundle_path)
    os.makedirs(work_dir or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".model", dir=work_dir or None)
    os.close(fd)
    torch.save(save_dict, tmp_path)
    log.debug("Extracted bundle weights to temp model: %s", tmp_path)
    return tmp_path
