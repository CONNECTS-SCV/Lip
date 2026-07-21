"""best step의 전체 학습 상태 스냅샷 / 복원.

REINVENT4 표준 `.chkpt`는 네트워크 가중치만 담아서, 이걸로 이어학습하면
Adam optimizer 모멘텀·diversity filter 메모리·inception replay buffer가 전부
리셋된다(= 현재 lip의 학습 손상 원인). 그래서 lip 엔진은 best step에서
"학습 상태 합본"을 스냅샷해 하나의 번들로 저장한다:

    - agent 가중치      : ModelAdapter.get_save_dict()  (public, network state_dict 포함)
    - Adam optimizer    : RLReward._optimizer.state_dict()  (torch 표준)
    - diversity filter  : deepcopy (BucketCounter + set — 순수 파이썬 자료구조)
    - inception buffer  : deepcopy (storage list + set — 단, runtime 참조는 strip)

파일 포맷은 REINVENT의 model save-dict가 아니라 {BUNDLE_KEY: {...}} 형태이므로,
이어학습 시 create_adapter에 바로 넘기면 check_metadata에서 실패한다. 따라서
로드 전 is_lip_bundle()로 감지하고, agent_save_dict만 떼어 임시 .model로 쓴 뒤
create_adapter로 조립하고, 나머지 상태를 restore_into로 덮어쓴다.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lip.engine.assembly import EngineContext

log = logging.getLogger(__name__)

BUNDLE_VERSION = 1
BUNDLE_KEY = "lip_rl_bundle"


def _strip_inception_runtime_refs(inception):
    """inception deepcopy에서 pickle 곤란한 runtime 참조(scoring_function/prior)를 제거.

    scoring_function은 RDKit/subprocess 핸들을 들고 있어 직렬화가 위험하고, prior는
    복원 시점의 live 객체로 다시 바인딩하는 게 맞다. 둘 다 None으로 비우고, 복원
    후 inception.update(scorer)로 재바인딩한다.
    """
    inception.scoring_function = None
    inception.prior = None
    return inception


def take_snapshot(ctx: "EngineContext") -> dict:
    """새 best 시점의 in-memory 스냅샷. 이후 step이 변형하지 못하도록 deepcopy한다."""
    inception_copy = None
    if ctx.inception is not None:
        inception_copy = _strip_inception_runtime_refs(copy.deepcopy(ctx.inception))

    return {
        "version": BUNDLE_VERSION,
        "agent_save_dict": copy.deepcopy(ctx.agent.get_save_dict()),
        "optimizer_state": copy.deepcopy(ctx.reward_strategy._optimizer.state_dict()),
        "diversity_filter": copy.deepcopy(ctx.diversity_filter),
        "inception": inception_copy,
    }


def save_bundle(snapshot: dict, path: str) -> None:
    """best 스냅샷을 composite 번들로 디스크에 저장(REINVENT .chkpt가 아님)."""
    torch.save({BUNDLE_KEY: snapshot}, path)
    log.info("Saved lip best-checkpoint bundle to %s", path)


def is_lip_bundle(path: str) -> bool:
    """주어진 파일이 lip composite 번들인지 감지."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 — 여기서 삼키지 않고 False로 판정만, 호출부가 로깅
        log.debug("Not a loadable torch file (%s): %s", path, e)
        return False
    return isinstance(obj, dict) and BUNDLE_KEY in obj


def extract_agent_save_dict(path: str) -> dict:
    """번들에서 agent_save_dict만 떼어낸다(create_adapter에 넘길 임시 .model 용)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    snap = obj[BUNDLE_KEY]
    _check_version(snap)
    return snap["agent_save_dict"]


def restore_into(ctx: "EngineContext", path: str) -> None:
    """번들의 optimizer/diversity_filter/inception 상태를 live context에 복원.

    가중치는 create_adapter가 이미 로드했다고 가정하고, 여기서는 Adam 모멘텀·DF·
    inception을 덮어쓴다. inception은 scoring_function을 strip한 채 저장했으므로
    live scorer로 재바인딩한다.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    snap = obj[BUNDLE_KEY]
    _check_version(snap)

    # 가중치도 명시적으로 재로드(임시 .model 경로를 안 거친 경우 대비)
    ctx.agent.network.load_state_dict(snap["agent_save_dict"]["network"])
    ctx.reward_strategy._optimizer.load_state_dict(snap["optimizer_state"])

    ctx.diversity_filter = snap["diversity_filter"]
    ctx.state.diversity_filter = ctx.diversity_filter  # ModelState 동기화

    if snap["inception"] is not None and ctx.inception is not None:
        restored = snap["inception"]
        restored.prior = ctx.inception.prior       # live prior 재바인딩
        ctx.inception = restored
        ctx.inception.update(ctx.scorer)           # live scorer 재바인딩
    log.info("Restored full learning state (optimizer + DF + inception) from %s", path)


def _check_version(snap: dict) -> None:
    version = snap.get("version")
    if version != BUNDLE_VERSION:
        raise RuntimeError(
            f"Unsupported lip bundle version {version} (expected {BUNDLE_VERSION})"
        )
