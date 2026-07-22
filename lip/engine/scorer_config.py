"""ScoringComponent 리스트 → REINVENT4 Scorer 입력 dict 변환.

기존 subprocess 경로는 ComponentBuilder가 만든 ScoringComponent를 TOML 문자열로
직렬화해 REINVENT4에 넘겼다. in-process 경로에서는 TOML 없이 REINVENT4의
`Scorer(input_config: dict)`가 그대로 받는 dict를 직접 만든다.

REINVENT4 `get_components`(reinvent/scoring/config.py:47-68)가 기대하는 구조:
    {
      "type": "geometric_mean",
      "parallel": 1,
      "component": [
        { "<ComponentType>": {
            "endpoint": [ {name, weight, transform?, params?}, ... ],
            "params": {...}  # component-level (ExternalProcess의 executable/args)
        }},
        ...
      ]
    }

ExternalProcess(docking/interaction)는 동일 (executable, args)끼리 하나의 component
블록으로 묶어 REINVENT4가 subprocess를 한 번만 띄우고 여러 property를 읽게 한다
(기존 TOML 그룹핑과 동일한 최적화).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lip.generator.reinvent import ScoringComponent

log = logging.getLogger(__name__)

# 기본 집계 방식 — 기존 TOML의 [stage.scoring] type = "geometric_mean"와 동일
_AGGREGATION = "geometric_mean"
_EXTERNAL_PROCESS = "ExternalProcess"


def _endpoint_block(comp: "ScoringComponent") -> dict:
    """ScoringComponent 하나를 endpoint dict로 변환."""
    name = comp.name or comp.type
    ep: dict = {"name": name, "weight": comp.weight}
    if comp.transform:
        # lip transform dict는 이미 {"type": ..., ...} 형태라 그대로 사용
        ep["transform"] = dict(comp.transform)
    if comp.type == _EXTERNAL_PROCESS:
        # REINVENT ExternalProcess.Parameters는 endpoint마다 executable/args/property를
        # List[str]로 요구한다(comp_external_process.py:39-41). get_components가 각
        # endpoint params를 collect_params로 리스트화하므로, 각 endpoint에 셋 다 넣는다.
        # property는 CLI payload에서 읽을 점수 키 = 이 endpoint의 name.
        ep["params"] = {
            "executable": comp.params.get("executable", ""),
            "args": comp.params.get("args", ""),
            "property": name,
        }
    elif comp.params:
        ep["params"] = dict(comp.params)
    return ep


def build_scorer_config(components: list["ScoringComponent"]) -> dict:
    """ScoringComponent 리스트 → Scorer 입력 dict.

    Args:
        components: ComponentBuilder.build_all()이 만든 ScoringComponent 리스트.

    Returns:
        REINVENT4 Scorer(input_config)가 받는 dict.
    """
    comp_blocks: list[dict] = []

    # ExternalProcess는 (executable, args)로 그룹핑, 나머지는 개별 블록
    ext_groups: dict[tuple, list["ScoringComponent"]] = {}

    for comp in components:
        if comp.type == _EXTERNAL_PROCESS:
            key = (comp.params.get("executable", ""), comp.params.get("args", ""))
            ext_groups.setdefault(key, []).append(comp)
        else:
            comp_blocks.append(
                {comp.type: {"endpoint": [_endpoint_block(comp)]}}
            )

    for (_executable, _args), comps in ext_groups.items():
        # executable/args/property는 각 endpoint params에 들어간다(_endpoint_block).
        # component-level params는 두지 않는다(중복 방지).
        comp_blocks.append(
            {
                _EXTERNAL_PROCESS: {
                    "endpoint": [_endpoint_block(c) for c in comps],
                }
            }
        )

    config = {
        "type": _AGGREGATION,
        "parallel": 1,
        "component": comp_blocks,
    }
    log.debug(
        "Built in-process scorer config: %d components (%d ExternalProcess groups)",
        len(comp_blocks),
        len(ext_groups),
    )
    return config
