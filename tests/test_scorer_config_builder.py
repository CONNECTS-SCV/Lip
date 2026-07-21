"""scorer_config.build_scorer_config 로직 검증 (REINVENT4/torch 불필요)."""

from dataclasses import dataclass, field

from lip.engine.scorer_config import build_scorer_config


@dataclass
class _SC:
    """ScoringComponent stub — 실제 dataclass와 동일 필드."""
    type: str
    name: str = ""
    weight: float = 1.0
    transform: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)


def test_groups_external_process_by_exe_args():
    comps = [
        _SC("QED", "qed", 0.5),
        _SC("MolecularWeight", "mw", 0.5,
            {"type": "double_sigmoid", "low": 250, "high": 550}),
        _SC("ExternalProcess", "docking_score", 0.5,
            {"type": "reverse_sigmoid", "high": -6, "low": -12},
            {"executable": "python", "args": "dock.py --receptor r.pdb"}),
        _SC("ExternalProcess", "interaction_count", 0.3,
            {"type": "sigmoid", "low": 0, "high": 10},
            {"executable": "python", "args": "dock.py --receptor r.pdb"}),
    ]
    cfg = build_scorer_config(comps)

    assert cfg["type"] == "geometric_mean"
    assert cfg["parallel"] == 1
    # QED, MW, ExternalProcess(그룹 1개) = 3 블록
    assert len(cfg["component"]) == 3

    ext = _find(cfg, "ExternalProcess")
    # 동일 exe/args의 docking + interaction이 하나로 그룹핑, endpoint 2개
    assert len(ext["endpoint"]) == 2
    assert ext["params"]["executable"] == "python"
    assert {e["name"] for e in ext["endpoint"]} == {"docking_score", "interaction_count"}


def test_endpoint_transform_and_params():
    comps = [
        _SC("MolecularWeight", "mw", 0.4,
            {"type": "double_sigmoid", "low": 200, "high": 500}),
    ]
    cfg = build_scorer_config(comps)
    mw = _find(cfg, "MolecularWeight")["endpoint"][0]
    assert mw["name"] == "mw"
    assert mw["weight"] == 0.4
    assert mw["transform"]["type"] == "double_sigmoid"


def test_external_process_has_no_endpoint_params():
    # ExternalProcess의 executable/args는 component-level params로만, endpoint엔 없음
    comps = [
        _SC("ExternalProcess", "d", 1.0, {},
            {"executable": "python", "args": "x"}),
    ]
    cfg = build_scorer_config(comps)
    ext = _find(cfg, "ExternalProcess")
    assert "params" not in ext["endpoint"][0]
    assert ext["params"] == {"executable": "python", "args": "x"}


def _find(cfg: dict, comp_type: str) -> dict:
    for block in cfg["component"]:
        if comp_type in block:
            return block[comp_type]
    raise AssertionError(f"{comp_type} not found in {cfg}")
