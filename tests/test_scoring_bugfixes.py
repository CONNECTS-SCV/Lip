"""계산식 버그 수정 검증 (rdkit/torch 불필요한 순수 로직 부분).

- _parse_energy: 파싱 성공/실패 구분 (실패 시 None)
- _docking_args: 공백 포함 경로가 shlex.split로 복원 가능하게 quote되는가
- property_range._score_mol: min만/max만/양쪽/역전 케이스
"""

import shlex

from lip.scoring.docking import UniDockScorer


# ---------------------------------------------------------------------------
# [2] _parse_energy 실패 sentinel
# ---------------------------------------------------------------------------

def test_parse_energy_valid():
    pose = "REMARK VINA RESULT:    -8.5   0.0   0.0\nATOM ...\n"
    assert UniDockScorer._parse_energy(pose) == -8.5


def test_parse_energy_missing_remark_returns_none():
    pose = "ATOM  1  C  LIG  1  0.0 0.0 0.0\nENDMDL\n"
    assert UniDockScorer._parse_energy(pose) is None


def test_parse_energy_malformed_returns_none():
    # REMARK는 있으나 필드가 부족/비수치 → None (예전엔 0.0으로 삼켜졌음)
    assert UniDockScorer._parse_energy("REMARK VINA RESULT: xx") is None
    assert UniDockScorer._parse_energy("REMARK VINA RESULT: a b c") is None


def test_parse_energy_zero_is_valid_not_failure():
    # 진짜 0.0 결합은 유효값으로 반환되어야 한다(실패 None과 구분)
    pose = "REMARK VINA RESULT:    0.0   0.0   0.0\n"
    assert UniDockScorer._parse_energy(pose) == 0.0


# ---------------------------------------------------------------------------
# [1] 도킹 args 공백 이스케이프
# ---------------------------------------------------------------------------

def test_docking_args_quotes_receptor_with_spaces():
    from lip.config import LipConfig
    from lip.component_builder import ComponentBuilder

    cfg = LipConfig()
    cfg.receptor_pdb = r"C:\Users\My Documents\rec eptor.pdb"  # 공백 2군데
    cfg.pocket_center = [1.0, 2.0, 3.0]
    cfg.docking.enabled = True
    cfg.docking.analyze_interactions = False

    builder = ComponentBuilder(cfg)
    args = builder._docking_args()
    assert args is not None

    # REINVENT ExternalProcess와 동일하게 shlex.split → --receptor의 값이 온전히 복원
    tokens = shlex.split(args)
    ri = tokens.index("--receptor")
    assert tokens[ri + 1] == cfg.receptor_pdb, tokens


# ---------------------------------------------------------------------------
# [3] property_range max-only / min-only / 역전
# ---------------------------------------------------------------------------

def _score(value, *, mn=None, mx=None):
    """_score_mol을 rdkit 없이 호출: _func가 상수 반환하도록 인스턴스를 조립."""
    from lip.constraints.property_range import PropertyRangeConstraint

    params = {"property": "molecular_weight"}
    if mn is not None:
        params["min"] = mn
    if mx is not None:
        params["max"] = mx
    c = PropertyRangeConstraint(1.0, params)
    c._func = lambda mol: value  # mol 무시하고 고정값
    return c._score_mol(object(), "X")[0]


def test_property_range_within_bounds():
    assert _score(300, mn=250, mx=550) == 1.0


def test_property_range_max_only_violation_is_penalized():
    # max만 설정. 예전 버그: 위반해도 1.0. 이제는 1.0보다 작아야 함.
    s = _score(2000, mx=500)  # 500 상한을 크게 초과
    assert s < 1.0, f"max-only violation must be penalized, got {s}"


def test_property_range_min_only_violation_is_penalized():
    s = _score(10, mn=250)  # 250 하한 미달
    assert s < 1.0, f"min-only violation must be penalized, got {s}"


def test_property_range_inverted_bounds_no_crash():
    # min > max 역전 — 음수 range로 score>1.0이 되면 안 됨
    s = _score(300, mn=500, mx=100)
    assert 0.0 <= s <= 1.0, f"inverted bounds must stay in [0,1], got {s}"
