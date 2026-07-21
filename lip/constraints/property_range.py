"""Property range constraint - MW, LogP, TPSA, HBD, HBA, etc."""

from __future__ import annotations

import math
from typing import Any

from lip.constraints.base import BaseConstraint, ConstraintResult
from lip.constraints.registry import register
from lip.utils.chem import PROPERTY_FUNCTIONS


@register("property_range")
class PropertyRangeConstraint(BaseConstraint):
    """Score molecules based on whether a property falls within [min, max].

    Params:
        property (str): Property name (e.g. "molecular_weight", "logp").
        min (float): Minimum value (default: -inf).
        max (float): Maximum value (default: +inf).
        margin (float): Fractional margin for soft scoring (default: 0.1).
    """

    def __init__(self, weight: float = 1.0, params: dict | None = None):
        super().__init__(weight, params)
        prop_name = self.params.get("property", "molecular_weight")
        self._func = PROPERTY_FUNCTIONS.get(prop_name)
        self._lo = self.params.get("min", -math.inf)
        self._hi = self.params.get("max", math.inf)
        self._margin = self.params.get("margin", 0.1)

    def score(self, smiles_list: list[str]) -> ConstraintResult:
        if self._func is None:
            return ConstraintResult(
                scores=[0.0] * len(smiles_list),
                passed=[False] * len(smiles_list),
            )
        return super().score(smiles_list)

    # 한쪽 바운드만 설정된 경우(min만/max만) soft-scoring에 쓸 기본 스케일.
    # 예전에는 max=inf일 때 range_size = inf - lo = inf가 되어 dist≈0 → 위반해도
    # score=1.0이 되는 버그가 있었다. 유한 range가 없을 때 이 값으로 대체한다.
    _DEFAULT_SCALE = 100.0

    def _score_mol(self, mol: Any, smi: str) -> tuple[float, bool, Any]:
        value = float(self._func(mol))
        if self._lo <= value <= self._hi:
            return 1.0, True, value

        # 유효한 soft-scoring 스케일 계산.
        # - 양쪽 유한: 실제 범위 폭
        # - 한쪽만 유한: 위반한 바운드의 절댓값 기반(0이면 기본 스케일) — inf가 range로
        #   새어들어가 dist가 0이 되는 것을 막는다
        # - 스케일이 0/음수면(min>max 역전 포함) 기본 스케일로 폴백
        lo_fin = math.isfinite(self._lo)
        hi_fin = math.isfinite(self._hi)
        if lo_fin and hi_fin:
            range_size = self._hi - self._lo
        elif hi_fin:  # max만 설정
            range_size = abs(self._hi)
        elif lo_fin:  # min만 설정
            range_size = abs(self._lo)
        else:  # 양쪽 무한대 — 사실상 제약 없음
            range_size = self._DEFAULT_SCALE
        if range_size <= 0:
            range_size = self._DEFAULT_SCALE

        if value < self._lo:
            dist = (self._lo - value) / (range_size * self._margin + 1e-8)
        else:
            dist = (value - self._hi) / (range_size * self._margin + 1e-8)
        return max(0.0, 1.0 - dist), False, value

    @classmethod
    def get_ui_schema(cls) -> dict:
        return {
            "property": {"type": "select", "options": list(PROPERTY_FUNCTIONS.keys())},
            "min": {"type": "float", "default": 0},
            "max": {"type": "float", "default": 500},
            "margin": {"type": "float", "default": 0.1},
        }
