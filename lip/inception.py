"""Inception buffer for RL optimization — dedup + retention policy."""

from __future__ import annotations

import random
from typing import Any


class InceptionBuffer:
    """RL 학습용 inception 메모리 버퍼.

    SMILES 기준 dedup, retention 정책(top/random) 관리.
    """

    def __init__(self, memory_size: int, retention: str = "top"):
        self.memory_size = memory_size
        self.retention = retention
        self._buffer: dict[str, dict[str, Any]] = {}

    def update(self, molecules: list[dict[str, Any]]) -> int:
        """새 분자 추가 (dedup + retention 적용). 추가된 수 반환."""
        added = 0
        for mol in molecules:
            smi = mol["smiles"]
            if smi not in self._buffer or mol["score"] > self._buffer[smi]["score"]:
                self._buffer[smi] = {"smiles": smi, "score": mol["score"]}
                added += 1
        self._apply_retention()
        return added

    def _apply_retention(self):
        if len(self._buffer) <= self.memory_size:
            return
        entries = list(self._buffer.values())
        if self.retention == "random":
            kept = random.sample(entries, self.memory_size)
        else:  # "top"
            entries.sort(key=lambda e: e["score"], reverse=True)
            kept = entries[:self.memory_size]
        self._buffer = {e["smiles"]: e for e in kept}

    def get_entries(self) -> list[dict[str, Any]]:
        return list(self._buffer.values())

    @property
    def size(self) -> int:
        return len(self._buffer)
