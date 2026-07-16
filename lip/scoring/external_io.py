"""Input helpers for REINVENT4 ExternalProcess scoring scripts."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from typing import Any


def parse_external_smiles(text: str) -> list[str]:
    """Parse SMILES batch input from REINVENT ExternalProcess stdin.

    REINVENT deployments differ in how they pass batches to external scorers:
    some send one SMILES per line, while others send a JSON or Python repr list.
    Accept all of those forms so scoring payload lengths match the batch size.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    parsed = _parse_structured(raw)
    if parsed is not None:
        return parsed

    return [line.strip() for line in raw.splitlines() if line.strip()]


def _parse_structured(raw: str) -> list[str] | None:
    for parser in (json.loads, ast.literal_eval):
        try:
            data = parser(raw)
        except Exception:
            continue
        smiles = _extract_smiles(data)
        if smiles is not None:
            return smiles
    return None


def _extract_smiles(data: Any) -> list[str] | None:
    if isinstance(data, str):
        return [data] if data.strip() else []

    if isinstance(data, dict):
        for key in ("smiles", "SMILES"):
            smiles = _extract_smiles(data.get(key))
            if smiles is not None:
                return smiles

        payload = data.get("payload")
        if isinstance(payload, dict):
            for key in ("smiles", "SMILES"):
                smiles = _extract_smiles(payload.get(key))
                if smiles is not None:
                    return smiles
        elif payload is not None:
            return _extract_smiles(payload)
        return None

    if isinstance(data, Iterable):
        out: list[str] = []
        for item in data:
            if isinstance(item, str):
                smi = item.strip()
            elif isinstance(item, dict):
                smi = str(item.get("smiles") or item.get("SMILES") or "").strip()
            else:
                smi = ""
            if smi:
                out.append(smi)
        return out

    return None
