from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.canonical import CanonicalReceipt


NULL_LIKE_STRINGS = {"", "null", "none", "n/a", "na", "not found", "unknown"}


def _sanitize_null_like(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in NULL_LIKE_STRINGS:
            return None
        return cleaned

    if isinstance(value, list):
        return [_sanitize_null_like(v) for v in value]

    if isinstance(value, dict):
        return {k: _sanitize_null_like(v) for k, v in value.items()}

    return value


def load_gold_receipt(path: str | Path) -> CanonicalReceipt:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = _sanitize_null_like(data)
    return CanonicalReceipt.model_validate(data)
