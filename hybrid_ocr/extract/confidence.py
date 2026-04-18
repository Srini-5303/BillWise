from __future__ import annotations

from typing import Iterable, List


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def final_field_confidence(model_scores: List[float], ocr_scores: List[float]) -> float:
    model_conf = safe_mean(model_scores)
    ocr_conf = safe_mean(ocr_scores)
    return 0.65 * model_conf + 0.35 * ocr_conf
