from __future__ import annotations

import re
from typing import Any

from paddleocr import PaddleOCR

from billwise.preprocessing.schemas import VariantScore


KEYWORDS = ["total", "subtotal", "tax", "visa", "mastercard", "debit", "credit"]

_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = PaddleOCR(use_angle_cls=False, lang="en")
    return _OCR


def _extract_texts_and_confidences(result: Any) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    confidences: list[float] = []

    if result is None:
        return texts, confidences

    # Newer PaddleOCR pipeline often returns iterable of dicts
    if isinstance(result, list):
        for page in result:
            if isinstance(page, dict):
                rec_texts = page.get("rec_texts") or page.get("texts") or []
                rec_scores = page.get("rec_scores") or page.get("scores") or []

                for i, text in enumerate(rec_texts):
                    texts.append(str(text))
                    score = rec_scores[i] if i < len(rec_scores) else 0.0
                    try:
                        confidences.append(float(score))
                    except (TypeError, ValueError):
                        confidences.append(0.0)

            # Older PaddleOCR nested list format
            elif isinstance(page, list):
                for line in page:
                    try:
                        text = str(line[1][0])
                        conf = float(line[1][1])
                        texts.append(text)
                        confidences.append(conf)
                    except Exception:
                        continue

    return texts, confidences


def score_variant(name: str, image_path: str) -> VariantScore:
    ocr = get_ocr()

    # IMPORTANT:
    # Newer PaddleOCR versions route ocr() -> predict(), and predict()
    # does not accept cls=False in this environment.
    result = ocr.ocr(image_path)

    texts, confidences = _extract_texts_and_confidences(result)

    joined = " ".join(texts).lower()
    keyword_hits = sum(1 for kw in KEYWORDS if kw in joined)

    money_hits = len(re.findall(r"\d+\.\d{2}", joined))
    token_count = sum(len(t.split()) for t in texts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    overall = (
        0.45 * avg_conf +
        0.20 * min(len(texts) / 25, 1.0) +
        0.15 * min(token_count / 80, 1.0) +
        0.10 * min(keyword_hits / 4, 1.0) +
        0.10 * min(money_hits / 6, 1.0)
    )

    return VariantScore(
        name=name,
        path=image_path,
        ocr_line_count=len(texts),
        ocr_token_count=token_count,
        avg_confidence=avg_conf,
        keyword_hits=keyword_hits,
        overall_score=overall,
    )