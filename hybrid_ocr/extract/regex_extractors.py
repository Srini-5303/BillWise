from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from app.schemas import ExtractedField, OCRToken


DATE_PATTERNS = [
    r"\d{1,2}/\d{1,2}/\d{4}",
    r"\d{1,2}/\d{1,2}/\d{2}",
    r"\d{1,2}-\d{1,2}-\d{4}",
    r"\d{1,2}-\d{1,2}-\d{2}",
]

TIME_PATTERNS = [
    r"\d{1,2}:\d{2}\s?(?:AM|PM)",
    r"\d{1,2}:\d{2}",
    r"\b(?:AM|PM)\b",
]

MONEY_PATTERN = r"\$?\d+\.\d{2}"

PAYMENT_KEYWORDS = [
    "VISA",
    "MASTERCARD",
    "MASTERCARD",
    "MC",
    "AMEX",
    "AMERICAN EXPRESS",
    "DISCOVER",
    "DEBIT",
    "CREDIT",
    "EBT",
    "SNAP",
    "CASH",
    "APPLE PAY",
    "GOOGLE PAY",
]

CARD_PATTERNS = [
    r"(?:\*{2,}|X{2,}|x{2,})\s*\d{4}",
    r"(?:ENDING IN|ENDING|END|CARD|ACCT|ACCOUNT)\D{0,12}(\d{4})",
    r"(?:VISA|MASTERCARD|MC|AMEX|DISCOVER|DEBIT|CREDIT)\D{0,12}(\d{4})",
]


def _looks_like_tax_or_rate(text: str) -> bool:
    upper = text.upper()
    return (
        "%" in upper
        or "TAX" in upper
        or "SAVINGS" in upper
        or "SAVED" in upper
        or re.search(r"\d+\.\d+% ?", upper) is not None
    )


def _has_card_context(text: str) -> bool:
    upper = text.upper()
    card_keywords = [
        "VISA", "MASTERCARD", "MC", "AMEX", "DISCOVER",
        "DEBIT", "CREDIT", "CARD", "ACCT", "ACCOUNT",
        "ENDING", "END IN", "END"
    ]
    return any(k in upper for k in card_keywords) or bool(re.search(r"(\*{2,}|X{2,}|x{2,})\s*\d{4}", text))


def _find_first_match(text: str, patterns: List[str]) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def _find_all_money(text: str) -> List[str]:
    return re.findall(MONEY_PATTERN, text)


def _normalize_money(value: str) -> str:
    match = re.search(MONEY_PATTERN, value)
    if not match:
        return value.strip()
    money = match.group(0)
    if not money.startswith("$"):
        money = f"${money}"
    return money


def _clone_with_new_value(field: ExtractedField, new_value: str, confidence_boost: float = 0.0) -> ExtractedField:
    updated = field.model_copy(deep=True)
    updated.value = new_value
    updated.confidence = min(0.999, updated.confidence + confidence_boost)
    return updated


def clean_field_value(field: ExtractedField) -> ExtractedField:
    name = field.field_name
    value = field.value.strip()

    if name == "date":
        date_match = _find_first_match(value, DATE_PATTERNS)
        if date_match:
            return _clone_with_new_value(field, date_match, confidence_boost=0.02)

    if name == "time":
        time_match = _find_first_match(value, TIME_PATTERNS)
        if time_match:
            return _clone_with_new_value(field, time_match, confidence_boost=0.02)

    if name in {"subtotal", "tax", "tips", "total"}:
        monies = _find_all_money(value)
        if monies:
            cleaned = _normalize_money(monies[0])
            return _clone_with_new_value(field, cleaned, confidence_boost=0.02)

    return field


def choose_best_field(fields: List[ExtractedField], prefer_bottom: bool = False) -> Optional[ExtractedField]:
    if not fields:
        return None

    def score(field: ExtractedField) -> float:
        base = field.confidence
        if prefer_bottom:
            base += field.bbox[1] / 10000.0
        return base

    return max(fields, key=score)


def consolidate_fields(fields: List[ExtractedField]) -> List[ExtractedField]:
    original_grouped: Dict[str, List[ExtractedField]] = {}
    for field in fields:
        original_grouped.setdefault(field.field_name, []).append(field)

    cleaned = [clean_field_value(field) for field in fields]

    grouped: Dict[str, List[ExtractedField]] = {}
    for field in cleaned:
        grouped.setdefault(field.field_name, []).append(field)

    final_fields: List[ExtractedField] = []

    for field_name, candidates in grouped.items():
        if field_name in {"subtotal", "tax", "tips", "total"}:
            best = choose_best_field(candidates, prefer_bottom=True)
        else:
            best = choose_best_field(candidates, prefer_bottom=False)

        if best is not None:
            final_fields.append(best)

    final_fields.sort(key=lambda f: (f.bbox[1], f.bbox[0]))

    date_field = next((f for f in final_fields if f.field_name == "date"), None)
    time_field = next((f for f in final_fields if f.field_name == "time"), None)

    if date_field is not None and time_field is not None:
        if time_field.value.upper() in {"AM", "PM"}:
            raw_date_candidates = original_grouped.get("date", [])
            suffix = time_field.value.upper()

            for cand in raw_date_candidates:
                hhmm = re.search(r"(\d{1,2}:\d{2})", cand.value)
                if hhmm:
                    time_field.value = f"{hhmm.group(1)} {suffix}"
                    time_field.confidence = min(0.999, time_field.confidence + 0.02)
                    break

    return final_fields


def _union_bbox(boxes: List[List[int]]) -> List[int]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def _bottom_region_tokens(tokens: List[OCRToken], image_height: int, min_y_ratio: float = 0.72) -> List[OCRToken]:
    cutoff = int(image_height * min_y_ratio)
    return [t for t in tokens if t.bbox[1] >= cutoff]


def _window_tokens(tokens: List[OCRToken], window_size: int = 3) -> List[Tuple[List[OCRToken], str]]:
    ordered = sorted(tokens, key=lambda t: (t.bbox[1], t.bbox[0]))
    windows: List[Tuple[List[OCRToken], str]] = []

    for i in range(len(ordered)):
        for size in range(1, window_size + 1):
            chunk = ordered[i:i + size]
            if len(chunk) != size:
                continue
            text = " ".join(t.text for t in chunk)
            windows.append((chunk, text))

    return windows


def _score_bottomness(bbox: List[int], image_height: int) -> float:
    return min(1.0, bbox[1] / max(1, image_height))


def _make_regex_field(
    field_name: str,
    value: str,
    matched_tokens: List[OCRToken],
    image_height: int,
    pattern_score: float,
) -> ExtractedField:
    bbox = _union_bbox([t.bbox for t in matched_tokens])
    ocr_conf = sum(t.ocr_confidence for t in matched_tokens) / len(matched_tokens)
    bottom_bonus = 0.05 * _score_bottomness(bbox, image_height)
    confidence = min(0.999, 0.75 * pattern_score + 0.25 * ocr_conf + bottom_bonus)

    return ExtractedField(
        field_name=field_name,
        value=value,
        confidence=confidence,
        bbox=bbox,
        source="regex+ocr",
        ocr_confidence=ocr_conf,
        model_confidence=None,
        token_ids=[t.id for t in matched_tokens],
    )


def extract_payment_fields_from_ocr(tokens: List[OCRToken], image_height: int) -> List[ExtractedField]:
    bottom_tokens = _bottom_region_tokens(tokens, image_height=image_height, min_y_ratio=0.72)
    windows = _window_tokens(bottom_tokens, window_size=3)

    payment_candidates: List[ExtractedField] = []
    card_candidates: List[ExtractedField] = []

    for chunk_tokens, text in windows:
        upper_text = text.upper()

        for keyword in PAYMENT_KEYWORDS:
            if keyword in upper_text:
                payment_candidates.append(
                    _make_regex_field(
                        field_name="payment_method",
                        value=keyword,
                        matched_tokens=chunk_tokens,
                        image_height=image_height,
                        pattern_score=0.95,
                    )
                )
                break

        for pattern in CARD_PATTERNS:
            if _looks_like_tax_or_rate(text):
                continue

            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue

            if not _has_card_context(text):
                continue

            digits_match = re.search(r"(\d{4})", match.group(0))
            if digits_match:
                last4 = digits_match.group(1)
                card_candidates.append(
                    _make_regex_field(
                        field_name="card_last4",
                        value=last4,
                        matched_tokens=chunk_tokens,
                        image_height=image_height,
                        pattern_score=0.95,
                    )
                )
                break

    final_fields: List[ExtractedField] = []

    if payment_candidates:
        best_payment = max(payment_candidates, key=lambda f: f.confidence)
        final_fields.append(best_payment)

    if card_candidates:
        best_card = max(card_candidates, key=lambda f: f.confidence)
        final_fields.append(best_card)

    return final_fields
