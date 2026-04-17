from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Dict, List

from evaluation.canonical import CanonicalReceipt
from evaluation.normalize import (
    normalize_amount,
    normalize_card_last4,
    normalize_date,
    normalize_item_name,
    normalize_payment_method,
    normalize_text,
    normalize_time,
)

CORE_FIELDS = [
    "merchant_name",
    "date",
    "time",
    "subtotal",
    "tax",
    "total",
    "payment_method",
    "card_last4",
    "receipt_number",
]


def _compare_field(field_name: str, pred_value, gold_value) -> float:
    if field_name == "merchant_name":
        return float(normalize_text(pred_value) == normalize_text(gold_value))
    if field_name == "date":
        return float(normalize_date(pred_value) == normalize_date(gold_value))
    if field_name == "time":
        return float(normalize_time(pred_value) == normalize_time(gold_value))
    if field_name in {"subtotal", "tax", "total"}:
        p = normalize_amount(pred_value)
        g = normalize_amount(gold_value)
        if p is None and g is None:
            return 1.0
        if p is None or g is None:
            return 0.0
        return float(abs(p - g) <= 0.01)
    if field_name == "payment_method":
        return float(normalize_payment_method(pred_value) == normalize_payment_method(gold_value))
    if field_name == "card_last4":
        return float(normalize_card_last4(pred_value) == normalize_card_last4(gold_value))
    if field_name == "receipt_number":
        return float(normalize_text(pred_value) == normalize_text(gold_value))
    return 0.0


def _item_name_f1(pred: CanonicalReceipt, gold: CanonicalReceipt) -> Dict[str, float]:
    pred_names = [normalize_item_name(i.name) for i in pred.items if normalize_item_name(i.name)]
    gold_names = [normalize_item_name(i.name) for i in gold.items if normalize_item_name(i.name)]

    pred_counter = Counter(pred_names)
    gold_counter = Counter(gold_names)

    tp = sum((pred_counter & gold_counter).values())
    fp = sum((pred_counter - gold_counter).values())
    fn = sum((gold_counter - pred_counter).values())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "item_name_precision": precision,
        "item_name_recall": recall,
        "item_name_f1": f1,
        "pred_item_count": len(pred_names),
        "gold_item_count": len(gold_names),
    }


def score_receipt(pred: CanonicalReceipt, gold: CanonicalReceipt) -> Dict[str, float]:
    field_scores: Dict[str, float] = {}

    for field_name in CORE_FIELDS:
        pred_val = getattr(pred.fields, field_name, None)
        gold_val = getattr(gold.fields, field_name, None)
        field_scores[field_name] = _compare_field(field_name, pred_val, gold_val)

    core_mean = mean(field_scores.values())
    item_scores = _item_name_f1(pred, gold)

    overall_score = 0.70 * core_mean + 0.30 * item_scores["item_name_f1"]

    return {
        **field_scores,
        "core_field_mean": core_mean,
        **item_scores,
        "overall_score": overall_score,
    }


def summarize_scores(rows: List[Dict]) -> Dict[str, float]:
    valid = [r for r in rows if "metrics" in r]

    if not valid:
        return {}

    keys = list(valid[0]["metrics"].keys())
    summary = {}

    for key in keys:
        vals = [r["metrics"][key] for r in valid]
        summary[key] = mean(vals)

    return summary