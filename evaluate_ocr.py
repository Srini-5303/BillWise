"""
evaluate_ocr.py — Evaluate Google Vision OCR pipeline against gold labels.

Usage:
    python evaluate_ocr.py

Reads from:
    receipts&labels/receipts/    — receipt images
    receipts&labels/gold_labels/ — ground truth JSON files

Writes to:
    evaluation/vision_per_receipt.json
    evaluation/vision_summary.json

Metrics match the LayoutLM evaluation format for direct comparison.
"""

import os
import re
import json
import glob
from datetime import datetime

from ocr_pipeline import process_image

RECEIPTS_DIR = os.path.join("receipts&labels", "receipts")
GOLD_DIR     = os.path.join("receipts&labels", "gold_labels")
OUTPUT_DIR   = "evaluation"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_str(s):
    if s is None:
        return ""
    return re.sub(r"[^A-Z0-9 ]", "", str(s).upper().strip())


def _parse_date(s):
    if not s:
        return None
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y",
                "%b %d %Y", "%B %d %Y", "%d %b %Y", "%d %B %Y"]:
        try:
            return datetime.strptime(s.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return s.strip()


def _parse_time(s):
    """Normalise to HH:MM (24-hour) for comparison."""
    if not s:
        return None
    s = s.strip().upper()
    for fmt in ["%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"]:
        try:
            return datetime.strptime(s, fmt).strftime("%H:%M")
        except ValueError:
            pass
    return s


def _parse_float(s):
    if s is None or s == "":
        return None
    try:
        return round(float(str(s).replace(",", "")), 2)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Token-overlap F1 for item names
# ---------------------------------------------------------------------------

def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _norm_str(pred).split()
    gold_tokens = _norm_str(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)


def _match_items(pred_names, gold_names, threshold=0.5):
    if not gold_names and not pred_names:
        return 1.0, 1.0, 1.0
    if not gold_names or not pred_names:
        return 0.0, 0.0, 0.0

    used_pred = set()
    matched = 0

    for g in gold_names:
        best_score, best_idx = 0.0, -1
        for i, p in enumerate(pred_names):
            if i in used_pred:
                continue
            score = _token_f1(p, g)
            if score > best_score:
                best_score, best_idx = score, i
        if best_score >= threshold:
            matched += 1
            used_pred.add(best_idx)

    precision = matched / len(pred_names) if pred_names else 0.0
    recall    = matched / len(gold_names) if gold_names else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Per-receipt evaluation
# ---------------------------------------------------------------------------

def _field_score(pred_val, gold_val, kind="exact"):
    """
    Returns 1.0 / 0.0 / None (None = gold has no value, skip from mean).
    kind: 'exact' | 'float' | 'time' | 'fuzzy'
    """
    if gold_val is None or gold_val == "":
        return None

    if kind == "float":
        g = _parse_float(gold_val)
        p = _parse_float(pred_val)
        if g is None:
            return None
        return 1.0 if (p is not None and abs(g - p) < 0.02) else 0.0

    if kind == "time":
        g = _parse_time(str(gold_val))
        p = _parse_time(str(pred_val)) if pred_val else None
        if g is None:
            return None
        return 1.0 if g == p else 0.0

    if kind == "fuzzy":
        g = _norm_str(gold_val)
        p = _norm_str(pred_val)
        if not g:
            return None
        return 1.0 if (g in p or p in g or _token_f1(p, g) >= 0.5) else 0.0

    # exact
    g = _norm_str(gold_val)
    p = _norm_str(pred_val)
    if not g:
        return None
    return 1.0 if g == p else 0.0


def evaluate_receipt(image_path, gold):
    result = process_image(image_path)

    gold_fields = gold.get("fields", {})
    gold_items  = [it["name"] for it in gold.get("items", [])]
    pred_items  = [name for name, _ in result.get("items", [])]

    scores = {
        "merchant_name":  _field_score(result.get("store"),          gold_fields.get("merchant_name"), "fuzzy"),
        "date":           _field_score(result.get("date"),           gold_fields.get("date"),           "exact"),
        "time":           _field_score(result.get("time"),           gold_fields.get("time"),           "time"),
        "subtotal":       _field_score(result.get("subtotal"),       gold_fields.get("subtotal"),       "float"),
        "tax":            _field_score(result.get("tax"),            gold_fields.get("tax"),            "float"),
        "total":          _field_score(result.get("total"),          gold_fields.get("total"),          "float"),
        "payment_method": _field_score(result.get("payment_method"), gold_fields.get("payment_method"), "fuzzy"),
        "card_last4":     _field_score(result.get("card"),           gold_fields.get("card_last4"),     "exact"),
        "receipt_number": _field_score(result.get("receipt_number"), gold_fields.get("receipt_number"), "exact"),
    }

    # date needs YYYY-MM-DD normalisation before exact compare
    if gold_fields.get("date"):
        gold_date_norm = _parse_date(str(gold_fields["date"]))
        pred_date_norm = _parse_date(result.get("date", "") or "")
        scores["date"] = 1.0 if gold_date_norm and gold_date_norm == pred_date_norm else 0.0

    # card_last4 — compare last 4 digits only
    if gold_fields.get("card_last4") is not None:
        gold_card4 = str(gold_fields["card_last4"]).strip()[-4:]
        pred_card4 = str(result.get("card", "") or "").strip()[-4:]
        scores["card_last4"] = 1.0 if gold_card4 == pred_card4 else 0.0

    core_vals  = [v for v in scores.values() if v is not None]
    core_mean  = sum(core_vals) / len(core_vals) if core_vals else 0.0

    item_p, item_r, item_f1 = _match_items(pred_items, gold_items)

    overall = core_mean * 0.7 + item_f1 * 0.3

    metrics = {
        **{k: (v if v is not None else 0.0) for k, v in scores.items()},
        "core_field_mean":     core_mean,
        "item_name_precision": item_p,
        "item_name_recall":    item_r,
        "item_name_f1":        item_f1,
        "pred_item_count":     len(pred_items),
        "gold_item_count":     len(gold_items),
        "overall_score":       overall,
    }

    debug = {
        "pred_store":          result.get("store"),
        "pred_date":           result.get("date"),
        "pred_time":           result.get("time"),
        "pred_subtotal":       result.get("subtotal"),
        "pred_tax":            result.get("tax"),
        "pred_total":          result.get("total"),
        "pred_payment_method": result.get("payment_method"),
        "pred_card":           result.get("card"),
        "pred_receipt_number": result.get("receipt_number"),
        "pred_items":          pred_items,
    }

    return metrics, debug


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gold_files = sorted(glob.glob(os.path.join(GOLD_DIR, "*.json")))
    if not gold_files:
        print("No gold label files found in", GOLD_DIR)
        return

    per_receipt = []
    all_metrics = {}

    for gf in gold_files:
        with open(gf, encoding="utf-8") as f:
            gold = json.load(f)

        receipt_id = gold["receipt_id"]
        image_file = gold["image_file"]
        image_path = os.path.join(RECEIPTS_DIR, image_file)

        if not os.path.exists(image_path):
            print(f"  [SKIP] {receipt_id} — image not found: {image_path}")
            continue

        print(f"  Evaluating {receipt_id} ...", end=" ", flush=True)
        try:
            metrics, debug = evaluate_receipt(image_path, gold)
            print(f"overall={metrics['overall_score']:.3f}  "
                  f"items_f1={metrics['item_name_f1']:.3f}  "
                  f"store={debug['pred_store']!r}")

            per_receipt.append({
                "receipt_id": receipt_id,
                "image_file": image_file,
                "metrics":    metrics,
            })

            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)

        except Exception as e:
            print(f"ERROR — {e}")

    if not per_receipt:
        print("No receipts evaluated.")
        return

    summary = {k: sum(v) / len(v) for k, v in all_metrics.items()}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    per_path = os.path.join(OUTPUT_DIR, "vision_per_receipt.json")
    sum_path = os.path.join(OUTPUT_DIR, "vision_summary.json")

    with open(per_path, "w", encoding="utf-8") as f:
        json.dump(per_receipt, f, indent=2)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n── Summary ─────────────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<25} {v:.4f}")
    print(f"\nSaved:\n  {per_path}\n  {sum_path}")


if __name__ == "__main__":
    main()
