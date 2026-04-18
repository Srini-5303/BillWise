from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


CANONICAL_FIELD_NAMES = [
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

CORE_FIELDS = ["merchant_name", "date", "time", "subtotal", "tax", "total"]
PAYMENT_FIELDS = ["payment_method", "card_last4"]

NUMERIC_FIELDS = {"subtotal", "tax", "total"}
TEXT_FIELDS = {"merchant_name", "receipt_number"}
DATE_FIELDS = {"date"}
TIME_FIELDS = {"time"}
PAYMENT_METHOD_FIELDS = {"payment_method"}
CARD_FIELDS = {"card_last4"}

MONEY_RE = re.compile(r"-?\$?\s*\d[\d,]*\.\d{2}")
FOUR_DIGIT_RE = re.compile(r"\b(\d{4})\b")


@dataclass
class MethodSpec:
    name: str
    pred_dir: Path
    adapter: str


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def normalize_spaces(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text if text else None


def normalize_text(text: Optional[str]) -> Optional[str]:
    text = normalize_spaces(text)
    if text is None:
        return None
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def normalize_receipt_id(text: Optional[str]) -> Optional[str]:
    text = normalize_spaces(text)
    if text is None:
        return None
    text = re.sub(r"^(REF|RECEIPT|TRANS|TXN|TRANSACTION|ID|NO|NUMBER|#)\s*[:#-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text)
    return text if text else None


def parse_money(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return round(float(value), 2)

    text = str(value).strip()
    if not text:
        return None

    match = MONEY_RE.search(text.replace(" ", ""))
    if not match:
        return None

    cleaned = match.group(0).replace("$", "").replace(",", "").strip()
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def parse_date(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = normalize_spaces(str(value))
    if text is None:
        return None

    candidates = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m-%d-%Y",
        "%m-%d-%y",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%b %d %Y",
        "%B %d %Y",
        "%d %b %Y",
        "%d %B %Y",
    ]

    for fmt in candidates:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    m = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", text)
    if m:
        return parse_date(m.group(1))

    return None


def parse_time(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = normalize_spaces(str(value))
    if text is None:
        return None

    patterns = [
        "%I:%M %p",
        "%I:%M%p",
        "%H:%M",
    ]

    for fmt in patterns:
        try:
            dt = datetime.strptime(text.upper(), fmt)
            return dt.strftime("%I:%M %p")
        except ValueError:
            pass

    m = re.search(r"(\d{1,2}:\d{2})\s*(AM|PM)?", text, flags=re.IGNORECASE)
    if m:
        hhmm = m.group(1)
        suffix = m.group(2)
        if suffix:
            return f"{datetime.strptime(hhmm + ' ' + suffix.upper(), '%I:%M %p').strftime('%I:%M %p')}"
        return hhmm

    return None


def normalize_payment_method(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = normalize_text(str(value))
    if text is None:
        return None

    if text in {"CASH"}:
        return "cash"
    if any(x in text for x in ["APPLE PAY", "GOOGLE PAY", "SAMSUNG PAY"]):
        return "mobile_wallet"
    if "CONTACTLESS" in text or "TAP" in text:
        return "contactless"
    if any(x in text for x in ["VISA", "MASTERCARD", "MASTERCARD", "AMEX", "DISCOVER", "DEBIT", "CREDIT", "CARD"]):
        return "card"
    if "EBT" in text or "SNAP" in text:
        return "ebt"
    if "GIFT CARD" in text or "GIFTCARD" in text:
        return "gift_card"
    if "CHECK" in text:
        return "check"

    if text in {"OTHER"}:
        return "other"

    return None


def normalize_card_last4(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = normalize_spaces(str(value))
    if text is None:
        return None

    if text.lower() == "cash":
        return None

    m = FOUR_DIGIT_RE.search(text)
    if m:
        return m.group(1)

    m = re.search(r"(?:\*{2,}|X{2,}|x{2,})\s*(\d{4})", text)
    if m:
        return m.group(1)

    return None


def normalize_quantity(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = normalize_spaces(str(value))
    if text is None:
        return None

    m = re.search(r"(\d+(?:\.\d+)?)\s*[@xX]", text)
    if m:
        return float(m.group(1))

    m = re.search(r"^\d+(?:\.\d+)?$", text)
    if m:
        return float(m.group(0))

    return None


def parse_quantity_and_unit_price_from_text(text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    text = normalize_spaces(text)
    if text is None:
        return None, None

    m = re.search(r"(\d+(?:\.\d+)?)\s*[@xX]\s*\$?\s*(\d+(?:\.\d{2})?)", text)
    if m:
        return float(m.group(1)), float(m.group(2))

    return normalize_quantity(text), None


def normalize_item_name(value: Any) -> Optional[str]:
    return normalize_text(value)


def to_empty_canonical(receipt_id: str, image_file: str) -> Dict[str, Any]:
    return {
        "receipt_id": receipt_id,
        "image_file": image_file,
        "fields": {k: None for k in CANONICAL_FIELD_NAMES},
        "items": [],
        "evaluation_metadata": {
            "currency": "USD",
            "visible_payment_info": False,
            "has_discounts_or_coupons": False,
            "quality": None,
            "notes": None,
        },
    }


def canonicalize_gold(raw: Dict[str, Any]) -> Dict[str, Any]:
    receipt_id = raw.get("receipt_id")
    image_file = raw.get("image_file")

    fields = raw.get("fields", {})
    items = raw.get("items", [])
    meta = raw.get("evaluation_metadata", {})

    canonical = to_empty_canonical(receipt_id=receipt_id, image_file=image_file)
    canonical["fields"].update({
        "merchant_name": normalize_spaces(fields.get("merchant_name")),
        "date": parse_date(fields.get("date")),
        "time": parse_time(fields.get("time")),
        "subtotal": parse_money(fields.get("subtotal")),
        "tax": parse_money(fields.get("tax")),
        "total": parse_money(fields.get("total")),
        "payment_method": normalize_payment_method(fields.get("payment_method")),
        "card_last4": normalize_card_last4(fields.get("card_last4")),
        "receipt_number": normalize_receipt_id(fields.get("receipt_number")),
    })

    canonical_items = []
    for i, item in enumerate(items, start=1):
        canonical_items.append({
            "line_id": int(item.get("line_id", i)),
            "name": normalize_spaces(item.get("name")),
            "quantity": normalize_quantity(item.get("quantity")),
            "unit_price": parse_money(item.get("unit_price")),
            "item_total": parse_money(item.get("item_total")),
        })
    canonical["items"] = canonical_items

    canonical["evaluation_metadata"] = {
        "currency": meta.get("currency", "USD"),
        "visible_payment_info": bool(meta.get("visible_payment_info", False)),
        "has_discounts_or_coupons": bool(meta.get("has_discounts_or_coupons", False)),
        "quality": meta.get("quality"),
        "notes": meta.get("notes"),
    }

    return canonical


def adapt_google_regex(raw: Dict[str, Any], receipt_id: str, image_file: str) -> Dict[str, Any]:
    canonical = to_empty_canonical(receipt_id, image_file)

    store = raw.get("store")
    date = raw.get("date")
    total = raw.get("total")
    card = raw.get("card")
    items = raw.get("items")

    payment_method = None
    card_last4 = None

    if isinstance(card, str):
        if card.strip().lower() == "cash":
            payment_method = "cash"
        else:
            maybe_last4 = normalize_card_last4(card)
            if maybe_last4 is not None:
                payment_method = "card"
                card_last4 = maybe_last4

    canonical["fields"].update({
        "merchant_name": normalize_spaces(store),
        "date": parse_date(date),
        "time": None,
        "subtotal": None,
        "tax": None,
        "total": parse_money(total),
        "payment_method": payment_method,
        "card_last4": card_last4,
        "receipt_number": None,
    })

    canonical_items = []
    if isinstance(items, str):
        split_items = [x.strip() for x in items.split(";") if x.strip() and x.strip().lower() != "not found"]
        for idx, name in enumerate(split_items, start=1):
            canonical_items.append({
                "line_id": idx,
                "name": normalize_spaces(name),
                "quantity": None,
                "unit_price": None,
                "item_total": None,
            })

    canonical["items"] = canonical_items
    return canonical


def adapt_groq(raw: Dict[str, Any], receipt_id: str, image_file: str) -> Dict[str, Any]:
    canonical = to_empty_canonical(receipt_id, image_file)

    card_last4 = raw.get("card_last4")
    payment_method = raw.get("payment_method")

    if isinstance(card_last4, str) and card_last4.lower() == "cash":
        payment_method = "cash"
        card_last4 = None

    canonical["fields"].update({
        "merchant_name": normalize_spaces(raw.get("store_name")),
        "date": parse_date(raw.get("invoice_date")),
        "time": None,
        "subtotal": parse_money(raw.get("subtotal")),
        "tax": parse_money(raw.get("tax_amount")),
        "total": parse_money(raw.get("total_amount")),
        "payment_method": normalize_payment_method(payment_method),
        "card_last4": normalize_card_last4(card_last4),
        "receipt_number": normalize_receipt_id(raw.get("receipt_number")),
    })

    canonical_items = []
    for idx, item in enumerate(raw.get("items", []), start=1):
        canonical_items.append({
            "line_id": idx,
            "name": normalize_spaces(item.get("name")),
            "quantity": normalize_quantity(item.get("quantity")),
            "unit_price": parse_money(item.get("unit_price")),
            "item_total": parse_money(item.get("item_total")),
        })

    canonical["items"] = canonical_items
    return canonical


def adapt_prototype(raw: Dict[str, Any], receipt_id: str, image_file: str) -> Dict[str, Any]:
    canonical = to_empty_canonical(receipt_id, image_file)

    field_map: Dict[str, Any] = {}
    for entry in raw.get("fields", []):
        key = entry.get("field_name")
        value = entry.get("value")
        if key not in field_map:
            field_map[key] = value

    canonical["fields"].update({
        "merchant_name": normalize_spaces(field_map.get("merchant_name")),
        "date": parse_date(field_map.get("date")),
        "time": parse_time(field_map.get("time")),
        "subtotal": parse_money(field_map.get("subtotal")),
        "tax": parse_money(field_map.get("tax")),
        "total": parse_money(field_map.get("total")),
        "payment_method": normalize_payment_method(field_map.get("payment_method")),
        "card_last4": normalize_card_last4(field_map.get("card_last4")),
        "receipt_number": normalize_receipt_id(field_map.get("receipt_number")),
    })

    canonical_items = []
    for idx, item in enumerate(raw.get("items", []), start=1):
        description = item.get("description", {}) or {}
        quantity_obj = item.get("quantity", {}) or {}
        price_obj = item.get("price", {}) or {}

        quantity_text = quantity_obj.get("value")
        qty, unit_price_from_qty = parse_quantity_and_unit_price_from_text(quantity_text)

        canonical_items.append({
            "line_id": idx,
            "name": normalize_spaces(description.get("value")),
            "quantity": qty,
            "unit_price": unit_price_from_qty,
            "item_total": parse_money(price_obj.get("value")),
        })

    canonical["items"] = canonical_items
    return canonical


ADAPTERS: Dict[str, Callable[[Dict[str, Any], str, str], Dict[str, Any]]] = {
    "google_regex": adapt_google_regex,
    "groq": adapt_groq,
    "prototype": adapt_prototype,
    "canonical": lambda raw, receipt_id, image_file: canonicalize_gold(raw),
}


def field_equal(field_name: str, gold_value: Any, pred_value: Any, money_tol: float = 0.01) -> bool:
    if field_name in NUMERIC_FIELDS:
        g = parse_money(gold_value)
        p = parse_money(pred_value)
        if g is None and p is None:
            return True
        if g is None or p is None:
            return False
        return abs(g - p) <= money_tol

    if field_name in DATE_FIELDS:
        return parse_date(gold_value) == parse_date(pred_value)

    if field_name in TIME_FIELDS:
        return parse_time(gold_value) == parse_time(pred_value)

    if field_name in PAYMENT_METHOD_FIELDS:
        return normalize_payment_method(gold_value) == normalize_payment_method(pred_value)

    if field_name in CARD_FIELDS:
        return normalize_card_last4(gold_value) == normalize_card_last4(pred_value)

    if field_name == "receipt_number":
        return normalize_receipt_id(gold_value) == normalize_receipt_id(pred_value)

    return normalize_text(gold_value) == normalize_text(pred_value)


def item_name_similarity(name_a: Optional[str], name_b: Optional[str]) -> float:
    a = normalize_item_name(name_a)
    b = normalize_item_name(name_b)
    if a is None or b is None:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def greedy_item_match(
    gold_items: List[Dict[str, Any]],
    pred_items: List[Dict[str, Any]],
    threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gold_items):
        for pi, p in enumerate(pred_items):
            sim = item_name_similarity(g.get("name"), p.get("name"))
            if sim >= threshold:
                candidates.append((sim, gi, pi))

    candidates.sort(reverse=True, key=lambda x: (x[0], -x[1], -x[2]))

    matched_gold = set()
    matched_pred = set()
    matches: List[Tuple[int, int, float]] = []

    for sim, gi, pi in candidates:
        if gi in matched_gold or pi in matched_pred:
            continue
        matched_gold.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, sim))

    unmatched_gold = [i for i in range(len(gold_items)) if i not in matched_gold]
    unmatched_pred = [i for i in range(len(pred_items)) if i not in matched_pred]
    return matches, unmatched_gold, unmatched_pred


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def evaluate_items(
    gold_items: List[Dict[str, Any]],
    pred_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    exact_matches, exact_unmatched_gold, exact_unmatched_pred = greedy_item_match(gold_items, pred_items, threshold=1.0)
    soft_matches, soft_unmatched_gold, soft_unmatched_pred = greedy_item_match(gold_items, pred_items, threshold=0.85)

    exact_tp = len(exact_matches)
    exact_fp = len(exact_unmatched_pred)
    exact_fn = len(exact_unmatched_gold)

    soft_tp = len(soft_matches)
    soft_fp = len(soft_unmatched_pred)
    soft_fn = len(soft_unmatched_gold)

    def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    soft_subfield = {
        "quantity_support": 0,
        "quantity_correct": 0,
        "unit_price_support": 0,
        "unit_price_correct": 0,
        "item_total_support": 0,
        "item_total_correct": 0,
    }

    for gi, pi, _ in soft_matches:
        g = gold_items[gi]
        p = pred_items[pi]

        if g.get("quantity") is not None:
            soft_subfield["quantity_support"] += 1
            if normalize_quantity(g.get("quantity")) == normalize_quantity(p.get("quantity")):
                soft_subfield["quantity_correct"] += 1

        if g.get("unit_price") is not None:
            soft_subfield["unit_price_support"] += 1
            if field_equal("subtotal", g.get("unit_price"), p.get("unit_price")):
                soft_subfield["unit_price_correct"] += 1

        if g.get("item_total") is not None:
            soft_subfield["item_total_support"] += 1
            if field_equal("subtotal", g.get("item_total"), p.get("item_total")):
                soft_subfield["item_total_correct"] += 1

    return {
        "item_name_exact": prf(exact_tp, exact_fp, exact_fn),
        "item_name_soft": prf(soft_tp, soft_fp, soft_fn),
        "quantity_accuracy_on_labeled": safe_div(
            soft_subfield["quantity_correct"],
            soft_subfield["quantity_support"],
        ),
        "unit_price_accuracy_on_labeled": safe_div(
            soft_subfield["unit_price_correct"],
            soft_subfield["unit_price_support"],
        ),
        "item_total_accuracy_on_labeled": safe_div(
            soft_subfield["item_total_correct"],
            soft_subfield["item_total_support"],
        ),
        "quantity_support": soft_subfield["quantity_support"],
        "unit_price_support": soft_subfield["unit_price_support"],
        "item_total_support": soft_subfield["item_total_support"],
    }


def evaluate_receipt(gold: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    field_results = {}
    correct_count = 0

    for field_name in CANONICAL_FIELD_NAMES:
        gold_value = gold["fields"].get(field_name)
        pred_value = pred["fields"].get(field_name)
        correct = field_equal(field_name, gold_value, pred_value)
        field_results[field_name] = {
            "gold": gold_value,
            "pred": pred_value,
            "correct": correct,
        }
        correct_count += int(correct)

    items_result = evaluate_items(gold.get("items", []), pred.get("items", []))

    return {
        "receipt_id": gold["receipt_id"],
        "image_file": gold["image_file"],
        "field_results": field_results,
        "field_accuracy": safe_div(correct_count, len(CANONICAL_FIELD_NAMES)),
        "core_field_accuracy": safe_div(
            sum(int(field_results[f]["correct"]) for f in CORE_FIELDS),
            len(CORE_FIELDS),
        ),
        "payment_field_accuracy": safe_div(
            sum(int(field_results[f]["correct"]) for f in PAYMENT_FIELDS),
            len(PAYMENT_FIELDS),
        ),
        "items_result": items_result,
    }


def aggregate_results(per_receipt: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_field_correct = Counter()
    per_field_total = Counter()

    field_acc_sum = 0.0
    core_acc_sum = 0.0
    payment_acc_sum = 0.0

    item_exact_precision = []
    item_exact_recall = []
    item_exact_f1 = []

    item_soft_precision = []
    item_soft_recall = []
    item_soft_f1 = []

    qty_acc = []
    unit_price_acc = []
    item_total_acc = []

    qty_support_total = 0
    unit_price_support_total = 0
    item_total_support_total = 0

    for row in per_receipt:
        field_acc_sum += row["field_accuracy"]
        core_acc_sum += row["core_field_accuracy"]
        payment_acc_sum += row["payment_field_accuracy"]

        for field_name, result in row["field_results"].items():
            per_field_total[field_name] += 1
            per_field_correct[field_name] += int(result["correct"])

        exact = row["items_result"]["item_name_exact"]
        soft = row["items_result"]["item_name_soft"]

        item_exact_precision.append(exact["precision"])
        item_exact_recall.append(exact["recall"])
        item_exact_f1.append(exact["f1"])

        item_soft_precision.append(soft["precision"])
        item_soft_recall.append(soft["recall"])
        item_soft_f1.append(soft["f1"])

        qty_support = row["items_result"]["quantity_support"]
        unit_support = row["items_result"]["unit_price_support"]
        total_support = row["items_result"]["item_total_support"]

        qty_support_total += qty_support
        unit_price_support_total += unit_support
        item_total_support_total += total_support

        if qty_support > 0:
            qty_acc.append(row["items_result"]["quantity_accuracy_on_labeled"])
        if unit_support > 0:
            unit_price_acc.append(row["items_result"]["unit_price_accuracy_on_labeled"])
        if total_support > 0:
            item_total_acc.append(row["items_result"]["item_total_accuracy_on_labeled"])

    n = max(1, len(per_receipt))

    return {
        "receipts_evaluated": len(per_receipt),
        "overall_field_accuracy": field_acc_sum / n,
        "overall_core_field_accuracy": core_acc_sum / n,
        "overall_payment_field_accuracy": payment_acc_sum / n,
        "per_field_accuracy": {
            f: safe_div(per_field_correct[f], per_field_total[f])
            for f in CANONICAL_FIELD_NAMES
        },
        "item_name_exact_precision_macro": sum(item_exact_precision) / n,
        "item_name_exact_recall_macro": sum(item_exact_recall) / n,
        "item_name_exact_f1_macro": sum(item_exact_f1) / n,
        "item_name_soft_precision_macro": sum(item_soft_precision) / n,
        "item_name_soft_recall_macro": sum(item_soft_recall) / n,
        "item_name_soft_f1_macro": sum(item_soft_f1) / n,
        "quantity_accuracy_on_labeled_macro": sum(qty_acc) / len(qty_acc) if qty_acc else 0.0,
        "unit_price_accuracy_on_labeled_macro": sum(unit_price_acc) / len(unit_price_acc) if unit_price_acc else 0.0,
        "item_total_accuracy_on_labeled_macro": sum(item_total_acc) / len(item_total_acc) if item_total_acc else 0.0,
        "quantity_support_total": qty_support_total,
        "unit_price_support_total": unit_price_support_total,
        "item_total_support_total": item_total_support_total,
    }


def parse_pred_spec(spec: str) -> MethodSpec:
    # format: method_name=pred_dir:adapter_name
    if "=" not in spec or ":" not in spec:
        raise ValueError(
            f"Invalid --pred spec: {spec!r}. Expected format name=path:adapter"
        )

    name, rest = spec.split("=", 1)
    pred_dir_str, adapter = rest.rsplit(":", 1)

    if adapter not in ADAPTERS:
        raise ValueError(
            f"Unknown adapter {adapter!r}. Supported: {sorted(ADAPTERS.keys())}"
        )

    return MethodSpec(name=name, pred_dir=Path(pred_dir_str), adapter=adapter)


def write_summary_csv(summary_by_method: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, summary in summary_by_method.items():
        row = {"method": method}
        row.update(summary)
        row["per_field_accuracy"] = json.dumps(summary["per_field_accuracy"])
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["method"]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_per_receipt_csv(method_name: str, per_receipt: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in per_receipt:
        flat = {
            "method": method_name,
            "receipt_id": row["receipt_id"],
            "image_file": row["image_file"],
            "field_accuracy": row["field_accuracy"],
            "core_field_accuracy": row["core_field_accuracy"],
            "payment_field_accuracy": row["payment_field_accuracy"],
            "item_name_exact_f1": row["items_result"]["item_name_exact"]["f1"],
            "item_name_soft_f1": row["items_result"]["item_name_soft"]["f1"],
            "quantity_accuracy_on_labeled": row["items_result"]["quantity_accuracy_on_labeled"],
            "unit_price_accuracy_on_labeled": row["items_result"]["unit_price_accuracy_on_labeled"],
            "item_total_accuracy_on_labeled": row["items_result"]["item_total_accuracy_on_labeled"],
        }
        for field_name, result in row["field_results"].items():
            flat[f"{field_name}_correct"] = int(result["correct"])
        rows.append(flat)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3 receipt extraction methods against canonical gold labels.")
    parser.add_argument("--gold-dir", required=True, help="Folder containing canonical gold JSON files.")
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Prediction spec in the form method_name=pred_dir:adapter_name. "
             "Adapter options: google_regex, groq, prototype, canonical",
    )
    parser.add_argument("--output-dir", default="evaluation_output", help="Folder to write evaluation artifacts.")
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir)
    output_dir = Path(args.output_dir)

    method_specs = [parse_pred_spec(spec) for spec in args.pred]
    gold_files = sorted(gold_dir.glob("*.json"))

    if not gold_files:
        raise FileNotFoundError(f"No gold JSON files found in {gold_dir}")

    summary_by_method: Dict[str, Dict[str, Any]] = {}

    for method in method_specs:
        adapter_fn = ADAPTERS[method.adapter]
        per_receipt_results = []
        missing_prediction_files = []

        for gold_path in gold_files:
            gold_raw = load_json(gold_path)
            gold = canonicalize_gold(gold_raw)

            pred_path = method.pred_dir / gold_path.name
            if pred_path.exists():
                pred_raw = load_json(pred_path)
                pred = adapter_fn(pred_raw, gold["receipt_id"], gold["image_file"])
            else:
                pred = to_empty_canonical(gold["receipt_id"], gold["image_file"])
                missing_prediction_files.append(gold_path.name)

            result = evaluate_receipt(gold, pred)
            per_receipt_results.append(result)

        summary = aggregate_results(per_receipt_results)
        summary["missing_prediction_files"] = missing_prediction_files
        summary["missing_prediction_count"] = len(missing_prediction_files)

        summary_by_method[method.name] = summary

        dump_json(output_dir / f"{method.name}_per_receipt.json", per_receipt_results)
        write_per_receipt_csv(method.name, per_receipt_results, output_dir / f"{method.name}_per_receipt.csv")

    dump_json(output_dir / "summary.json", summary_by_method)
    write_summary_csv(summary_by_method, output_dir / "summary.csv")

    print("\n=== Evaluation complete ===")
    print(f"Gold dir: {gold_dir}")
    print(f"Output dir: {output_dir}\n")

    for method_name, summary in summary_by_method.items():
        print(f"[{method_name}]")
        print(f"  receipts_evaluated: {summary['receipts_evaluated']}")
        print(f"  missing_prediction_count: {summary['missing_prediction_count']}")
        print(f"  overall_field_accuracy: {summary['overall_field_accuracy']:.4f}")
        print(f"  overall_core_field_accuracy: {summary['overall_core_field_accuracy']:.4f}")
        print(f"  overall_payment_field_accuracy: {summary['overall_payment_field_accuracy']:.4f}")
        print(f"  item_name_exact_f1_macro: {summary['item_name_exact_f1_macro']:.4f}")
        print(f"  item_name_soft_f1_macro: {summary['item_name_soft_f1_macro']:.4f}")
        print(f"  item_total_accuracy_on_labeled_macro: {summary['item_total_accuracy_on_labeled_macro']:.4f}")
        print()


if __name__ == "__main__":
    main()