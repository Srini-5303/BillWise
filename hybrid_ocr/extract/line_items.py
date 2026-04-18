from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from app.schemas import LineItem, LineItemField
from extract.confidence import final_field_confidence
from kie.label_mapper import to_canonical_field


def _union_bbox(boxes: List[List[int]]) -> List[int]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def _y_center(box: List[int]) -> float:
    return (box[1] + box[3]) / 2.0


def _vertical_distance(box1: List[int], box2: List[int]) -> float:
    return abs(_y_center(box1) - _y_center(box2))


def _make_field(preds: List[Dict[str, Any]]) -> Optional[LineItemField]:
    if not preds:
        return None

    preds = sorted(preds, key=lambda p: p["bbox"][0])
    text = " ".join(p["text"] for p in preds).strip()
    bbox = _union_bbox([p["bbox"] for p in preds])
    confidence = final_field_confidence(
        [p["label_confidence"] for p in preds],
        [p["ocr_confidence"] for p in preds],
    )

    return LineItemField(
        value=text,
        confidence=confidence,
        bbox=bbox,
    )


def _best_description_for_price(
    price_pred: Dict[str, Any],
    desc_preds: List[Dict[str, Any]],
    used_desc_ids: Set[int],
    max_y_distance: int = 12,
) -> Optional[Dict[str, Any]]:
    candidates = []

    for desc in desc_preds:
        if desc["token_id"] in used_desc_ids:
            continue

        if desc["bbox"][0] >= price_pred["bbox"][0]:
            continue

        y_dist = _vertical_distance(desc["bbox"], price_pred["bbox"])
        if y_dist > max_y_distance:
            continue

        candidates.append((y_dist, desc["bbox"][0], desc))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def _best_quantity_for_pair(
    desc_pred: Optional[Dict[str, Any]],
    price_pred: Dict[str, Any],
    qty_preds: List[Dict[str, Any]],
    used_qty_ids: Set[int],
    max_y_distance: int = 16,
) -> Optional[Dict[str, Any]]:
    candidates = []

    for qty in qty_preds:
        if qty["token_id"] in used_qty_ids:
            continue

        if qty["bbox"][0] >= price_pred["bbox"][0]:
            continue

        ref_box = desc_pred["bbox"] if desc_pred is not None else price_pred["bbox"]
        y_dist = _vertical_distance(qty["bbox"], ref_box)
        if y_dist > max_y_distance:
            continue

        candidates.append((y_dist, -qty["bbox"][0], qty))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def extract_line_items(predictions: List[Dict[str, Any]]) -> List[LineItem]:
    desc_preds: List[Dict[str, Any]] = []
    qty_preds: List[Dict[str, Any]] = []
    price_preds: List[Dict[str, Any]] = []

    for pred in predictions:
        field_name = to_canonical_field(pred["label"])

        enriched = dict(pred)
        enriched["field_name"] = field_name

        if field_name == "item_description":
            desc_preds.append(enriched)
        elif field_name == "item_quantity":
            qty_preds.append(enriched)
        elif field_name == "item_price":
            price_preds.append(enriched)

    price_preds.sort(key=lambda p: (_y_center(p["bbox"]), p["bbox"][0]))

    used_desc_ids: Set[int] = set()
    used_qty_ids: Set[int] = set()

    items: List[LineItem] = []

    for price in price_preds:
        desc = _best_description_for_price(price, desc_preds, used_desc_ids)
        qty = _best_quantity_for_pair(desc, price, qty_preds, used_qty_ids)

        if desc is None:
            continue

        used_desc_ids.add(desc["token_id"])
        if qty is not None:
            used_qty_ids.add(qty["token_id"])

        item = LineItem(
            description=_make_field([desc]),
            quantity=_make_field([qty]) if qty is not None else None,
            price=_make_field([price]),
        )
        items.append(item)

    return items
