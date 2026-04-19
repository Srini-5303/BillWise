from __future__ import annotations

from typing import Any

from billwise.common.ids import new_id
from billwise.common.schemas import BoundingBox, ExtractedField, LineItem, ReceiptRecord


FIELD_ORDER = [
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


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _bbox_to_model(bbox: Any) -> BoundingBox | None:
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = bbox
        return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
    except (TypeError, ValueError):
        return None


def _normalize_item_name(name: Any) -> str | None:
    text = _safe_str(name)
    if text is None:
        return None
    return " ".join(text.lower().split())


def _field_entry(payload: dict, field_name: str) -> dict:
    return (payload.get("fields", {}) or {}).get(field_name, {}) or {}


def _field_final_value(payload: dict, field_name: str) -> Any:
    return _field_entry(payload, field_name).get("final_value")


def _field_confidence(payload: dict, field_name: str) -> float | None:
    return _safe_float(_field_entry(payload, field_name).get("final_confidence"))


def _field_source(payload: dict, field_name: str) -> str | None:
    return _safe_str(_field_entry(payload, field_name).get("final_source"))


def _field_bbox(payload: dict, field_name: str):
    entry = _field_entry(payload, field_name)
    final_bbox = entry.get("final_bbox")
    if final_bbox:
        return final_bbox

    prototype = entry.get("prototype") or {}
    return prototype.get("bbox")


def build_receipt_from_hybrid_payload(
    payload: dict,
    image_path: str,
    receipt_id: str,
) -> ReceiptRecord:
    fields: list[ExtractedField] = []

    for field_name in FIELD_ORDER:
        value = _field_final_value(payload, field_name)

        fields.append(
            ExtractedField(
                field_name=field_name,
                field_value=_safe_str(value),
                confidence=_field_confidence(payload, field_name),
                bbox=_bbox_to_model(_field_bbox(payload, field_name)),
                source_model=_field_source(payload, field_name),
            )
        )

    prototype_item_lookup = {}
    for row in payload.get("prototype_items", []) or []:
        line_id = row.get("line_id")
        if line_id is not None:
            prototype_item_lookup[line_id] = row

    items: list[LineItem] = []
    for idx, row in enumerate(payload.get("hybrid_items", []) or [], start=1):
        line_id = row.get("line_id", idx)
        proto_row = prototype_item_lookup.get(line_id, {})

        raw_name = _safe_str(row.get("name"))
        items.append(
            LineItem(
                item_id=new_id("item"),
                raw_name=raw_name or f"item_{idx}",
                normalized_name=_normalize_item_name(raw_name),
                quantity=_safe_float(row.get("quantity")),
                unit_price=_safe_float(row.get("unit_price")),
                item_total=_safe_float(row.get("item_total")),
                item_confidence=_safe_float(proto_row.get("confidence")),
                item_source="hybrid-selected",
            )
        )

    requires_review = bool(payload.get("review_required", False))

    return ReceiptRecord(
        receipt_id=receipt_id,
        image_path=image_path,
        source="local_upload",
        processing_status="extracted",
        review_status="pending" if requires_review else "approved",
        vendor_name=_safe_str(_field_final_value(payload, "merchant_name")),
        receipt_date=_safe_str(_field_final_value(payload, "date")),
        receipt_time=_safe_str(_field_final_value(payload, "time")),
        subtotal=_safe_float(_field_final_value(payload, "subtotal")),
        tax=_safe_float(_field_final_value(payload, "tax")),
        total=_safe_float(_field_final_value(payload, "total")),
        payment_method=_safe_str(_field_final_value(payload, "payment_method")),
        card_last4=_safe_str(_field_final_value(payload, "card_last4")),
        receipt_number=_safe_str(_field_final_value(payload, "receipt_number")),
        extraction_method="hybrid",
        requires_review=requires_review,
        fields=fields,
        items=items,
    )