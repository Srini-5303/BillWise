from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.annotation_config import ANNOTATION_GUIDELINES, LABELS
from app.pipeline import run_prototype_debug_pipeline
from kie.label_mapper import to_canonical_field


def _map_prediction_label(raw_label: str) -> str:
    canonical = to_canonical_field(raw_label)

    mapping = {
        "merchant_name": "merchant_name",
        "date": "date",
        "time": "time",
        "subtotal": "subtotal",
        "tax": "tax",
        "total": "total",
        "payment_method": "payment_method",
        "card_last4": "card_last4",
        "receipt_number": "receipt_number",
        "item_description": "item_description",
        "item_quantity": "item_quantity",
        "item_price": "item_total",   # default mapping; user can relabel to item_unit_price if needed
        "store_address": "other",
        "phone_number": "other",
        "tips": "other",
    }

    return mapping.get(canonical, "other")


def build_annotation_payload(image_path: str) -> Dict[str, Any]:
    receipt_id = Path(image_path).stem
    debug = run_prototype_debug_pipeline(image_path)

    tokens = debug["tokens"]
    predictions = debug["predictions"]

    pred_map = {p["token_id"]: p for p in predictions}

    token_rows: List[Dict[str, Any]] = []

    for tok in tokens:
        pred = pred_map.get(tok.id)

        if pred is None:
            predicted_label = "other"
            predicted_label_confidence = None
            predicted_raw_label = None
            status = "unreviewed"
        else:
            predicted_raw_label = pred.get("label")
            predicted_label = _map_prediction_label(predicted_raw_label)
            predicted_label_confidence = pred.get("label_confidence")
            status = "predicted" if predicted_label != "other" else "unreviewed"

        token_rows.append(
            {
                "token_id": tok.id,
                "text": tok.text,
                "bbox": tok.bbox,
                "ocr_confidence": tok.ocr_confidence,
                "predicted_raw_label": predicted_raw_label,
                "predicted_label": predicted_label,
                "predicted_label_confidence": predicted_label_confidence,
                "corrected_label": predicted_label,
                "status": status,
            }
        )

    payload = {
        "receipt_id": receipt_id,
        "image_file": Path(image_path).name,
        "image_path": image_path,
        "image_width": debug["image_width"],
        "image_height": debug["image_height"],
        "label_set": LABELS,
        "guidelines": ANNOTATION_GUIDELINES,
        "tokens": token_rows,
        "manual_boxes": [],
        "prototype_result": debug["result"].model_dump(),
    }

    return payload
