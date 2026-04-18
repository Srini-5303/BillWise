from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.pipeline import run_prototype_pipeline
from evaluation.normalize import normalize_amount
from methods.groq_vlm import GroqVLMMethod
from methods.hybrid_method import HybridMethod
from methods.prototype_method import PrototypeMethod

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


def _receipt_id_from_path(image_path: str) -> str:
    return Path(image_path).stem


def _proto_field_map(proto_result) -> Dict[str, Any]:
    out = {}
    for f in proto_result.fields:
        out[f.field_name] = {
            "value": f.value,
            "confidence": f.confidence,
            "bbox": f.bbox,
            "source": f.source,
            "ocr_confidence": f.ocr_confidence,
            "model_confidence": f.model_confidence,
        }
    return out


def _canonical_field_dict(canonical_receipt) -> Dict[str, Any]:
    return canonical_receipt.fields.model_dump()


def _lineitems_from_proto(proto_result) -> List[Dict[str, Any]]:
    rows = []
    for idx, item in enumerate(proto_result.items, start=1):
        rows.append(
            {
                "line_id": idx,
                "name": item.description.value if item.description else None,
                "quantity": item.quantity.value if item.quantity else None,
                "item_total": item.price.value if item.price else None,
                "bbox": item.description.bbox if item.description else None,
                "confidence": item.description.confidence if item.description else None,
            }
        )
    return rows


def _lineitems_from_canonical(canonical_receipt) -> List[Dict[str, Any]]:
    rows = []
    for item in canonical_receipt.items:
        rows.append(
            {
                "line_id": item.line_id,
                "name": item.name,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "item_total": item.item_total,
            }
        )
    return rows


def build_hybrid_ui_payload(image_path: str) -> Dict[str, Any]:
    receipt_id = _receipt_id_from_path(image_path)

    proto_raw = run_prototype_pipeline(image_path)
    proto_method = PrototypeMethod()
    groq_method = GroqVLMMethod()
    hybrid_method = HybridMethod()

    proto_canonical = proto_method.extract(image_path, receipt_id)
    groq_canonical = groq_method.extract(image_path, receipt_id)
    hybrid_canonical = hybrid_method.extract(image_path, receipt_id)

    proto_fields = _proto_field_map(proto_raw)
    groq_fields = _canonical_field_dict(groq_canonical)
    hybrid_fields = _canonical_field_dict(hybrid_canonical)

    field_sources = getattr(hybrid_canonical, "field_sources", {})
    review_required = getattr(hybrid_canonical, "review_required", False)
    review_reasons = getattr(hybrid_canonical, "review_reasons", [])

    fields_payload = {}

    for field_name in FIELD_ORDER:
        proto_entry = proto_fields.get(field_name)
        groq_value = groq_fields.get(field_name)
        final_value = hybrid_fields.get(field_name)
        final_source = field_sources.get(field_name, None)

        if final_source == "prototype" and proto_entry is not None:
            final_confidence = proto_entry.get("confidence")
            final_bbox = proto_entry.get("bbox")
        else:
            final_confidence = None
            final_bbox = proto_entry.get("bbox") if proto_entry else None

        fields_payload[field_name] = {
            "final_value": final_value,
            "final_source": final_source,
            "final_confidence": final_confidence,
            "final_bbox": final_bbox,
            "prototype": proto_entry,
            "groq": {
                "value": groq_value,
                "confidence": None,
                "bbox": None,
                "source": "groq",
            },
        }

    payload = {
        "receipt_id": receipt_id,
        "image_file": Path(image_path).name,
        "image_path": image_path,
        "review_required": review_required,
        "review_reasons": review_reasons,
        "fields": fields_payload,
        "prototype_items": _lineitems_from_proto(proto_raw),
        "groq_items": _lineitems_from_canonical(groq_canonical),
        "hybrid_items": _lineitems_from_canonical(hybrid_canonical),
        "prototype_raw_output": proto_raw.model_dump(),
        "prototype_canonical_output": proto_canonical.model_dump(),
        "groq_canonical_output": groq_canonical.model_dump(),
        "hybrid_canonical_output": hybrid_canonical.model_dump(),
    }

    return payload
