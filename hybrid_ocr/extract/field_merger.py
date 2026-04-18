from __future__ import annotations

from typing import Any, Dict, List

from app.schemas import ExtractedField
from extract.confidence import final_field_confidence
from kie.label_mapper import is_value_label, to_canonical_field


NON_ITEM_FIELDS = {
    "merchant_name",
    "store_address",
    "phone_number",
    "date",
    "time",
    "subtotal",
    "tax",
    "tips",
    "total",
}


def _union_bbox(boxes: List[List[int]]) -> List[int]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def _vertical_overlap(box1: List[int], box2: List[int]) -> int:
    return max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))


def _box_height(box: List[int]) -> int:
    return max(1, box[3] - box[1])


def _same_line(box1: List[int], box2: List[int], min_overlap_ratio: float = 0.45) -> bool:
    overlap = _vertical_overlap(box1, box2)
    min_height = min(_box_height(box1), _box_height(box2))
    ratio = overlap / min_height
    return ratio >= min_overlap_ratio


def _horizontal_gap(box1: List[int], box2: List[int]) -> int:
    return box2[0] - box1[2]


def _should_merge(prev_pred: Dict[str, Any], curr_pred: Dict[str, Any]) -> bool:
    if prev_pred["field_name"] != curr_pred["field_name"]:
        return False

    prev_box = prev_pred["bbox"]
    curr_box = curr_pred["bbox"]

    if not _same_line(prev_box, curr_box):
        return False

    gap = _horizontal_gap(prev_box, curr_box)
    if gap > 60:
        return False

    return True


def merge_non_item_fields(predictions: List[Dict[str, Any]]) -> List[ExtractedField]:
    normalized_preds: List[Dict[str, Any]] = []

    for pred in predictions:
        label = pred["label"]

        if not is_value_label(label):
            continue

        field_name = to_canonical_field(label)

        if field_name not in NON_ITEM_FIELDS:
            continue

        normalized_preds.append(
            {
                "token_id": pred["token_id"],
                "text": pred["text"],
                "bbox": pred["bbox"],
                "ocr_confidence": pred["ocr_confidence"],
                "label_confidence": pred["label_confidence"],
                "field_name": field_name,
            }
        )

    normalized_preds.sort(key=lambda p: (p["field_name"], p["bbox"][1], p["bbox"][0]))

    merged_groups: List[List[Dict[str, Any]]] = []

    for pred in normalized_preds:
        if not merged_groups:
            merged_groups.append([pred])
            continue

        last_group = merged_groups[-1]
        last_pred = last_group[-1]

        if _should_merge(last_pred, pred):
            last_group.append(pred)
        else:
            merged_groups.append([pred])

    extracted_fields: List[ExtractedField] = []

    for group in merged_groups:
        field_name = group[0]["field_name"]
        texts = [g["text"] for g in group]
        boxes = [g["bbox"] for g in group]
        model_scores = [g["label_confidence"] for g in group]
        ocr_scores = [g["ocr_confidence"] for g in group]
        token_ids = [g["token_id"] for g in group]

        value = " ".join(texts).strip()
        bbox = _union_bbox(boxes)
        confidence = final_field_confidence(model_scores, ocr_scores)

        extracted_fields.append(
            ExtractedField(
                field_name=field_name,
                value=value,
                confidence=confidence,
                bbox=bbox,
                source="layoutlm",
                ocr_confidence=sum(ocr_scores) / len(ocr_scores),
                model_confidence=sum(model_scores) / len(model_scores),
                token_ids=token_ids,
            )
        )

    return extracted_fields
