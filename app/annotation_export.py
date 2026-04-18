from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _split_manual_text(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"\s+", text.strip())
    return [p for p in parts if p]


def _is_deleted_token(token: Dict[str, Any]) -> bool:
    status = str(token.get("status", "")).lower()
    corrected_label = str(token.get("corrected_label", "")).lower()
    return status == "deleted" or corrected_label == "ignore"


def convert_annotation_to_layoutlm(annotation: Dict[str, Any]) -> Dict[str, Any]:
    image_path = annotation["image_path"]

    words = []
    bboxes = []
    labels = []

    # Existing OCR/LayoutLM tokens
    for tok in annotation.get("tokens", []):
        if _is_deleted_token(tok):
            continue

        text = tok.get("text")
        bbox = tok.get("bbox")
        label = tok.get("corrected_label", "other")

        if not text or not bbox or len(bbox) != 4:
            continue

        words.append(text)
        bboxes.append(bbox)
        labels.append(label)

    # Manual boxes
    for box in annotation.get("manual_boxes", []):
        bbox = box.get("bbox")
        label = box.get("label")
        text = box.get("text")

        if not bbox or len(bbox) != 4 or not label or not text:
            continue

        manual_words = _split_manual_text(text)
        for w in manual_words:
            words.append(w)
            bboxes.append(bbox)
            labels.append(label)

    return {
        "receipt_id": annotation["receipt_id"],
        "image_file": annotation["image_file"],
        "image_path": image_path,
        "words": words,
        "bboxes": bboxes,
        "labels": labels,
    }


def export_annotation_folder(
    annotation_dir: str = "assets/dataset/annotation_feedback",
    out_dir: str = "assets/dataset/layoutlm_training",
):
    annotation_dir = Path(annotation_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(annotation_dir.glob("*_annotation.json"))
    exported = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        converted = convert_annotation_to_layoutlm(data)

        out_path = out_dir / f"{data['receipt_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=2)

        exported.append(str(out_path))

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"files": exported}, f, indent=2)

    return exported, str(manifest_path)