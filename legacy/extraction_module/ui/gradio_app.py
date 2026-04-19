from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from app.annotation_config import LABELS
from app.annotation_payload import build_annotation_payload
from app.hybrid_ui_payload import build_hybrid_ui_payload


REVIEW_FIELD_NAMES = [
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


LABEL_COLORS = {
    "merchant_name": "red",
    "date": "orange",
    "time": "gold",
    "subtotal": "cyan",
    "tax": "deepskyblue",
    "total": "blue",
    "payment_method": "purple",
    "card_last4": "magenta",
    "receipt_number": "brown",
    "item_description": "lime",
    "item_quantity": "green",
    "item_unit_price": "lightgreen",
    "item_total": "darkgreen",
    "other": "gray",
    "ignore": "black",
}


def _to_pil(img):
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return Image.fromarray(img)
    if isinstance(img, str):
        return Image.open(img).convert("RGBA")
    return None


def _format_conf(conf: Optional[float]) -> Optional[str]:
    if conf is None:
        return None
    return f"{conf:.2f}"


def _draw_conf_badge(draw: ImageDraw.ImageDraw, bbox: List[int], conf_text: str, fill="red"):
    x1, y1, x2, y2 = bbox
    try:
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), conf_text)
        tw, th = tx1 - tx0, ty1 - ty0
    except Exception:
        tw, th = 42, 14

    pad = 4
    rx2 = x2
    rx1 = max(0, rx2 - tw - 2 * pad)
    ry1 = max(0, y1)
    ry2 = ry1 + th + 2 * pad
    draw.rectangle([rx1, ry1, rx2, ry2], fill=fill)
    draw.text((rx1 + pad, ry1 + pad), conf_text, fill="white")


def _editor_value_from_image(image: Image.Image):
    return {
        "background": image,
        "layers": [],
        "composite": image,
    }


def _extract_bbox_from_editor(editor_value) -> Optional[List[int]]:
    if not editor_value:
        return None

    layers = editor_value.get("layers", [])
    if not layers:
        return None

    layer = _to_pil(layers[-1])
    if layer is None:
        return None

    layer = layer.convert("RGBA")
    alpha = layer.getchannel("A")
    bbox = alpha.getbbox()

    if not bbox:
        return None

    x1, y1, x2, y2 = bbox
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    return [int(x1), int(y1), int(x2), int(y2)]


# ---------------- REVIEW TAB HELPERS ----------------

def render_review_preview(image_path: str, payload: Dict[str, Any], selected_field: str):
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)

    field = payload["fields"][selected_field]
    proto = field.get("prototype") or {}

    bbox = proto.get("bbox")
    conf = proto.get("confidence")

    if bbox and len(bbox) == 4:
        draw.rectangle(bbox, outline="red", width=4)
        draw.text((bbox[0], max(0, bbox[1] - 18)), selected_field, fill="red")

        conf_text = _format_conf(conf)
        if conf_text:
            _draw_conf_badge(draw, bbox, conf_text, fill="red")

    return image.convert("RGB")


def build_review_field_explanation(payload: Dict[str, Any], field_name: str) -> str:
    field = payload["fields"][field_name]
    proto = field.get("prototype") or {}
    groq = field.get("groq") or {}

    lines = [
        f"Field: {field_name}",
        f"Final value: {field.get('final_value')}",
        f"Chosen source: {field.get('final_source')}",
        f"Final confidence: {field.get('final_confidence')}",
        "",
        "Prototype candidate:",
        f"  value: {proto.get('value')}",
        f"  confidence: {proto.get('confidence')}",
        f"  bbox: {proto.get('bbox')}",
        "",
        "Groq candidate:",
        f"  value: {groq.get('value')}",
        "  confidence: semantic-only",
        "  bbox: None",
    ]

    if field.get("final_source") == "groq":
        lines += [
            "",
            "Note: final value came from Groq semantic extraction.",
            "Any prototype bbox shown is supporting evidence only.",
        ]

    return "\n".join(lines)


def review_items_to_df(payload: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    for item in payload["hybrid_items"]:
        rows.append(
            [
                item.get("line_id"),
                item.get("name"),
                item.get("quantity"),
                item.get("unit_price"),
                item.get("item_total"),
            ]
        )
    return rows


def update_review_field(selected_field: str, review_payload: Dict[str, Any]):
    if not review_payload or not selected_field:
        return None, ""

    image = render_review_preview(review_payload["image_path"], review_payload, selected_field)
    expl = build_review_field_explanation(review_payload, selected_field)
    return image, expl


def save_review_feedback(
    review_payload: Dict[str, Any],
    merchant_name,
    date,
    time,
    subtotal,
    tax,
    total,
    payment_method,
    card_last4,
    receipt_number,
    items_df,
    notes,
):
    if not review_payload:
        raise gr.Error("Run extraction first.")

    corrected = deepcopy(review_payload["hybrid_canonical_output"])

    corrected["fields"]["merchant_name"] = merchant_name or None
    corrected["fields"]["date"] = date or None
    corrected["fields"]["time"] = time or None
    corrected["fields"]["subtotal"] = subtotal if subtotal not in ("", None) else None
    corrected["fields"]["tax"] = tax if tax not in ("", None) else None
    corrected["fields"]["total"] = total if total not in ("", None) else None
    corrected["fields"]["payment_method"] = payment_method or None
    corrected["fields"]["card_last4"] = card_last4 or None
    corrected["fields"]["receipt_number"] = receipt_number or None

    corrected_items = []
    if items_df:
        for row in items_df:
            if not row:
                continue
            corrected_items.append(
                {
                    "line_id": row[0],
                    "name": row[1] if len(row) > 1 and row[1] != "" else None,
                    "quantity": row[2] if len(row) > 2 and row[2] != "" else None,
                    "unit_price": row[3] if len(row) > 3 and row[3] != "" else None,
                    "item_total": row[4] if len(row) > 4 and row[4] != "" else None,
                }
            )
    corrected["items"] = corrected_items

    feedback = {
        "receipt_id": review_payload["receipt_id"],
        "image_file": review_payload["image_file"],
        "image_path": review_payload["image_path"],
        "timestamp": datetime.now().isoformat(),
        "notes": notes or None,
        "review_required_before_edit": review_payload.get("review_required"),
        "review_reasons_before_edit": review_payload.get("review_reasons", []),
        "original_hybrid_output": review_payload["hybrid_canonical_output"],
        "prototype_output": review_payload["prototype_canonical_output"],
        "groq_output": review_payload["groq_canonical_output"],
        "corrected_output": corrected,
    }

    out_dir = Path("assets/dataset/review_feedback")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{review_payload['receipt_id']}_feedback.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2)

    return f"Saved review feedback to {out_path}"


# ---------------- ANNOTATION TAB HELPERS ----------------

def annotation_tokens_to_df(annotation_payload: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    for tok in annotation_payload["tokens"]:
        bbox = tok["bbox"]
        rows.append(
            [
                tok["token_id"],
                tok["text"],
                tok["predicted_label"],
                tok["predicted_label_confidence"],
                tok["corrected_label"],
                tok["status"],
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
            ]
        )
    return rows


def manual_boxes_to_df(annotation_payload: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    for box in annotation_payload["manual_boxes"]:
        bbox = box["bbox"]
        rows.append(
            [
                box["box_id"],
                box["label"],
                box.get("text"),
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                box.get("status", "added"),
            ]
        )
    return rows


def token_dropdown_choices(annotation_payload: Dict[str, Any]):
    choices = []
    for tok in annotation_payload["tokens"]:
        short_text = tok["text"][:30]
        label = tok["corrected_label"]
        choices.append((f'{tok["token_id"]} | {short_text} | {label}', tok["token_id"]))
    return choices


def manual_box_dropdown_choices(annotation_payload: Dict[str, Any]):
    choices = []
    for box in annotation_payload["manual_boxes"]:
        choices.append((f'{box["box_id"]} | {box["label"]} | {box.get("text")}', box["box_id"]))
    return choices


def _find_token(annotation_payload: Dict[str, Any], token_id: int):
    for tok in annotation_payload["tokens"]:
        if tok["token_id"] == token_id:
            return tok
    return None


def _find_manual_box(annotation_payload: Dict[str, Any], box_id: int):
    for box in annotation_payload["manual_boxes"]:
        if box["box_id"] == box_id:
            return box
    return None


def build_token_info(annotation_payload: Dict[str, Any], token_id: Optional[int]) -> str:
    if token_id is None:
        return "No token selected."

    tok = _find_token(annotation_payload, token_id)
    if tok is None:
        return "Token not found."

    return "\n".join(
        [
            f"token_id: {tok['token_id']}",
            f"text: {tok['text']}",
            f"bbox: {tok['bbox']}",
            f"ocr_confidence: {tok['ocr_confidence']}",
            f"predicted_raw_label: {tok['predicted_raw_label']}",
            f"predicted_label: {tok['predicted_label']}",
            f"predicted_label_confidence: {tok['predicted_label_confidence']}",
            f"corrected_label: {tok['corrected_label']}",
            f"status: {tok['status']}",
        ]
    )


def render_annotation_preview(image_path: str, annotation_payload: Dict[str, Any], selected_token_id: Optional[int]):
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)

    # Draw token boxes
    for tok in annotation_payload["tokens"]:
        bbox = tok["bbox"]
        label = tok["corrected_label"]
        status = tok["status"]

        if status == "deleted":
            continue

        color = LABEL_COLORS.get(label, "gray")
        width = 2

        if selected_token_id is not None and tok["token_id"] == selected_token_id:
            color = "blue"
            width = 4

        draw.rectangle(bbox, outline=color, width=width)
        draw.text((bbox[0], max(0, bbox[1] - 14)), str(tok["token_id"]), fill=color)

        if selected_token_id is not None and tok["token_id"] == selected_token_id:
            conf_text = _format_conf(tok.get("predicted_label_confidence"))
            if conf_text:
                _draw_conf_badge(draw, bbox, conf_text, fill="blue")

    # Draw manual boxes in lime
    for box in annotation_payload["manual_boxes"]:
        bbox = box["bbox"]
        label = box["label"]
        draw.rectangle(bbox, outline="lime", width=4)
        draw.text((bbox[0], max(0, bbox[1] - 16)), label, fill="lime")

    return _editor_value_from_image(image.convert("RGB"))


def update_annotation_selected_token(token_id: int, annotation_payload: Dict[str, Any]):
    if not annotation_payload:
        return None, "", gr.update(), [], [], gr.update()

    preview = render_annotation_preview(annotation_payload["image_path"], annotation_payload, token_id)
    token_info = build_token_info(annotation_payload, token_id)

    tok = _find_token(annotation_payload, token_id)
    corrected_label = tok["corrected_label"] if tok else "other"

    return (
        preview,
        token_info,
        gr.update(choices=LABELS, value=corrected_label),
        annotation_tokens_to_df(annotation_payload),
        manual_boxes_to_df(annotation_payload),
        gr.update(choices=manual_box_dropdown_choices(annotation_payload)),
    )


def apply_token_label(token_id: int, new_label: str, annotation_payload: Dict[str, Any]):
    if not annotation_payload:
        raise gr.Error("Run extraction first.")
    if token_id is None:
        raise gr.Error("Select a token first.")
    if not new_label:
        raise gr.Error("Choose a label.")

    tok = _find_token(annotation_payload, token_id)
    if tok is None:
        raise gr.Error("Token not found.")

    tok["corrected_label"] = new_label
    tok["status"] = "corrected"

    preview = render_annotation_preview(annotation_payload["image_path"], annotation_payload, token_id)
    token_info = build_token_info(annotation_payload, token_id)

    return (
        annotation_payload,
        preview,
        token_info,
        annotation_tokens_to_df(annotation_payload),
        "Updated token label.",
    )


def replace_token_bbox(editor_value, token_id: int, annotation_payload: Dict[str, Any]):
    if not annotation_payload:
        raise gr.Error("Run extraction first.")
    if token_id is None:
        raise gr.Error("Select a token first.")

    bbox = _extract_bbox_from_editor(editor_value)
    if bbox is None:
        raise gr.Error("Draw a replacement box on the annotation preview first.")

    tok = _find_token(annotation_payload, token_id)
    if tok is None:
        raise gr.Error("Token not found.")

    tok["bbox"] = bbox
    tok["status"] = "bbox_corrected"

    preview = render_annotation_preview(annotation_payload["image_path"], annotation_payload, token_id)
    token_info = build_token_info(annotation_payload, token_id)

    return (
        annotation_payload,
        preview,
        token_info,
        annotation_tokens_to_df(annotation_payload),
        f"Updated bbox for token {token_id}.",
    )


def delete_selected_token(token_id: int, annotation_payload: Dict[str, Any]):
    if not annotation_payload:
        raise gr.Error("Run extraction first.")
    if token_id is None:
        raise gr.Error("Select a token first.")

    tok = _find_token(annotation_payload, token_id)
    if tok is None:
        raise gr.Error("Token not found.")

    tok["corrected_label"] = "ignore"
    tok["status"] = "deleted"

    preview = render_annotation_preview(annotation_payload["image_path"], annotation_payload, token_id)
    token_info = build_token_info(annotation_payload, token_id)

    return (
        annotation_payload,
        preview,
        token_info,
        annotation_tokens_to_df(annotation_payload),
        "Selected token marked as deleted.",
    )


def restore_selected_token(token_id: int, annotation_payload: Dict[str, Any]):
    if not annotation_payload:
        raise gr.Error("Run extraction first.")
    if token_id is None:
        raise gr.Error("Select a token first.")

    tok = _find_token(annotation_payload, token_id)
    if tok is None:
        raise gr.Error("Token not found.")

    tok["corrected_label"] = tok["predicted_label"] or "other"
    tok["status"] = "restored"

    preview = render_annotation_preview(annotation_payload["image_path"], annotation_payload, token_id)
    token_info = build_token_info(annotation_payload, token_id)

    return (
        annotation_payload,
        preview,
        token_info,
        annotation_tokens_to_df(annotation_payload),
        "Selected token restored.",
    )


def bulk_relabel_tokens(from_label: str, to_label: str, annotation_payload: Dict[str, Any], selected_token_id: Optional[int]):
    if not annotation_payload:
        raise gr.Error("Run extraction first.")
    if not from_label or not to_label:
        raise gr.Error("Choose both source label and target label.")

    changed = 0
    for tok in annotation_payload["tokens"]:
        if tok["status"] == "deleted":
            continue
        if tok["corrected_label"] == from_label:
            tok["corrected_label"] = to_label
            tok["status"] = "bulk_corrected"
            changed += 1

    preview = render_annotation_preview(annotation_payload["image_path"], annotation_payload, selected_token_id)

    return (
        annotation_payload,
        preview,
        annotation_tokens_to_df(annotation_payload),
        f"Bulk relabeled {changed} tokens from '{from_label}' to '{to_label}'.",
    )


def add_manual_box(editor_value, manual_label: str, manual_text: str, annotation_payload: Dict[str, Any], selected_token_id: Optional[int]):
    if not annotation_payload:
        raise gr.Error("Run extraction first.")
    if not manual_label:
        raise gr.Error("Choose a label for the new manual box.")
    if not manual_text or not str(manual_text).strip():
        raise gr.Error("Enter the text inside the manual box.")

    bbox = _extract_bbox_from_editor(editor_value)
    if bbox is None:
        raise gr.Error("Draw a new box on the annotation preview first.")

    next_id = 1
    if annotation_payload["manual_boxes"]:
        next_id = max(box["box_id"] for box in annotation_payload["manual_boxes"]) + 1

    annotation_payload["manual_boxes"].append(
        {
            "box_id": next_id,
            "label": manual_label,
            "text": str(manual_text).strip(),
            "bbox": bbox,
            "status": "added",
            "created_at": datetime.now().isoformat(),
        }
    )

    preview = render_annotation_preview(annotation_payload["image_path"], annotation_payload, selected_token_id)

    return (
        annotation_payload,
        preview,
        manual_boxes_to_df(annotation_payload),
        gr.update(choices=manual_box_dropdown_choices(annotation_payload)),
        "Added manual box.",
    )


def delete_manual_box(box_id: int, annotation_payload: Dict[str, Any], selected_token_id: Optional[int]):
    if not annotation_payload:
        raise gr.Error("Run extraction first.")
    if box_id is None:
        raise gr.Error("Select a manual box first.")

    kept = [b for b in annotation_payload["manual_boxes"] if b["box_id"] != box_id]
    annotation_payload["manual_boxes"] = kept

    preview = render_annotation_preview(annotation_payload["image_path"], annotation_payload, selected_token_id)

    return (
        annotation_payload,
        preview,
        manual_boxes_to_df(annotation_payload),
        gr.update(choices=manual_box_dropdown_choices(annotation_payload), value=None),
        "Deleted selected manual box.",
    )


def save_annotation(annotation_payload: Dict[str, Any], notes: str):
    if not annotation_payload:
        raise gr.Error("Run extraction first.")

    out = {
        "receipt_id": annotation_payload["receipt_id"],
        "image_file": annotation_payload["image_file"],
        "image_path": annotation_payload["image_path"],
        "timestamp": datetime.now().isoformat(),
        "notes": notes or None,
        "tokens": annotation_payload["tokens"],
        "manual_boxes": annotation_payload["manual_boxes"],
        "label_set": annotation_payload["label_set"],
    }

    out_dir = Path("assets/dataset/annotation_feedback")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{annotation_payload['receipt_id']}_annotation.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return f"Saved annotation to {out_path}"


# ---------------- LOAD BOTH TAB PAYLOADS ----------------

def run_all_pipelines(image_path: str):
    if not image_path:
        raise gr.Error("Please upload a receipt image.")

    review_payload = build_hybrid_ui_payload(image_path)
    annotation_payload = build_annotation_payload(image_path)

    # Review defaults
    default_review_field = "merchant_name"
    review_preview = render_review_preview(image_path, review_payload, default_review_field)
    review_expl = build_review_field_explanation(review_payload, default_review_field)
    review_required = "Yes" if review_payload.get("review_required") else "No"
    review_reasons = ", ".join(review_payload.get("review_reasons", []))
    review_field_values = [review_payload["fields"][f]["final_value"] for f in REVIEW_FIELD_NAMES]
    review_items_df = review_items_to_df(review_payload)

    # Annotation defaults
    token_choices = token_dropdown_choices(annotation_payload)
    default_token = token_choices[0][1] if token_choices else None
    annotation_preview = render_annotation_preview(image_path, annotation_payload, default_token)
    token_info = build_token_info(annotation_payload, default_token)
    corrected_label_value = _find_token(annotation_payload, default_token)["corrected_label"] if default_token is not None else "other"
    token_table = annotation_tokens_to_df(annotation_payload)
    manual_table = manual_boxes_to_df(annotation_payload)
    manual_choices = manual_box_dropdown_choices(annotation_payload)
    guidelines = annotation_payload["guidelines"]

    return (
        review_payload,
        annotation_payload,
        review_preview,
        gr.update(choices=REVIEW_FIELD_NAMES, value=default_review_field),
        review_expl,
        review_required,
        review_reasons,
        *review_field_values,
        review_items_df,
        review_payload["prototype_canonical_output"],
        review_payload["groq_canonical_output"],
        review_payload["hybrid_canonical_output"],
        annotation_preview,
        gr.update(choices=token_choices, value=default_token),
        token_info,
        gr.update(choices=LABELS, value=corrected_label_value),
        token_table,
        manual_table,
        gr.update(choices=manual_choices, value=None),
        guidelines,
        "Ready.",
    )


def create_app():
    with gr.Blocks(title="Receipt Review + Annotation UI") as demo:
        gr.Markdown("# Hybrid Receipt Review and Training Annotation")

        review_payload_state = gr.State()
        annotation_payload_state = gr.State()

        with gr.Row():
            image_input = gr.Image(type="filepath", label="Upload Receipt")
            run_btn = gr.Button("Run Extraction + Load Annotation", variant="primary")

        with gr.Tabs():
            # -------- TAB 1: FINAL REVIEW --------
            with gr.Tab("Final Review"):
                with gr.Row():
                    with gr.Column(scale=1):
                        review_field_selector = gr.Dropdown(label="Highlight Field", choices=[])
                        review_required = gr.Textbox(label="Review Required", interactive=False)
                        review_reasons = gr.Textbox(label="Review Reasons", interactive=False)

                    with gr.Column(scale=1):
                        review_preview = gr.Image(label="Evidence Overlay Preview")
                        review_explanation = gr.Textbox(label="Field Explainability", lines=16, interactive=False)

                with gr.Row():
                    with gr.Column():
                        merchant_name = gr.Textbox(label="merchant_name")
                        date = gr.Textbox(label="date")
                        time = gr.Textbox(label="time")
                        subtotal = gr.Textbox(label="subtotal")
                        tax = gr.Textbox(label="tax")
                        total = gr.Textbox(label="total")
                        payment_method = gr.Textbox(label="payment_method")
                        card_last4 = gr.Textbox(label="card_last4")
                        receipt_number = gr.Textbox(label="receipt_number")
                        review_notes = gr.Textbox(label="Reviewer Notes", lines=4)
                        save_review_btn = gr.Button("Save Final Review Feedback")

                    with gr.Column():
                        review_items_df = gr.Dataframe(
                            headers=["line_id", "name", "quantity", "unit_price", "item_total"],
                            datatype=["number", "str", "number", "number", "number"],
                            row_count=(1, "dynamic"),
                            col_count=(5, "fixed"),
                            label="Editable Final Items",
                        )

                with gr.Accordion("Prototype Output", open=False):
                    prototype_json = gr.JSON()
                with gr.Accordion("Groq Output", open=False):
                    groq_json = gr.JSON()
                with gr.Accordion("Hybrid Output", open=False):
                    hybrid_json = gr.JSON()

                review_save_status = gr.Textbox(label="Review Save Status", interactive=False)

            # -------- TAB 2: TRAINING ANNOTATION --------
            with gr.Tab("Training Annotation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        annotation_editor = gr.ImageEditor(
                            label="Annotation Preview (all predicted boxes)",
                            type="pil",
                            interactive=True,
                            height=700,
                        )
                    with gr.Column(scale=1):
                        guidelines_box = gr.Textbox(label="Annotation Guidelines", lines=20, interactive=False)
                        token_selector = gr.Dropdown(label="Select Token", choices=[])
                        token_info = gr.Textbox(label="Selected Token Info", lines=12, interactive=False)

                        corrected_label_dropdown = gr.Dropdown(label="Corrected Label", choices=LABELS)
                        apply_label_btn = gr.Button("Apply Label to Selected Token")
                        replace_bbox_btn = gr.Button("Replace Selected Token BBox with Drawn Box")

                        with gr.Row():
                            delete_token_btn = gr.Button("Delete Selected Token")
                            restore_token_btn = gr.Button("Restore Selected Token")

                        with gr.Row():
                            bulk_from_label = gr.Dropdown(label="Bulk From Label", choices=LABELS)
                            bulk_to_label = gr.Dropdown(label="Bulk To Label", choices=LABELS)
                        bulk_relabel_btn = gr.Button("Bulk Relabel")

                        manual_label_dropdown = gr.Dropdown(label="New Manual Box Label", choices=LABELS)
                        manual_text = gr.Textbox(label="New Manual Box Text")
                        add_manual_box_btn = gr.Button("Add Drawn Box as Manual Annotation")

                        manual_box_selector = gr.Dropdown(label="Select Manual Box", choices=[])
                        delete_manual_box_btn = gr.Button("Delete Selected Manual Box")

                        annotation_notes = gr.Textbox(label="Annotation Notes", lines=4)
                        save_annotation_btn = gr.Button("Save Training Annotation")
                        annotation_status = gr.Textbox(label="Annotation Status", interactive=False)

                with gr.Row():
                    token_table = gr.Dataframe(
                        headers=["token_id", "text", "predicted_label", "pred_conf", "corrected_label", "status", "x1", "y1", "x2", "y2"],
                        datatype=["number", "str", "str", "number", "str", "str", "number", "number", "number", "number"],
                        row_count=(1, "dynamic"),
                        col_count=(10, "fixed"),
                        label="Token-Level Annotation Table",
                        interactive=False,
                    )

                with gr.Row():
                    manual_boxes_table = gr.Dataframe(
                        headers=["id", "label", "text", "x1", "y1", "x2", "y2", "status"],
                        datatype=["number", "str", "str", "number", "number", "number", "number", "str"],
                        row_count=(1, "dynamic"),
                        col_count=(8, "fixed"),
                        label="Manual Added Boxes",
                        interactive=False,
                    )

        run_btn.click(
            fn=run_all_pipelines,
            inputs=[image_input],
            outputs=[
                review_payload_state,
                annotation_payload_state,
                review_preview,
                review_field_selector,
                review_explanation,
                review_required,
                review_reasons,
                merchant_name,
                date,
                time,
                subtotal,
                tax,
                total,
                payment_method,
                card_last4,
                receipt_number,
                review_items_df,
                prototype_json,
                groq_json,
                hybrid_json,
                annotation_editor,
                token_selector,
                token_info,
                corrected_label_dropdown,
                token_table,
                manual_boxes_table,
                manual_box_selector,
                guidelines_box,
                annotation_status,
            ],
        )

        # Review events
        review_field_selector.change(
            fn=update_review_field,
            inputs=[review_field_selector, review_payload_state],
            outputs=[review_preview, review_explanation],
        )

        save_review_btn.click(
            fn=save_review_feedback,
            inputs=[
                review_payload_state,
                merchant_name,
                date,
                time,
                subtotal,
                tax,
                total,
                payment_method,
                card_last4,
                receipt_number,
                review_items_df,
                review_notes,
            ],
            outputs=[review_save_status],
        )

        # Annotation events
        token_selector.change(
            fn=update_annotation_selected_token,
            inputs=[token_selector, annotation_payload_state],
            outputs=[annotation_editor, token_info, corrected_label_dropdown, token_table, manual_boxes_table, manual_box_selector],
        )

        apply_label_btn.click(
            fn=apply_token_label,
            inputs=[token_selector, corrected_label_dropdown, annotation_payload_state],
            outputs=[annotation_payload_state, annotation_editor, token_info, token_table, annotation_status],
        )

        replace_bbox_btn.click(
            fn=replace_token_bbox,
            inputs=[annotation_editor, token_selector, annotation_payload_state],
            outputs=[annotation_payload_state, annotation_editor, token_info, token_table, annotation_status],
        )

        delete_token_btn.click(
            fn=delete_selected_token,
            inputs=[token_selector, annotation_payload_state],
            outputs=[annotation_payload_state, annotation_editor, token_info, token_table, annotation_status],
        )

        restore_token_btn.click(
            fn=restore_selected_token,
            inputs=[token_selector, annotation_payload_state],
            outputs=[annotation_payload_state, annotation_editor, token_info, token_table, annotation_status],
        )

        bulk_relabel_btn.click(
            fn=bulk_relabel_tokens,
            inputs=[bulk_from_label, bulk_to_label, annotation_payload_state, token_selector],
            outputs=[annotation_payload_state, annotation_editor, token_table, annotation_status],
        )

        add_manual_box_btn.click(
            fn=add_manual_box,
            inputs=[annotation_editor, manual_label_dropdown, manual_text, annotation_payload_state, token_selector],
            outputs=[annotation_payload_state, annotation_editor, manual_boxes_table, manual_box_selector, annotation_status],
        )

        delete_manual_box_btn.click(
            fn=delete_manual_box,
            inputs=[manual_box_selector, annotation_payload_state, token_selector],
            outputs=[annotation_payload_state, annotation_editor, manual_boxes_table, manual_box_selector, annotation_status],
        )

        save_annotation_btn.click(
            fn=save_annotation,
            inputs=[annotation_payload_state, annotation_notes],
            outputs=[annotation_status],
        )

    return demo