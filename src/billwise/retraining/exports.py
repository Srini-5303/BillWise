from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from billwise.common.config import get_config
from billwise.common.db import get_connection
from billwise.common.gcs_storage import build_blob_path, gcs_enabled, upload_file, upload_json
from billwise.common.logging import get_logger


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_category_feedback_df() -> pd.DataFrame:
    with get_connection() as conn:
        query = """
        SELECT
            hv.item_id,
            hv.receipt_id,
            hv.raw_item_text,
            li.normalized_name,
            hv.original_category,
            hv.validated_category,
            hv.validator_note,
            hv.validated_at,
            r.vendor_name,
            r.receipt_date,
            li.quantity,
            li.unit_price,
            li.item_total
        FROM human_validations hv
        LEFT JOIN line_items li
            ON hv.item_id = li.item_id
        LEFT JOIN receipts r
            ON hv.receipt_id = r.receipt_id
        WHERE hv.validation_type = 'category'
        ORDER BY hv.validated_at DESC
        """
        return pd.read_sql(query, conn)


def _load_ocr_feedback_df() -> pd.DataFrame:
    with get_connection() as conn:
        query = """
        SELECT
            oc.receipt_id,
            oc.field_name,
            oc.original_value,
            oc.corrected_value,
            oc.corrected_at,
            r.vendor_name,
            r.receipt_date,
            r.image_path,
            r.extraction_method
        FROM ocr_corrections oc
        LEFT JOIN receipts r
            ON oc.receipt_id = r.receipt_id
        ORDER BY oc.corrected_at DESC
        """
        return pd.read_sql(query, conn)


def _category_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "task": "receipt_item_categorization",
                "item_id": str(row.get("item_id", "")),
                "receipt_id": str(row.get("receipt_id", "")),
                "raw_item_text": row.get("raw_item_text"),
                "normalized_item_text": row.get("normalized_name"),
                "original_category": row.get("original_category"),
                "validated_category": row.get("validated_category"),
                "validator_note": row.get("validator_note"),
                "validated_at": row.get("validated_at"),
                "vendor_name": row.get("vendor_name"),
                "receipt_date": row.get("receipt_date"),
                "quantity": row.get("quantity"),
                "unit_price": row.get("unit_price"),
                "item_total": row.get("item_total"),
                "source": "dashboard_validation",
            }
        )
    return records


def _ocr_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "task": "receipt_field_correction",
                "receipt_id": str(row.get("receipt_id", "")),
                "field_name": row.get("field_name"),
                "original_value": row.get("original_value"),
                "corrected_value": row.get("corrected_value"),
                "corrected_at": row.get("corrected_at"),
                "vendor_name": row.get("vendor_name"),
                "receipt_date": row.get("receipt_date"),
                "image_path": row.get("image_path"),
                "extraction_method": row.get("extraction_method"),
                "source": "dashboard_ocr_correction",
            }
        )
    return records


def export_retraining_feedback() -> dict:
    logger = get_logger("billwise.retraining.export")
    cfg = get_config()

    ts = datetime.utcnow()
    stamp = ts.strftime("%Y%m%dT%H%M%S")

    export_root = cfg.paths.exports_dir / "retraining" / stamp
    export_root.mkdir(parents=True, exist_ok=True)

    df_cat = _load_category_feedback_df()
    df_ocr = _load_ocr_feedback_df()

    cat_records = _category_records(df_cat)
    ocr_records = _ocr_records(df_ocr)

    cat_jsonl = export_root / "category_feedback.jsonl"
    cat_csv = export_root / "category_feedback.csv"
    ocr_jsonl = export_root / "ocr_feedback.jsonl"
    ocr_csv = export_root / "ocr_feedback.csv"
    manifest_path = export_root / "manifest.json"

    _write_jsonl(cat_jsonl, cat_records)
    _write_jsonl(ocr_jsonl, ocr_records)

    df_cat.to_csv(cat_csv, index=False)
    df_ocr.to_csv(ocr_csv, index=False)

    manifest = {
        "generated_at": ts.isoformat(),
        "category_feedback_count": len(cat_records),
        "ocr_feedback_count": len(ocr_records),
        "local_paths": {
            "category_feedback_jsonl": str(cat_jsonl),
            "category_feedback_csv": str(cat_csv),
            "ocr_feedback_jsonl": str(ocr_jsonl),
            "ocr_feedback_csv": str(ocr_csv),
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Category feedback rows: %s", len(cat_records))
    logger.info("OCR feedback rows: %s", len(ocr_records))
    logger.info("Local retraining export folder: %s", export_root)

    gcs_manifest = {}

    if gcs_enabled():
        category_prefix = build_blob_path("training", "snapshots", stamp, "category")
        ocr_prefix = build_blob_path("training", "snapshots", stamp, "ocr")
        latest_prefix = build_blob_path("training", "latest")

        gcs_manifest["category_feedback_jsonl"] = upload_file(
            cat_jsonl, f"{category_prefix}/category_feedback.jsonl", "application/x-ndjson"
        )
        gcs_manifest["category_feedback_csv"] = upload_file(
            cat_csv, f"{category_prefix}/category_feedback.csv", "text/csv"
        )
        gcs_manifest["ocr_feedback_jsonl"] = upload_file(
            ocr_jsonl, f"{ocr_prefix}/ocr_feedback.jsonl", "application/x-ndjson"
        )
        gcs_manifest["ocr_feedback_csv"] = upload_file(
            ocr_csv, f"{ocr_prefix}/ocr_feedback.csv", "text/csv"
        )

        upload_file(cat_jsonl, f"{latest_prefix}/category_feedback.jsonl", "application/x-ndjson")
        upload_file(cat_csv, f"{latest_prefix}/category_feedback.csv", "text/csv")
        upload_file(ocr_jsonl, f"{latest_prefix}/ocr_feedback.jsonl", "application/x-ndjson")
        upload_file(ocr_csv, f"{latest_prefix}/ocr_feedback.csv", "text/csv")

        gcs_manifest["manifest_json"] = upload_json(
            {
                **manifest,
                "gcs_paths": gcs_manifest,
            },
            build_blob_path("training", "snapshots", stamp, "manifest.json"),
        )

    return {
        "generated_at": ts.isoformat(),
        "category_feedback_count": len(cat_records),
        "ocr_feedback_count": len(ocr_records),
        "export_dir": str(export_root),
        "gcs_paths": gcs_manifest,
    }