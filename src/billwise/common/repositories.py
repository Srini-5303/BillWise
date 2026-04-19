from __future__ import annotations

import json
from typing import Iterable

from billwise.common.db import get_connection
from billwise.common.schemas import (
    CategoryPrediction,
    ExtractedField,
    LineItem,
    PipelineRun,
    ReceiptRecord,
    ReviewChange,
)


def _to_json(value: dict | list | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _bool_to_int(value: bool) -> int:
    return 1 if value else 0


class ReceiptRepository:
    @staticmethod
    def upsert_receipt(receipt: ReceiptRecord) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO receipts (
                    receipt_id, image_path, source, upload_timestamp,
                    processing_status, review_status, vendor_name,
                    receipt_date, receipt_time, subtotal, tax, total,
                    payment_method, card_last4, receipt_number,
                    extraction_method, requires_review
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.receipt_id,
                    receipt.image_path,
                    receipt.source,
                    receipt.upload_timestamp.isoformat(),
                    receipt.processing_status,
                    receipt.review_status,
                    receipt.vendor_name,
                    receipt.receipt_date,
                    receipt.receipt_time,
                    receipt.subtotal,
                    receipt.tax,
                    receipt.total,
                    receipt.payment_method,
                    receipt.card_last4,
                    receipt.receipt_number,
                    receipt.extraction_method,
                    _bool_to_int(receipt.requires_review),
                ),
            )

            conn.execute("DELETE FROM receipt_fields WHERE receipt_id = ?", (receipt.receipt_id,))
            conn.execute("DELETE FROM line_items WHERE receipt_id = ?", (receipt.receipt_id,))

            for field in receipt.fields:
                conn.execute(
                    """
                    INSERT INTO receipt_fields (
                        receipt_id, field_name, field_value, confidence, bbox_json, source_model
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt.receipt_id,
                        field.field_name,
                        field.field_value,
                        field.confidence,
                        _to_json(field.bbox.model_dump() if field.bbox else None),
                        field.source_model,
                    ),
                )

            for item in receipt.items:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO line_items (
                        item_id, receipt_id, raw_name, normalized_name,
                        quantity, unit_price, item_total, item_confidence, item_source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.item_id,
                        receipt.receipt_id,
                        item.raw_name,
                        item.normalized_name,
                        item.quantity,
                        item.unit_price,
                        item.item_total,
                        item.item_confidence,
                        item.item_source,
                    ),
                )

            conn.commit()

    @staticmethod
    def get_receipt(receipt_id: str) -> ReceiptRecord | None:
        with get_connection() as conn:
            receipt_row = conn.execute(
                "SELECT * FROM receipts WHERE receipt_id = ?",
                (receipt_id,),
            ).fetchone()

            if receipt_row is None:
                return None

            field_rows = conn.execute(
                "SELECT * FROM receipt_fields WHERE receipt_id = ? ORDER BY id",
                (receipt_id,),
            ).fetchall()

            item_rows = conn.execute(
                "SELECT * FROM line_items WHERE receipt_id = ? ORDER BY item_id",
                (receipt_id,),
            ).fetchall()

        fields: list[ExtractedField] = []
        for row in field_rows:
            bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else None
            fields.append(
                ExtractedField(
                    field_name=row["field_name"],
                    field_value=row["field_value"],
                    confidence=row["confidence"],
                    bbox=bbox,
                    source_model=row["source_model"],
                )
            )

        items: list[LineItem] = []
        for row in item_rows:
            items.append(
                LineItem(
                    item_id=row["item_id"],
                    raw_name=row["raw_name"],
                    normalized_name=row["normalized_name"],
                    quantity=row["quantity"],
                    unit_price=row["unit_price"],
                    item_total=row["item_total"],
                    item_confidence=row["item_confidence"],
                    item_source=row["item_source"],
                )
            )

        return ReceiptRecord(
            receipt_id=receipt_row["receipt_id"],
            image_path=receipt_row["image_path"],
            source=receipt_row["source"],
            upload_timestamp=receipt_row["upload_timestamp"],
            processing_status=receipt_row["processing_status"],
            review_status=receipt_row["review_status"],
            vendor_name=receipt_row["vendor_name"],
            receipt_date=receipt_row["receipt_date"],
            receipt_time=receipt_row["receipt_time"],
            subtotal=receipt_row["subtotal"],
            tax=receipt_row["tax"],
            total=receipt_row["total"],
            payment_method=receipt_row["payment_method"],
            card_last4=receipt_row["card_last4"],
            receipt_number=receipt_row["receipt_number"],
            extraction_method=receipt_row["extraction_method"],
            requires_review=bool(receipt_row["requires_review"]),
            fields=fields,
            items=items,
        )

    @staticmethod
    def list_receipts(limit: int = 100) -> list[ReceiptRecord]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT receipt_id FROM receipts ORDER BY upload_timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()

        results: list[ReceiptRecord] = []
        for row in rows:
            receipt = ReceiptRepository.get_receipt(row["receipt_id"])
            if receipt is not None:
                results.append(receipt)
        return results

    @staticmethod
    def update_receipt_status(
        receipt_id: str,
        processing_status: str | None = None,
        review_status: str | None = None,
        requires_review: bool | None = None,
    ) -> None:
        sets = []
        values = []

        if processing_status is not None:
            sets.append("processing_status = ?")
            values.append(processing_status)

        if review_status is not None:
            sets.append("review_status = ?")
            values.append(review_status)

        if requires_review is not None:
            sets.append("requires_review = ?")
            values.append(_bool_to_int(requires_review))

        if not sets:
            return

        values.append(receipt_id)

        with get_connection() as conn:
            conn.execute(
                f"UPDATE receipts SET {', '.join(sets)} WHERE receipt_id = ?",
                tuple(values),
            )
            conn.commit()


class CategorizationRepository:
    @staticmethod
    def replace_for_items(predictions: Iterable[CategoryPrediction]) -> None:
        predictions = list(predictions)
        item_ids = [p.item_id for p in predictions]

        with get_connection() as conn:
            if item_ids:
                placeholders = ",".join("?" for _ in item_ids)
                conn.execute(
                    f"DELETE FROM categorizations WHERE item_id IN ({placeholders})",
                    tuple(item_ids),
                )

            for pred in predictions:
                conn.execute(
                    """
                    INSERT INTO categorizations (
                        item_id, predicted_category, category_confidence,
                        top_k_scores_json, categorizer_model,
                        needs_human_review, final_category
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pred.item_id,
                        pred.predicted_category,
                        pred.category_confidence,
                        _to_json(pred.top_k_scores),
                        pred.categorizer_model,
                        _bool_to_int(pred.needs_human_review),
                        pred.final_category,
                    ),
                )

            conn.commit()

    @staticmethod
    def get_for_receipt(receipt_id: str) -> list[CategoryPrediction]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT c.*
                FROM categorizations c
                INNER JOIN line_items li ON li.item_id = c.item_id
                WHERE li.receipt_id = ?
                ORDER BY c.id
                """,
                (receipt_id,),
            ).fetchall()

        results: list[CategoryPrediction] = []
        for row in rows:
            top_k_scores = json.loads(row["top_k_scores_json"]) if row["top_k_scores_json"] else {}
            results.append(
                CategoryPrediction(
                    item_id=row["item_id"],
                    predicted_category=row["predicted_category"],
                    category_confidence=row["category_confidence"],
                    top_k_scores=top_k_scores,
                    categorizer_model=row["categorizer_model"],
                    needs_human_review=bool(row["needs_human_review"]),
                    final_category=row["final_category"],
                )
            )
        return results


class ReviewRepository:
    @staticmethod
    def log_change(change: ReviewChange) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO review_logs (
                    entity_type, entity_id, field_name, old_value, new_value,
                    reviewed_at, review_source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    change.entity_type,
                    change.entity_id,
                    change.field_name,
                    None if change.old_value is None else str(change.old_value),
                    None if change.new_value is None else str(change.new_value),
                    change.reviewed_at.isoformat(),
                    change.review_source,
                ),
            )
            conn.commit()


class PipelineRunRepository:
    @staticmethod
    def save_run(run: PipelineRun) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pipeline_runs (
                    run_id, receipt_id, stage, status,
                    started_at, finished_at, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.receipt_id,
                    run.stage,
                    run.status,
                    run.started_at.isoformat(),
                    run.finished_at.isoformat() if run.finished_at else None,
                    run.error_message,
                ),
            )
            conn.commit()

    @staticmethod
    def list_runs_for_receipt(receipt_id: str) -> list[PipelineRun]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM pipeline_runs
                WHERE receipt_id = ?
                ORDER BY started_at
                """,
                (receipt_id,),
            ).fetchall()

        runs: list[PipelineRun] = []
        for row in rows:
            runs.append(
                PipelineRun(
                    run_id=row["run_id"],
                    receipt_id=row["receipt_id"],
                    stage=row["stage"],
                    status=row["status"],
                    started_at=row["started_at"],
                    finished_at=row["finished_at"],
                    error_message=row["error_message"],
                )
            )
        return runs