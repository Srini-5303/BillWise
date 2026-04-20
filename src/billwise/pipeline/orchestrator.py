from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from billwise.categorization.service import categorize_receipt_items
from billwise.common.db import init_db
from billwise.common.gcs_storage import build_blob_path, gcs_enabled, upload_json
from billwise.common.ids import new_id
from billwise.common.logging import get_logger
from billwise.common.repositories import PipelineRunRepository, ReceiptRepository
from billwise.common.schemas import PipelineRun
from billwise.common.storage import ensure_directories
from billwise.extraction.service import extract_receipt_with_hybrid


@dataclass
class PipelineResult:
    receipt_id: str
    vendor_name: str | None
    item_count: int
    category_count: int
    requires_review: bool
    processing_status: str
    review_status: str


def _save_stage(
    receipt_id: str,
    pipeline_run_id: str,
    stage: str,
    status: str,
    error_message: str | None = None,
) -> None:
    now = datetime.utcnow()
    run = PipelineRun(
        run_id=f"{pipeline_run_id}_{stage}",
        receipt_id=receipt_id,
        stage=stage,
        status=status,
        started_at=now,
        finished_at=now,
        error_message=error_message,
    )
    PipelineRunRepository.save_run(run)

    if gcs_enabled():
        blob_path = build_blob_path(
            "pipeline_runs",
            now.strftime("%Y"),
            now.strftime("%m"),
            now.strftime("%d"),
            f"{run.run_id}.json",
        )
        upload_json(run.model_dump(mode="json"), blob_path)


def determine_final_status(
    extraction_requires_review: bool,
    categorization_requires_review: bool,
) -> tuple[str, str, bool]:
    final_requires_review = extraction_requires_review or categorization_requires_review
    processing_status = "ready_for_dashboard"
    review_status = "pending" if final_requires_review else "approved"
    return processing_status, review_status, final_requires_review


def run_billwise_pipeline(image_path: str | Path) -> PipelineResult:
    logger = get_logger("billwise.pipeline")

    ensure_directories()
    init_db()

    pipeline_run_id = new_id("run")

    receipt, _payload = extract_receipt_with_hybrid(image_path=image_path, persist=True)
    _save_stage(receipt.receipt_id, pipeline_run_id, "extraction", "success")

    logger.info("Extraction completed for receipt_id=%s", receipt.receipt_id)

    predictions = categorize_receipt_items(receipt.receipt_id)
    _save_stage(receipt.receipt_id, pipeline_run_id, "categorization", "success")

    category_review_needed = any(p.needs_human_review for p in predictions)
    processing_status, review_status, final_requires_review = determine_final_status(
        extraction_requires_review=receipt.requires_review,
        categorization_requires_review=category_review_needed,
    )

    ReceiptRepository.update_receipt_status(
        receipt_id=receipt.receipt_id,
        processing_status=processing_status,
        review_status=review_status,
        requires_review=final_requires_review,
    )
    _save_stage(receipt.receipt_id, pipeline_run_id, "finalize", "success")

    logger.info(
        "Pipeline completed | receipt_id=%s | items=%s | categories=%s | requires_review=%s",
        receipt.receipt_id,
        len(receipt.items),
        len(predictions),
        final_requires_review,
    )

    return PipelineResult(
        receipt_id=receipt.receipt_id,
        vendor_name=receipt.vendor_name,
        item_count=len(receipt.items),
        category_count=len(predictions),
        requires_review=final_requires_review,
        processing_status=processing_status,
        review_status=review_status,
    )