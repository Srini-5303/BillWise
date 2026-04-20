from __future__ import annotations

import json
from pathlib import Path

from billwise.common.artifacts import save_processed_receipt_artifact
from billwise.common.config import get_config
from billwise.common.gcs_storage import build_blob_path, gcs_enabled, upload_json
from billwise.common.ids import new_id
from billwise.common.logging import get_logger
from billwise.common.repositories import ReceiptRepository
from billwise.common.schemas import ReceiptRecord
from billwise.common.storage import copy_to_raw_storage, ensure_directories
from billwise.extraction.legacy_runtime import load_hybrid_payload_builder
from billwise.extraction.transform import build_receipt_from_hybrid_payload
from billwise.preprocessing.artifacts import save_preprocessing_artifact
from billwise.preprocessing.router import run_preprocessing_router


def _save_hybrid_payload_artifact(receipt_id: str, payload: dict) -> Path:
    cfg = get_config()
    path = cfg.paths.processed_dir / f"{receipt_id}.hybrid_payload.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if gcs_enabled():
        upload_json(
            payload,
            build_blob_path("processed", "hybrid_payloads", f"{receipt_id}.json"),
        )

    return path


def extract_receipt_with_hybrid(
    image_path: str | Path,
    persist: bool = True,
) -> tuple[ReceiptRecord, dict]:
    logger = get_logger("billwise.extraction.service")
    ensure_directories()

    source_path = Path(image_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Receipt image not found: {source_path}")

    receipt_id = new_id("rcpt")
    suffix = source_path.suffix if source_path.suffix else ".jpg"

    # Always preserve the original input in raw storage first.
    stored_image_path = copy_to_raw_storage(source_path, f"{receipt_id}{suffix}")

    # Run preprocessing router on the stored raw image.
    preprocessing_result = run_preprocessing_router(stored_image_path)

    selected_input_path = Path(preprocessing_result.selected_ocr_path)
    logger.info(
        "Preprocessing | performed=%s | selected_ocr_variant=%s | selected_vlm_variant=%s",
        preprocessing_result.performed,
        preprocessing_result.selected_ocr_variant,
        preprocessing_result.selected_vlm_variant,
    )

    build_hybrid_ui_payload = load_hybrid_payload_builder()
    payload = build_hybrid_ui_payload(str(selected_input_path))

    receipt = build_receipt_from_hybrid_payload(
        payload=payload,
        image_path=str(stored_image_path),  # keep original stored raw path as canonical image
        receipt_id=receipt_id,
    )

    if persist:
        ReceiptRepository.upsert_receipt(receipt)
        save_processed_receipt_artifact(receipt)
        _save_hybrid_payload_artifact(receipt_id, payload)
        save_preprocessing_artifact(receipt_id, preprocessing_result)

    return receipt, payload