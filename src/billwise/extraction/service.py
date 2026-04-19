from __future__ import annotations

import json
from pathlib import Path

from billwise.common.artifacts import save_processed_receipt_artifact
from billwise.common.config import get_config
from billwise.common.ids import new_id
from billwise.common.repositories import ReceiptRepository
from billwise.common.schemas import ReceiptRecord
from billwise.common.storage import copy_to_raw_storage, ensure_directories
from billwise.extraction.legacy_runtime import load_hybrid_payload_builder
from billwise.extraction.transform import build_receipt_from_hybrid_payload


def _save_hybrid_payload_artifact(receipt_id: str, payload: dict) -> Path:
    cfg = get_config()
    path = cfg.paths.processed_dir / f"{receipt_id}.hybrid_payload.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path


def extract_receipt_with_hybrid(
    image_path: str | Path,
    persist: bool = True,
) -> tuple[ReceiptRecord, dict]:
    ensure_directories()

    source_path = Path(image_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Receipt image not found: {source_path}")

    receipt_id = new_id("rcpt")
    suffix = source_path.suffix if source_path.suffix else ".jpg"
    stored_image_path = copy_to_raw_storage(source_path, f"{receipt_id}{suffix}")

    build_hybrid_ui_payload = load_hybrid_payload_builder()
    payload = build_hybrid_ui_payload(str(stored_image_path))

    receipt = build_receipt_from_hybrid_payload(
        payload=payload,
        image_path=str(stored_image_path),
        receipt_id=receipt_id,
    )

    if persist:
        ReceiptRepository.upsert_receipt(receipt)
        save_processed_receipt_artifact(receipt)
        _save_hybrid_payload_artifact(receipt_id, payload)

    return receipt, payload