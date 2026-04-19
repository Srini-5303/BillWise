from __future__ import annotations

import json
from pathlib import Path

from billwise.common.config import get_config
from billwise.common.schemas import ReceiptRecord
from billwise.common.storage import ensure_directories


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def save_processed_receipt_artifact(receipt: ReceiptRecord) -> Path:
    ensure_directories()
    cfg = get_config()
    path = cfg.paths.processed_dir / f"{receipt.receipt_id}.json"
    payload = receipt.model_dump(mode="json")
    return _write_json(path, payload)


def save_reviewed_receipt_artifact(receipt: ReceiptRecord) -> Path:
    ensure_directories()
    cfg = get_config()
    path = cfg.paths.reviewed_dir / f"{receipt.receipt_id}.json"
    payload = receipt.model_dump(mode="json")
    return _write_json(path, payload)


def load_receipt_artifact(receipt_id: str, reviewed: bool = False) -> dict | None:
    cfg = get_config()
    base_dir = cfg.paths.reviewed_dir if reviewed else cfg.paths.processed_dir
    path = base_dir / f"{receipt_id}.json"

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)