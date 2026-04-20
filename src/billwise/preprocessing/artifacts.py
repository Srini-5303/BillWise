from __future__ import annotations

import json
from pathlib import Path

from billwise.common.config import get_config
from billwise.common.gcs_storage import build_blob_path, gcs_enabled, upload_file, upload_json
from billwise.preprocessing.schemas import PreprocessingResult


def save_preprocessing_artifact(
    receipt_id: str,
    result: PreprocessingResult,
) -> Path:
    cfg = get_config()

    artifact_path = cfg.paths.processed_dir / f"{receipt_id}.preprocessing.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    payload = result.model_dump(mode="json")
    payload["receipt_id"] = receipt_id

    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if gcs_enabled():
        upload_json(
            payload,
            build_blob_path("preprocessing", "receipts", receipt_id, "metadata.json"),
        )

        for variant in result.variants:
            variant_path = Path(variant.path)
            if variant_path.exists():
                upload_file(
                    variant_path,
                    build_blob_path("preprocessing", "receipts", receipt_id, "variants", variant_path.name),
                )

    return artifact_path