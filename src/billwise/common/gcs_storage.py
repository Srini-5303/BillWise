from __future__ import annotations

import json
import mimetypes
from pathlib import Path

from google.cloud import storage

from billwise.common.config import get_config


def gcs_enabled() -> bool:
    cfg = get_config()

    if cfg.gcs.storage_backend not in {"hybrid", "gcs", "gcs_only"}:
        return False

    if not cfg.gcs.bucket_name:
        return False

    cred_path = cfg.google_application_credentials
    if not cred_path:
        return False

    return Path(cred_path).exists()


def get_gcs_client() -> storage.Client:
    cfg = get_config()
    if not cfg.gcs.bucket_name:
        raise ValueError("GCS bucket is not configured")
    return storage.Client(project=cfg.gcs.project_id or None)


def get_bucket():
    cfg = get_config()
    client = get_gcs_client()
    return client.bucket(cfg.gcs.bucket_name)


def build_blob_path(*parts: str) -> str:
    cfg = get_config()
    clean_parts = [p.strip("/").replace("\\", "/") for p in parts if p]
    return "/".join([cfg.gcs.prefix, *clean_parts])


def upload_file(local_path: str | Path, blob_path: str, content_type: str | None = None) -> str:
    local_path = Path(local_path)
    bucket = get_bucket()
    blob = bucket.blob(blob_path)

    if content_type is None:
        guessed, _ = mimetypes.guess_type(str(local_path))
        content_type = guessed or "application/octet-stream"

    blob.upload_from_filename(str(local_path), content_type=content_type)
    return f"gs://{bucket.name}/{blob_path}"


def upload_json(payload: dict, blob_path: str) -> str:
    bucket = get_bucket()
    blob = bucket.blob(blob_path)
    blob.upload_from_string(
        json.dumps(payload, ensure_ascii=False, indent=2),
        content_type="application/json",
    )
    return f"gs://{bucket.name}/{blob_path}"


def upload_text(text: str, blob_path: str, content_type: str = "text/plain") -> str:
    bucket = get_bucket()
    blob = bucket.blob(blob_path)
    blob.upload_from_string(text, content_type=content_type)
    return f"gs://{bucket.name}/{blob_path}"