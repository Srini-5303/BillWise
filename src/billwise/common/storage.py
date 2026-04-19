from __future__ import annotations

from pathlib import Path
import hashlib
import shutil

from billwise.common.config import get_config


def ensure_directories() -> None:
    cfg = get_config()
    cfg.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.reviewed_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.exports_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.database_path.parent.mkdir(parents=True, exist_ok=True)


def sha256_of_file(file_path: str | Path) -> str:
    file_path = Path(file_path)
    digest = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def copy_to_raw_storage(source_path: str | Path, target_name: str | None = None) -> Path:
    ensure_directories()
    cfg = get_config()

    source_path = Path(source_path)
    destination = cfg.paths.raw_dir / (target_name or source_path.name)
    shutil.copy2(source_path, destination)
    return destination