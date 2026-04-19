from __future__ import annotations

import sys
from pathlib import Path

from billwise.common.config import find_project_root


REQUIRED_SUBPATHS = [
    "app",
    "evaluation",
    "extract",
    "kie",
    "methods",
    "ocr",
    "output",
    "ui",
]


def get_legacy_extraction_root() -> Path:
    return find_project_root() / "legacy" / "extraction_module"


def ensure_legacy_extraction_on_path() -> Path:
    root = get_legacy_extraction_root()

    if not root.exists():
        raise FileNotFoundError(
            f"Legacy extraction module not found at: {root}\n"
            "Place Extraction_module.zip contents under legacy/extraction_module/"
        )

    missing = [name for name in REQUIRED_SUBPATHS if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Legacy extraction module is incomplete. Missing: {missing}\n"
            f"Checked under: {root}"
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    return root


def load_hybrid_payload_builder():
    ensure_legacy_extraction_on_path()
    from app.hybrid_ui_payload import build_hybrid_ui_payload

    return build_hybrid_ui_payload


def load_gradio_app_factory():
    ensure_legacy_extraction_on_path()
    from ui.gradio_app import create_app

    return create_app