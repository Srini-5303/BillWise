from __future__ import annotations

import sys
from pathlib import Path

from billwise.common.config import find_project_root


def get_legacy_categorization_root() -> Path:
    return find_project_root() / "legacy" / "categorization_module"


def ensure_legacy_categorization_on_path() -> Path:
    root = get_legacy_categorization_root()

    if not root.exists():
        raise FileNotFoundError(
            f"Legacy categorization module not found at: {root}\n"
            "Place Categorization_module.zip contents under legacy/categorization_module/"
        )

    required = ["Abbreviation_Normalization.py", "Categorization.py"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Legacy categorization module is incomplete. Missing: {missing}\n"
            f"Checked under: {root}"
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    return root


def load_legacy_normalization_symbols():
    ensure_legacy_categorization_on_path()
    from Abbreviation_Normalization import init_pipeline, smart_normalize

    return init_pipeline, smart_normalize


def load_legacy_classifier_symbols():
    ensure_legacy_categorization_on_path()
    from Categorization import load_classifier, classify

    return load_classifier, classify