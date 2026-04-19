from __future__ import annotations

from functools import lru_cache

from billwise.common.config import find_project_root
from billwise.categorization.legacy_runtime import load_legacy_normalization_symbols
from billwise.categorization.model import get_categorization_config


@lru_cache(maxsize=1)
def load_normalization_pipeline():
    cfg = get_categorization_config()
    root = find_project_root()
    dataset_path = root / cfg["normalization"]["inventory_dataset_path"]

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Normalization inventory dataset not found: {dataset_path}\n"
            "Place the labeled CSV at data/reference/merged_labeled.csv"
        )

    init_pipeline, _ = load_legacy_normalization_symbols()
    inventory, vectorizer, tfidf_matrix = init_pipeline(str(dataset_path))

    return {
        "inventory": inventory,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
    }


def normalize_item_text(text: str) -> tuple[str, bool]:
    _, smart_normalize = load_legacy_normalization_symbols()
    normalized_text, was_changed = smart_normalize(text)
    return normalized_text, was_changed