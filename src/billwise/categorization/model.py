from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from billwise.common.config import find_project_root


@lru_cache(maxsize=1)
def get_categorization_config() -> dict:
    root = find_project_root()
    path = root / "configs" / "categorization.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_state_dict(raw):
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"]
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


@lru_cache(maxsize=1)
def load_billwise_classifier():
    cfg = get_categorization_config()
    root = find_project_root()

    weights_path = root / cfg["model"]["weights_path"]
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Categorization weights not found: {weights_path}"
        )

    labels = cfg["labels"]
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_name"])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["base_model_name"],
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    raw = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(raw)

    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Unexpected checkpoint format at {weights_path}: {type(state_dict)}"
        )

    state_dict = _strip_module_prefix(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    classifier_missing = [
        k for k in missing if k in {
            "pre_classifier.weight",
            "pre_classifier.bias",
            "classifier.weight",
            "classifier.bias",
        }
    ]
    if classifier_missing:
        raise ValueError(
            f"Checkpoint load failed for classification head. Missing: {classifier_missing}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "labels": labels,
        "max_length": cfg["model"]["max_length"],
        "runtime_cfg": cfg,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }