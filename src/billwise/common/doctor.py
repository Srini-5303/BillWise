from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from billwise.common.config import find_project_root, get_config
from billwise.categorization.legacy_runtime import get_legacy_categorization_root
from billwise.extraction.legacy_runtime import REQUIRED_SUBPATHS, get_legacy_extraction_root


def _path_check(name: str, path: Path, required: bool = True) -> dict:
    exists = path.exists()
    if exists:
        return {
            "name": name,
            "status": "ok",
            "detail": str(path),
        }
    return {
        "name": name,
        "status": "fail" if required else "warn",
        "detail": f"Missing: {path}",
    }


def _env_check(name: str, required: bool = False) -> dict:
    value = os.getenv(name)
    present = bool(value)
    if present:
        return {
            "name": f"env:{name}",
            "status": "ok",
            "detail": "present",
        }
    return {
        "name": f"env:{name}",
        "status": "fail" if required else "warn",
        "detail": "not set",
    }


def _import_check(module_name: str, required: bool = True) -> dict:
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        return {
            "name": f"import:{module_name}",
            "status": "ok",
            "detail": "importable",
        }
    return {
        "name": f"import:{module_name}",
        "status": "fail" if required else "warn",
        "detail": "not importable",
    }


def collect_doctor_report() -> dict:
    root = find_project_root()
    cfg = get_config()

    checks: list[dict] = []

    checks.append({
        "name": "project_root",
        "status": "ok",
        "detail": str(root),
    })

    checks.extend([
        _path_check("db_path", cfg.paths.database_path, required=False),
        _path_check("raw_dir", cfg.paths.raw_dir),
        _path_check("processed_dir", cfg.paths.processed_dir),
        _path_check("reviewed_dir", cfg.paths.reviewed_dir),
        _path_check("exports_dir", cfg.paths.exports_dir),
        _path_check("logs_dir", cfg.paths.logs_dir),
    ])

    extraction_root = get_legacy_extraction_root()
    checks.append(_path_check("legacy_extraction_root", extraction_root))
    for subdir in REQUIRED_SUBPATHS:
        checks.append(_path_check(f"legacy_extraction:{subdir}", extraction_root / subdir))

    categorization_root = get_legacy_categorization_root()
    checks.append(_path_check("legacy_categorization_root", categorization_root))
    checks.append(_path_check(
        "legacy_categorization:Abbreviation_Normalization.py",
        categorization_root / "Abbreviation_Normalization.py",
    ))
    checks.append(_path_check(
        "legacy_categorization:Categorization.py",
        categorization_root / "Categorization.py",
    ))

    checks.append(_path_check(
        "categorization_weights",
        root / "models" / "categorization" / "distilbert_receipt_classifier.pt",
    ))
    checks.append(_path_check(
        "normalization_inventory_csv",
        root / "data" / "reference" / "merged_labeled.csv",
    ))

    checks.extend([
        _env_check("GROQ_API_KEY", required=False),
        _env_check("GEMINI_API_KEY", required=False),
        _env_check("GOOGLE_APPLICATION_CREDENTIALS", required=False),
    ])

    checks.extend([
        _import_check("pandas"),
        _import_check("streamlit"),
        _import_check("torch"),
        _import_check("transformers"),
        _import_check("gradio"),
        _import_check("duckdb"),
        _import_check("yaml"),
        _import_check("dotenv"),
    ])

    summary = {
        "ok": sum(1 for c in checks if c["status"] == "ok"),
        "warn": sum(1 for c in checks if c["status"] == "warn"),
        "fail": sum(1 for c in checks if c["status"] == "fail"),
    }

    return {
        "project_root": str(root),
        "checks": checks,
        "summary": summary,
    }