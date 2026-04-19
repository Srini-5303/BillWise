from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    reviewed_dir: Path
    exports_dir: Path
    logs_dir: Path
    database_path: Path


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    paths: AppPaths
    groq_api_key: str | None
    google_application_credentials: str | None


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing pyproject.toml")


@lru_cache(maxsize=1)
def load_yaml_config(relative_path: str) -> dict:
    root = find_project_root()
    config_path = root / relative_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty or invalid YAML: {config_path}")

    if not isinstance(data, dict):
        raise ValueError(f"Config file must parse to a dictionary: {config_path}")

    return data


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    root = find_project_root()

    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        example_env_path = root / ".env.example"
        if example_env_path.exists():
            load_dotenv(example_env_path)

    storage_cfg = load_yaml_config("configs/storage.yaml")

    if "database" not in storage_cfg or "paths" not in storage_cfg:
        raise KeyError("configs/storage.yaml must contain 'database' and 'paths' sections")

    paths_cfg = storage_cfg["paths"]
    db_cfg = storage_cfg["database"]

    required_path_keys = ["raw_dir", "processed_dir", "reviewed_dir", "exports_dir", "logs_dir"]
    for key in required_path_keys:
        if key not in paths_cfg:
            raise KeyError(f"Missing '{key}' in configs/storage.yaml -> paths")

    if "sqlite_path" not in db_cfg:
        raise KeyError("Missing 'sqlite_path' in configs/storage.yaml -> database")

    paths = AppPaths(
        project_root=root,
        raw_dir=root / paths_cfg["raw_dir"],
        processed_dir=root / paths_cfg["processed_dir"],
        reviewed_dir=root / paths_cfg["reviewed_dir"],
        exports_dir=root / paths_cfg["exports_dir"],
        logs_dir=root / paths_cfg["logs_dir"],
        database_path=root / db_cfg["sqlite_path"],
    )

    return AppConfig(
        project_name=storage_cfg.get("project_name", "BillWise"),
        paths=paths,
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        google_application_credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or None,
    )