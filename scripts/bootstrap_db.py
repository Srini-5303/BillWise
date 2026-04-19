from __future__ import annotations

from billwise.common.config import get_config
from billwise.common.db import init_db
from billwise.common.logging import get_logger
from billwise.common.storage import ensure_directories


def main() -> None:
    logger = get_logger("billwise.bootstrap")
    cfg = get_config()

    ensure_directories()
    init_db()

    logger.info("Project root: %s", cfg.paths.project_root)
    logger.info("SQLite DB: %s", cfg.paths.database_path)
    logger.info("Bootstrap completed successfully.")


if __name__ == "__main__":
    main()