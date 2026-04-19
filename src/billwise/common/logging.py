from __future__ import annotations

import logging
import os


def get_logger(name: str = "billwise") -> logging.Logger:
    level_name = os.getenv("BILLWISE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger