from __future__ import annotations

from billwise.common.logging import get_logger
from billwise.dashboard.exports import export_dashboard_views


def main() -> None:
    logger = get_logger("billwise.export_dashboard_views")
    result = export_dashboard_views()

    logger.info("Receipts rows: %s", result["receipts_rows"])
    logger.info("Items rows: %s", result["items_rows"])
    logger.info("Joined rows: %s", result["joined_rows"])

    if result["gcs_paths"]:
        for key, value in result["gcs_paths"].items():
            logger.info("%s -> %s", key, value)


if __name__ == "__main__":
    main()