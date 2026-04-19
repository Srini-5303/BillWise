from __future__ import annotations

from pathlib import Path

from billwise.common.config import get_config
from billwise.common.logging import get_logger
from billwise.common.storage import ensure_directories
from billwise.dashboard.bridge import load_all_data


def main() -> None:
    logger = get_logger("billwise.export_dashboard_views")

    ensure_directories()
    cfg = get_config()

    df_receipts, df_items, df_joined = load_all_data()

    exports_dir = cfg.paths.exports_dir
    exports_dir.mkdir(parents=True, exist_ok=True)

    receipts_path = exports_dir / "receipts_view.csv"
    items_path = exports_dir / "items_view.csv"
    joined_path = exports_dir / "joined_view.csv"

    df_receipts.to_csv(receipts_path, index=False)
    df_items.to_csv(items_path, index=False)
    df_joined.to_csv(joined_path, index=False)

    logger.info("Exported receipts view -> %s (%s rows)", receipts_path, len(df_receipts))
    logger.info("Exported items view -> %s (%s rows)", items_path, len(df_items))
    logger.info("Exported joined view -> %s (%s rows)", joined_path, len(df_joined))


if __name__ == "__main__":
    main()