from __future__ import annotations

from billwise.common.config import get_config
from billwise.common.gcs_storage import build_blob_path, gcs_enabled, upload_file
from billwise.common.logging import get_logger
from billwise.common.storage import ensure_directories
from billwise.dashboard.bridge import load_all_data


def export_dashboard_views() -> dict:
    logger = get_logger("billwise.dashboard.exports")

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

    result = {
        "receipts_path": str(receipts_path),
        "items_path": str(items_path),
        "joined_path": str(joined_path),
        "receipts_rows": len(df_receipts),
        "items_rows": len(df_items),
        "joined_rows": len(df_joined),
        "gcs_paths": {},
    }

    logger.info("Exported receipts view -> %s (%s rows)", receipts_path, len(df_receipts))
    logger.info("Exported items view -> %s (%s rows)", items_path, len(df_items))
    logger.info("Exported joined view -> %s (%s rows)", joined_path, len(df_joined))

    if gcs_enabled():
        result["gcs_paths"]["receipts_view"] = upload_file(
            receipts_path, build_blob_path("views", "receipts_view.csv"), "text/csv"
        )
        result["gcs_paths"]["items_view"] = upload_file(
            items_path, build_blob_path("views", "items_view.csv"), "text/csv"
        )
        result["gcs_paths"]["joined_view"] = upload_file(
            joined_path, build_blob_path("views", "joined_view.csv"), "text/csv"
        )

        logger.info("Uploaded receipts view -> %s", result["gcs_paths"]["receipts_view"])
        logger.info("Uploaded items view -> %s", result["gcs_paths"]["items_view"])
        logger.info("Uploaded joined view -> %s", result["gcs_paths"]["joined_view"])

    return result