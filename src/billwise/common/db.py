from __future__ import annotations

import sqlite3

from billwise.common.config import get_config
from billwise.common.storage import ensure_directories


DDL = """
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id TEXT PRIMARY KEY,
    image_path TEXT NOT NULL,
    source TEXT,
    upload_timestamp TEXT,
    processing_status TEXT,
    review_status TEXT,
    vendor_name TEXT,
    receipt_date TEXT,
    receipt_time TEXT,
    subtotal REAL,
    tax REAL,
    total REAL,
    payment_method TEXT,
    card_last4 TEXT,
    receipt_number TEXT,
    extraction_method TEXT,
    requires_review INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS receipt_fields (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    field_value TEXT,
    confidence REAL,
    bbox_json TEXT,
    source_model TEXT,
    FOREIGN KEY(receipt_id) REFERENCES receipts(receipt_id)
);

CREATE TABLE IF NOT EXISTS line_items (
    item_id TEXT PRIMARY KEY,
    receipt_id TEXT NOT NULL,
    raw_name TEXT NOT NULL,
    normalized_name TEXT,
    quantity REAL,
    unit_price REAL,
    item_total REAL,
    item_confidence REAL,
    item_source TEXT,
    FOREIGN KEY(receipt_id) REFERENCES receipts(receipt_id)
);

CREATE TABLE IF NOT EXISTS categorizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL,
    predicted_category TEXT,
    category_confidence REAL,
    top_k_scores_json TEXT,
    categorizer_model TEXT,
    needs_human_review INTEGER DEFAULT 0,
    final_category TEXT,
    FOREIGN KEY(item_id) REFERENCES line_items(item_id)
);

CREATE TABLE IF NOT EXISTS review_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reviewed_at TEXT,
    review_source TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    receipt_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error_message TEXT,
    FOREIGN KEY(receipt_id) REFERENCES receipts(receipt_id)
);
"""


def get_connection() -> sqlite3.Connection:
    ensure_directories()
    db_path = get_config().paths.database_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(DDL)
        conn.commit()