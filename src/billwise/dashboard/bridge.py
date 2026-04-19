from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "billwise.db"


_DDL_VALIDATIONS = """
CREATE TABLE IF NOT EXISTS human_validations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id            TEXT,
    receipt_id         TEXT,
    raw_item_text      TEXT,
    original_category  TEXT,
    validated_category TEXT,
    validator_note     TEXT,
    validated_at       TEXT,
    validation_type    TEXT DEFAULT 'category'
);
"""

_DDL_OCR_CORRECTIONS = """
CREATE TABLE IF NOT EXISTS ocr_corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id      TEXT,
    field_name      TEXT,
    original_value  TEXT,
    corrected_value TEXT,
    corrected_at    TEXT
);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_aux_tables(conn: sqlite3.Connection) -> None:
    conn.execute(_DDL_VALIDATIONS)
    conn.execute(_DDL_OCR_CORRECTIONS)
    conn.commit()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_receipts_df(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    SELECT
        r.receipt_id,
        r.vendor_name,
        r.receipt_date,
        COALESCE(r.total, 0.0) AS receipt_total,
        COALESCE(r.tax, 0.0) AS tax,
        CASE
            WHEN COALESCE(r.payment_method, '') <> '' AND COALESCE(r.card_last4, '') <> ''
                THEN r.payment_method || ' ••••' || r.card_last4
            WHEN COALESCE(r.card_last4, '') <> ''
                THEN '••••' || r.card_last4
            ELSE COALESCE(r.payment_method, '')
        END AS card_used,
        COALESCE(AVG(rf.confidence), 0.0) AS extraction_confidence,
        COALESCE(r.source, 'local_upload') AS source,
        r.processing_status,
        r.review_status,
        COALESCE(r.requires_review, 0) AS requires_review
    FROM receipts r
    LEFT JOIN receipt_fields rf
        ON r.receipt_id = rf.receipt_id
    GROUP BY
        r.receipt_id, r.vendor_name, r.receipt_date, r.total, r.tax,
        r.payment_method, r.card_last4, r.source,
        r.processing_status, r.review_status, r.requires_review, r.upload_timestamp
    ORDER BY r.upload_timestamp DESC
    """
    df = pd.read_sql(query, conn)

    if df.empty:
        return pd.DataFrame(columns=[
            "receipt_id", "vendor_name", "receipt_date", "receipt_total", "tax",
            "card_used", "extraction_confidence", "source",
            "processing_status", "review_status", "requires_review",
        ])

    df["receipt_date"] = pd.to_datetime(df["receipt_date"], errors="coerce")
    df["receipt_total"] = df["receipt_total"].apply(_safe_float)
    df["tax"] = df["tax"].apply(_safe_float)
    df["extraction_confidence"] = df["extraction_confidence"].apply(_safe_float)

    df["vendor_name"] = df["vendor_name"].fillna("Unknown Vendor").astype(str).str.strip()
    df.loc[df["vendor_name"].isin(["", "None", "nan"]), "vendor_name"] = "Unknown Vendor"

    df["card_used"] = df["card_used"].fillna("").astype(str).str.strip()
    df["source"] = df["source"].fillna("local_upload").astype(str).str.strip()

    return df


def _load_items_df(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    SELECT
        li.item_id AS id,
        li.receipt_id,
        li.raw_name AS raw_item_text,
        COALESCE(li.normalized_name, LOWER(li.raw_name)) AS normalized_item_text,
        COALESCE(li.normalized_name, li.raw_name) AS matched_canonical_item,
        COALESCE(c.final_category, c.predicted_category, 'Other') AS category,
        COALESCE(li.quantity, 1.0) AS quantity,
        'unit' AS unit,
        COALESCE(li.unit_price, 0.0) AS unit_price,
        COALESCE(li.item_total, COALESCE(li.quantity, 1.0) * COALESCE(li.unit_price, 0.0), 0.0) AS line_total,
        COALESCE(c.category_confidence, 0.0) AS category_confidence
    FROM line_items li
    LEFT JOIN categorizations c
        ON li.item_id = c.item_id
    """
    df = pd.read_sql(query, conn)

    if df.empty:
        return pd.DataFrame(columns=[
            "id", "receipt_id", "raw_item_text", "normalized_item_text",
            "matched_canonical_item", "category", "quantity", "unit",
            "unit_price", "line_total", "category_confidence",
        ])

    df["id"] = df["id"].astype(str)
    df["quantity"] = df["quantity"].apply(_safe_float, default=1.0)
    df["unit_price"] = df["unit_price"].apply(_safe_float)
    df["line_total"] = df["line_total"].apply(_safe_float)
    df["category_confidence"] = df["category_confidence"].apply(_safe_float)
    return df


def _apply_ocr_corrections(df_receipts: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    if df_receipts.empty:
        return df_receipts

    corrections = pd.read_sql(
        """
        SELECT receipt_id, field_name, corrected_value, corrected_at
        FROM ocr_corrections
        ORDER BY corrected_at ASC
        """,
        conn,
    )

    if corrections.empty:
        return df_receipts

    df = df_receipts.copy()

    for _, row in corrections.iterrows():
        rid = row["receipt_id"]
        field = row["field_name"]
        value = row["corrected_value"]

        mask = df["receipt_id"] == rid
        if not mask.any():
            continue

        if field == "vendor_name":
            df.loc[mask, "vendor_name"] = value
        elif field == "receipt_date":
            df.loc[mask, "receipt_date"] = pd.to_datetime(value, errors="coerce")
        elif field == "receipt_total":
            df.loc[mask, "receipt_total"] = _safe_float(value)
        elif field == "card_used":
            df.loc[mask, "card_used"] = value

    return df


def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = _get_conn()
    _init_aux_tables(conn)

    try:
        df_receipts = _load_receipts_df(conn)
        df_receipts = _apply_ocr_corrections(df_receipts, conn)

        df_items = _load_items_df(conn)

        if df_receipts.empty or df_items.empty:
            df_joined = pd.DataFrame(columns=[
                "id", "receipt_id", "raw_item_text", "normalized_item_text",
                "matched_canonical_item", "category", "quantity", "unit",
                "unit_price", "line_total", "category_confidence",
                "vendor_name", "receipt_date", "receipt_total", "tax",
                "card_used", "extraction_confidence", "month",
            ])
            return df_receipts, df_items, df_joined

        df_joined = df_items.merge(
            df_receipts[[
                "receipt_id", "vendor_name", "receipt_date", "receipt_total",
                "tax", "card_used", "extraction_confidence",
            ]],
            on="receipt_id",
            how="left",
        )

        df_joined["month"] = (
            pd.to_datetime(df_joined["receipt_date"], errors="coerce")
            .dt.to_period("M")
            .astype(str)
        )

        return df_receipts, df_items, df_joined

    finally:
        conn.close()


def get_data_source_label() -> str:
    return "🗄️ BillWise SQLite"


def save_category_validation(
    item_id: str,
    receipt_id: str,
    raw_item_text: str,
    original_category: str,
    validated_category: str,
    validator_note: str = "",
) -> None:
    conn = _get_conn()
    _init_aux_tables(conn)

    try:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        conn.execute(
            """
            INSERT INTO human_validations
                (item_id, receipt_id, raw_item_text, original_category,
                 validated_category, validator_note, validated_at, validation_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'category')
            """,
            (
                str(item_id),
                receipt_id,
                raw_item_text,
                original_category,
                validated_category,
                validator_note,
                now,
            ),
        )

        conn.execute(
            """
            UPDATE categorizations
            SET final_category = ?, needs_human_review = 0
            WHERE item_id = ?
            """,
            (validated_category, str(item_id)),
        )

        conn.execute(
            """
            INSERT INTO review_logs
                (entity_type, entity_id, field_name, old_value, new_value, reviewed_at, review_source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "line_item",
                str(item_id),
                "category",
                original_category,
                validated_category,
                now,
                "dashboard_validation",
            ),
        )

        conn.commit()
    finally:
        conn.close()


def save_ocr_correction(
    receipt_id: str,
    field_name: str,
    original_value: str,
    corrected_value: str,
) -> None:
    conn = _get_conn()
    _init_aux_tables(conn)

    try:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        conn.execute(
            """
            INSERT INTO ocr_corrections
                (receipt_id, field_name, original_value, corrected_value, corrected_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                field_name,
                original_value,
                corrected_value,
                now,
            ),
        )

        conn.execute(
            """
            INSERT INTO review_logs
                (entity_type, entity_id, field_name, old_value, new_value, reviewed_at, review_source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "receipt",
                receipt_id,
                field_name,
                original_value,
                corrected_value,
                now,
                "dashboard_ocr_correction",
            ),
        )

        conn.commit()
    finally:
        conn.close()


def load_validated_item_ids() -> set[str]:
    conn = _get_conn()
    _init_aux_tables(conn)

    try:
        rows = conn.execute(
            "SELECT item_id FROM human_validations WHERE validation_type = 'category'"
        ).fetchall()
        return {str(r["item_id"]) for r in rows if r["item_id"] is not None}
    finally:
        conn.close()


def load_all_validations() -> pd.DataFrame:
    conn = _get_conn()
    _init_aux_tables(conn)

    try:
        return pd.read_sql(
            "SELECT * FROM human_validations ORDER BY validated_at DESC",
            conn,
        )
    finally:
        conn.close()


def load_all_ocr_corrections() -> pd.DataFrame:
    conn = _get_conn()
    _init_aux_tables(conn)

    try:
        return pd.read_sql(
            "SELECT * FROM ocr_corrections ORDER BY corrected_at DESC",
            conn,
        )
    finally:
        conn.close()