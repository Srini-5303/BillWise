"""
data_loader.py — Data access layer for BillWise Dashboard.

Priority order for data loading:
  1. GCS CSV  (if GCS_BUCKET_NAME + GCS_BILLS_BLOB env vars are set)
  2. Local bills_output.csv  (if file exists next to this script)
  3. Seed demo data  (always works, no credentials needed)

The loaded data is normalised into two pandas DataFrames:
  df_receipts  — one row per receipt
  df_items     — one row per line item
  df_joined    — df_items LEFT JOIN df_receipts (used by most analytics)
"""
from __future__ import annotations

import ast
import io
import os
import random
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils import categorize_item

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DB_PATH   = BASE_DIR / "billwise.db"

# Point to the Flask bot's CSV so WhatsApp-scanned receipts show up here too.
# Falls back to a local copy if the sibling folder doesn't exist.
_flask_csv = BASE_DIR.parent / "BillWise-main" / "bills_output.csv"
LOCAL_CSV  = _flask_csv if _flask_csv.exists() else BASE_DIR / "bills_output.csv"

# ── SQLite schema ──────────────────────────────────────────────────────────────
_DDL_VALIDATIONS = """
CREATE TABLE IF NOT EXISTS human_validations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id            INTEGER,
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

_DDL_RECEIPTS = """
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id            TEXT PRIMARY KEY,
    vendor_name           TEXT,
    receipt_date          TEXT,
    receipt_total         REAL,
    tax                   REAL DEFAULT 0,
    card_used             TEXT,
    sender                TEXT,
    extraction_confidence REAL DEFAULT 0.85,
    source                TEXT DEFAULT 'demo'
);
"""

_DDL_ITEMS = """
CREATE TABLE IF NOT EXISTS line_items (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id             TEXT,
    raw_item_text          TEXT,
    normalized_item_text   TEXT,
    matched_canonical_item TEXT,
    category               TEXT,
    quantity               REAL DEFAULT 1.0,
    unit                   TEXT DEFAULT 'unit',
    unit_price             REAL,
    line_total             REAL,
    category_confidence    REAL DEFAULT 0.75,
    FOREIGN KEY (receipt_id) REFERENCES receipts(receipt_id)
);
"""

# ── Seed catalogue ─────────────────────────────────────────────────────────────
_VENDORS = [
    "Sysco Foods", "US Foods", "Metro Distributors", "Chef's Warehouse",
    "Restaurant Depot", "FreshMart Supplies", "Pacific Coast Seafood",
    "Dairy Direct", "Green Leaf Produce", "Valley Fresh Farms",
    "Premier Supply Co", "Spice Emporium", "Continental Bakers",
    "Golden Gate Provisions", "CleanPro Supplies",
]

# (raw_text, normalized, canonical, unit, base_price)
_ITEMS: dict[str, list[tuple]] = {
    "Meat": [
        ("chkn brst 5kg",      "Chicken Breast 5kg",       "Chicken Breast",  "kg",   8.50),
        ("grnd beef 80/20",    "Ground Beef 80/20",        "Ground Beef",     "kg",   6.75),
        ("pork loin bnls",     "Pork Loin Boneless",       "Pork Loin",       "kg",   7.20),
        ("lamb chops rack",    "Lamb Rack Chops",          "Lamb Chops",      "kg",  22.00),
        ("beef ribeye stk",    "Beef Ribeye Steak",        "Beef Ribeye",     "kg",  35.00),
        ("veal cutlet",        "Veal Cutlet",              "Veal Cutlet",     "kg",  28.00),
    ],
    "Dairy": [
        ("mozz fresh kg",      "Fresh Mozzarella",         "Mozzarella",      "kg",  12.00),
        ("hvy cream 1L",       "Heavy Whipping Cream 1L",  "Heavy Cream",     "liter", 4.50),
        ("butter unslted",     "Unsalted Butter Block",    "Butter",          "kg",   9.00),
        ("parmigiano kg",      "Parmigiano Reggiano",      "Parmesan",        "kg",  24.00),
        ("whole milk 4L",      "Whole Milk 4L",            "Whole Milk",      "liter", 1.80),
        ("ricotta 500g",       "Ricotta Cheese 500g",      "Ricotta",         "kg",  10.00),
    ],
    "Produce": [
        ("roma toms 10kg",     "Roma Tomatoes 10kg",       "Roma Tomatoes",   "kg",   2.50),
        ("baby spinach",       "Baby Spinach Leaves",      "Baby Spinach",    "kg",   6.00),
        ("yel onions 25lb",    "Yellow Onions 25lb",       "Yellow Onions",   "kg",   1.20),
        ("garlic fresh",       "Fresh Garlic Bulbs",       "Garlic",          "kg",   4.50),
        ("bell peppers mix",   "Mixed Bell Peppers",       "Bell Peppers",    "kg",   3.80),
        ("evoo 5L",            "Extra Virgin Olive Oil 5L","Olive Oil",       "liter", 9.50),
        ("lemons fresh",       "Fresh Lemons",             "Lemons",          "kg",   2.20),
    ],
    "Dry Goods": [
        ("penne rigate 5kg",   "Penne Rigate 5kg",         "Penne Pasta",     "kg",   2.20),
        ("ap flour 25kg",      "All Purpose Flour 25kg",   "AP Flour",        "kg",   0.90),
        ("arborio rice 5kg",   "Arborio Rice 5kg",         "Arborio Rice",    "kg",   3.50),
        ("blk pepper grnd",    "Ground Black Pepper",      "Black Pepper",    "kg",  14.00),
        ("sea salt fine",      "Fine Sea Salt",            "Sea Salt",        "kg",   1.50),
        ("san marz toms",      "San Marzano Tomatoes 400g","Canned Tomatoes", "can",  1.80),
        ("breadcrumbs itln",   "Italian Breadcrumbs",      "Breadcrumbs",     "kg",   3.00),
    ],
    "Seafood": [
        ("salmon flt atl",     "Atlantic Salmon Fillet",   "Salmon",          "kg",  26.00),
        ("tiger shrimp 16/20", "Tiger Shrimp 16/20",       "Tiger Shrimp",    "kg",  30.00),
        ("sea bass euro",      "European Sea Bass",        "Sea Bass",        "kg",  34.00),
        ("cod flt atl",        "Atlantic Cod Fillet",      "Cod Fish",        "kg",  19.00),
        ("tuna stk yllwfin",   "Yellowfin Tuna Steak",     "Tuna Steak",      "kg",  38.00),
        ("scallops u10",       "Scallops U10",             "Scallops",        "kg",  42.00),
    ],
    "Bakery": [
        ("sourdough loaf",     "Artisan Sourdough Loaf",   "Sourdough Bread", "unit",  4.50),
        ("brioche buns x6",    "Brioche Burger Buns x6",   "Brioche Buns",    "pack",  5.80),
        ("butter croiss x12",  "Butter Croissants x12",    "Croissants",      "pack", 12.00),
        ("rosemary foc",       "Rosemary Focaccia Sheet",  "Focaccia",        "unit",  8.00),
        ("dinner rolls x24",   "Dinner Rolls x24",         "Dinner Rolls",    "pack",  9.50),
    ],
    "Beverages": [
        ("sprklng water cs",   "Sparkling Water Case",     "Sparkling Water", "case", 18.00),
        ("still water cs",     "Still Water Case",         "Still Water",     "case", 14.00),
        ("fresh oj 4L",        "Fresh Squeezed OJ 4L",     "Orange Juice",    "liter", 5.50),
        ("espresso blend",     "Arabica Espresso Blend",   "Espresso Beans",  "kg",   22.00),
        ("chianti 750ml",      "Chianti Classico 750ml",   "Red Wine",        "bottle",14.00),
        ("pinot grigio",       "Pinot Grigio 750ml",       "White Wine",      "bottle",12.00),
    ],
    "Cleaning Supplies": [
        ("dish soap 5L",       "Commercial Dish Soap 5L",  "Dish Soap",       "bottle",12.50),
        ("food sanitizer",     "Food Safe Sanitizer 1L",   "Sanitizer",       "bottle", 8.00),
        ("paper twls x6",      "Paper Towels x6",          "Paper Towels",    "pack",  18.00),
        ("latex glvs M",       "Food Grade Gloves M 100ct","Latex Gloves",    "box",   14.00),
        ("trash bags 60L",     "Trash Bags 60L x50",       "Trash Bags",      "box",   16.00),
    ],
}

# Vendor → categories it sells
_VENDOR_CATS: dict[str, list[str]] = {
    "Sysco Foods":           ["Meat", "Dairy", "Dry Goods", "Beverages"],
    "US Foods":              ["Meat", "Produce", "Dry Goods", "Seafood"],
    "Metro Distributors":    ["Meat", "Dairy", "Produce", "Dry Goods"],
    "Chef's Warehouse":      ["Meat", "Dairy", "Seafood", "Dry Goods"],
    "Restaurant Depot":      ["Meat", "Dry Goods", "Cleaning Supplies", "Beverages"],
    "FreshMart Supplies":    ["Produce", "Dairy", "Bakery"],
    "Pacific Coast Seafood": ["Seafood"],
    "Dairy Direct":          ["Dairy"],
    "Green Leaf Produce":    ["Produce"],
    "Valley Fresh Farms":    ["Produce", "Dairy"],
    "Premier Supply Co":     ["Dry Goods", "Cleaning Supplies", "Beverages"],
    "Spice Emporium":        ["Dry Goods"],
    "Continental Bakers":    ["Bakery"],
    "Golden Gate Provisions":["Meat", "Seafood", "Dairy"],
    "CleanPro Supplies":     ["Cleaning Supplies"],
}


# ── SQLite helpers ─────────────────────────────────────────────────────────────
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_DDL_RECEIPTS)
    conn.execute(_DDL_ITEMS)
    conn.execute(_DDL_VALIDATIONS)
    conn.execute(_DDL_OCR_CORRECTIONS)
    conn.commit()


def _db_has_data(conn: sqlite3.Connection) -> bool:
    try:
        n = conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
        return n > 0
    except Exception:
        return False


# ── GCS loader ─────────────────────────────────────────────────────────────────
def _load_from_gcs() -> pd.DataFrame | None:
    """Try to download bills_output.csv from GCS. Returns raw DataFrame or None."""
    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    blob_name   = os.environ.get("GCS_BILLS_BLOB", "bills_output.csv")
    if not bucket_name:
        return None
    try:
        from google.cloud import storage  # type: ignore
        client  = storage.Client()
        blob    = client.bucket(bucket_name).blob(blob_name)
        content = blob.download_as_bytes()
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        return None


def _load_from_local_csv() -> pd.DataFrame | None:
    """Try to load a local bills_output.csv. Returns raw DataFrame or None."""
    if LOCAL_CSV.exists():
        try:
            return pd.read_csv(str(LOCAL_CSV))
        except Exception:
            return None
    return None


# ── GCS CSV → normalised DataFrames ───────────────────────────────────────────
def _parse_items_column(raw: str) -> list[dict]:
    """
    Parse the Items column from the GCS CSV into a list of item dicts.
    The column may be stored as a Python list repr or JSON array.
    Each element is a string like 'Item Name $2.50' or 'Item Name: 2.50'.
    """
    if not raw or pd.isna(raw):
        return []
    items = []
    # Try to eval as Python list
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list):
            for entry in parsed:
                text  = str(entry).strip()
                price = _extract_price(text)
                items.append({"text": text, "price": price})
            return items
    except Exception:
        pass
    # Fallback: split by comma or newline
    for part in re.split(r"[,\n]+", str(raw)):
        part = part.strip().strip("[]'\"")
        if part:
            price = _extract_price(part)
            items.append({"text": part, "price": price})
    return items


def _extract_price(text: str) -> float:
    """Extract a dollar amount from item text. Returns 0.0 if not found."""
    match = re.search(r"\$?([\d,]+\.?\d*)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    return 0.0


def _gcs_df_to_normalised(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map GCS CSV columns → receipts + line_items DataFrames.

    Supports two GCS CSV layouts automatically:

    Layout A — one row per receipt (legacy):
        Serial_No, Bill_File, Store_Name, Invoice_Date, Total,
        Card_Used, Received_At, Sender, Image_Hash, Items

    Layout B — one row per item (current):
        Serial_No, Bill_File, Store_Name, Invoice_Date, Total,
        Card_Used, Received_At, Sender, Image_Hash,
        Item_Name, Item_Price, Grocery_Category
    """
    # Detect layout
    per_item_layout = "Item_Name" in raw_df.columns

    if per_item_layout:
        return _gcs_per_item_layout(raw_df)
    else:
        return _gcs_per_receipt_layout(raw_df)


def _gcs_per_item_layout(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle Layout B: one CSV row per line item.
    Groups by (Serial_No, Bill_File) to build unique receipts, then maps
    each row to a line item.
    """
    receipts_rows: list[dict] = []
    items_rows:    list[dict] = []

    # Use a fixed random seed so extraction_confidence is stable between reloads
    rng = random.Random(99)

    # Build unique receipts by grouping on Serial_No (+ Bill_File for safety)
    group_cols = ["Serial_No"]
    if "Bill_File" in raw_df.columns:
        group_cols.append("Bill_File")

    seen_ids: set[str] = set()

    for _, row in raw_df.iterrows():
        serial    = str(row.get("Serial_No", "")).strip()
        bill_file = str(row.get("Bill_File", "")).strip()
        # Build a stable unique receipt ID from serial + file hash
        uid       = f"{serial}_{bill_file}"
        receipt_id = f"R{str(serial).zfill(5)}"

        # Deduplicate — only add receipt row once per unique receipt
        if uid not in seen_ids:
            seen_ids.add(uid)
            vendor   = str(row.get("Store_Name", "Unknown")).strip() or "Unknown"
            raw_date = str(row.get("Invoice_Date", ""))
            total    = _to_float(row.get("Total", 0))
            card     = str(row.get("Card_Used", ""))
            sender   = str(row.get("Sender",    ""))

            receipts_rows.append({
                "receipt_id":            receipt_id,
                "vendor_name":           vendor,
                "receipt_date":          _normalise_date(raw_date),
                "receipt_total":         total,
                "tax":                   round(total * 0.08, 2),
                "card_used":             card,
                "sender":                sender,
                "extraction_confidence": round(rng.uniform(0.75, 0.98), 2),
                "source":                "gcs",
            })

        # Always add the item row
        raw_text  = str(row.get("Item_Name", "")).strip()
        price     = _to_float(row.get("Item_Price", 0))
        gcs_cat   = str(row.get("Grocery_Category", "")).strip()

        # Use GCS-provided category if valid, otherwise fall back to keyword model
        if gcs_cat and gcs_cat not in ("nan", "", "None"):
            category = gcs_cat
            conf     = 0.90   # treat GCS-supplied label as high confidence
        else:
            category, conf = categorize_item(raw_text)

        norm_text = raw_text.title()
        canonical = re.sub(r"\s*[@$#]\s*.*$", "", norm_text).strip() or norm_text

        items_rows.append({
            "receipt_id":             receipt_id,
            "raw_item_text":          raw_text,
            "normalized_item_text":   norm_text,
            "matched_canonical_item": canonical,
            "category":               category,
            "quantity":               1.0,
            "unit":                   "unit",
            "unit_price":             price,
            "line_total":             price,
            "category_confidence":    conf,
        })

    return pd.DataFrame(receipts_rows), pd.DataFrame(items_rows)


def _gcs_per_receipt_layout(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle Layout A: one CSV row per receipt with an Items blob column.
    """
    receipts_rows: list[dict] = []
    items_rows:    list[dict] = []
    rng = random.Random(99)

    for _, row in raw_df.iterrows():
        receipt_id = f"R{str(row.get('Serial_No', '')).zfill(5)}"
        vendor     = str(row.get("Store_Name", "Unknown")).strip() or "Unknown"
        raw_date   = str(row.get("Invoice_Date", ""))
        total      = _to_float(row.get("Total", 0))
        card       = str(row.get("Card_Used", ""))
        sender     = str(row.get("Sender",    ""))

        receipts_rows.append({
            "receipt_id":            receipt_id,
            "vendor_name":           vendor,
            "receipt_date":          _normalise_date(raw_date),
            "receipt_total":         total,
            "tax":                   round(total * 0.08, 2),
            "card_used":             card,
            "sender":                sender,
            "extraction_confidence": round(rng.uniform(0.75, 0.98), 2),
            "source":                "gcs",
        })

        parsed_items = _parse_items_column(row.get("Items", ""))
        if not parsed_items:
            parsed_items = [{"text": "General Purchase", "price": total}]

        for item in parsed_items:
            text           = item["text"]
            price          = item["price"] or (total / len(parsed_items))
            category, conf = categorize_item(text)
            norm_text      = text.title()
            canonical      = norm_text.split("$")[0].strip()

            items_rows.append({
                "receipt_id":             receipt_id,
                "raw_item_text":          text,
                "normalized_item_text":   norm_text,
                "matched_canonical_item": canonical,
                "category":               category,
                "quantity":               1.0,
                "unit":                   "unit",
                "unit_price":             price,
                "line_total":             price,
                "category_confidence":    conf,
            })

    return pd.DataFrame(receipts_rows), pd.DataFrame(items_rows)


def _to_float(val) -> float:
    try:
        return float(str(val).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _normalise_date(raw: str) -> str:
    """Try to parse a date string into YYYY-MM-DD. Falls back to today."""
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y",
                "%b %d %Y", "%B %d %Y", "%d %b %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date.today().strftime("%Y-%m-%d")


# ── Seed data generator ────────────────────────────────────────────────────────
def _seed_demo_data(conn: sqlite3.Connection) -> None:
    """Generate 120 realistic receipts over 180 days and insert into SQLite."""
    rng   = random.Random(42)   # fixed seed → reproducible demo
    today = date.today()
    start = today - timedelta(days=180)

    receipt_rows = []
    item_rows    = []

    # Build date pool — skew towards Mon/Tue (restaurant prep days)
    date_pool: list[date] = []
    d = start
    while d <= today:
        weight = 3 if d.weekday() in (0, 1) else 1
        date_pool.extend([d] * weight)
        d += timedelta(days=1)

    # Flatten items catalogue
    all_items_by_cat: dict[str, list[tuple]] = _ITEMS

    for i in range(120):
        receipt_date = rng.choice(date_pool)

        # Spend increases slightly over time (growth story for the demo)
        age_days    = (today - receipt_date).days
        growth_mult = 1 + 0.15 * (1 - age_days / 180)

        vendor      = rng.choice(_VENDORS)
        vendor_cats = _VENDOR_CATS.get(vendor, list(all_items_by_cat.keys()))

        # 3–6 line items per receipt
        n_items    = rng.randint(3, 6)
        chosen_cat = rng.choice(vendor_cats)
        cat_pool   = all_items_by_cat.get(chosen_cat, [])

        # Mix items from 1-3 categories
        all_pool: list[tuple] = []
        for cat in rng.sample(vendor_cats, k=min(3, len(vendor_cats))):
            all_pool.extend([(cat,) + itm for itm in all_items_by_cat.get(cat, [])])

        chosen_items = rng.choices(all_pool, k=n_items) if all_pool else []

        # Build line items
        receipt_total = 0.0
        receipt_item_rows = []
        for itm in chosen_items:
            cat, raw, norm, canonical, unit, base_price = itm
            qty        = round(rng.uniform(1, 8), 1) if unit == "kg" else rng.randint(1, 4)
            price_var  = rng.uniform(0.92, 1.10) * growth_mult
            unit_price = round(base_price * price_var, 2)
            line_total = round(unit_price * qty, 2)
            receipt_total += line_total

            _, cat_conf = categorize_item(raw)
            receipt_item_rows.append({
                "raw_item_text":          raw,
                "normalized_item_text":   norm,
                "matched_canonical_item": canonical,
                "category":               cat,
                "quantity":               qty,
                "unit":                   unit,
                "unit_price":             unit_price,
                "line_total":             line_total,
                "category_confidence":    cat_conf,
            })

        receipt_total = round(receipt_total, 2)
        tax           = round(receipt_total * 0.08, 2)
        receipt_id    = f"R{(i + 1):05d}"

        # Deliberate anomalies for demo
        extraction_conf = 0.92
        if i in (45, 78):           # 2 high-spend outliers
            receipt_total  = round(receipt_total * 4.5, 2)
            tax            = round(receipt_total * 0.08, 2)
        if i in (10, 11):           # 1 duplicate pair (same vendor, date, near-identical total)
            if i == 11 and receipt_rows:
                prev          = receipt_rows[-1]
                vendor        = prev["vendor_name"]
                receipt_date  = datetime.strptime(prev["receipt_date"], "%Y-%m-%d").date()
                receipt_total = round(prev["receipt_total"] * 1.02, 2)
        if i in (20, 35, 50, 65, 80):  # 5 low-confidence extractions
            extraction_conf = round(rng.uniform(0.40, 0.59), 2)

        receipt_rows.append({
            "receipt_id":            receipt_id,
            "vendor_name":           vendor,
            "receipt_date":          receipt_date.strftime("%Y-%m-%d"),
            "receipt_total":         receipt_total,
            "tax":                   tax,
            "card_used":             rng.choice(["VISA XXXX1234", "MASTERCARD XXXX5678",
                                                 "AMEX XXXX9012", "DEBIT XXXX3456"]),
            "sender":                "+1555000000" + str(rng.randint(10, 99)),
            "extraction_confidence": extraction_conf,
            "source":                "demo",
        })
        for ir in receipt_item_rows:
            ir["receipt_id"] = receipt_id
            item_rows.append(ir)

    # Insert into SQLite
    df_r = pd.DataFrame(receipt_rows)
    df_i = pd.DataFrame(item_rows)
    df_r.to_sql("receipts",   conn, if_exists="append", index=False)
    df_i.to_sql("line_items", conn, if_exists="append", index=False)
    conn.commit()


# ── Public API ─────────────────────────────────────────────────────────────────
def init_db() -> None:
    """Create schema and populate data. Always reloads from GCS when configured."""
    conn = _get_conn()
    _init_schema(conn)

    gcs_configured = bool(os.environ.get("GCS_BUCKET_NAME", "").strip())

    # If GCS is configured, always reload fresh from GCS (skip stale cache)
    if _db_has_data(conn) and not gcs_configured:
        conn.close()
        return

    # If GCS is configured, wipe existing data so we get a clean reload
    if gcs_configured:
        conn.execute("DELETE FROM line_items")
        conn.execute("DELETE FROM receipts")
        conn.commit()

    # Try real data sources first
    raw_df = _load_from_gcs()
    if raw_df is None:
        raw_df = _load_from_local_csv()

    if raw_df is not None and not raw_df.empty:
        df_r, df_i = _gcs_df_to_normalised(raw_df)
        df_r.to_sql("receipts",   conn, if_exists="append", index=False)
        df_i.to_sql("line_items", conn, if_exists="append", index=False)
        conn.commit()
    else:
        # Fall back to seed data — dashboard always works
        _seed_demo_data(conn)

    conn.close()


def reset_db() -> None:
    """Drop and recreate the database (useful for demo resets)."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()


@st.cache_data(ttl=300, show_spinner="Loading BillWise data…")
def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and return (df_receipts, df_items, df_joined).
    Cached for 5 minutes. Call st.cache_data.clear() to force reload.
    """
    init_db()
    conn = _get_conn()

    df_receipts = pd.read_sql(
        "SELECT * FROM receipts ORDER BY receipt_date DESC",
        conn,
    )
    df_items = pd.read_sql(
        "SELECT * FROM line_items",
        conn,
    )
    conn.close()

    # Parse dates
    df_receipts["receipt_date"] = pd.to_datetime(
        df_receipts["receipt_date"], errors="coerce"
    )
    df_items["id"] = df_items["id"].astype(int)

    # Build joined DataFrame (most analytics use this)
    df_joined = df_items.merge(
        df_receipts[[
            "receipt_id", "vendor_name", "receipt_date",
            "receipt_total", "tax", "card_used", "extraction_confidence",
        ]],
        on="receipt_id",
        how="left",
    )
    df_joined["month"] = (
        df_joined["receipt_date"].dt.to_period("M").astype(str)
    )

    return df_receipts, df_items, df_joined


def get_data_source_label() -> str:
    """Return a human-readable label for the current data source."""
    if os.environ.get("GCS_BUCKET_NAME") and _load_from_gcs() is not None:
        return "☁️ Google Cloud Storage"
    if LOCAL_CSV.exists():
        return "📁 Local CSV"
    return "🌱 Demo Seed Data"


# ── Human Validation helpers ───────────────────────────────────────────────────

def save_category_validation(
    item_id: int,
    receipt_id: str,
    raw_item_text: str,
    original_category: str,
    validated_category: str,
    validator_note: str = "",
) -> None:
    """Persist a human category correction for a flagged line item."""
    conn = _get_conn()
    _init_schema(conn)
    conn.execute(
        """
        INSERT INTO human_validations
            (item_id, receipt_id, raw_item_text, original_category,
             validated_category, validator_note, validated_at, validation_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'category')
        """,
        (
            item_id,
            receipt_id,
            raw_item_text,
            original_category,
            validated_category,
            validator_note,
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def save_ocr_correction(
    receipt_id: str,
    field_name: str,
    original_value: str,
    corrected_value: str,
) -> None:
    """Persist a human correction for an OCR-extracted receipt field."""
    conn = _get_conn()
    _init_schema(conn)
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
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def load_validated_item_ids() -> set[int]:
    """Return the set of line_item IDs that have already been validated."""
    conn = _get_conn()
    _init_schema(conn)
    try:
        rows = conn.execute(
            "SELECT item_id FROM human_validations WHERE validation_type = 'category'"
        ).fetchall()
        return {r[0] for r in rows if r[0] is not None}
    except Exception:
        return set()
    finally:
        conn.close()


def load_all_validations() -> pd.DataFrame:
    """Return the full human_validations table as a DataFrame."""
    conn = _get_conn()
    _init_schema(conn)
    try:
        df = pd.read_sql(
            "SELECT * FROM human_validations ORDER BY validated_at DESC", conn
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def load_all_ocr_corrections() -> pd.DataFrame:
    """Return the full ocr_corrections table as a DataFrame."""
    conn = _get_conn()
    _init_schema(conn)
    try:
        df = pd.read_sql(
            "SELECT * FROM ocr_corrections ORDER BY corrected_at DESC", conn
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df
