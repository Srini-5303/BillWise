import csv
import io
import os
import logging
import threading
from datetime import datetime
from google.cloud import storage

log = logging.getLogger("csv_writer")


BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
CSV_BLOB    = os.environ.get("GCS_BILLS_BLOB", "bills_output.csv")
HEADERS     = ["Serial_No", "Bill_File", "Store_Name", "Invoice_Date",
               "Total", "Card_Used", "Received_At", "Sender", "Image_Hash",
               "Item_Name", "Item_Price", "Grocery_Category"]

_lock       = threading.Lock()
_gcs_client = storage.Client()


def _get_bucket():
    return _gcs_client.bucket(BUCKET_NAME)


def _read_rows() -> list[list]:
    """Download current CSV from GCS and return all rows (no header)."""
    bucket = _get_bucket()
    blob   = bucket.blob(CSV_BLOB)

    if not blob.exists():
        return []

    content = blob.download_as_text(encoding="utf-8")
    reader  = csv.reader(io.StringIO(content))
    rows    = list(reader)
    return rows[1:] if rows else []


def _write_rows(rows: list[list]):
    """Upload full CSV (header + all rows) back to GCS."""
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(HEADERS)
    w.writerows(rows)

    bucket = _get_bucket()
    blob   = bucket.blob(CSV_BLOB)
    blob.upload_from_string(buf.getvalue(), content_type="text/csv")


def _fuzzy_score(s1: str, s2: str) -> float:
    """
    Simple character-level fuzzy match score between 0.0 and 1.0.
    Uses longest common subsequence ratio — no extra libraries needed.
    """
    s1, s2 = s1.upper().strip(), s2.upper().strip()
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    # Build LCS matrix
    m, n   = len(s1), len(s2)
    dp     = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[m][n]
    return (2.0 * lcs_len) / (m + n)


# Minimum fuzzy score to consider store names a match
STORE_SIMILARITY_THRESHOLD = 0.85


def is_duplicate(image_hash: str, store: str, date: str, total: str):
    """
    Check both duplicate layers against existing CSV rows.

    Layer 1 — exact image hash match (any sender)
    Layer 2 — fuzzy store + exact date + exact total match

    Returns (True, matching_row) if duplicate, else (False, None).
    Col indices: 0=Serial, 1=File, 2=Store, 3=Date, 4=Total, 5=Card,
                 6=ReceivedAt, 7=Sender, 8=Hash, 9=Item_Name, 10=Item_Price, 11=Category
    """
    rows = _read_rows()

    for row in rows:
        if len(row) < 9:
            continue

        existing_hash  = row[8].strip()
        existing_store = row[2].strip()
        existing_date  = row[3].strip()
        existing_total = row[4].strip()

        # Layer 1 — same image
        if image_hash and existing_hash == image_hash:
            return True, row

        # Layer 2 — same bill content
        date_match  = (date  and existing_date  == date)
        total_match = (total and existing_total == str(total))
        store_score = _fuzzy_score(store, existing_store)
        store_match = store_score >= STORE_SIMILARITY_THRESHOLD

        if date_match and total_match and store_match:
            return True, row

    return False, None


def append_bill(filename, store, date, total, card, sender, image_hash, items) -> int:
    """
    Thread-safe append of one bill to GCS CSV.
    Writes one row per item. items is a list of (item_name, item_price, grocery_category) tuples.
    Returns the bill's serial number.
    """
    if not items:
        items = [("Unknown", "", "")]

    with _lock:
        existing = _read_rows()

        # Serial is bill-level — find max existing serial rather than row count
        if existing:
            try:
                serial = max(int(row[0]) for row in existing if row) + 1
            except (ValueError, IndexError):
                serial = len(existing) + 1
        else:
            serial = 1

        received = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        for item_name, item_price, item_category in items:
            new_row = [
                serial,
                filename,
                store or "Not found",
                date  or "Not found",
                total or "Not found",
                card,
                received,
                sender,
                image_hash,
                item_name,
                item_price,
                item_category,
            ]
            existing.append(new_row)

        _write_rows(existing)
        log.info("Bill #%d written — store=%r | date=%s | total=%s | %d item rows → GCS",
                 serial, store, date, total, len(items))

    return serial


def reset_csv():
    """Overwrite GCS CSV with just the headers (wipes all data)."""
    with _lock:
        _write_rows([])
