import csv
import io
import os
import threading
from datetime import datetime
from google.cloud import storage

# ── Set your bucket name as a Cloud Run env var ──
BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
CSV_BLOB    = "bills_output.csv"
HEADERS     = ["Serial_No", "Bill_File", "Store_Name", "Invoice_Date",
               "Total", "Card_Used", "Received_At", "Sender"]

_lock          = threading.Lock()
_gcs_client    = storage.Client()          # uses Cloud Run's service account automatically


def _get_bucket():
    return _gcs_client.bucket(BUCKET_NAME)


def _read_rows() -> list[list]:
    """Download current CSV from GCS and parse it. Returns all rows (no header)."""
    bucket = _get_bucket()
    blob   = bucket.blob(CSV_BLOB)

    if not blob.exists():
        return []

    content = blob.download_as_text(encoding="utf-8")
    reader  = csv.reader(io.StringIO(content))
    rows    = list(reader)
    return rows[1:] if rows else []  


def _write_rows(rows: list[list]):
    """Upload the full CSV (header + all rows) back to GCS."""
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(HEADERS)
    w.writerows(rows)

    bucket = _get_bucket()
    blob   = bucket.blob(CSV_BLOB)
    blob.upload_from_string(buf.getvalue(), content_type="text/csv")


def append_bill(filename, store, date, total, card, sender) -> int:
    """
    Thread-safe append of one bill row to the GCS CSV.
    Returns the assigned serial number.
    """
    with _lock:
        existing = _read_rows()
        serial   = len(existing) + 1
        received = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        new_row  = [serial, filename, store,
                    date  or "Not found",
                    total or "Not found",
                    card, received, sender]

        existing.append(new_row)
        _write_rows(existing)

    return serial