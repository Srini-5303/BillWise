"""
csv_loader.py
Loads any CSV file (GCS, local path) into a DuckDB in-memory connection
and registers it as a table called `data`.
Works for any CSV schema — column types are inferred automatically.
"""

import io
import tempfile
import os

import duckdb
import pandas as pd


def _bytes_to_conn(content: bytes) -> duckdb.DuckDBPyConnection:
    """Write bytes to a temp file, load into DuckDB, return connection."""
    conn = duckdb.connect()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        conn.execute(
            f"CREATE TABLE data AS SELECT * FROM read_csv_auto('{tmp_path}', header=true, sample_size=-1)"
        )
    finally:
        os.unlink(tmp_path)
    return conn


def load_from_gcs(bucket_name: str, blob_name: str) -> duckdb.DuckDBPyConnection:
    """Download a CSV blob from GCS and load into DuckDB."""
    from google.cloud import storage
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    content = blob.download_as_bytes()
    return _bytes_to_conn(content)


def load_from_local(file_path: str) -> duckdb.DuckDBPyConnection:
    """Load a local CSV file into DuckDB."""
    conn = duckdb.connect()
    conn.execute(
        f"CREATE TABLE data AS SELECT * FROM read_csv_auto('{file_path}', header=true, sample_size=-1)"
    )
    return conn


def load_from_dataframe(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """Load an existing pandas DataFrame into DuckDB (useful for testing)."""
    conn = duckdb.connect()
    conn.register("data", df)
    return conn


def get_connection(source: str, source_type: str = "gcs") -> duckdb.DuckDBPyConnection:
    """
    Generic entry point.
    source_type: "gcs" | "local"
    For "gcs": source is the blob name; reads GCS_BUCKET_NAME from env.
    For "local": source is the file path.
    """
    if source_type == "gcs":
        bucket = os.environ["GCS_BUCKET_NAME"]
        return load_from_gcs(bucket, source)
    elif source_type == "local":
        return load_from_local(source)
    else:
        raise ValueError(f"Unknown source_type: {source_type!r}. Use 'gcs' or 'local'.")
