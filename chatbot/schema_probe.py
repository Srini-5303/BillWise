"""
schema_probe.py
Introspects a DuckDB 'data' table and produces a compact context string
that is injected into the Gemini system prompt.
Works for any CSV schema — no hardcoded column names.
"""

import duckdb


def probe_schema(conn: duckdb.DuckDBPyConnection) -> dict:
    """
    Returns a dict describing the table:
      columns     : list of {name, type, samples, min, max}
      row_count   : int
      numeric_cols: list of column names
      date_cols   : list of column names
    """
    columns = []
    numeric_types = {"INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL",
                     "HUGEINT", "SMALLINT", "TINYINT", "UBIGINT", "UINTEGER"}
    date_types    = {"DATE", "TIMESTAMP", "TIMESTAMPTZ", "TIME"}

    raw_cols   = conn.execute("DESCRIBE data").fetchall()
    row_count  = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    numeric_cols, date_cols = [], []

    for row in raw_cols:
        col_name = row[0]
        col_type = row[1].upper().split("(")[0].strip()

        col_info = {"name": col_name, "type": col_type, "samples": [], "min": None, "max": None}

        # Sample distinct values (first 4)
        try:
            samples = conn.execute(
                f'SELECT DISTINCT "{col_name}" FROM data WHERE "{col_name}" IS NOT NULL LIMIT 4'
            ).fetchall()
            col_info["samples"] = [str(r[0]) for r in samples]
        except Exception:
            pass

        # Min/max for numeric and date columns
        if col_type in numeric_types:
            numeric_cols.append(col_name)
            try:
                mn, mx = conn.execute(
                    f'SELECT MIN("{col_name}"), MAX("{col_name}") FROM data'
                ).fetchone()
                col_info["min"] = mn
                col_info["max"] = mx
            except Exception:
                pass
        elif col_type in date_types:
            date_cols.append(col_name)
            try:
                mn, mx = conn.execute(
                    f'SELECT MIN("{col_name}"), MAX("{col_name}") FROM data'
                ).fetchone()
                col_info["min"] = str(mn)
                col_info["max"] = str(mx)
            except Exception:
                pass

        columns.append(col_info)

    return {
        "columns":      columns,
        "row_count":    row_count,
        "numeric_cols": numeric_cols,
        "date_cols":    date_cols,
    }


def build_context_string(schema: dict) -> str:
    """Formats the schema dict into a compact string for the LLM system prompt."""
    lines = [
        f"Table name : data",
        f"Total rows : {schema['row_count']:,}",
        "",
        "Columns:",
    ]
    for col in schema["columns"]:
        line = f"  - {col['name']} ({col['type']})"
        if col["min"] is not None and col["max"] is not None:
            line += f" | range: {col['min']} → {col['max']}"
        if col["samples"]:
            line += f" | e.g. {', '.join(col['samples'][:3])}"
        lines.append(line)

    if schema["date_cols"]:
        lines += ["", f"Date columns   : {', '.join(schema['date_cols'])}"]
    if schema["numeric_cols"]:
        lines += [f"Numeric columns: {', '.join(schema['numeric_cols'])}"]

    return "\n".join(lines)
