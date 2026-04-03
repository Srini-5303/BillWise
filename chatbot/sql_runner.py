"""
sql_runner.py
Executes LLM-generated SQL inside DuckDB and formats the result.
Includes a token-level safety check to block destructive statements.
"""

import duckdb


# Tokens that must never appear in LLM-generated queries
_FORBIDDEN = [
    "drop ", "delete ", "insert ", "update ", "create ",
    "alter ", "attach ", "copy ", "truncate ", "export ",
    "load ", "install ", "pragma ",
]


def _format_scalar(value, col_name: str) -> str:
    """Pretty-print a single scalar result."""
    if value is None:
        return "No data found."
    col_lower = col_name.lower()
    if isinstance(value, float):
        # Treat as currency if the column name suggests money
        if any(w in col_lower for w in ["total", "sum", "amount", "spend", "cost", "price"]):
            return f"**${value:,.2f}**"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def run_sql(
    sql: str,
    conn: duckdb.DuckDBPyConnection,
) -> tuple[bool, str]:
    """
    Execute SQL in DuckDB and return a formatted string result.

    Returns:
        (success: bool, result_or_error: str)
        On success  : human-readable result (scalar, markdown table, or message)
        On failure  : error description
    """
    # ── Safety check ───────────────────────────────────────────────────────────
    lower = sql.lower()
    for token in _FORBIDDEN:
        if token in lower:
            return False, f"Query blocked: `{token.strip()}` statements are not allowed."

    # ── Execute ────────────────────────────────────────────────────────────────
    try:
        result_df = conn.execute(sql).df()
    except duckdb.Error as e:
        return False, f"SQL error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

    # ── Format result ──────────────────────────────────────────────────────────
    if result_df.empty:
        return True, "No matching records found for that query."

    rows, cols = result_df.shape

    # Single scalar (e.g. SELECT SUM(Total) → 1 row × 1 col)
    if rows == 1 and cols == 1:
        col_name = result_df.columns[0]
        return True, _format_scalar(result_df.iloc[0, 0], col_name)

    # Single-row multi-column (e.g. SELECT COUNT(*), AVG(Total))
    if rows == 1:
        parts = []
        for col in result_df.columns:
            parts.append(f"**{col}**: {_format_scalar(result_df.iloc[0][col], col)}")
        return True, "  |  ".join(parts)

    # Multi-row table — render as markdown
    try:
        table_md = result_df.to_markdown(index=False)
    except Exception:
        # tabulate not installed fallback
        table_md = result_df.to_string(index=False)

    # Cap output at 30 rows to keep SMS / chat responses readable
    display_rows = rows
    if rows > 30:
        lines = table_md.split("\n")
        header = "\n".join(lines[:2])          # header + separator
        body   = "\n".join(lines[2:32])        # first 30 data rows
        table_md = f"{header}\n{body}\n\n*... showing 30 of {rows} rows*"
        display_rows = 30

    return True, table_md
