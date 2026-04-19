"""
text_to_sql.py — Text-to-SQL engine for the Ask BillWise page.

Flow:
  User question
    → build_schema_context()     — describe the three DuckDB tables
    → generate_sql()             — Gemini writes a SELECT query (with few-shot examples)
    → run_query()                — DuckDB executes it safely
    → [self-correct if error]    — Gemini fixes its own SQL on failure (up to 3 retries)
    → format_result()            — returns (answer_text, result_df, chart)

Improvements over the naive approach:
  1. Few-shot examples  — 6 hand-written Q→SQL pairs in the system prompt
  2. Self-correction    — failed SQL + error message sent back to Gemini for repair
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any

import duckdb
import pandas as pd

from charts import result_bar_chart, result_line_chart

# ── Safety: only read-only SQL is allowed ─────────────────────────────────────
_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE|ATTACH)\b",
    re.IGNORECASE,
)

MAX_RETRIES = 3   # max self-correction attempts per question


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class SQLResult:
    answer_text:  str
    sql:          str  = ""
    result_df:    Any  = None   # pd.DataFrame | None
    chart:        Any  = None   # go.Figure | None
    error:        str  = ""
    attempts:     int  = 1      # how many generation attempts were needed
    was_corrected: bool = False  # True if self-correction fired at least once


# ── Few-shot examples (hand-written, always correct) ─────────────────────────
# Each pair teaches the model the preferred SQL style for BillWise.
# Covers: SUM, COUNT DISTINCT, GROUP BY + ORDER, WHERE filter, date filter, MAX.

_FEW_SHOT_EXAMPLES = """
-- EXAMPLE 1: simple total spend
-- Q: What is the total spend across all receipts?
<sql>
SELECT SUM(receipt_total) AS total_spend
FROM receipts;
</sql>

-- EXAMPLE 2: group by with ordering
-- Q: Which vendor has the highest total spend?
<sql>
SELECT vendor_name, SUM(receipt_total) AS total_spend
FROM receipts
GROUP BY vendor_name
ORDER BY total_spend DESC
LIMIT 1;
</sql>

-- EXAMPLE 3: category filter
-- Q: How much did we spend on Dairy items?
<sql>
SELECT SUM(line_total) AS total_spend
FROM joined
WHERE category = 'Dairy';
</sql>

-- EXAMPLE 4: count distinct
-- Q: How many unique vendors do we have?
<sql>
SELECT COUNT(DISTINCT vendor_name) AS unique_vendors
FROM receipts;
</sql>

-- EXAMPLE 5: date range filter
-- Q: What was the total spend in the last 30 days?
<sql>
SELECT SUM(receipt_total) AS total_spend
FROM receipts
WHERE receipt_date >= CURRENT_DATE - INTERVAL '30 days';
</sql>

-- EXAMPLE 6: top-N items
-- Q: Show the top 5 most expensive items by line total
<sql>
SELECT normalized_item_text, category, line_total
FROM joined
ORDER BY line_total DESC
LIMIT 5;
</sql>
"""


# ── Schema context builder ────────────────────────────────────────────────────

def build_schema_context(
    df_receipts: pd.DataFrame,
    df_items:    pd.DataFrame,
    df_joined:   pd.DataFrame,
) -> str:
    """
    Build a schema description + few-shot examples for Gemini.
    Explicitly enumerates all column names (reduces column hallucination).
    """
    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    def _col_lines(df: pd.DataFrame) -> str:
        lines = []
        for col in df.columns:
            dtype  = str(df[col].dtype)
            sample = df[col].dropna().head(2).tolist()
            s      = ", ".join(str(v)[:25] for v in sample)
            lines.append(f"  {col} ({dtype})  -- e.g. {s}")
        return "\n".join(lines)

    return f"""Today is {today}.

=== DATABASE SCHEMA ===

TABLE receipts   (one row per receipt)
{_col_lines(df_receipts)}

TABLE line_items  (one row per purchased item)
{_col_lines(df_items)}

TABLE joined      (line_items LEFT JOIN receipts — use this for most questions)
{_col_lines(df_joined)}

=== RULES ===
1. Only SELECT statements — never INSERT/UPDATE/DELETE/DROP.
2. receipt_date is TIMESTAMP — use receipt_date >= CURRENT_DATE - INTERVAL '30 days'.
3. line_total and receipt_total are DOUBLE — no casting needed.
4. Use "joined" for anything involving items + receipt details together.
5. Use "receipts" for receipt-level aggregation (vendor spend, receipt counts).
6. Always alias aggregated columns: SUM(line_total) AS total_spend.
7. LIMIT 50 unless the user asks for more or wants a single value.
8. vendor_name in "receipts" and "joined" refers to the store/supplier name.

=== EXAMPLES ===
{_FEW_SHOT_EXAMPLES}
"""


# ── Gemini SQL generator ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a SQL analyst for BillWise, a restaurant supply expense tracker.
Given the database schema and examples below, write a single correct DuckDB SQL
SELECT query that answers the user's question.

Wrap ONLY the SQL in <sql> ... </sql> tags. No explanation, no markdown, no commentary.
If the question is ambiguous, make a reasonable assumption and write the query.
"""

_CORRECTION_PROMPT = """\
The SQL query you wrote failed to execute. Here is the error:

ERROR: {error}

Original SQL:
{sql}

Fix the SQL so it works correctly with DuckDB. Return ONLY the corrected SQL
wrapped in <sql> ... </sql> tags. No explanation needed.
"""


def _call_gemini(prompt: str, api_key: str, temperature: float = 0.1) -> str:
    """Low-level Gemini call. Returns raw response text."""
    from google import genai                          # type: ignore
    from google.genai import types as gtypes          # type: ignore

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=gtypes.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


def _extract_sql(raw: str) -> str | None:
    """Pull SQL out of <sql>...</sql> tags or bare SELECT."""
    m = re.search(r"<sql>(.*?)</sql>", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(SELECT\s.+)", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def generate_sql(question: str, schema_context: str, api_key: str) -> tuple[str, str]:
    """
    Ask Gemini to generate DuckDB SQL for *question*.
    Returns (sql, raw_response). Raises ValueError if nothing extracted.
    """
    prompt = f"{schema_context}\n\nUser question: {question}"
    raw    = _call_gemini(prompt, api_key)
    sql    = _extract_sql(raw)
    if not sql:
        raise ValueError(f"Gemini returned no SQL.\nResponse: {raw}")
    return sql, raw


def correct_sql(
    bad_sql: str,
    error:   str,
    schema_context: str,
    api_key: str,
) -> str:
    """
    Self-correction pass: send the broken SQL + error back to Gemini and ask
    for a fix. Returns corrected SQL string or raises ValueError.
    """
    correction_prompt = (
        schema_context
        + "\n\n"
        + _CORRECTION_PROMPT.format(error=error.strip(), sql=bad_sql.strip())
    )
    raw = _call_gemini(correction_prompt, api_key, temperature=0.05)
    sql = _extract_sql(raw)
    if not sql:
        raise ValueError(f"Correction attempt returned no SQL.\nResponse: {raw}")
    return sql


# ── Safe DuckDB executor ──────────────────────────────────────────────────────

def run_query(
    sql:         str,
    df_receipts: pd.DataFrame,
    df_items:    pd.DataFrame,
    df_joined:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Execute a SELECT query against in-memory DuckDB tables.
    Raises ValueError on blocked or failed queries.
    """
    if _BLOCKED.search(sql):
        raise ValueError("Blocked: query contains a write statement.")

    conn = duckdb.connect()
    conn.register("receipts",   df_receipts)
    conn.register("line_items", df_items)
    conn.register("joined",     df_joined)
    try:
        return conn.execute(sql).fetchdf()
    finally:
        conn.close()


# ── Result formatter ──────────────────────────────────────────────────────────

def _fmt_currency(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


def format_result(result_df: pd.DataFrame, question: str) -> tuple[str, Any]:
    """
    Build a human-readable answer + optional Plotly chart from query results.
    Returns (answer_text, chart_or_None).
    """
    if result_df is None or result_df.empty:
        return "The query returned no results.", None

    rows, cols = result_df.shape

    # Single scalar
    if rows == 1 and cols == 1:
        val = result_df.iloc[0, 0]
        col = result_df.columns[0]
        if isinstance(val, float) and math.isnan(val):
            return "No data found for that query.", None
        if any(k in col.lower() for k in ("spend", "total", "price", "amount", "cost", "tax")):
            return f"**{col.replace('_',' ').title()}**: {_fmt_currency(val)}", None
        return (
            f"**{col.replace('_',' ').title()}**: {val:,}"
            if isinstance(val, (int, float))
            else f"**{col.replace('_',' ').title()}**: {val}"
        ), None

    chart = None

    # Two-column (label + numeric) → auto bar chart
    if cols == 2:
        label_col = result_df.columns[0]
        value_col = result_df.columns[1]
        if pd.api.types.is_numeric_dtype(result_df[value_col]):
            chart = result_bar_chart(
                result_df, label_col, value_col,
                f"{value_col.replace('_',' ').title()} by {label_col.replace('_',' ').title()}",
            )

    # Single-row multi-column
    if rows == 1:
        parts = []
        for c in result_df.columns:
            val = result_df.iloc[0][c]
            if any(k in c.lower() for k in ("spend", "total", "price", "amount", "cost", "tax")):
                parts.append(f"**{c.replace('_',' ').title()}**: {_fmt_currency(val)}")
            elif isinstance(val, float):
                parts.append(f"**{c.replace('_',' ').title()}**: {val:,.2f}")
            elif isinstance(val, int):
                parts.append(f"**{c.replace('_',' ').title()}**: {val:,}")
            else:
                parts.append(f"**{c.replace('_',' ').title()}**: {val}")
        return " · ".join(parts), chart

    # Multi-row: summarise top result
    numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        top_numeric = numeric_cols[0]
        top_row     = result_df.sort_values(top_numeric, ascending=False).iloc[0]
        label_col   = result_df.columns[0]
        top_val     = top_row[top_numeric]
        top_label   = top_row[label_col]
        val_str = (
            _fmt_currency(top_val)
            if any(k in top_numeric.lower() for k in ("spend","total","price","amount","cost","tax"))
            else f"{top_val:,.2f}" if isinstance(top_val, float) else f"{top_val:,}"
        )
        return f"**{rows} rows** returned. Top result: **{top_label}** — {val_str}.", chart

    return f"**{rows} rows** returned.", chart


# ── Public entry point ────────────────────────────────────────────────────────

def ask_billwise(
    question:    str,
    df_receipts: pd.DataFrame,
    df_items:    pd.DataFrame,
    df_joined:   pd.DataFrame,
    api_key:     str | None = None,
    max_retries: int = MAX_RETRIES,
) -> SQLResult:
    """
    Full Text-to-SQL pipeline with few-shot prompting and self-correction.

    Steps:
      1. Build schema context (with few-shot examples injected).
      2. Ask Gemini to generate SQL.
      3. Execute against DuckDB.
      4. If execution fails → send error back to Gemini → get corrected SQL.
      5. Repeat step 3-4 up to max_retries times.
      6. Format the final result into a human-readable answer + optional chart.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return SQLResult(
            answer_text="⚠️ No Gemini API key configured. Set GEMINI_API_KEY in .env.",
        )

    schema_ctx = build_schema_context(df_receipts, df_items, df_joined)

    # ── Step 1: initial SQL generation ────────────────────────────────────────
    try:
        sql, _ = generate_sql(question, schema_ctx, key)
    except Exception as e:
        return SQLResult(answer_text=f"⚠️ Gemini could not generate SQL: {e}", error=str(e))

    # ── Steps 2–4: execute with self-correction loop ──────────────────────────
    last_error  = ""
    attempts    = 1
    was_corrected = False

    for attempt in range(max_retries):
        try:
            result_df = run_query(sql, df_receipts, df_items, df_joined)
            # Success
            answer, chart = format_result(result_df, question)
            return SQLResult(
                answer_text=answer,
                sql=sql,
                result_df=result_df,
                chart=chart,
                attempts=attempts,
                was_corrected=was_corrected,
            )
        except Exception as exec_err:
            last_error = str(exec_err)
            if attempt < max_retries - 1:
                # Self-correction: ask Gemini to fix its own SQL
                try:
                    sql          = correct_sql(sql, last_error, schema_ctx, key)
                    attempts    += 1
                    was_corrected = True
                except Exception as corr_err:
                    last_error = str(corr_err)
                    break

    return SQLResult(
        answer_text=f"⚠️ Query failed after {attempts} attempt(s): {last_error}",
        sql=sql,
        error=last_error,
        attempts=attempts,
        was_corrected=was_corrected,
    )


# ── Interactive test entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import logging
    from pathlib import Path

    _here = Path(__file__).parent.resolve()

    # Load .env so GEMINI_API_KEY is available
    _env_file = _here / ".env"
    if _env_file.exists():
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

    # Change to dashboard dir so data_loader can find billwise.db / seed data
    os.chdir(_here)

    # Suppress Streamlit's "No runtime found" warnings — harmless outside the app
    logging.disable(logging.WARNING)
    from data_loader import load_all_data  # type: ignore
    logging.disable(logging.NOTSET)

    print("Loading data …")
    df_receipts, df_items, df_joined = load_all_data()
    print(
        f"Loaded: {len(df_receipts)} receipt(s), "
        f"{len(df_items)} line item(s)\n"
        + "-" * 60
    )

    _api_key = os.environ.get("GEMINI_API_KEY", "")
    if not _api_key:
        print("⚠️  GEMINI_API_KEY not set — set it in .env or the environment.")
        sys.exit(1)

    # Use CLI arguments as questions, or fall back to built-in sample questions
    _questions = sys.argv[1:] or [
        "What is the total spend across all receipts?",
        "Which vendor has the highest total spend?",
        "Show spend broken down by category.",
        "What are the top 5 most expensive items?",
        "How many unique vendors do we have?",
    ]

    for _q in _questions:
        print(f"\n❓ {_q}")
        _res = ask_billwise(_q, df_receipts, df_items, df_joined, api_key=_api_key)

        print(f"SQL:\n{_res.sql or '(none)'}")
        if _res.was_corrected:
            print(f"⚠️  Self-corrected ({_res.attempts} attempt(s))")

        print(f"\n💬 {_res.answer_text}")

        if _res.result_df is not None and not _res.result_df.empty:
            print()
            print(_res.result_df.to_string(index=False))

        if _res.error:
            print(f"\n🔴 Error: {_res.error}")

        print("-" * 60)
