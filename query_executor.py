"""
query_executor.py — Execute a parsed intent dict against the DataFrames.

Flow:
  intent dict
    → apply_filters()          (pandas masking — no LLM)
    → _handle_<intent_type>()  (deterministic aggregation)
    → build answer text        (deterministic string formatting)
    → optional Gemini explanation (narration only, never arithmetic)

Returns a QueryResult dataclass with:
  answer_text  — plain-English answer (deterministic)
  filters_used — dict of filters actually applied
  result_df    — supporting table DataFrame
  chart        — optional Plotly figure
  explanation  — optional short LLM explanation (if api_key set)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from utils import apply_date_filter, fmt_currency, fmt_number, resolve_date_range
from charts import (
    result_bar_chart,
    result_line_chart,
    spend_trend_chart,
    vendor_bar_chart,
    category_bar_chart,
    item_bar_chart,
)


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    answer_text:  str
    filters_used: dict      = field(default_factory=dict)
    result_df:    Any       = None     # pd.DataFrame | None
    chart:        Any       = None     # go.Figure | None
    explanation:  str       = ""
    intent:       dict      = field(default_factory=dict)


# ── Filter helpers ─────────────────────────────────────────────────────────────

def apply_filters(
    df:      pd.DataFrame,
    filters: dict,
    date_col: str = "receipt_date",
) -> tuple[pd.DataFrame, dict]:
    """
    Apply intent filters to a DataFrame.
    Returns (filtered_df, filters_actually_applied).
    """
    applied: dict = {}
    df = df.copy()

    cat = filters.get("category")
    if cat:
        mask = df["category"].str.lower() == cat.lower() if "category" in df.columns else pd.Series(True, index=df.index)
        df = df[mask]
        applied["category"] = cat

    vendor = filters.get("vendor")
    if vendor and "vendor_name" in df.columns:
        df = df[df["vendor_name"].str.contains(vendor, case=False, na=False)]
        applied["vendor"] = vendor

    item_text = filters.get("item_text")
    if item_text:
        text_cols = [c for c in ("normalized_item_text", "raw_item_text",
                                  "matched_canonical_item") if c in df.columns]
        if text_cols:
            mask = pd.Series(False, index=df.index)
            for col in text_cols:
                mask = mask | df[col].str.contains(item_text, case=False, na=False)
            df = df[mask]
            applied["item_text"] = item_text

    date_range = filters.get("date_range")
    if date_range:
        start, end = resolve_date_range(date_range)
        if date_col in df.columns:
            df = apply_date_filter(df, date_col, start, end)
            applied["date_range"] = date_range

    return df, applied


# ── Intent handlers ────────────────────────────────────────────────────────────

def _handle_spend_total(
    intent: dict,
    df_joined: pd.DataFrame,
    df_receipts: pd.DataFrame,
) -> QueryResult:
    df, applied = apply_filters(df_joined, intent["filters"])
    group_by = intent.get("group_by")

    if df.empty:
        return QueryResult(
            answer_text="No matching data found for those filters.",
            filters_used=applied,
            result_df=pd.DataFrame(),
            intent=intent,
        )

    if group_by and group_by in df.columns:
        result = (
            df.groupby(group_by, as_index=False)["line_total"]
            .sum()
            .rename(columns={"line_total": "spend"})
            .sort_values("spend", ascending=intent.get("order", "desc") == "desc")
        )
        limit = intent.get("limit")
        if limit:
            result = result.head(int(limit))
        result["spend"] = result["spend"].round(2)

        top_name  = result.iloc[0][group_by]
        top_spend = result.iloc[0]["spend"]
        total     = result["spend"].sum()

        answer = (
            f"Top {group_by.replace('_', ' ')} by spend: "
            f"**{top_name}** leads with **{fmt_currency(top_spend)}**. "
            f"Total across shown rows: **{fmt_currency(total)}**."
        )
        chart = result_bar_chart(result, group_by, "spend", f"Spend by {group_by.replace('_', ' ').title()}")
    else:
        total = df["line_total"].sum()
        answer = f"Total spend: **{fmt_currency(total)}**"
        if applied.get("category"):
            answer += f" on **{applied['category']}**"
        if applied.get("vendor"):
            answer += f" from **{applied['vendor']}**"
        if applied.get("date_range"):
            answer += f" ({applied['date_range'].replace('_', ' ')})"
        answer += "."
        result = pd.DataFrame({"Metric": ["Total Spend"], "Value": [fmt_currency(total)]})
        chart  = None

    return QueryResult(answer_text=answer, filters_used=applied,
                       result_df=result, chart=chart, intent=intent)


def _handle_quantity_total(intent: dict, df_joined: pd.DataFrame) -> QueryResult:
    df, applied = apply_filters(df_joined, intent["filters"])
    group_by    = intent.get("group_by")

    if df.empty:
        return QueryResult("No matching items found.", filters_used=applied,
                           result_df=pd.DataFrame(), intent=intent)

    if group_by and group_by in df.columns:
        result = (
            df.groupby(group_by, as_index=False)["quantity"]
            .sum()
            .sort_values("quantity", ascending=False)
        )
        limit = intent.get("limit")
        if limit:
            result = result.head(int(limit))
        top     = result.iloc[0]
        answer  = (
            f"Top item by quantity: **{top[group_by]}** with "
            f"**{fmt_number(top['quantity'])} units**."
        )
        chart = result_bar_chart(result, group_by, "quantity", "Quantity by Item")
    else:
        total = df["quantity"].sum()
        unit  = df["unit"].mode()[0] if "unit" in df.columns and not df["unit"].empty else "units"
        item_hint = applied.get("item_text") or applied.get("category") or "items"
        answer = (
            f"Total quantity of **{item_hint}** purchased: "
            f"**{total:,.1f} {unit}**"
        )
        if applied.get("date_range"):
            answer += f" ({applied['date_range'].replace('_', ' ')})"
        answer += "."
        result = pd.DataFrame({"Metric": ["Total Quantity"], "Value": [f"{total:,.1f} {unit}"]})
        chart  = None

    return QueryResult(answer_text=answer, filters_used=applied,
                       result_df=result, chart=chart, intent=intent)


def _handle_top_vendors(intent: dict, df_receipts: pd.DataFrame) -> QueryResult:
    df, applied = apply_filters(df_receipts, intent["filters"])
    if df.empty:
        return QueryResult("No vendor data found.", filters_used=applied,
                           result_df=pd.DataFrame(), intent=intent)

    limit  = int(intent.get("limit") or 5)
    result = (
        df.groupby("vendor_name", as_index=False)
        .agg(spend=("receipt_total", "sum"), receipts=("receipt_id", "count"))
        .sort_values("spend", ascending=False)
        .head(limit)
    )
    result["spend"] = result["spend"].round(2)

    top = result.iloc[0]
    answer = (
        f"Top vendor by spend: **{top['vendor_name']}** "
        f"with **{fmt_currency(top['spend'])}** across {int(top['receipts'])} receipts."
    )
    if applied.get("date_range"):
        answer += f" ({applied['date_range'].replace('_', ' ')})"

    chart = result_bar_chart(result, "vendor_name", "spend",
                             f"Top {limit} Vendors by Spend")
    return QueryResult(answer_text=answer, filters_used=applied,
                       result_df=result, chart=chart, intent=intent)


def _handle_top_items(intent: dict, df_joined: pd.DataFrame) -> QueryResult:
    df, applied = apply_filters(df_joined, intent["filters"])
    if df.empty:
        return QueryResult("No item data found.", filters_used=applied,
                           result_df=pd.DataFrame(), intent=intent)

    metric = "quantity" if intent.get("metric") == "sum_quantity" else "spend"
    limit  = int(intent.get("limit") or 5)
    result = (
        df.groupby("matched_canonical_item", as_index=False)
        .agg(spend=("line_total", "sum"), quantity=("quantity", "sum"),
             category=("category", "first"))
        .sort_values(metric, ascending=False)
        .head(limit)
    )
    result["spend"]    = result["spend"].round(2)
    result["quantity"] = result["quantity"].round(1)

    top    = result.iloc[0]
    val    = fmt_currency(top["spend"]) if metric == "spend" else f"{top['quantity']:,.1f} units"
    answer = f"Top item: **{top['matched_canonical_item']}** — **{val}**"
    if applied.get("date_range"):
        answer += f" ({applied['date_range'].replace('_', ' ')})"
    answer += "."

    chart = item_bar_chart(result, metric)
    return QueryResult(answer_text=answer, filters_used=applied,
                       result_df=result, chart=chart, intent=intent)


def _handle_category_breakdown(intent: dict, df_joined: pd.DataFrame) -> QueryResult:
    df, applied = apply_filters(df_joined, intent["filters"])
    if df.empty:
        return QueryResult("No category data found.", filters_used=applied,
                           result_df=pd.DataFrame(), intent=intent)

    result = (
        df.groupby("category", as_index=False)
        .agg(spend=("line_total", "sum"), items=("id", "count"))
        .sort_values("spend", ascending=False)
    )
    result["spend"] = result["spend"].round(2)
    total = result["spend"].sum()
    result["pct"] = (result["spend"] / total * 100).round(1) if total > 0 else 0.0

    top    = result.iloc[0]
    answer = (
        f"Largest spending category: **{top['category']}** "
        f"({fmt_currency(top['spend'])}, {top['pct']:.1f}% of total)."
    )
    chart = result_bar_chart(result, "category", "spend", "Spend by Category")
    return QueryResult(answer_text=answer, filters_used=applied,
                       result_df=result, chart=chart, intent=intent)


def _handle_receipt_lookup(
    intent: dict,
    df_receipts: pd.DataFrame,
    df_joined: pd.DataFrame,
) -> QueryResult:
    df, applied = apply_filters(df_receipts, intent["filters"])
    if df.empty:
        return QueryResult("No receipts matched those filters.", filters_used=applied,
                           result_df=pd.DataFrame(), intent=intent)

    limit  = int(intent.get("limit") or 20)
    result = df.sort_values("receipt_date", ascending=False).head(limit)
    n      = len(result)
    total  = result["receipt_total"].sum()

    answer = f"Found **{n} receipt(s)** totalling **{fmt_currency(total)}**."
    if applied.get("vendor"):
        answer = f"Receipts from **{applied['vendor']}**: {answer}"
    if applied.get("date_range"):
        answer += f" ({applied['date_range'].replace('_', ' ')})"

    return QueryResult(answer_text=answer, filters_used=applied,
                       result_df=result, chart=None, intent=intent)


def _handle_item_search(intent: dict, df_joined: pd.DataFrame) -> QueryResult:
    df, applied = apply_filters(df_joined, intent["filters"])
    if df.empty:
        return QueryResult(
            f"No items found containing **'{intent['filters'].get('item_text', '')}'**.",
            filters_used=applied,
            result_df=pd.DataFrame(),
            intent=intent,
        )

    cols   = ["vendor_name", "receipt_date", "raw_item_text",
              "normalized_item_text", "matched_canonical_item",
              "category", "quantity", "unit", "line_total"]
    cols   = [c for c in cols if c in df.columns]
    result = df[cols].sort_values("receipt_date", ascending=False).head(30)

    total_spend = df["line_total"].sum()
    total_qty   = df["quantity"].sum()
    answer = (
        f"Found **{len(df)} line items** matching "
        f"**'{applied.get('item_text', '')}'** — "
        f"total spend: **{fmt_currency(total_spend)}**, "
        f"total quantity: **{total_qty:,.1f} units**."
    )
    return QueryResult(answer_text=answer, filters_used=applied,
                       result_df=result, chart=None, intent=intent)


def _handle_compare_periods(
    intent: dict,
    df_joined: pd.DataFrame,
) -> QueryResult:
    """Compare this month vs last month (or any two periods)."""
    filters = intent.get("filters", {})

    # Build two filter dicts
    f_current  = {**filters, "date_range": "this_month"}
    f_previous = {**filters, "date_range": "last_month"}

    df_cur,  _ = apply_filters(df_joined, f_current)
    df_prev, _ = apply_filters(df_joined, f_previous)

    spend_cur  = df_cur["line_total"].sum()
    spend_prev = df_prev["line_total"].sum()
    delta      = spend_cur - spend_prev
    delta_pct  = (delta / spend_prev * 100) if spend_prev > 0 else 0.0
    direction  = "up" if delta > 0 else "down"

    result = pd.DataFrame({
        "Period":      ["This Month",      "Last Month"],
        "Spend":       [round(spend_cur, 2), round(spend_prev, 2)],
        "Change ($)":  [round(delta, 2),     "—"],
        "Change (%)":  [f"{delta_pct:+.1f}%","—"],
    })
    answer = (
        f"This month: **{fmt_currency(spend_cur)}** vs "
        f"last month: **{fmt_currency(spend_prev)}** — "
        f"spend is **{direction} {abs(delta_pct):.1f}%** "
        f"({fmt_currency(abs(delta))})."
    )
    if filters.get("category"):
        answer = f"**{filters['category']}** — " + answer

    # Monthly trend chart
    df_joined_copy = df_joined.copy()
    df_joined_copy["month"] = df_joined_copy["receipt_date"].dt.to_period("M").astype(str)
    trend = (
        df_joined_copy.groupby("month", as_index=False)["line_total"]
        .sum()
        .rename(columns={"line_total": "spend"})
        .tail(6)
    )
    chart = result_line_chart(trend, "month", "spend", "Monthly Spend Trend")
    return QueryResult(answer_text=answer, filters_used={"periods": "this_month vs last_month"},
                       result_df=result, chart=chart, intent=intent)


# ── Optional LLM explanation ───────────────────────────────────────────────────

def generate_explanation(
    question:  str,
    result:    QueryResult,
    api_key:   str | None = None,
) -> str:
    """
    Use Gemini to generate a short plain-English explanation of the result.
    The LLM is given only the answer text — it never does arithmetic.
    Returns empty string on failure or if no API key.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return ""

    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=120,
            ),
        )
        prompt = (
            f"A user asked: \"{question}\"\n\n"
            f"The analytics system found: {result.answer_text}\n\n"
            "Write ONE short sentence (max 30 words) giving a business insight "
            "or context about this result. Do not repeat the numbers — "
            "just add meaningful interpretation. Plain text only, no markdown."
        )
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return ""


# ── Public API ─────────────────────────────────────────────────────────────────

def execute_intent(
    intent:      dict,
    df_receipts: pd.DataFrame,
    df_items:    pd.DataFrame,
    df_joined:   pd.DataFrame,
) -> QueryResult:
    """
    Route an intent dict to the correct handler.
    All arithmetic is done with pandas — the LLM never touches numbers.
    """
    it = intent.get("intent_type", "spend_total")

    try:
        if it == "spend_total":
            return _handle_spend_total(intent, df_joined, df_receipts)
        elif it == "quantity_total":
            return _handle_quantity_total(intent, df_joined)
        elif it == "top_vendors":
            return _handle_top_vendors(intent, df_receipts)
        elif it == "top_items":
            return _handle_top_items(intent, df_joined)
        elif it == "category_breakdown":
            return _handle_category_breakdown(intent, df_joined)
        elif it == "receipt_lookup":
            return _handle_receipt_lookup(intent, df_receipts, df_joined)
        elif it == "item_search":
            return _handle_item_search(intent, df_joined)
        elif it == "compare_periods":
            return _handle_compare_periods(intent, df_joined)
        else:
            return _handle_spend_total(intent, df_joined, df_receipts)
    except Exception as exc:
        return QueryResult(
            answer_text=f"⚠️ Could not process that query: {exc}",
            intent=intent,
        )
