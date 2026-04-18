"""
analytics.py — All deterministic pandas analytics for BillWise Dashboard.

Every function takes DataFrames (never raw SQL strings) and returns a
clean DataFrame ready for display or charting. No LLM calls here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import apply_date_filter, resolve_date_range


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe(df: pd.DataFrame) -> pd.DataFrame:
    """Return empty DataFrame on None input."""
    return df if df is not None else pd.DataFrame()


def _prev_period(
    df: pd.DataFrame,
    date_col: str,
    start,
    end,
) -> pd.DataFrame:
    """Return rows from the previous period of equal length."""
    if start is None or end is None:
        return pd.DataFrame()
    delta = (end - start).days + 1
    prev_end   = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=delta - 1)
    return apply_date_filter(df.copy(), date_col, prev_start.date(), prev_end.date())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def get_kpis(
    df_receipts: pd.DataFrame,
    df_joined:   pd.DataFrame,
    date_range:  str = "All Time",
) -> dict:
    """
    Returns KPI dict:
      total_spend, total_receipts, total_vendors,
      total_items, avg_bill, spend_delta_pct
    """
    start, end = resolve_date_range(date_range)

    dfr = apply_date_filter(df_receipts.copy(), "receipt_date", start, end)
    dfj = apply_date_filter(df_joined.copy(),   "receipt_date", start, end)

    total_spend    = dfj["line_total"].sum()
    total_receipts = dfr["receipt_id"].nunique()
    total_vendors  = dfr["vendor_name"].nunique()
    total_items    = len(dfj)
    avg_bill       = (total_spend / total_receipts) if total_receipts else 0

    # Delta vs previous period
    prev_r = _prev_period(df_receipts, "receipt_date",
                          pd.Timestamp(start) if start else None,
                          pd.Timestamp(end)   if end   else None)
    prev_j = _prev_period(df_joined, "receipt_date",
                          pd.Timestamp(start) if start else None,
                          pd.Timestamp(end)   if end   else None)

    prev_spend = prev_j["line_total"].sum() if not prev_j.empty else None
    if prev_spend and prev_spend > 0:
        spend_delta_pct = round((total_spend - prev_spend) / prev_spend * 100, 1)
    else:
        spend_delta_pct = None

    return {
        "total_spend":    round(total_spend, 2),
        "total_receipts": int(total_receipts),
        "total_vendors":  int(total_vendors),
        "total_items":    int(total_items),
        "avg_bill":       round(avg_bill, 2),
        "spend_delta_pct":spend_delta_pct,
    }


def get_spend_trend(
    df_joined:  pd.DataFrame,
    date_range: str = "Last 90 Days",
    freq:       str = "W",          # 'D' daily  'W' weekly  'ME' monthly
) -> pd.DataFrame:
    """Weekly (default) spend aggregated over time. Returns date + spend."""
    start, end = resolve_date_range(date_range)
    df = apply_date_filter(df_joined.copy(), "receipt_date", start, end)
    if df.empty:
        return pd.DataFrame(columns=["receipt_date", "spend"])

    df = (
        df.set_index("receipt_date")["line_total"]
        .resample(freq)
        .sum()
        .reset_index()
    )
    df.columns = ["receipt_date", "spend"]
    return df


def get_recent_receipts(df_receipts: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Latest N receipts sorted by date desc."""
    cols = ["receipt_id", "vendor_name", "receipt_date",
            "receipt_total", "card_used", "extraction_confidence"]
    available = [c for c in cols if c in df_receipts.columns]
    df = df_receipts[available].sort_values("receipt_date", ascending=False).head(n).copy()
    return df


def get_anomalies(df_receipts: pd.DataFrame) -> pd.DataFrame:
    """
    Receipts where receipt_total > mean + 1.5 * std.
    Also flags low extraction confidence (< 0.65).
    """
    if df_receipts.empty:
        return pd.DataFrame()

    mean = df_receipts["receipt_total"].mean()
    std  = df_receipts["receipt_total"].std()
    threshold = mean + 1.5 * std

    high_spend = df_receipts[df_receipts["receipt_total"] > threshold].copy()
    high_spend["alert_type"] = "💰 High Spend"

    low_conf = df_receipts[df_receipts["extraction_confidence"] < 0.65].copy()
    low_conf["alert_type"] = "⚠️ Low Confidence"

    combined = pd.concat([high_spend, low_conf]).drop_duplicates(subset="receipt_id")
    cols     = ["receipt_id", "vendor_name", "receipt_date",
                "receipt_total", "extraction_confidence", "alert_type"]
    available = [c for c in cols if c in combined.columns]
    return combined[available].sort_values("receipt_total", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════

def get_category_spend(
    df_joined:   pd.DataFrame,
    date_range:  str = "All Time",
    categories:  list[str] | None = None,
) -> pd.DataFrame:
    """Spend and quantity per category. Returns category + spend + quantity + pct."""
    start, end = resolve_date_range(date_range)
    df = apply_date_filter(df_joined.copy(), "receipt_date", start, end)

    if categories:
        df = df[df["category"].isin(categories)]
    if df.empty:
        return pd.DataFrame(columns=["category", "spend", "quantity", "pct"])

    agg = (
        df.groupby("category", as_index=False)
        .agg(spend=("line_total", "sum"), quantity=("quantity", "sum"))
        .sort_values("spend", ascending=False)
    )
    total = agg["spend"].sum()
    agg["pct"] = (agg["spend"] / total * 100).round(1) if total > 0 else 0.0
    agg["spend"]    = agg["spend"].round(2)
    agg["quantity"] = agg["quantity"].round(1)
    return agg


def get_category_by_month(
    df_joined:  pd.DataFrame,
    date_range: str = "Last 90 Days",
) -> pd.DataFrame:
    """Pivot table: rows = month, columns = category, values = spend."""
    start, end = resolve_date_range(date_range)
    df = apply_date_filter(df_joined.copy(), "receipt_date", start, end)
    if df.empty:
        return pd.DataFrame()

    df["month"] = df["receipt_date"].dt.to_period("M").astype(str)
    pivot = (
        df.groupby(["month", "category"])["line_total"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    return pivot


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VENDORS
# ══════════════════════════════════════════════════════════════════════════════

def get_top_vendors(
    df_receipts: pd.DataFrame,
    date_range:  str = "All Time",
    n:           int = 10,
) -> pd.DataFrame:
    """Top N vendors by total receipt spend."""
    start, end = resolve_date_range(date_range)
    df = apply_date_filter(df_receipts.copy(), "receipt_date", start, end)
    if df.empty:
        return pd.DataFrame(columns=["vendor_name", "spend", "receipts", "avg_spend"])

    agg = (
        df.groupby("vendor_name", as_index=False)
        .agg(spend=("receipt_total", "sum"), receipts=("receipt_id", "count"))
        .assign(avg_spend=lambda x: (x["spend"] / x["receipts"]).round(2))
        .sort_values("spend", ascending=False)
        .head(n)
    )
    agg["spend"] = agg["spend"].round(2)
    return agg


def get_vendor_frequency(
    df_receipts: pd.DataFrame,
    date_range:  str = "All Time",
) -> pd.DataFrame:
    """Receipt count per vendor, sorted by frequency."""
    start, end = resolve_date_range(date_range)
    df = apply_date_filter(df_receipts.copy(), "receipt_date", start, end)
    if df.empty:
        return pd.DataFrame()

    freq = (
        df.groupby("vendor_name", as_index=False)
        .agg(receipt_count=("receipt_id", "count"),
             total_spend=("receipt_total", "sum"),
             avg_spend=("receipt_total", "mean"))
        .sort_values("receipt_count", ascending=False)
    )
    freq["total_spend"] = freq["total_spend"].round(2)
    freq["avg_spend"]   = freq["avg_spend"].round(2)
    return freq


def get_vendor_trend(
    df_receipts: pd.DataFrame,
    vendors:     list[str] | None = None,
    date_range:  str = "Last 90 Days",
) -> pd.DataFrame:
    """Monthly spend per vendor. Long-format DataFrame: month, vendor, spend."""
    start, end = resolve_date_range(date_range)
    df = apply_date_filter(df_receipts.copy(), "receipt_date", start, end)

    if vendors:
        df = df[df["vendor_name"].isin(vendors)]
    if df.empty:
        return pd.DataFrame(columns=["month", "vendor_name", "spend"])

    df["month"] = df["receipt_date"].dt.to_period("M").astype(str)
    trend = (
        df.groupby(["month", "vendor_name"], as_index=False)["receipt_total"]
        .sum()
        .rename(columns={"receipt_total": "spend"})
        .sort_values("month")
    )
    return trend


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ITEMS
# ══════════════════════════════════════════════════════════════════════════════

def get_top_items(
    df_joined:  pd.DataFrame,
    metric:     str = "spend",      # 'spend' or 'quantity'
    date_range: str = "All Time",
    n:          int = 20,
) -> pd.DataFrame:
    """Top N items by spend or quantity."""
    start, end = resolve_date_range(date_range)
    df = apply_date_filter(df_joined.copy(), "receipt_date", start, end)
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby("matched_canonical_item", as_index=False)
        .agg(
            spend=("line_total", "sum"),
            quantity=("quantity", "sum"),
            frequency=("receipt_id", "nunique"),
            category=("category", "first"),
        )
    )
    sort_col = "spend" if metric == "spend" else "quantity"
    agg = agg.sort_values(sort_col, ascending=False).head(n)
    agg["spend"]    = agg["spend"].round(2)
    agg["quantity"] = agg["quantity"].round(1)
    return agg


def get_item_trend(
    df_joined: pd.DataFrame,
    item:      str,
) -> pd.DataFrame:
    """Monthly spend and quantity for a single canonical item."""
    df = df_joined[df_joined["matched_canonical_item"] == item].copy()
    if df.empty:
        return pd.DataFrame(columns=["month", "spend", "quantity"])

    df["month"] = df["receipt_date"].dt.to_period("M").astype(str)
    trend = (
        df.groupby("month", as_index=False)
        .agg(spend=("line_total", "sum"), quantity=("quantity", "sum"))
        .sort_values("month")
    )
    trend["spend"]    = trend["spend"].round(2)
    trend["quantity"] = trend["quantity"].round(1)
    return trend


def get_raw_vs_normalized(
    df_joined:  pd.DataFrame,
    search:     str = "",
    n:          int = 50,
) -> pd.DataFrame:
    """Side-by-side raw OCR text vs normalized text with confidence scores."""
    df = df_joined.copy()
    if search:
        mask = (
            df["raw_item_text"].str.contains(search, case=False, na=False) |
            df["normalized_item_text"].str.contains(search, case=False, na=False)
        )
        df = df[mask]

    cols = ["raw_item_text", "normalized_item_text", "matched_canonical_item",
            "category", "category_confidence", "vendor_name", "receipt_date"]
    available = [c for c in cols if c in df.columns]
    return df[available].drop_duplicates(subset=["raw_item_text"]).head(n)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RECEIPT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def get_filtered_receipts(
    df_receipts: pd.DataFrame,
    vendor:      str | None = None,
    date_range:  str        = "All Time",
    min_conf:    float      = 0.0,
) -> pd.DataFrame:
    """Filterable receipt table."""
    start, end = resolve_date_range(date_range)
    df = apply_date_filter(df_receipts.copy(), "receipt_date", start, end)

    if vendor and vendor != "All":
        df = df[df["vendor_name"] == vendor]
    if min_conf > 0:
        df = df[df["extraction_confidence"] >= min_conf]

    return df.sort_values("receipt_date", ascending=False)


def get_receipt_items(
    df_joined:  pd.DataFrame,
    receipt_id: str,
) -> pd.DataFrame:
    """All line items for a single receipt."""
    cols = ["raw_item_text", "normalized_item_text", "matched_canonical_item",
            "category", "quantity", "unit", "unit_price", "line_total",
            "category_confidence"]
    df = df_joined[df_joined["receipt_id"] == receipt_id].copy()
    available = [c for c in cols if c in df.columns]
    return df[available].reset_index(drop=True)


def detect_duplicates(df_receipts: pd.DataFrame) -> pd.DataFrame:
    """
    Find receipts that share the same vendor + date with total within 5%.
    Returns pairs as a DataFrame.
    """
    if df_receipts.empty:
        return pd.DataFrame()

    df = df_receipts.copy()
    df["receipt_date_str"] = df["receipt_date"].dt.strftime("%Y-%m-%d")

    groups = df.groupby(["vendor_name", "receipt_date_str"])
    pairs  = []
    for (vendor, d), grp in groups:
        if len(grp) < 2:
            continue
        rows = grp.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                t1, t2 = rows[i]["receipt_total"], rows[j]["receipt_total"]
                if t1 > 0 and abs(t1 - t2) / t1 <= 0.05:
                    pairs.append({
                        "receipt_id_1": rows[i]["receipt_id"],
                        "receipt_id_2": rows[j]["receipt_id"],
                        "vendor_name":  vendor,
                        "date":         d,
                        "total_1":      t1,
                        "total_2":      t2,
                    })
    return pd.DataFrame(pairs)


def get_high_spend_alerts(
    df_receipts: pd.DataFrame,
    sigma:       float = 1.5,
) -> pd.DataFrame:
    """Receipts whose total exceeds mean + sigma * std."""
    if df_receipts.empty:
        return pd.DataFrame()
    mean      = df_receipts["receipt_total"].mean()
    std       = df_receipts["receipt_total"].std()
    threshold = mean + sigma * std
    df = df_receipts[df_receipts["receipt_total"] > threshold].copy()
    df["threshold"] = round(threshold, 2)
    df["excess"]    = (df["receipt_total"] - threshold).round(2)
    return df.sort_values("receipt_total", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — HUMAN VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def get_flagged_items(
    df_joined: pd.DataFrame,
    validated_ids: set | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split line items into two validation queues:

    • urgent   — category_confidence < 0.60  (model was uncertain, needs immediate review)
    • review   — 0.60 ≤ category_confidence < 0.75  (medium confidence, LLM may have failed)

    Items already validated (ids in validated_ids) are excluded.
    Returns (df_urgent, df_needs_review).
    """
    if df_joined.empty:
        empty = pd.DataFrame(columns=[
            "id", "receipt_id", "vendor_name", "receipt_date",
            "raw_item_text", "normalized_item_text", "matched_canonical_item",
            "category", "category_confidence",
        ])
        return empty, empty.copy()

    validated_ids = validated_ids or set()

    cols = [
        "id", "receipt_id", "vendor_name", "receipt_date",
        "raw_item_text", "normalized_item_text", "matched_canonical_item",
        "category", "category_confidence",
    ]
    available = [c for c in cols if c in df_joined.columns]
    df = df_joined[available].drop_duplicates(subset=["id"]).copy()

    # Exclude already-validated items
    if validated_ids:
        df = df[~df["id"].isin(validated_ids)]

    conf = df["category_confidence"] if "category_confidence" in df.columns else pd.Series([1.0] * len(df))

    df_urgent = df[df["category_confidence"] < 0.60].sort_values(
        "category_confidence", ascending=True
    ).reset_index(drop=True)

    df_review = df[
        (df["category_confidence"] >= 0.60) & (df["category_confidence"] < 0.75)
    ].sort_values("category_confidence", ascending=True).reset_index(drop=True)

    return df_urgent, df_review


def get_ocr_issues(df_receipts: pd.DataFrame) -> pd.DataFrame:
    """
    Return receipts that have missing or suspicious OCR-extracted fields:
      - vendor_name is null / 'Unknown' / very short (< 3 chars)
      - receipt_date is null
      - receipt_total is 0 or null
      - extraction_confidence < 0.65

    Returns a DataFrame with a 'issues' column listing what's wrong.
    """
    if df_receipts.empty:
        return pd.DataFrame()

    df = df_receipts.copy()
    issue_rows = []

    for _, row in df.iterrows():
        issues = []

        vendor = str(row.get("vendor_name", "") or "")
        if not vendor or vendor.lower() in ("unknown", "nan", "") or len(vendor) < 3:
            issues.append("Missing vendor")

        raw_date = row.get("receipt_date")
        if raw_date is None or (isinstance(raw_date, float) and np.isnan(raw_date)):
            issues.append("Missing date")

        total = row.get("receipt_total", 0) or 0
        if float(total) <= 0:
            issues.append("Missing/zero total")

        conf = row.get("extraction_confidence", 1.0) or 1.0
        if float(conf) < 0.65:
            issues.append(f"Low OCR confidence ({float(conf):.0%})")

        if issues:
            issue_rows.append({
                "receipt_id":            row.get("receipt_id", ""),
                "vendor_name":           vendor or "—",
                "receipt_date":          row.get("receipt_date"),
                "receipt_total":         total,
                "extraction_confidence": conf,
                "card_used":             row.get("card_used", ""),
                "issues":                ", ".join(issues),
                "issue_count":           len(issues),
            })

    if not issue_rows:
        return pd.DataFrame()

    result = pd.DataFrame(issue_rows).sort_values(
        ["issue_count", "extraction_confidence"], ascending=[False, True]
    ).reset_index(drop=True)
    return result
