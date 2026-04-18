"""
app.py — BillWise Analytics Dashboard (Streamlit)

7 pages:
  1. Overview          — KPIs, spend trend, recent receipts, anomaly alerts
  2. Categories        — spend/quantity by category, monthly heatmap, table
  3. Vendors           — top vendors, frequency, trends, searchable table
  4. Items             — top items, monthly trend, raw vs normalized comparison
  5. Receipt Explorer  — filterable table, line-item drilldown, duplicate alerts
  6. Ask BillWise      — natural language query interface (intent JSON → pandas)
  7. Human Validation  — two-tier review queue for low/medium confidence items + OCR fixes

Run: streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

# Load .env file so GEMINI_API_KEY is available without manual export
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="BillWise Analytics",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ──────────────────────────────────────────────────────────────
from analytics import (
    get_anomalies,
    get_category_by_month,
    get_category_spend,
    detect_duplicates,
    get_filtered_receipts,
    get_flagged_items,
    get_high_spend_alerts,
    get_item_trend,
    get_kpis,
    get_ocr_issues,
    get_raw_vs_normalized,
    get_receipt_items,
    get_recent_receipts,
    get_spend_trend,
    get_top_items,
    get_top_vendors,
    get_vendor_frequency,
    get_vendor_trend,
)
from charts import (
    category_bar_chart,
    category_donut_chart,
    category_heatmap,
    category_line_chart,
    item_bar_chart,
    item_trend_chart,
    spend_distribution_chart,
    spend_trend_chart,
    vendor_bar_chart,
    vendor_trend_chart,
)
from data_loader import (
    get_data_source_label,
    load_all_data,
    load_all_ocr_corrections,
    load_all_validations,
    load_validated_item_ids,
    save_category_validation,
    save_ocr_correction,
)
from text_to_sql import ask_billwise
from utils import (
    CATEGORY_LIST,
    DATE_RANGE_OPTIONS,
    confidence_label,
    fmt_currency,
    fmt_number,
    get_top3_predictions,
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] { background: #1c1917; }
[data-testid="stSidebar"] * { color: #fafaf9 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* KPI cards — force light background with dark text regardless of theme */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e8e5e0 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetric"] * { color: #1c1917 !important; }
[data-testid="stMetricLabel"] p { font-size: 0.78rem !important; color: #78716c !important; }
[data-testid="stMetricValue"]   { font-size: 1.6rem  !important; font-weight: 700 !important; color: #1c1917 !important; }
[data-testid="stMetricDelta"]   { font-size: 0.82rem !important; }
/* Keep delta colours readable */
[data-testid="stMetricDeltaIcon-Up"]   { color: #059669 !important; }
[data-testid="stMetricDeltaIcon-Down"] { color: #dc2626 !important; }

/* Answer card */
.answer-card {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 1.05rem;
    line-height: 1.6;
    color: #1c1917 !important;
}
.answer-card * { color: #1c1917 !important; }
.explanation-card {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 12px 16px;
    font-style: italic;
    color: #78716c;
    margin-top: 8px;
}
.filters-pill {
    display: inline-block;
    background: #f5f3ef;
    border: 1px solid #e8e5e0;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 2px;
    color: #1c1917;
}
.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #78716c;
    margin: 18px 0 8px 0;
}
/* Alert badges */
.alert-high  { color: #dc2626; font-weight: 600; }
.alert-warn  { color: #f59e0b; font-weight: 600; }
.alert-ok    { color: #059669; font-weight: 600; }

/* Human Validation item cards */
.val-card {
    background: #ffffff;
    border: 1px solid #e8e5e0;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.val-card-urgent {
    background: #fff7f7;
    border: 1.5px solid #fca5a5;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.val-card-review {
    background: #fffbeb;
    border: 1.5px solid #fde68a;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.conf-bar-wrap {
    background: #f3f4f6;
    border-radius: 6px;
    height: 8px;
    margin-top: 4px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 6px;
    background: #d97706;
}
.badge-urgent { background:#fee2e2; color:#b91c1c; padding:2px 8px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-review { background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-ok     { background:#d1fae5; color:#065f46; padding:2px 8px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.ocr-issue-tag { background:#ede9fe; color:#5b21b6; padding:2px 8px; border-radius:12px; font-size:0.78rem; margin:2px; display:inline-block; }
</style>
""", unsafe_allow_html=True)


# ── Load data (cached) ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _data():
    return load_all_data()


def get_data():
    return _data()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧾 BillWise")
    st.caption("Receipt Analytics Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🏷️ Categories", "🏪 Vendors",
         "📦 Items", "🔍 Receipt Explorer", "💬 Ask BillWise",
         "🚨 Human Validation"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<p class="section-header">Global Date Filter</p>', unsafe_allow_html=True)
    global_date = st.selectbox("Date Range", DATE_RANGE_OPTIONS,
                               index=6, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p class="section-header">Data Source</p>', unsafe_allow_html=True)
    st.caption(get_data_source_label())

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    gemini_key = os.environ.get("GEMINI_API_KEY", "")

# ── Load data ──────────────────────────────────────────────────────────────────
df_receipts, df_items, df_joined = get_data()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("📊 Overview")
    st.caption(f"Showing: **{global_date}**")

    # ── KPI cards ──────────────────────────────────────────────────────────────
    kpis = get_kpis(df_receipts, df_joined, global_date)
    c1, c2, c3, c4, c5 = st.columns(5)

    delta_str = (
        f"{kpis['spend_delta_pct']:+.1f}% vs prev. period"
        if kpis["spend_delta_pct"] is not None else None
    )
    c1.metric("💰 Total Spend",    fmt_currency(kpis["total_spend"]),   delta_str)
    c2.metric("🧾 Receipts",       fmt_number(kpis["total_receipts"]))
    c3.metric("🏪 Vendors",        fmt_number(kpis["total_vendors"]))
    c4.metric("📦 Line Items",     fmt_number(kpis["total_items"]))
    c5.metric("📈 Avg Bill Size",  fmt_currency(kpis["avg_bill"]))

    st.markdown("---")

    # ── Spend trend ────────────────────────────────────────────────────────────
    trend_df = get_spend_trend(df_joined, global_date, freq="W")
    st.plotly_chart(spend_trend_chart(trend_df, "Weekly Spend Trend"),
                    use_container_width=True, key="chart_spend_trend")

    st.markdown("---")

    col_recent, col_alerts = st.columns([0.6, 0.4])

    with col_recent:
        st.markdown("#### 🕐 Recent Receipts")
        recent = get_recent_receipts(df_receipts, n=8)
        if not recent.empty:
            display = recent.copy()
            display["receipt_date"] = display["receipt_date"].dt.strftime("%Y-%m-%d")
            display["receipt_total"] = display["receipt_total"].apply(fmt_currency)
            if "extraction_confidence" in display.columns:
                display["confidence"] = display["extraction_confidence"].apply(confidence_label)
                display = display.drop(columns=["extraction_confidence"], errors="ignore")
            st.dataframe(
                display[["receipt_id", "vendor_name", "receipt_date",
                          "receipt_total", "confidence"]].rename(columns={
                    "receipt_id": "ID", "vendor_name": "Vendor",
                    "receipt_date": "Date", "receipt_total": "Total",
                }),
                hide_index=True, use_container_width=True,
            )
        else:
            st.info("No receipts found.")

    with col_alerts:
        st.markdown("#### ⚠️ Anomaly Alerts")
        anomalies = get_anomalies(df_receipts)
        if anomalies.empty:
            st.success("✅ No anomalies detected.")
        else:
            for _, row in anomalies.head(6).iterrows():
                alert_class = "alert-high" if "High" in str(row.get("alert_type","")) else "alert-warn"
                st.markdown(
                    f'<span class="{alert_class}">{row.get("alert_type","⚠️")}</span> '
                    f'**{row["vendor_name"]}** — {fmt_currency(row["receipt_total"])} '
                    f'({str(row["receipt_date"])[:10]})',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏷️ Categories":
    st.title("🏷️ Category Analytics")

    # Filters
    col_f1, col_f2 = st.columns([0.4, 0.6])
    with col_f1:
        cat_date = st.selectbox("Date Range", DATE_RANGE_OPTIONS,
                                index=DATE_RANGE_OPTIONS.index(global_date))
    with col_f2:
        cat_filter = st.multiselect("Filter Categories", CATEGORY_LIST,
                                    placeholder="All categories")

    cat_spend = get_category_spend(df_joined, cat_date,
                                   cat_filter if cat_filter else None)

    # Charts row 1
    col_pie, col_bar = st.columns([0.45, 0.55])
    with col_pie:
        st.plotly_chart(category_donut_chart(cat_spend), key="chart_cat_donut",
                        use_container_width=True)
    with col_bar:
        metric_toggle = st.radio("Metric", ["spend", "quantity"],
                                 horizontal=True, label_visibility="collapsed")
        st.plotly_chart(category_bar_chart(cat_spend, metric_toggle), key="chart_cat_bar",
                        use_container_width=True)

    # Heatmap
    st.markdown("---")
    pivot_data = get_category_by_month(df_joined, cat_date)
    tab_heat, tab_line = st.tabs(["🟧 Heatmap", "📈 Line Chart"])
    with tab_heat:
        st.plotly_chart(category_heatmap(pivot_data), use_container_width=True, key="chart_cat_heatmap")
    with tab_line:
        st.plotly_chart(category_line_chart(pivot_data), use_container_width=True, key="chart_cat_line")

    # Table
    st.markdown("---")
    st.markdown("#### Category Summary Table")
    if not cat_spend.empty:
        display = cat_spend.copy()
        display["spend"]    = display["spend"].apply(fmt_currency)
        display["quantity"] = display["quantity"].apply(lambda x: f"{x:,.1f}")
        display["pct"]      = display["pct"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(
            display.rename(columns={
                "category": "Category", "spend": "Total Spend",
                "quantity": "Quantity", "pct": "% of Total",
            }),
            hide_index=True, use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VENDORS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏪 Vendors":
    st.title("🏪 Vendor Analytics")

    vend_date = st.selectbox("Date Range", DATE_RANGE_OPTIONS,
                             index=DATE_RANGE_OPTIONS.index(global_date),
                             key="vend_date")

    top_vendors = get_top_vendors(df_receipts, vend_date, n=10)
    vend_freq   = get_vendor_frequency(df_receipts, vend_date)

    # Top chart
    v_metric = st.radio("Rank by", ["spend", "receipts"], horizontal=True,
                        label_visibility="collapsed")
    st.plotly_chart(vendor_bar_chart(top_vendors, v_metric), use_container_width=True, key="chart_vendor_bar")

    st.markdown("---")
    col_freq, col_avg = st.columns(2)

    with col_freq:
        st.markdown("#### Visit Frequency")
        if not vend_freq.empty:
            freq_display = vend_freq[["vendor_name", "receipt_count", "total_spend"]].copy()
            freq_display["total_spend"] = freq_display["total_spend"].apply(fmt_currency)
            st.dataframe(
                freq_display.rename(columns={
                    "vendor_name": "Vendor",
                    "receipt_count": "Receipts",
                    "total_spend": "Total Spend",
                }),
                hide_index=True, use_container_width=True,
            )

    with col_avg:
        st.markdown("#### Avg Spend per Visit")
        if not vend_freq.empty:
            avg_display = vend_freq[["vendor_name", "avg_spend", "receipt_count"]].copy()
            avg_display["avg_spend"] = avg_display["avg_spend"].apply(fmt_currency)
            avg_display = avg_display.sort_values("avg_spend", ascending=False)
            st.dataframe(
                avg_display.rename(columns={
                    "vendor_name": "Vendor",
                    "avg_spend": "Avg / Visit",
                    "receipt_count": "# Receipts",
                }),
                hide_index=True, use_container_width=True,
            )

    # Trend
    st.markdown("---")
    st.markdown("#### Vendor Spend Trend")
    all_vendors = sorted(df_receipts["vendor_name"].unique().tolist())
    default_vendors = top_vendors["vendor_name"].head(4).tolist() if not top_vendors.empty else []
    selected_vendors = st.multiselect(
        "Select vendors to compare", all_vendors,
        default=default_vendors, key="vend_trend_select",
    )
    if selected_vendors:
        trend_df = get_vendor_trend(df_receipts, selected_vendors, vend_date)
        st.plotly_chart(vendor_trend_chart(trend_df), use_container_width=True, key="chart_vendor_trend")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ITEMS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📦 Items":
    st.title("📦 Item Analytics")

    col_s, col_m, col_d = st.columns([0.4, 0.3, 0.3])
    with col_s:
        item_search_q = st.text_input("🔍 Search items", placeholder="e.g. mozzarella")
    with col_m:
        item_metric = st.radio("Rank by", ["spend", "quantity"], horizontal=True)
    with col_d:
        item_date = st.selectbox("Date Range", DATE_RANGE_OPTIONS,
                                 index=DATE_RANGE_OPTIONS.index(global_date),
                                 key="item_date")

    top_items = get_top_items(df_joined, item_metric, item_date, n=20)

    # Filter by search if provided
    if item_search_q and not top_items.empty:
        mask = top_items["matched_canonical_item"].str.contains(
            item_search_q, case=False, na=False)
        top_items = top_items[mask]

    st.plotly_chart(item_bar_chart(top_items, item_metric), use_container_width=True, key="chart_item_bar")

    st.markdown("---")
    col_trend, col_cat = st.columns([0.55, 0.45])

    with col_trend:
        st.markdown("#### Monthly Item Trend")
        all_items = sorted(df_joined["matched_canonical_item"].dropna().unique().tolist())
        sel_item  = st.selectbox("Select item", all_items,
                                 index=0 if all_items else 0)
        if sel_item:
            item_trend_df = get_item_trend(df_joined, sel_item)
            st.plotly_chart(item_trend_chart(item_trend_df, sel_item), key="chart_item_trend",
                            use_container_width=True)

    with col_cat:
        st.markdown("#### Category Breakdown of Items")
        if not top_items.empty:
            cat_counts = (
                top_items.groupby("category")["spend"]
                .sum()
                .reset_index()
                .rename(columns={"spend": "Total Spend"})
                .sort_values("Total Spend", ascending=False)
            )
            cat_counts["Total Spend"] = cat_counts["Total Spend"].apply(fmt_currency)
            st.dataframe(cat_counts, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔬 Raw OCR vs Normalized Comparison")
    raw_norm = get_raw_vs_normalized(df_joined, item_search_q or "")
    if not raw_norm.empty:
        display = raw_norm.copy()
        if "receipt_date" in display.columns:
            display["receipt_date"] = display["receipt_date"].dt.strftime("%Y-%m-%d")
        if "category_confidence" in display.columns:
            display["category_confidence"] = display["category_confidence"].apply(
                lambda x: confidence_label(x))
        st.dataframe(display.rename(columns={
            "raw_item_text":          "Raw OCR Text",
            "normalized_item_text":   "Normalized",
            "matched_canonical_item": "Canonical Item",
            "category":               "Category",
            "category_confidence":    "Confidence",
            "vendor_name":            "Vendor",
            "receipt_date":           "Date",
        }), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RECEIPT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Receipt Explorer":
    st.title("🔍 Receipt Explorer")

    # ── Filters ────────────────────────────────────────────────────────────────
    col_v, col_d2, col_c = st.columns(3)
    with col_v:
        vendors_list = ["All"] + sorted(df_receipts["vendor_name"].unique().tolist())
        sel_vendor   = st.selectbox("Vendor", vendors_list)
    with col_d2:
        exp_date = st.selectbox("Date Range", DATE_RANGE_OPTIONS,
                                index=DATE_RANGE_OPTIONS.index(global_date),
                                key="exp_date")
    with col_c:
        min_conf = st.slider("Min. Confidence", 0.0, 1.0, 0.0, 0.05)

    filtered = get_filtered_receipts(
        df_receipts,
        vendor=None if sel_vendor == "All" else sel_vendor,
        date_range=exp_date,
        min_conf=min_conf,
    )

    st.markdown(f"**{len(filtered)} receipts** match your filters — "
                f"total: **{fmt_currency(filtered['receipt_total'].sum())}**")

    # ── Receipt table ──────────────────────────────────────────────────────────
    if not filtered.empty:
        display_r = filtered.copy()
        display_r["receipt_date"] = display_r["receipt_date"].dt.strftime("%Y-%m-%d")
        display_r["receipt_total"] = display_r["receipt_total"].apply(fmt_currency)
        if "extraction_confidence" in display_r.columns:
            display_r["extraction_confidence"] = display_r["extraction_confidence"].apply(
                confidence_label)
        st.dataframe(
            display_r[[c for c in ["receipt_id", "vendor_name", "receipt_date",
                                    "receipt_total", "card_used",
                                    "extraction_confidence"] if c in display_r.columns]].rename(
                columns={
                    "receipt_id": "ID", "vendor_name": "Vendor",
                    "receipt_date": "Date", "receipt_total": "Total",
                    "card_used": "Card", "extraction_confidence": "Confidence",
                }),
            hide_index=True, use_container_width=True,
        )
        # Spend distribution
        st.plotly_chart(spend_distribution_chart(filtered), use_container_width=True, key="chart_spend_dist")
    else:
        st.info("No receipts match the current filters.")

    # ── Line-item drilldown ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔎 Line-Item Drilldown")
    receipt_ids = filtered["receipt_id"].tolist() if not filtered.empty else []
    if receipt_ids:
        sel_rid = st.selectbox("Select Receipt ID", receipt_ids)
        if sel_rid:
            items_df = get_receipt_items(df_joined, sel_rid)
            vendor   = filtered[filtered["receipt_id"] == sel_rid]["vendor_name"].iloc[0]
            r_date   = filtered[filtered["receipt_id"] == sel_rid]["receipt_date"].iloc[0]
            total    = filtered[filtered["receipt_id"] == sel_rid]["receipt_total"].iloc[0]

            with st.expander(
                f"📋 {sel_rid} — {vendor} | {str(r_date)[:10]} | {fmt_currency(total)}",
                expanded=True,
            ):
                if not items_df.empty:
                    disp = items_df.copy()
                    if "unit_price" in disp.columns:
                        disp["unit_price"] = disp["unit_price"].apply(fmt_currency)
                    if "line_total" in disp.columns:
                        disp["line_total"] = disp["line_total"].apply(fmt_currency)
                    if "category_confidence" in disp.columns:
                        disp["category_confidence"] = disp["category_confidence"].apply(
                            confidence_label)
                    st.dataframe(disp, hide_index=True, use_container_width=True)
                else:
                    st.warning("No line items found for this receipt.")

    # ── Alerts ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    col_dup, col_high = st.columns(2)

    with col_dup:
        st.markdown("#### 🔁 Duplicate Receipts")
        dupes = detect_duplicates(df_receipts)
        if dupes.empty:
            st.success("✅ No duplicates detected.")
        else:
            st.warning(f"⚠️ {len(dupes)} potential duplicate(s) found")
            disp_dup = dupes.copy()
            disp_dup["total_1"] = disp_dup["total_1"].apply(fmt_currency)
            disp_dup["total_2"] = disp_dup["total_2"].apply(fmt_currency)
            st.dataframe(disp_dup.rename(columns={
                "receipt_id_1": "ID 1", "receipt_id_2": "ID 2",
                "vendor_name": "Vendor", "date": "Date",
                "total_1": "Total 1", "total_2": "Total 2",
            }), hide_index=True, use_container_width=True)

    with col_high:
        st.markdown("#### 💰 High-Spend Alerts")
        highs = get_high_spend_alerts(df_receipts)
        if highs.empty:
            st.success("✅ No high-spend outliers.")
        else:
            st.warning(f"⚠️ {len(highs)} high-spend receipt(s)")
            disp_h = highs[["receipt_id", "vendor_name", "receipt_date",
                             "receipt_total", "excess"]].copy()
            disp_h["receipt_date"]  = disp_h["receipt_date"].dt.strftime("%Y-%m-%d")
            disp_h["receipt_total"] = disp_h["receipt_total"].apply(fmt_currency)
            disp_h["excess"]        = disp_h["excess"].apply(
                lambda x: f"+{fmt_currency(x)}")
            st.dataframe(disp_h.rename(columns={
                "receipt_id": "ID", "vendor_name": "Vendor",
                "receipt_date": "Date", "receipt_total": "Total",
                "excess": "Above Avg",
            }), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ASK BILLWISE  (Text-to-SQL)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💬 Ask BillWise":
    st.title("💬 Ask BillWise")
    st.caption(
        "Ask anything about your spending in plain English. "
        "Gemini translates your question into a DuckDB SQL query, "
        "which runs directly against your receipt data."
    )

    # ── Example chips ──────────────────────────────────────────────────────────
    st.markdown("**Try these:**")
    examples = [
        "How much did we spend on dairy last month?",
        "Top 5 vendors by total spend",
        "Which category had the highest spend?",
        "Show all items from SUBZI MANDI",
        "Total spend per category",
        "How many receipts do we have?",
        "Top 10 items by line total",
        "What was the most expensive single item?",
    ]

    cols = st.columns(4)

    if "ask_input_proxy" not in st.session_state:
        st.session_state["ask_input_proxy"] = ""

    for i, ex in enumerate(examples):
        if cols[i % 4].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["ask_input_proxy"] = ex
            st.session_state.pop("ask_input", None)
            st.rerun()

    # ── Question input ─────────────────────────────────────────────────────────
    st.markdown("---")
    question = st.text_input(
        "Your question",
        placeholder="e.g. What did we spend on seafood in the last 30 days?",
        value=st.session_state["ask_input_proxy"],
        key="ask_input",
    )

    col_ask, col_clear = st.columns([0.15, 0.85])
    with col_ask:
        ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col_clear:
        if st.button("✕ Clear", use_container_width=False):
            st.session_state["ask_input_proxy"] = ""
            st.session_state.pop("ask_input", None)
            st.rerun()

    st.session_state["ask_input_proxy"] = question

    # ── Process query ──────────────────────────────────────────────────────────
    if ask_btn and question.strip():
        with st.spinner("Gemini is writing the SQL query…"):
            result = ask_billwise(
                question, df_receipts, df_items, df_joined,
                api_key=gemini_key or None,
            )

        # Answer card
        st.markdown(
            f'<div class="answer-card">📊 {result.answer_text}</div>',
            unsafe_allow_html=True,
        )

        # Show the generated SQL (collapsible)
        if result.sql:
            with st.expander("🔍 View generated SQL", expanded=False):
                st.code(result.sql, language="sql")

        # Error detail
        if result.error:
            with st.expander("⚠️ Error detail", expanded=False):
                st.text(result.error)

        # Chart
        if result.chart is not None:
            st.markdown("---")
            st.plotly_chart(result.chart, use_container_width=True, key="chart_ask_result")

        # Full result table
        if result.result_df is not None and not result.result_df.empty:
            st.markdown("---")
            st.markdown(f"**Query results** — {len(result.result_df)} row(s)")
            display_df = result.result_df.copy()
            for col in display_df.columns:
                if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                    display_df[col] = display_df[col].dt.strftime("%Y-%m-%d")
            st.dataframe(display_df, hide_index=True, use_container_width=True)

    elif ask_btn:
        st.warning("Please enter a question first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — HUMAN VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🚨 Human Validation":
    st.title("🚨 Human Validation Queue")
    st.caption(
        "Review flagged items where the model's confidence was too low to auto-accept. "
        "Your corrections are saved locally and improve future categorization accuracy."
    )

    # ── Summary stats strip ────────────────────────────────────────────────────
    validated_ids = load_validated_item_ids()
    df_urgent, df_review = get_flagged_items(df_joined, validated_ids)
    df_ocr = get_ocr_issues(df_receipts)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🚨 Urgent Review",   len(df_urgent),    help="confidence < 60%")
    m2.metric("⚠️ Needs Review",    len(df_review),    help="confidence 60–75%")
    m3.metric("🧾 OCR Issues",      len(df_ocr),       help="missing or suspicious fields")
    m4.metric("✅ Validated So Far", len(validated_ids), help="items already reviewed")

    st.markdown("---")

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_urgent, tab_review, tab_ocr, tab_log = st.tabs([
        "🚨 Urgent Review",
        "⚠️ Needs Review",
        "🔬 OCR Issues",
        "📋 Validation Log",
    ])

    # ── helper: render one item card ──────────────────────────────────────────
    def _render_item_card(row, card_style: str, tab_prefix: str) -> None:
        """Render a single flagged-item validation card."""
        item_id    = int(row.get("id", 0))
        receipt_id = str(row.get("receipt_id", ""))
        raw_text   = str(row.get("raw_item_text", ""))
        norm_text  = str(row.get("normalized_item_text", raw_text))
        current_cat = str(row.get("category", "Other"))
        conf        = float(row.get("category_confidence", 0.0))
        vendor      = str(row.get("vendor_name", ""))
        r_date      = row.get("receipt_date")
        date_str    = str(r_date)[:10] if r_date is not None else "—"

        badge_class = "badge-urgent" if conf < 0.60 else "badge-review"
        badge_label = f"{conf:.0%} confidence"

        top3 = get_top3_predictions(raw_text)

        with st.container():
            st.markdown(
                f'<div class="{card_style}">'
                f'<b style="font-size:1.05rem;">{norm_text or raw_text}</b>&nbsp;&nbsp;'
                f'<span class="{badge_class}">{badge_label}</span><br>'
                f'<span style="color:#78716c;font-size:0.85rem;">'
                f'Receipt {receipt_id} &middot; {vendor} &middot; {date_str}'
                f'</span><br>'
                f'<span style="color:#78716c;font-size:0.82rem;">Raw OCR: <code>{raw_text}</code></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Top-3 predictions with inline confidence bars
            st.markdown("**Model's top 3 predictions:**")
            for rank, (pred_cat, pred_conf) in enumerate(top3):
                bar_w    = int(pred_conf * 100)
                bar_col  = "#059669" if rank == 0 else ("#d97706" if rank == 1 else "#94a3b8")
                rank_icon = ("🥇" if rank == 0 else "🥈" if rank == 1 else "🥉")
                st.markdown(
                    f'{rank_icon} **{pred_cat}** — {pred_conf:.0%}'
                    f'<div class="conf-bar-wrap">'
                    f'<div class="conf-bar-fill" style="width:{bar_w}%;background:{bar_col};"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Category selector — default to model's top prediction
            default_cat = top3[0][0] if top3 else current_cat
            default_idx = CATEGORY_LIST.index(default_cat) if default_cat in CATEGORY_LIST else 0

            col_sel, col_note, col_btn = st.columns([0.4, 0.4, 0.2])
            with col_sel:
                chosen_cat = st.selectbox(
                    "Correct category",
                    CATEGORY_LIST,
                    index=default_idx,
                    key=f"{tab_prefix}_cat_{item_id}",
                )
            with col_note:
                note = st.text_input(
                    "Note (optional)",
                    placeholder="e.g. 'actually a herb'",
                    key=f"{tab_prefix}_note_{item_id}",
                )
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)   # vertical align
                if st.button("✔ Confirm", key=f"{tab_prefix}_save_{item_id}",
                             type="primary", use_container_width=True):
                    save_category_validation(
                        item_id=item_id,
                        receipt_id=receipt_id,
                        raw_item_text=raw_text,
                        original_category=current_cat,
                        validated_category=chosen_cat,
                        validator_note=note,
                    )
                    st.success(f"✅ Saved: **{raw_text}** → **{chosen_cat}**")
                    st.rerun()

            st.markdown("---")

    # ── TAB: Urgent Review ────────────────────────────────────────────────────
    with tab_urgent:
        st.markdown(
            "Items where the keyword model assigned **< 60% confidence**. "
            "These are the most uncertain classifications and require immediate attention."
        )

        if df_urgent.empty:
            st.success("🎉 No urgent items to review! All items either exceed the 60% confidence "
                       "threshold or have already been validated.")
        else:
            st.info(f"**{len(df_urgent)} items** need urgent review. "
                    f"Showing items sorted by lowest confidence first.")

            # Optional: filter by vendor
            vendors_in_urgent = sorted(df_urgent["vendor_name"].dropna().unique().tolist())
            if len(vendors_in_urgent) > 1:
                sel_v = st.selectbox(
                    "Filter by vendor",
                    ["All vendors"] + vendors_in_urgent,
                    key="urgent_vendor_filter",
                )
                if sel_v != "All vendors":
                    df_urgent = df_urgent[df_urgent["vendor_name"] == sel_v]

            for _, row in df_urgent.iterrows():
                _render_item_card(row, "val-card-urgent", "urg")

    # ── TAB: Needs Review ─────────────────────────────────────────────────────
    with tab_review:
        st.markdown(
            "Items with **60–75% confidence** — the model made a prediction but wasn't "
            "confident enough to auto-accept. These may have passed initial keyword matching "
            "but failed secondary LLM verification."
        )

        if df_review.empty:
            st.success("🎉 No items in the review queue. Everything above 75% confidence or "
                       "already validated.")
        else:
            st.info(f"**{len(df_review)} items** in the review queue. "
                    f"Sorted by lowest confidence first.")

            vendors_in_review = sorted(df_review["vendor_name"].dropna().unique().tolist())
            if len(vendors_in_review) > 1:
                sel_rv = st.selectbox(
                    "Filter by vendor",
                    ["All vendors"] + vendors_in_review,
                    key="review_vendor_filter",
                )
                if sel_rv != "All vendors":
                    df_review = df_review[df_review["vendor_name"] == sel_rv]

            for _, row in df_review.iterrows():
                _render_item_card(row, "val-card-review", "rev")

    # ── TAB: OCR Issues ───────────────────────────────────────────────────────
    with tab_ocr:
        st.markdown(
            "Receipts with **missing or suspicious OCR-extracted fields** — "
            "vendor name not detected, date parse failed, total is zero, "
            "or overall extraction confidence was low."
        )

        if df_ocr.empty:
            st.success("✅ All receipts have complete OCR fields.")
        else:
            st.warning(f"**{len(df_ocr)} receipt(s)** have OCR issues.")

            for _, row in df_ocr.iterrows():
                receipt_id  = str(row.get("receipt_id", ""))
                vendor      = str(row.get("vendor_name", "—"))
                r_date      = row.get("receipt_date")
                date_str    = str(r_date)[:10] if r_date is not None else "—"
                total       = float(row.get("receipt_total", 0))
                conf        = float(row.get("extraction_confidence", 0))
                issues_str  = str(row.get("issues", ""))

                # Issue tags
                issue_tags = "".join(
                    f'<span class="ocr-issue-tag">⚠️ {iss.strip()}</span>'
                    for iss in issues_str.split(",")
                    if iss.strip()
                )

                st.markdown(
                    f'<div class="val-card">'
                    f'<b>Receipt {receipt_id}</b> &nbsp;'
                    f'<span style="color:#78716c;font-size:0.85rem;">'
                    f'{vendor} &middot; {date_str} &middot; {fmt_currency(total)} &middot; '
                    f'OCR conf: {conf:.0%}</span><br>'
                    f'<div style="margin-top:6px;">{issue_tags}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                with st.expander(f"✏️ Correct fields for {receipt_id}", expanded=False):
                    ocr_cols = st.columns(2)

                    # Vendor correction
                    with ocr_cols[0]:
                        new_vendor = st.text_input(
                            "Vendor name",
                            value=vendor if vendor != "—" else "",
                            key=f"ocr_vendor_{receipt_id}",
                        )
                    # Date correction
                    with ocr_cols[1]:
                        new_date = st.text_input(
                            "Invoice date (YYYY-MM-DD)",
                            value=date_str if date_str != "—" else "",
                            key=f"ocr_date_{receipt_id}",
                        )

                    ocr_cols2 = st.columns(2)
                    with ocr_cols2[0]:
                        new_total = st.text_input(
                            "Receipt total ($)",
                            value=str(total) if total > 0 else "",
                            key=f"ocr_total_{receipt_id}",
                        )
                    with ocr_cols2[1]:
                        card_val = str(row.get("card_used", "") or "")
                        new_card = st.text_input(
                            "Card used",
                            value=card_val,
                            key=f"ocr_card_{receipt_id}",
                        )

                    if st.button(f"💾 Save corrections for {receipt_id}",
                                 key=f"ocr_save_{receipt_id}"):
                        saved_any = False
                        field_map = {
                            "vendor_name":    (vendor,   new_vendor),
                            "receipt_date":   (date_str, new_date),
                            "receipt_total":  (str(total), new_total),
                            "card_used":      (card_val, new_card),
                        }
                        for field, (orig, corrected) in field_map.items():
                            if corrected and corrected != orig and corrected != "—":
                                save_ocr_correction(
                                    receipt_id=receipt_id,
                                    field_name=field,
                                    original_value=orig,
                                    corrected_value=corrected,
                                )
                                saved_any = True
                        if saved_any:
                            st.success(f"✅ Corrections saved for receipt {receipt_id}")
                        else:
                            st.info("No changes detected.")

                st.markdown("---")

    # ── TAB: Validation Log ───────────────────────────────────────────────────
    with tab_log:
        st.markdown("#### Category Validation History")
        df_val_log = load_all_validations()
        if df_val_log.empty:
            st.info("No validations recorded yet.")
        else:
            display_log = df_val_log[[c for c in [
                "validated_at", "receipt_id", "raw_item_text",
                "original_category", "validated_category", "validator_note",
            ] if c in df_val_log.columns]].copy()
            st.dataframe(
                display_log.rename(columns={
                    "validated_at":       "Validated At",
                    "receipt_id":         "Receipt",
                    "raw_item_text":      "Raw OCR Text",
                    "original_category":  "Original",
                    "validated_category": "Corrected To",
                    "validator_note":     "Note",
                }),
                hide_index=True, use_container_width=True,
            )

        st.markdown("---")
        st.markdown("#### OCR Correction History")
        df_ocr_log = load_all_ocr_corrections()
        if df_ocr_log.empty:
            st.info("No OCR corrections recorded yet.")
        else:
            display_ocr = df_ocr_log[[c for c in [
                "corrected_at", "receipt_id", "field_name",
                "original_value", "corrected_value",
            ] if c in df_ocr_log.columns]].copy()
            st.dataframe(
                display_ocr.rename(columns={
                    "corrected_at":    "Corrected At",
                    "receipt_id":      "Receipt",
                    "field_name":      "Field",
                    "original_value":  "Original",
                    "corrected_value": "Corrected To",
                }),
                hide_index=True, use_container_width=True,
            )
