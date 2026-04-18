"""
analytics_routes.py — Flask Blueprint for all BillWise dashboard analytics pages.

Mounts the dashboard/ Python backend (analytics.py, charts.py, data_loader.py,
text_to_sql.py, utils.py) into the main Flask app without touching those files.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make dashboard/ modules importable (they use bare imports like `from utils import …`)
_DASHBOARD_DIR = Path(__file__).parent / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import analytics as an          # type: ignore[import]
import charts as ch              # type: ignore[import]
import data_loader as dl         # type: ignore[import]
from text_to_sql import ask_billwise   # type: ignore[import]
from utils import CATEGORY_LIST, DATE_RANGE_OPTIONS  # type: ignore[import]

from flask import Blueprint, jsonify, render_template, request

analytics_bp = Blueprint("analytics", __name__)

# ── 5-minute in-memory data cache ──────────────────────────────────────────────
_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 300.0


def get_data():
    now = time.time()
    if _cache["data"] is None or (now - _cache["ts"]) > _CACHE_TTL:
        _cache["data"] = dl.load_all_data()
        _cache["ts"] = now
    return _cache["data"]


def invalidate_cache():
    """Invalidate after a new bill is saved so the next page load reloads GCS data."""
    _cache["ts"] = 0.0
    _cache["data"] = None


# ── Serialisation helpers ───────────────────────────────────────────────────────

def _fig_json(fig) -> str:
    return fig.to_json()


def _df_records(df) -> list:
    if df is None or df.empty:
        return []
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    # Convert any remaining non-serialisable types
    return json.loads(df.to_json(orient="records", date_format="iso"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@analytics_bp.route("/overview")
def overview():
    df_r, df_i, df_j = get_data()
    date_range = request.args.get("date_range", "All Time")

    kpis        = an.get_kpis(df_r, df_j, date_range)
    trend_df    = an.get_spend_trend(df_j, "Last 90 Days")
    recent_df   = an.get_recent_receipts(df_r)
    anomaly_df  = an.get_anomalies(df_r)

    return render_template(
        "overview.html",
        kpis=kpis,
        spend_trend_json=_fig_json(ch.spend_trend_chart(trend_df)),
        recent=_df_records(recent_df),
        anomalies=_df_records(anomaly_df),
        date_range=date_range,
        date_range_options=DATE_RANGE_OPTIONS,
        active="overview",
    )


@analytics_bp.route("/categories")
def categories():
    df_r, df_i, df_j = get_data()
    date_range  = request.args.get("date_range", "All Time")
    metric      = request.args.get("metric", "spend")

    cat_df      = an.get_category_spend(df_j, date_range)
    month_df    = an.get_category_by_month(df_j, date_range)
    all_cats    = sorted(df_j["category"].dropna().unique().tolist()) if not df_j.empty else []

    return render_template(
        "categories.html",
        donut_json=_fig_json(ch.category_donut_chart(cat_df)),
        bar_json=_fig_json(ch.category_bar_chart(cat_df, metric)),
        heatmap_json=_fig_json(ch.category_heatmap(month_df)),
        line_json=_fig_json(ch.category_line_chart(month_df)),
        cat_table=_df_records(cat_df),
        all_cats=all_cats,
        date_range=date_range,
        metric=metric,
        date_range_options=DATE_RANGE_OPTIONS,
        active="categories",
    )


@analytics_bp.route("/vendors")
def vendors():
    df_r, df_i, df_j = get_data()
    date_range = request.args.get("date_range", "All Time")
    metric     = request.args.get("metric", "spend")

    top_df   = an.get_top_vendors(df_r, date_range)
    freq_df  = an.get_vendor_frequency(df_r, date_range)
    all_vendors = sorted(df_r["vendor_name"].dropna().unique().tolist()) if not df_r.empty else []

    return render_template(
        "vendors.html",
        bar_json=_fig_json(ch.vendor_bar_chart(top_df, metric)),
        freq_table=_df_records(freq_df),
        top_table=_df_records(top_df),
        all_vendors=all_vendors,
        date_range=date_range,
        metric=metric,
        date_range_options=DATE_RANGE_OPTIONS,
        active="vendors",
    )


@analytics_bp.route("/items")
def items():
    df_r, df_i, df_j = get_data()
    date_range = request.args.get("date_range", "All Time")
    metric     = request.args.get("metric", "spend")

    top_df   = an.get_top_items(df_j, metric, date_range)
    raw_df   = an.get_raw_vs_normalized(df_j)
    all_items = sorted(df_j["matched_canonical_item"].dropna().unique().tolist()) if not df_j.empty else []

    return render_template(
        "items.html",
        bar_json=_fig_json(ch.item_bar_chart(top_df, metric)),
        top_table=_df_records(top_df),
        raw_table=_df_records(raw_df),
        all_items=all_items,
        date_range=date_range,
        metric=metric,
        date_range_options=DATE_RANGE_OPTIONS,
        active="items",
    )


@analytics_bp.route("/receipts")
def receipts():
    df_r, df_i, df_j = get_data()
    date_range = request.args.get("date_range", "All Time")
    vendor     = request.args.get("vendor", "All")
    min_conf   = float(request.args.get("min_conf", "0"))

    filtered_df   = an.get_filtered_receipts(df_r, vendor if vendor != "All" else None, date_range, min_conf)
    dup_df        = an.detect_duplicates(df_r)
    high_spend_df = an.get_high_spend_alerts(df_r)
    all_vendors   = ["All"] + sorted(df_r["vendor_name"].dropna().unique().tolist()) if not df_r.empty else ["All"]

    return render_template(
        "receipts.html",
        dist_json=_fig_json(ch.spend_distribution_chart(filtered_df)),
        receipts_table=_df_records(filtered_df),
        duplicates=_df_records(dup_df),
        high_spend=_df_records(high_spend_df),
        all_vendors=all_vendors,
        date_range=date_range,
        vendor=vendor,
        min_conf=min_conf,
        date_range_options=DATE_RANGE_OPTIONS,
        active="receipts",
    )


@analytics_bp.route("/validation")
def validation():
    df_r, df_i, df_j = get_data()
    validated_ids = dl.load_validated_item_ids()
    df_urgent, df_review = an.get_flagged_items(df_j, validated_ids)
    df_ocr   = an.get_ocr_issues(df_r)
    df_val_log = dl.load_all_validations()
    df_ocr_log = dl.load_all_ocr_corrections()

    from utils import get_top3_predictions

    def _with_preds(df):
        records = _df_records(df)
        for row in records:
            row["top3"] = get_top3_predictions(row.get("raw_item_text", ""))
        return records

    urgent_records = _with_preds(df_urgent)
    review_records = _with_preds(df_review)

    return render_template(
        "validation.html",
        urgent=urgent_records,
        review=review_records,
        ocr_issues=_df_records(df_ocr),
        val_log=_df_records(df_val_log),
        ocr_log=_df_records(df_ocr_log),
        category_list=CATEGORY_LIST,
        active="validation",
    )


# ══════════════════════════════════════════════════════════════════════════════
# API — Ask BillWise
# ══════════════════════════════════════════════════════════════════════════════

@analytics_bp.route("/api/ask", methods=["POST"])
def api_ask():
    body     = request.get_json(force=True) or {}
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400

    df_r, df_i, df_j = get_data()
    result = ask_billwise(question, df_r, df_i, df_j)

    table_rows = _df_records(result.result_df) if result.result_df is not None else []
    chart_json = result.chart.to_json() if result.chart else None

    return jsonify({
        "answer":        result.answer_text,
        "chart_json":    chart_json,
        "table":         table_rows,
        "sql":           result.sql or "",
        "error":         result.error or None,
        "was_corrected": result.was_corrected,
    })


# ══════════════════════════════════════════════════════════════════════════════
# API — Analytics filter endpoints (AJAX)
# ══════════════════════════════════════════════════════════════════════════════

@analytics_bp.route("/api/analytics/categories", methods=["POST"])
def api_categories():
    body       = request.get_json(force=True) or {}
    date_range = body.get("date_range", "All Time")
    metric     = body.get("metric", "spend")
    cats_filter = body.get("categories", [])

    df_r, df_i, df_j = get_data()
    cat_df   = an.get_category_spend(df_j, date_range, cats_filter or None)
    month_df = an.get_category_by_month(df_j, date_range)

    return jsonify({
        "donut_json":   _fig_json(ch.category_donut_chart(cat_df)),
        "bar_json":     _fig_json(ch.category_bar_chart(cat_df, metric)),
        "heatmap_json": _fig_json(ch.category_heatmap(month_df)),
        "line_json":    _fig_json(ch.category_line_chart(month_df)),
        "table":        _df_records(cat_df),
    })


@analytics_bp.route("/api/analytics/vendors", methods=["POST"])
def api_vendors():
    body       = request.get_json(force=True) or {}
    date_range = body.get("date_range", "All Time")
    metric     = body.get("metric", "spend")
    sel_vendors = body.get("vendors", [])

    df_r, df_i, df_j = get_data()
    top_df   = an.get_top_vendors(df_r, date_range)
    freq_df  = an.get_vendor_frequency(df_r, date_range)
    trend_df = an.get_vendor_trend(df_r, sel_vendors or None, date_range) if sel_vendors else None

    resp = {
        "bar_json":   _fig_json(ch.vendor_bar_chart(top_df, metric)),
        "freq_table": _df_records(freq_df),
        "top_table":  _df_records(top_df),
    }
    if trend_df is not None and not trend_df.empty:
        resp["trend_json"] = _fig_json(ch.vendor_trend_chart(trend_df))
    return jsonify(resp)


@analytics_bp.route("/api/analytics/items", methods=["POST"])
def api_items():
    body       = request.get_json(force=True) or {}
    date_range = body.get("date_range", "All Time")
    metric     = body.get("metric", "spend")
    sel_item   = body.get("item", "")

    df_r, df_i, df_j = get_data()
    top_df = an.get_top_items(df_j, metric, date_range)

    resp = {
        "bar_json":  _fig_json(ch.item_bar_chart(top_df, metric)),
        "top_table": _df_records(top_df),
    }
    if sel_item:
        trend_df = an.get_item_trend(df_j, sel_item)
        resp["trend_json"] = _fig_json(ch.item_trend_chart(trend_df, sel_item))
    return jsonify(resp)


@analytics_bp.route("/api/analytics/receipts", methods=["POST"])
def api_receipts():
    body       = request.get_json(force=True) or {}
    date_range = body.get("date_range", "All Time")
    vendor     = body.get("vendor", "All")
    min_conf   = float(body.get("min_conf", 0))

    df_r, df_i, df_j = get_data()
    filtered = an.get_filtered_receipts(df_r, vendor if vendor != "All" else None, date_range, min_conf)
    return jsonify({
        "dist_json":      _fig_json(ch.spend_distribution_chart(filtered)),
        "receipts_table": _df_records(filtered),
    })


@analytics_bp.route("/api/analytics/receipt/<receipt_id>/items")
def api_receipt_items(receipt_id: str):
    df_r, df_i, df_j = get_data()
    items_df = an.get_receipt_items(df_j, receipt_id)
    return jsonify({"items": _df_records(items_df)})


# ══════════════════════════════════════════════════════════════════════════════
# API — Human Validation writes
# ══════════════════════════════════════════════════════════════════════════════

@analytics_bp.route("/api/validation/category", methods=["POST"])
def api_save_category():
    # JS sends: {id, receipt_id, raw_item_text, original_category, category, note}
    body = request.get_json(force=True) or {}
    try:
        dl.save_category_validation(
            item_id            = int(body.get("id", 0)),
            receipt_id         = str(body.get("receipt_id", "")),
            raw_item_text      = str(body.get("raw_item_text", "")),
            original_category  = str(body.get("original_category", "")),
            validated_category = str(body.get("category", "")),
            validator_note     = str(body.get("note", "")),
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@analytics_bp.route("/api/validation/ocr", methods=["POST"])
def api_save_ocr():
    # JS sends: {receipt_id, original_value, corrected_text, note}
    body = request.get_json(force=True) or {}
    try:
        dl.save_ocr_correction(
            receipt_id      = str(body.get("receipt_id", "")),
            field_name      = "item_text",
            original_value  = str(body.get("original_value", "")),
            corrected_value = str(body.get("corrected_text", "")),
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
