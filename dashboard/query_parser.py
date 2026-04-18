"""
query_parser.py — Convert a natural-language question into a structured intent JSON.

Two modes:
  1. Gemini  (when GEMINI_API_KEY is set) — reliable, uses gemini-2.5-flash
  2. Keyword fallback   — always works, no API key needed, covers demo queries

The LLM NEVER writes SQL. It only outputs a structured intent dict.
All arithmetic happens downstream in query_executor.py.
"""
from __future__ import annotations

import json
import os
import re

from utils import CATEGORY_LIST

# ── Intent schema ──────────────────────────────────────────────────────────────
#
# intent_type:   what the user wants
# filters:       what to filter the data by
# group_by:      how to aggregate (None = no grouping, return scalar)
# metric:        what to measure
# limit:         top-N (None = return all)
# order:         "desc" or "asc"
#
INTENT_TYPES = [
    "spend_total",       # total spend with optional filters
    "quantity_total",    # total quantity of an item/category
    "top_vendors",       # rank vendors by spend or frequency
    "top_items",         # rank items by spend or quantity
    "category_breakdown",# spend breakdown by category
    "receipt_lookup",    # find receipts matching criteria
    "item_search",       # find line items containing text
    "compare_periods",   # this month vs last month spend
]

METRICS = [
    "sum_line_total",    # sum of line_total
    "sum_quantity",      # sum of quantity
    "count_receipts",    # count distinct receipts
    "avg_unit_price",    # average unit price
    "count_items",       # count line item rows
]

DATE_RANGES = [
    "today", "this_week", "last_week",
    "this_month", "last_month",
    "last_30_days", "last_90_days", "this_year",
]

_GEMINI_SYSTEM = f"""You are an intent parser for BillWise, a restaurant expense analytics system.
Convert the user's question into a structured JSON intent.

INTENT TYPES: {', '.join(INTENT_TYPES)}

METRICS: {', '.join(METRICS)}

DATE RANGES (use exact strings): {', '.join(DATE_RANGES)}

CATEGORIES (use exact spelling): {', '.join(CATEGORY_LIST)}

Return ONLY valid JSON matching this exact schema — no extra text:
{{
  "intent_type": "<one of the intent types above>",
  "filters": {{
    "category":   "<category name or null>",
    "vendor":     "<vendor name or null>",
    "item_text":  "<item search text or null>",
    "date_range": "<date range string or null>"
  }},
  "group_by": "<vendor_name | category | matched_canonical_item | month | null>",
  "metric":   "<one of the metrics above>",
  "limit":    <integer or null>,
  "order":    "desc"
}}

Rules:
- For "how much ... spend" questions use intent_type=spend_total, metric=sum_line_total
- For "how much ... bought/purchased/used" questions use quantity_total, metric=sum_quantity
- For "top N vendors" use top_vendors, group_by=vendor_name, limit=N
- For "top N items/products" use top_items, group_by=matched_canonical_item, limit=N
- For "show/find receipts" use receipt_lookup
- For "show/find items containing X" use item_search, filters.item_text=X
- For "breakdown by category" use category_breakdown, group_by=category
- For "compare this month vs last month" use compare_periods
- If no limit is specified, use null for limit (except top_vendors/top_items default to 5)
- Always return valid JSON — never explain yourself"""


# ── Gemini parser ──────────────────────────────────────────────────────────────

def _parse_with_gemini(question: str, api_key: str) -> dict | None:
    """Call Gemini to parse intent. Returns dict or None on failure."""
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.05,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
            system_instruction=_GEMINI_SYSTEM,
        )
        response = model.generate_content(question)
        raw      = response.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$",           "", raw)
        return _validate_intent(json.loads(raw))
    except Exception:
        return None


# ── Keyword fallback ───────────────────────────────────────────────────────────

def _keyword_parse(question: str) -> dict:
    """Rule-based intent parser. Covers all common demo questions."""
    q = question.lower().strip()

    # ── intent_type ──────────────────────────────────────────────────────────
    if "compare" in q or ("vs" in q and ("month" in q or "week" in q)):
        intent_type = "compare_periods"
    elif "breakdown" in q or ("by category" in q):
        intent_type = "category_breakdown"
    elif any(w in q for w in ["top", "highest", "most", "best", "largest"]) and "vendor" in q:
        intent_type = "top_vendors"
    elif any(w in q for w in ["top", "highest", "most"]) and any(
            w in q for w in ["item", "product", "bought", "purchased", "ingredient"]):
        intent_type = "top_items"
    elif any(w in q for w in ["show receipt", "find receipt", "list receipt",
                               "which receipt", "receipts from", "receipts containing"]):
        intent_type = "receipt_lookup"
    elif any(w in q for w in ["show", "find", "search", "containing", "contains",
                               "with", "include"]) and any(
            w in q for w in ["item", "product", "ingredient", "cheese",
                              "oil", "meat", "fish", "milk"]):
        intent_type = "item_search"
    elif any(w in q for w in ["how much", "quantity", "how many", "how often",
                               "units", "kg", "liters", "kilos"]):
        # Distinguish spend question from quantity question
        if any(w in q for w in ["spend", "cost", "pay", "paid", "charge", "bill", "price", "dollar"]):
            intent_type = "spend_total"
        else:
            intent_type = "quantity_total"
    elif any(w in q for w in ["spend", "spent", "cost", "total", "amount",
                               "expense", "bill", "invoice", "paid"]):
        intent_type = "spend_total"
    else:
        intent_type = "spend_total"   # safe default

    # ── metric ───────────────────────────────────────────────────────────────
    if intent_type == "quantity_total":
        metric = "sum_quantity"
    elif intent_type in ("top_vendors", "receipt_lookup"):
        metric = "sum_line_total"
    elif "count" in q or "how many receipt" in q:
        metric = "count_receipts"
    else:
        metric = "sum_line_total"

    # ── category filter ───────────────────────────────────────────────────────
    category = None
    for cat in CATEGORY_LIST:
        if cat.lower() in q:
            category = cat
            break
    # Also catch common ingredient-to-category mappings
    _ingredient_map = {
        "oil": "Produce", "olive oil": "Produce", "evoo": "Produce",
        "milk": "Dairy", "butter": "Dairy", "cream": "Dairy",
        "cheese": "Dairy", "mozzarella": "Dairy", "parmesan": "Dairy",
        "chicken": "Meat", "beef": "Meat", "pork": "Meat", "lamb": "Meat",
        "salmon": "Seafood", "shrimp": "Seafood", "fish": "Seafood",
        "bread": "Bakery", "croissant": "Bakery", "sourdough": "Bakery",
        "pasta": "Dry Goods", "flour": "Dry Goods", "rice": "Dry Goods",
        "coffee": "Beverages", "espresso": "Beverages", "wine": "Beverages",
        "soap": "Cleaning Supplies", "sanitizer": "Cleaning Supplies",
    }
    if category is None:
        for keyword, cat in _ingredient_map.items():
            if keyword in q:
                category = cat
                break

    # ── item_text filter ──────────────────────────────────────────────────────
    item_text = None
    if intent_type == "item_search":
        for trigger in ["containing ", "contains ", "with ", "for ", "about ",
                         "called ", "named ", "like "]:
            if trigger in q:
                rest      = q.split(trigger, 1)[1].strip()
                item_text = " ".join(rest.split()[:4]).strip("?.,!")
                break
        if not item_text:
            # Fall back to last few words of question
            words     = q.rstrip("?").split()
            item_text = " ".join(words[-3:]) if len(words) >= 3 else q

    # ── vendor filter ─────────────────────────────────────────────────────────
    vendor = None
    # Cannot reliably extract vendor names from keywords alone;
    # leave for Gemini to handle if needed.

    # ── date_range ────────────────────────────────────────────────────────────
    date_range = None
    if "last month"   in q: date_range = "last_month"
    elif "this month" in q: date_range = "this_month"
    elif "last week"  in q: date_range = "last_week"
    elif "this week"  in q: date_range = "this_week"
    elif "today"      in q: date_range = "today"
    elif "last 30"    in q: date_range = "last_30_days"
    elif "last 90"    in q: date_range = "last_90_days"
    elif "this year"  in q: date_range = "this_year"
    elif "yesterday"  in q: date_range = "today"

    # ── limit & group_by ──────────────────────────────────────────────────────
    limit    = None
    group_by = None

    # Extract "top N"
    m = re.search(r"\btop\s+(\d+)\b", q)
    if m:
        limit = int(m.group(1))
    elif intent_type in ("top_vendors", "top_items") and limit is None:
        limit = 5

    if intent_type == "top_vendors":
        group_by = "vendor_name"
    elif intent_type == "top_items":
        group_by = "matched_canonical_item"
    elif intent_type == "category_breakdown":
        group_by = "category"
    elif intent_type == "spend_total" and "by vendor" in q:
        group_by = "vendor_name"
    elif intent_type == "spend_total" and "by category" in q:
        group_by = "category"
    elif intent_type in ("compare_periods",):
        group_by = "month"

    return {
        "intent_type": intent_type,
        "filters": {
            "category":   category,
            "vendor":     vendor,
            "item_text":  item_text,
            "date_range": date_range,
        },
        "group_by": group_by,
        "metric":   metric,
        "limit":    limit,
        "order":    "desc",
    }


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_intent(raw: dict) -> dict:
    """Ensure intent dict has all required keys with valid values."""
    defaults = {
        "intent_type": "spend_total",
        "filters": {"category": None, "vendor": None,
                    "item_text": None, "date_range": None},
        "group_by": None,
        "metric":   "sum_line_total",
        "limit":    None,
        "order":    "desc",
    }
    defaults.update(raw)
    if "filters" not in defaults or not isinstance(defaults["filters"], dict):
        defaults["filters"] = {"category": None, "vendor": None,
                               "item_text": None, "date_range": None}
    for k in ("category", "vendor", "item_text", "date_range"):
        defaults["filters"].setdefault(k, None)
    if defaults["intent_type"] not in INTENT_TYPES:
        defaults["intent_type"] = "spend_total"
    if defaults["metric"] not in METRICS:
        defaults["metric"] = "sum_line_total"
    return defaults


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_intent(question: str, api_key: str | None = None) -> dict:
    """
    Convert a natural-language question to a structured intent dict.

    Tries Gemini first if api_key is provided.
    Falls back to keyword parser automatically.

    Returns a validated intent dict — never raises.
    """
    if not question or not question.strip():
        return _validate_intent({})

    key = api_key or os.environ.get("GEMINI_API_KEY")

    if key:
        result = _parse_with_gemini(question, key)
        if result:
            return result

    # Keyword fallback — always succeeds
    return _keyword_parse(question)
