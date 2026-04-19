"""
hybrid_ocr_adapter.py — Adapter between hybrid_ocr pipeline and app.py.

Uses HybridMethod (PaddleOCR + LayoutLMv3 + Groq VLM fusion) and maps
CanonicalReceipt to the same dict schema that process_image() returned:
  { store, date, total, card, items: [(name, price), ...] }
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

# Make hybrid_ocr sub-modules importable
_HYBRID_DIR = Path(__file__).parent / "hybrid_ocr"
for _sub in ("app", "ocr", "kie", "extract", "output", "methods", "evaluation"):
    _p = str(_HYBRID_DIR / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_HYBRID_DIR) not in sys.path:
    sys.path.insert(0, str(_HYBRID_DIR))

log = logging.getLogger("hybrid_ocr_adapter")


def process_image(image_path: str) -> dict:
    """
    Drop-in replacement for ocr_pipeline.process_image().
    Runs PaddleOCR + LayoutLMv3 + Groq VLM hybrid fusion.
    Returns { store, date, total, card, items: [(name, price), ...] }
    """
    from methods.hybrid_method import HybridMethod

    receipt_id = Path(image_path).stem
    result = HybridMethod().extract(image_path, receipt_id)

    f = result.fields  # CanonicalFields

    # store
    store = f.merchant_name or ""

    # date — CanonicalFields.date is already normalised to YYYY-MM-DD
    date = f.date or ""

    # total
    total = str(f.total) if f.total is not None else ""

    # card — combine payment method + last 4 digits
    payment = f.payment_method or ""
    last4   = f.card_last4 or ""
    if last4:
        card = f"{payment} XXXX{last4}".strip()
    elif payment:
        card = payment
    else:
        card = "cash"

    # items — CanonicalLineItem has .name and .item_total (or .unit_price)
    items = []
    for item in result.items:
        name  = item.name or ""
        price = str(item.item_total or item.unit_price or "")
        if name:
            items.append((name, price))

    log.info(
        "Hybrid OCR — store=%r date=%r total=%r card=%r items=%d",
        store, date, total, card, len(items),
    )

    return {
        "store": store,
        "date":  date,
        "total": total,
        "card":  card,
        "items": items,
    }
