"""
hybrid_ocr_adapter.py — Adapter between hybrid_ocr pipeline and app.py.

Maps ReceiptResult (PaddleOCR + LayoutLMv3) to the same dict schema that
the old Google Vision process_image() returned:
  { store, date, total, card, items: [(name, price), ...] }
"""
from __future__ import annotations

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
    Runs PaddleOCR + LayoutLMv3, returns the same dict schema.
    """
    from pipeline import run_prototype_pipeline  # hybrid_ocr/app/pipeline.py

    result = run_prototype_pipeline(image_path)

    # Build a field lookup: field_name → value
    fields = {f.field_name: f.value for f in result.fields}

    # store
    store = fields.get("merchant_name") or fields.get("store_name") or ""

    # date — already normalized to YYYY-MM-DD by the pipeline
    date = fields.get("date") or ""

    # total
    total = fields.get("total") or fields.get("subtotal") or ""

    # card — combine payment method + last 4 digits
    payment = fields.get("payment_method") or ""
    last4   = fields.get("card_last4") or ""
    if last4:
        card = f"{payment} XXXX{last4}".strip()
    elif payment:
        card = payment
    else:
        card = "cash"

    # items — list of (name, price) tuples
    items = []
    for item in result.items:
        name  = item.description.value if item.description else ""
        price = item.price.value       if item.price       else ""
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
