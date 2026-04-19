"""
hybrid_ocr_adapter.py — Adapter between hybrid_ocr pipeline and app.py.

The hybrid_ocr package uses `from app.schemas import ...` etc. internally,
which conflicts with the root app.py (Flask entry point). We resolve this by
temporarily removing 'app' from sys.modules before importing hybrid_ocr so
Python picks up hybrid_ocr/app/ as the 'app' package instead.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

log = logging.getLogger("hybrid_ocr_adapter")

_HYBRID_DIR = Path(__file__).parent / "hybrid_ocr"

# Add hybrid_ocr/ to front of sys.path so its sub-packages (app, methods, etc.)
# are found before anything else.
if str(_HYBRID_DIR) not in sys.path:
    sys.path.insert(0, str(_HYBRID_DIR))

# hybrid_ocr internals all do `from app.schemas import ...` etc.
# The root app.py is already registered as 'app' in sys.modules (Flask loaded it).
# Clear it so that when hybrid_ocr modules import 'app', they find
# hybrid_ocr/app/ (a real package with __init__.py) instead of root app.py.
for _key in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
    sys.modules.pop(_key, None)

# Now import hybrid_ocr modules — 'app' will resolve to hybrid_ocr/app/
from methods.hybrid_method import HybridMethod  # type: ignore[import]


def process_image(image_path: str) -> dict:
    """
    Drop-in replacement for ocr_pipeline.process_image().
    Runs PaddleOCR + LayoutLMv3 + Groq VLM hybrid fusion.
    Returns { store, date, total, card, items: [(name, price), ...] }
    """
    receipt_id = Path(image_path).stem
    result = HybridMethod().extract(image_path, receipt_id)

    f = result.fields  # CanonicalFields

    store   = f.merchant_name or ""
    date    = f.date or ""
    total   = str(f.total) if f.total is not None else ""

    payment = f.payment_method or ""
    last4   = f.card_last4 or ""
    if last4:
        card = f"{payment} XXXX{last4}".strip()
    elif payment:
        card = payment
    else:
        card = "cash"

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

    return {"store": store, "date": date, "total": total, "card": card, "items": items}
