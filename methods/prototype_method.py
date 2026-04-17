from __future__ import annotations

import re
from pathlib import Path

from app.pipeline import run_prototype_pipeline
from evaluation.canonical import CanonicalFields, CanonicalLineItem, CanonicalReceipt
from evaluation.normalize import (
    normalize_amount,
    normalize_card_last4,
    normalize_payment_method,
)
from methods.base import BaseExtractionMethod


def _parse_quantity_and_unit_price(text: str | None):
    if not text:
        return None, None

    text = str(text)
    match = re.search(r"(\d+(?:\.\d+)?)\s*@\s*\$?(\d+\.\d{2})", text)
    if match:
        return float(match.group(1)), float(match.group(2))

    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1)), None

    return None, None


class PrototypeMethod(BaseExtractionMethod):
    name = "prototype_layoutlm"

    def extract(self, image_path: str, receipt_id: str) -> CanonicalReceipt:
        result = run_prototype_pipeline(image_path)

        field_map = {f.field_name: f.value for f in result.fields}

        items = []
        for idx, item in enumerate(result.items, start=1):
            desc = item.description.value if item.description else None
            qty_text = item.quantity.value if item.quantity else None
            price_text = item.price.value if item.price else None

            quantity, unit_price = _parse_quantity_and_unit_price(qty_text)
            item_total = normalize_amount(price_text)

            items.append(
                CanonicalLineItem(
                    line_id=idx,
                    name=desc,
                    quantity=quantity,
                    unit_price=unit_price,
                    item_total=item_total,
                )
            )

        raw_payment_method = field_map.get("payment_method")
        raw_card_last4 = field_map.get("card_last4")

        payment_method = normalize_payment_method(raw_payment_method)
        card_last4 = normalize_card_last4(raw_card_last4)

        return CanonicalReceipt(
            receipt_id=receipt_id,
            image_file=Path(image_path).name,
            fields=CanonicalFields(
                merchant_name=field_map.get("merchant_name"),
                date=field_map.get("date"),
                time=field_map.get("time"),
                subtotal=normalize_amount(field_map.get("subtotal")),
                tax=normalize_amount(field_map.get("tax")),
                total=normalize_amount(field_map.get("total")),
                payment_method=payment_method,
                card_last4=card_last4,
                receipt_number=field_map.get("receipt_number"),
            ),
            items=items,
        )
