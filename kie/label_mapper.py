from __future__ import annotations

from typing import Dict


CANONICAL_FIELD_MAP: Dict[str, str] = {
    "Store_name_value": "merchant_name",
    "Store_addr_value": "store_address",
    "Tel_value": "phone_number",
    "Date_value": "date",
    "Time_value": "time",
    "Prod_item_value": "item_description",
    "Prod_quantity_value": "item_quantity",
    "Prod_price_value": "item_price",
    "Subtotal_value": "subtotal",
    "Tax_value": "tax",
    "Tips_value": "tips",
    "Total_value": "total",
}


def to_canonical_field(label: str) -> str:
    return CANONICAL_FIELD_MAP.get(label, label)


def is_value_label(label: str) -> bool:
    return label.endswith("_value")


def is_key_label(label: str) -> bool:
    return label.endswith("_key")


def is_other_label(label: str) -> bool:
    return label.lower() in {"o", "other", "others", "ignore"}
