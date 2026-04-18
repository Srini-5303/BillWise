from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from dateutil import parser


def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    value = re.sub(r"\s+", " ", value)
    return value.upper()


def normalize_item_name(value: Optional[str]) -> Optional[str]:
    text = normalize_text(value)
    if text is None:
        return None
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_amount(value) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return round(float(value), 2)

    text = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return round(float(match.group(0)), 2)
    except ValueError:
        return None


def normalize_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    try:
        return parser.parse(text, fuzzy=True).strftime("%Y-%m-%d")
    except Exception:
        return None


def normalize_time(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None

    text = re.sub(r"\s+", " ", text)

    for fmt in ("%I:%M %p", "%H:%M", "%I:%M%p"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%I:%M %p")
        except ValueError:
            pass

    match = re.search(r"(\d{1,2}:\d{2})\s*(AM|PM)", text)
    if match:
        try:
            dt = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%I:%M %p")
            return dt.strftime("%I:%M %p")
        except ValueError:
            return None

    return None


def normalize_card_last4(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 4:
        return digits
    return None


def normalize_payment_method(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    text = normalize_text(value)
    if text is None:
        return None

    if text in {"CASH"}:
        return "cash"
    if text in {"CARD", "VISA", "MASTERCARD", "AMEX", "DISCOVER", "DEBIT", "CREDIT"}:
        return "card"
    if text in {"APPLE PAY", "GOOGLE PAY", "SAMSUNG PAY"}:
        return "mobile_wallet"
    if text in {"CONTACTLESS", "TAP"}:
        return "contactless"
    if text in {"EBT", "SNAP"}:
        return "ebt"
    if text in {"CHECK"}:
        return "check"
    if text in {"GIFT CARD"}:
        return "gift_card"

    return "other"