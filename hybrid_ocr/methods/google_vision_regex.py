from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from google.cloud import vision

from evaluation.canonical import CanonicalFields, CanonicalLineItem, CanonicalReceipt
from evaluation.normalize import normalize_amount
from methods.base import BaseExtractionMethod


class GoogleVisionRegexMethod(BaseExtractionMethod):
    name = "google_vision_regex"

    def __init__(self) -> None:
        self.client = vision.ImageAnnotatorClient()

    def extract_text(self, path: str) -> str:
        with open(path, "rb") as img:
            content = img.read()

        image = vision.Image(content=content)
        response = self.client.text_detection(image=image)

        if response.text_annotations:
            return response.text_annotations[0].description
        return ""

    def extract_date(self, text: str) -> str:
        text = text.replace("|", " ").replace(",", " ")
        text = re.sub(r"\s+", " ", text)

        patterns = [
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            r"\b\d{1,2}/\d{1,2}/\d{2}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}-\d{1,2}-\d{4}\b",
            r"\b\d{1,2}-\d{1,2}-\d{2}\b",
            r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\s+\d{1,2}\s+\d{4}\b",
            r"\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\s+\d{4}\b",
            r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*/\d{1,2}/\d{4}\b",
        ]

        found = []
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                found.append(m.group())

        for raw in found:
            raw_clean = raw.strip()
            for fmt in [
                "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d",
                "%m-%d-%Y", "%m-%d-%y", "%b %d %Y",
                "%B %d %Y", "%d %b %Y", "%d %B %Y",
                "%b/%d/%Y", "%B/%d/%Y",
            ]:
                try:
                    dt = datetime.strptime(raw_clean.title(), fmt)
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
        return ""

    def extract_card_last4(self, text: str):
        lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
        card_keywords = ["VISA", "MASTERCARD", "CARD", "DEBIT", "CREDIT", "REFERENCE#"]

        for line in lines:
            if any(k in line for k in card_keywords):
                match = re.search(r"\d{4}", line)
                if match:
                    return match.group()
            match = re.search(r"[X\*]{4,}\d{4}", line)
            if match:
                return match.group()[-4:]
        return "cash"

    def detect_store(self, lines):
        for line in lines[:10]:
            clean = line.strip()
            if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", clean):
                continue
            if re.search(r"\d+\.\d{2}", clean):
                continue
            if len(clean) > 3:
                return clean
        return "UNKNOWN"

    def extract_items(self, text: str):
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        skip_keywords = [
            "TOTAL", "SUBTOTAL", "TAX", "BALANCE", "CHANGE", "CASH",
            "CREDIT", "DEBIT", "VISA", "MASTERCARD", "CARD", "DUE",
            "AMOUNT", "PAYMENT", "THANK", "SAVE", "DISCOUNT", "COUPON",
            "MEMBER", "REWARD", "POINT", "RECEIPT", "STORE", "TEL",
            "ADDRESS", "PHONE", "WWW", "HTTP", "APPROVED", "AUTH"
        ]

        money_pattern = r"\d[\d,]*\.\d{2}"
        items = []

        for line in lines:
            upper = line.upper()

            if not re.search(money_pattern, line):
                continue

            if any(k in upper for k in skip_keywords):
                continue

            name = re.sub(money_pattern, "", line).strip(" .-@#*/\\")
            if len(name) < 2:
                continue

            items.append(name)

        return "; ".join(items) if items else "Not found"

    def extract_total_from_text(self, text: str):
        lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
        money_pattern = r"\d[\d,]*\.\d{2}"
        amounts = []
        keyword_amounts = []

        for line in lines:
            clean_line = line.replace(",", "")
            matches = re.findall(money_pattern, clean_line)
            for m in matches:
                value = float(m)
                if 0 < value < 20000:
                    amounts.append(value)
                    if any(k in line for k in ["TOTAL", "AMOUNT", "BALANCE", "DUE"]):
                        keyword_amounts.append(value)

        if keyword_amounts:
            return max(keyword_amounts)
        if amounts:
            return max(amounts)
        return ""

    def process_image(self, path: str):
        text = self.extract_text(path)
        lines = text.split("\n")
        return {
            "store": self.detect_store(lines),
            "date": self.extract_date(text),
            "total": self.extract_total_from_text(text),
            "card": self.extract_card_last4(text),
            "items": self.extract_items(text),
        }

    def extract(self, image_path: str, receipt_id: str) -> CanonicalReceipt:
        raw = self.process_image(image_path)

        card_raw = raw.get("card")
        payment_method = None
        card_last4 = None

        if isinstance(card_raw, str) and card_raw.lower() == "cash":
            payment_method = "cash"
        elif isinstance(card_raw, str) and re.fullmatch(r"\d{4}", card_raw):
            payment_method = "card"
            card_last4 = card_raw

        items_raw = raw.get("items", "")
        item_names = []
        if items_raw and items_raw != "Not found":
            item_names = [x.strip() for x in items_raw.split(";") if x.strip()]

        items = [
            CanonicalLineItem(
                line_id=i + 1,
                name=name,
                quantity=None,
                unit_price=None,
                item_total=None,
            )
            for i, name in enumerate(item_names)
        ]

        return CanonicalReceipt(
            receipt_id=receipt_id,
            image_file=Path(image_path).name,
            fields=CanonicalFields(
                merchant_name=raw.get("store"),
                date=raw.get("date") or None,
                time=None,
                subtotal=None,
                tax=None,
                total=normalize_amount(raw.get("total")),
                payment_method=payment_method,
                card_last4=card_last4,
                receipt_number=None,
            ),
            items=items,
        )
