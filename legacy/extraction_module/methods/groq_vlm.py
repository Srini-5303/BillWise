from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from groq import Groq
from PIL import Image

from evaluation.canonical import CanonicalFields, CanonicalLineItem, CanonicalReceipt
from evaluation.normalize import (
    normalize_amount,
    normalize_card_last4,
    normalize_date,
    normalize_item_name,
    normalize_payment_method,
)
from methods.base import BaseExtractionMethod


class GroqVLMMethod(BaseExtractionMethod):
    name = "groq_vlm"

    MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
    MAX_BASE64_BYTES = 3_500_000

    def __init__(self) -> None:
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

        self.system_prompt = """You are a receipt data extractor.
Read the receipt image and return a single JSON object.

Rules:
- Return JSON only.
- Use null for missing values.
- Do not guess.
- Extract all visible purchased line items.
- Fields to return:
  store_name, invoice_date, total_amount, subtotal, tax_amount,
  card_last4, payment_method, receipt_number, items

Item fields:
- name
- quantity
- unit_price
- item_total
"""

        self.user_prompt = "Extract all bill details from this receipt image and return the JSON."

    def _compress_image(self, path: str) -> bytes:
        img = Image.open(path).convert("RGB")
        quality = 85

        while True:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            raw = buf.getvalue()
            if len(raw) <= self.MAX_BASE64_BYTES or quality <= 30:
                return raw
            quality -= 10

    def _encode_image(self, path: str):
        file_size = os.path.getsize(path)

        if file_size > self.MAX_BASE64_BYTES:
            raw = self._compress_image(path)
            media_type = "image/jpeg"
        else:
            with open(path, "rb") as f:
                raw = f.read()
            ext = os.path.splitext(path)[-1].lower().lstrip(".")
            media_type = f"image/{ext}" if ext in {"jpeg", "jpg", "png", "webp"} else "image/jpeg"

        return base64.b64encode(raw).decode("utf-8"), media_type

    def _call_groq(self, b64: str, media_type: str) -> str:
        response = self.client.chat.completions.create(
            model=self.MODEL_ID,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"},
                        },
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _parse_loose_json(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Groq response is not a JSON object")
        return data

    def _safe_get_str(self, obj: dict[str, Any], key: str):
        value = obj.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in {"null", "none", "n/a", "na", "unknown", "not found"}:
                return None
            return value
        return str(value)

    def _normalize_card(self, value):
        if value is None:
            return None

        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "n/a", "na", "unknown", "not found"}:
            return None

        # masked placeholders like xxxx, ????, **** with no digits -> null
        if re.fullmatch(r"[xX\*\?]{4,}", text):
            return None

        return normalize_card_last4(text)

    def _normalize_items(self, raw_items):
        if not isinstance(raw_items, list):
            return []

        items = []
        line_id = 1

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            raw_name = item.get("name")
            name = normalize_item_name(raw_name)

            # Drop rows with no usable name
            if not name:
                continue

            quantity = normalize_amount(item.get("quantity"))
            unit_price = normalize_amount(item.get("unit_price"))
            item_total = normalize_amount(item.get("item_total"))

            items.append(
                CanonicalLineItem(
                    line_id=line_id,
                    name=name,
                    quantity=quantity,
                    unit_price=unit_price,
                    item_total=item_total,
                )
            )
            line_id += 1

        return items

    def extract(self, image_path: str, receipt_id: str) -> CanonicalReceipt:
        b64, media_type = self._encode_image(image_path)

        parsed = None
        last_error = None
        for _ in range(2):
            try:
                raw = self._call_groq(b64, media_type)
                parsed = self._parse_loose_json(raw)
                break
            except Exception as exc:
                last_error = exc

        if parsed is None:
            raise RuntimeError(f"Groq extraction failed after 2 attempts: {last_error}")

        raw_store = self._safe_get_str(parsed, "store_name")
        raw_date = self._safe_get_str(parsed, "invoice_date")
        raw_total = parsed.get("total_amount")
        raw_subtotal = parsed.get("subtotal")
        raw_tax = parsed.get("tax_amount")
        raw_payment_method = self._safe_get_str(parsed, "payment_method")
        raw_card_last4 = parsed.get("card_last4")
        raw_receipt_number = self._safe_get_str(parsed, "receipt_number")
        raw_items = parsed.get("items", [])

        # Special case from some model outputs
        if isinstance(raw_card_last4, str) and raw_card_last4.lower() == "cash":
            raw_payment_method = "cash"
            raw_card_last4 = None

        return CanonicalReceipt(
            receipt_id=receipt_id,
            image_file=Path(image_path).name,
            fields=CanonicalFields(
                merchant_name=raw_store,
                date=normalize_date(raw_date),
                time=None,
                subtotal=normalize_amount(raw_subtotal),
                tax=normalize_amount(raw_tax),
                total=normalize_amount(raw_total),
                payment_method=normalize_payment_method(raw_payment_method),
                card_last4=self._normalize_card(raw_card_last4),
                receipt_number=raw_receipt_number,
            ),
            items=self._normalize_items(raw_items),
        )
