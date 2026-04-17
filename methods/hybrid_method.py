from __future__ import annotations

from pathlib import Path
from typing import Optional

from evaluation.canonical import CanonicalFields, CanonicalReceipt
from evaluation.normalize import (
    normalize_amount,
    normalize_card_last4,
    normalize_date,
    normalize_item_name,
    normalize_payment_method,
    normalize_text,
    normalize_time,
)
from methods.base import BaseExtractionMethod
from methods.groq_vlm import GroqVLMMethod
from methods.prototype_method import PrototypeMethod


def _same_text(a: Optional[str], b: Optional[str]) -> bool:
    return normalize_text(a) == normalize_text(b)


def _same_date(a: Optional[str], b: Optional[str]) -> bool:
    return normalize_date(a) == normalize_date(b)


def _same_time(a: Optional[str], b: Optional[str]) -> bool:
    return normalize_time(a) == normalize_time(b)


def _same_amount(a, b) -> bool:
    na = normalize_amount(a)
    nb = normalize_amount(b)
    if na is None and nb is None:
        return True
    if na is None or nb is None:
        return False
    return abs(na - nb) <= 0.01


def _same_payment(a: Optional[str], b: Optional[str]) -> bool:
    return normalize_payment_method(a) == normalize_payment_method(b)


def _same_card(a: Optional[str], b: Optional[str]) -> bool:
    return normalize_card_last4(a) == normalize_card_last4(b)


def _item_name_set(items):
    names = []
    for item in items:
        norm = normalize_item_name(item.name)
        if norm:
            names.append(norm)
    return set(names)


class HybridMethod(BaseExtractionMethod):
    name = "hybrid_proto_groq"

    def __init__(self) -> None:
        self.prototype = PrototypeMethod()
        self.groq = GroqVLMMethod()

    def extract(self, image_path: str, receipt_id: str) -> CanonicalReceipt:
        proto = None
        groq = None
        proto_error = None
        groq_error = None

        try:
            proto = self.prototype.extract(image_path, receipt_id)
        except Exception as e:
            proto_error = str(e)

        try:
            groq = self.groq.extract(image_path, receipt_id)
        except Exception as e:
            groq_error = str(e)

        if proto is None and groq is None:
            raise RuntimeError(
                f"Both hybrid components failed. prototype_error={proto_error}; groq_error={groq_error}"
            )

        if proto is None:
            result = groq.model_copy(deep=True)
            result.review_required = True
            result.review_reasons = [f"prototype_failed: {proto_error}"]
            result.field_sources = {k: "groq" for k in result.fields.model_dump().keys()}
            return result

        if groq is None:
            result = proto.model_copy(deep=True)
            result.review_required = True
            result.review_reasons = [f"groq_failed: {groq_error}"]
            result.field_sources = {k: "prototype" for k in result.fields.model_dump().keys()}
            return result

        review_reasons = []
        field_sources = {}

        # Merge fields
        merged = CanonicalFields()

        # semantic-first
        merged.merchant_name = groq.fields.merchant_name or proto.fields.merchant_name
        field_sources["merchant_name"] = "groq" if groq.fields.merchant_name else "prototype"

        merged.date = groq.fields.date or proto.fields.date
        field_sources["date"] = "groq" if groq.fields.date else "prototype"

        merged.receipt_number = groq.fields.receipt_number or proto.fields.receipt_number
        field_sources["receipt_number"] = "groq" if groq.fields.receipt_number else "prototype"

        # structured-first
        merged.time = proto.fields.time or groq.fields.time
        field_sources["time"] = "prototype" if proto.fields.time else "groq"

        merged.subtotal = proto.fields.subtotal if proto.fields.subtotal is not None else groq.fields.subtotal
        field_sources["subtotal"] = "prototype" if proto.fields.subtotal is not None else "groq"

        merged.tax = proto.fields.tax if proto.fields.tax is not None else groq.fields.tax
        field_sources["tax"] = "prototype" if proto.fields.tax is not None else "groq"

        merged.total = proto.fields.total if proto.fields.total is not None else groq.fields.total
        field_sources["total"] = "prototype" if proto.fields.total is not None else "groq"

        # payment arbitration
        if proto.fields.card_last4:
            merged.card_last4 = proto.fields.card_last4
            field_sources["card_last4"] = "prototype"
        else:
            merged.card_last4 = groq.fields.card_last4
            field_sources["card_last4"] = "groq" if groq.fields.card_last4 else "none"

        if proto.fields.payment_method:
            merged.payment_method = proto.fields.payment_method
            field_sources["payment_method"] = "prototype"
        else:
            merged.payment_method = groq.fields.payment_method
            field_sources["payment_method"] = "groq" if groq.fields.payment_method else "none"

        # disagreement checks on important fields
        if not _same_text(proto.fields.merchant_name, groq.fields.merchant_name):
            if proto.fields.merchant_name and groq.fields.merchant_name:
                review_reasons.append("merchant_name_disagreement")

        if not _same_date(proto.fields.date, groq.fields.date):
            if proto.fields.date and groq.fields.date:
                review_reasons.append("date_disagreement")

        if not _same_time(proto.fields.time, groq.fields.time):
            if proto.fields.time and groq.fields.time:
                review_reasons.append("time_disagreement")

        if not _same_amount(proto.fields.subtotal, groq.fields.subtotal):
            if proto.fields.subtotal is not None and groq.fields.subtotal is not None:
                review_reasons.append("subtotal_disagreement")

        if not _same_amount(proto.fields.tax, groq.fields.tax):
            if proto.fields.tax is not None and groq.fields.tax is not None:
                review_reasons.append("tax_disagreement")

        if not _same_amount(proto.fields.total, groq.fields.total):
            if proto.fields.total is not None and groq.fields.total is not None:
                review_reasons.append("total_disagreement")

        if not _same_payment(proto.fields.payment_method, groq.fields.payment_method):
            if proto.fields.payment_method and groq.fields.payment_method:
                review_reasons.append("payment_method_disagreement")

        if not _same_card(proto.fields.card_last4, groq.fields.card_last4):
            if proto.fields.card_last4 and groq.fields.card_last4:
                review_reasons.append("card_last4_disagreement")

        # critical missing checks
        if not merged.merchant_name:
            review_reasons.append("missing_merchant_name")
        if not merged.date:
            review_reasons.append("missing_date")
        if merged.total is None:
            review_reasons.append("missing_total")

        # arithmetic sanity check
        if merged.subtotal is not None and merged.tax is not None and merged.total is not None:
            if abs((merged.subtotal + merged.tax) - merged.total) > 0.05:
                review_reasons.append("total_arithmetic_inconsistency")

        # items: prefer groq, fallback to prototype
        if groq.items:
            items = groq.items
            item_source = "groq"
        else:
            items = proto.items
            item_source = "prototype"

        proto_names = _item_name_set(proto.items)
        groq_names = _item_name_set(groq.items)

        if proto_names or groq_names:
            overlap = len(proto_names & groq_names)
            union = len(proto_names | groq_names)
            jaccard = overlap / union if union else 1.0
        else:
            jaccard = 1.0

        if abs(len(proto.items) - len(groq.items)) >= 4:
            review_reasons.append("item_count_disagreement")

        if jaccard < 0.40:
            review_reasons.append("item_semantic_disagreement")

        result = CanonicalReceipt(
            receipt_id=receipt_id,
            image_file=Path(image_path).name,
            fields=merged,
            items=items,
        )

        result.review_required = len(review_reasons) > 0
        result.review_reasons = review_reasons
        result.field_sources = field_sources
        result.item_source = item_source
        result.prototype_status = "ok"
        result.groq_status = "ok"

        return result
