from __future__ import annotations

from typing import List

from app.schemas import ExtractedField, LineItem, OCRToken, ReceiptResult


def build_receipt_result(
    image_path: str,
    image_width: int,
    image_height: int,
    fields: List[ExtractedField],
    items: List[LineItem],
    raw_ocr: List[OCRToken],
) -> ReceiptResult:
    return ReceiptResult(
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
        fields=fields,
        items=items,
        raw_ocr=raw_ocr,
    )
