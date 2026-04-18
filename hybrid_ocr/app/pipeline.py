from __future__ import annotations

from extract.field_merger import merge_non_item_fields
from extract.line_items import extract_line_items
from extract.regex_extractors import consolidate_fields, extract_payment_fields_from_ocr
from kie.layoutlm_infer import LayoutLMv3Inferencer
from ocr.paddle_ocr_engine import PaddleOCREngine
from output.formatter import build_receipt_result


def _run_prototype_core(image_path: str):
    ocr_engine = PaddleOCREngine()
    tokens, width, height = ocr_engine.predict(image_path)

    inferencer = LayoutLMv3Inferencer()
    predictions = inferencer.predict(image_path, tokens, width, height)

    merged_fields = merge_non_item_fields(predictions)
    cleaned_fields = consolidate_fields(merged_fields)
    payment_fields = extract_payment_fields_from_ocr(tokens, image_height=height)

    all_fields = cleaned_fields + payment_fields
    all_fields = sorted(all_fields, key=lambda f: (f.bbox[1], f.bbox[0]))

    line_items = extract_line_items(predictions)

    result = build_receipt_result(
        image_path=image_path,
        image_width=width,
        image_height=height,
        fields=all_fields,
        items=line_items,
        raw_ocr=tokens,
    )

    return {
        "tokens": tokens,
        "predictions": predictions,
        "image_width": width,
        "image_height": height,
        "result": result,
    }


def run_prototype_pipeline(image_path: str):
    return _run_prototype_core(image_path)["result"]


def run_prototype_debug_pipeline(image_path: str):
    return _run_prototype_core(image_path)
