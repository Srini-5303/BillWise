import json
from pathlib import Path
from collections import Counter
from pprint import pprint

from extract.field_merger import merge_non_item_fields
from extract.line_items import extract_line_items
from extract.regex_extractors import consolidate_fields, extract_payment_fields_from_ocr
from kie.layoutlm_infer import LayoutLMv3Inferencer
from ocr.paddle_ocr_engine import PaddleOCREngine
from output.formatter import build_receipt_result


def main() -> None:
    image_path = "assets/sample_receipt.jpg"

    ocr_engine = PaddleOCREngine()
    tokens, width, height = ocr_engine.predict(image_path)

    print(f"Image size: {width} x {height}")
    print(f"OCR tokens: {len(tokens)}")

    inferencer = LayoutLMv3Inferencer()
    predictions = inferencer.predict(image_path, tokens, width, height)

    print(f"LayoutLM predictions: {len(predictions)}")

    predicted_ids = {p['token_id'] for p in predictions}
    missing_ids = [t.id for t in tokens if t.id not in predicted_ids]

    print(f"Missing token predictions: {len(missing_ids)}")

    label_counts = Counter(p["label"] for p in predictions)
    print("Top labels:")
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count}")

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

    print("\nFinal extracted fields:")
    for field in result.fields:
        pprint(field.model_dump())

    print(f"\nExtracted line items count: {len(result.items)}")
    print("\nFirst 15 extracted line items:")
    for item in result.items[:15]:
        pprint(item.model_dump())

    print("\nFinal JSON:")
    print(json.dumps(result.model_dump(), indent=2))

    print("\nFinal JSON:")
    result_json = result.model_dump()
    print(json.dumps(result_json, indent=2))

    output_dir = Path("assets/debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "last_receipt_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    print(f"\nSaved JSON to: {output_path}")


if __name__ == "__main__":
    main()
