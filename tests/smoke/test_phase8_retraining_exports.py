from billwise.retraining.exports import _category_records, _ocr_records
import pandas as pd


def test_phase8_retraining_record_builders():
    df_cat = pd.DataFrame(
        [
            {
                "item_id": "item_1",
                "receipt_id": "rcpt_1",
                "raw_item_text": "BANANA",
                "normalized_name": "banana",
                "original_category": "Vegetables",
                "validated_category": "Fruits",
                "validator_note": "corrected",
                "validated_at": "2026-04-20 10:00:00",
                "vendor_name": "Trader Joe's",
                "receipt_date": "2026-04-20",
                "quantity": 2,
                "unit_price": 0.5,
                "item_total": 1.0,
            }
        ]
    )

    df_ocr = pd.DataFrame(
        [
            {
                "receipt_id": "rcpt_1",
                "field_name": "vendor_name",
                "original_value": "TRDER JOES",
                "corrected_value": "TRADER JOE'S",
                "corrected_at": "2026-04-20 10:05:00",
                "vendor_name": "Trader Joe's",
                "receipt_date": "2026-04-20",
                "image_path": "data/raw/rcpt_1.jpg",
                "extraction_method": "hybrid",
            }
        ]
    )

    cat_records = _category_records(df_cat)
    ocr_records = _ocr_records(df_ocr)

    assert len(cat_records) == 1
    assert len(ocr_records) == 1
    assert cat_records[0]["validated_category"] == "Fruits"
    assert ocr_records[0]["field_name"] == "vendor_name"