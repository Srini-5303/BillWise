from billwise.extraction.transform import build_receipt_from_hybrid_payload


def test_phase3_hybrid_payload_transform():
    payload = {
        "review_required": True,
        "fields": {
            "merchant_name": {
                "final_value": "Target",
                "final_source": "groq",
                "final_confidence": None,
                "final_bbox": [10, 20, 100, 50],
                "prototype": {"bbox": [10, 20, 100, 50]},
            },
            "date": {
                "final_value": "2026-04-19",
                "final_source": "groq",
                "final_confidence": None,
                "final_bbox": None,
                "prototype": {"bbox": [12, 70, 120, 100]},
            },
            "time": {
                "final_value": "14:11",
                "final_source": "prototype",
                "final_confidence": 0.91,
                "final_bbox": [200, 70, 260, 100],
                "prototype": {"bbox": [200, 70, 260, 100]},
            },
            "subtotal": {
                "final_value": 10.50,
                "final_source": "prototype",
                "final_confidence": 0.88,
                "final_bbox": [200, 600, 320, 630],
                "prototype": {"bbox": [200, 600, 320, 630]},
            },
            "tax": {
                "final_value": 0.84,
                "final_source": "prototype",
                "final_confidence": 0.86,
                "final_bbox": [200, 640, 320, 670],
                "prototype": {"bbox": [200, 640, 320, 670]},
            },
            "total": {
                "final_value": 11.34,
                "final_source": "prototype",
                "final_confidence": 0.95,
                "final_bbox": [200, 690, 320, 720],
                "prototype": {"bbox": [200, 690, 320, 720]},
            },
            "payment_method": {
                "final_value": "card",
                "final_source": "prototype",
                "final_confidence": 0.80,
                "final_bbox": [20, 730, 150, 760],
                "prototype": {"bbox": [20, 730, 150, 760]},
            },
            "card_last4": {
                "final_value": "1234",
                "final_source": "prototype",
                "final_confidence": 0.79,
                "final_bbox": [160, 730, 240, 760],
                "prototype": {"bbox": [160, 730, 240, 760]},
            },
            "receipt_number": {
                "final_value": "R-1001",
                "final_source": "groq",
                "final_confidence": None,
                "final_bbox": None,
                "prototype": {"bbox": [20, 120, 170, 150]},
            },
        },
        "prototype_items": [
            {
                "line_id": 1,
                "name": "BANANA",
                "quantity": "2",
                "item_total": "1.58",
                "bbox": [30, 200, 180, 225],
                "confidence": 0.87,
            }
        ],
        "hybrid_items": [
            {
                "line_id": 1,
                "name": "BANANA",
                "quantity": 2,
                "unit_price": 0.79,
                "item_total": 1.58,
            }
        ],
    }

    receipt = build_receipt_from_hybrid_payload(
        payload=payload,
        image_path="data/raw/rcpt_test.jpg",
        receipt_id="rcpt_test",
    )

    assert receipt.receipt_id == "rcpt_test"
    assert receipt.vendor_name == "Target"
    assert receipt.total == 11.34
    assert receipt.requires_review is True
    assert receipt.review_status == "pending"
    assert len(receipt.fields) == 9
    assert len(receipt.items) == 1
    assert receipt.items[0].raw_name == "BANANA"
    assert receipt.items[0].normalized_name == "banana"