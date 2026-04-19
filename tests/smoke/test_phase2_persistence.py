from billwise.common.artifacts import load_receipt_artifact, save_processed_receipt_artifact
from billwise.common.db import init_db
from billwise.common.ids import new_id
from billwise.common.repositories import CategorizationRepository, ReceiptRepository
from billwise.common.schemas import CategoryPrediction, LineItem, ReceiptRecord
from billwise.common.storage import ensure_directories


def test_phase2_receipt_persistence():
    ensure_directories()
    init_db()

    receipt_id = new_id("rcpt")
    item_id = new_id("item")

    receipt = ReceiptRecord(
        receipt_id=receipt_id,
        image_path="data/raw/test.jpg",
        extraction_method="hybrid",
        items=[
            LineItem(
                item_id=item_id,
                raw_name="APPLE",
                normalized_name="apple",
                quantity=1,
                unit_price=1.5,
                item_total=1.5,
                item_confidence=0.9,
                item_source="hybrid-selected",
            )
        ],
    )

    ReceiptRepository.upsert_receipt(receipt)
    save_processed_receipt_artifact(receipt)

    loaded_receipt = ReceiptRepository.get_receipt(receipt_id)
    assert loaded_receipt is not None
    assert loaded_receipt.receipt_id == receipt_id
    assert len(loaded_receipt.items) == 1
    assert loaded_receipt.items[0].raw_name == "APPLE"

    CategorizationRepository.replace_for_items(
        [
            CategoryPrediction(
                item_id=item_id,
                predicted_category="Produce",
                category_confidence=0.93,
                top_k_scores={"Produce": 0.93, "Bakery": 0.07},
                categorizer_model="distilbert_receipt_classifier",
                needs_human_review=False,
                final_category="Produce",
            )
        ]
    )

    preds = CategorizationRepository.get_for_receipt(receipt_id)
    assert len(preds) == 1
    assert preds[0].predicted_category == "Produce"

    artifact = load_receipt_artifact(receipt_id)
    assert artifact is not None
    assert artifact["receipt_id"] == receipt_id