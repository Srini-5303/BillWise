from __future__ import annotations

from billwise.common.artifacts import save_processed_receipt_artifact
from billwise.common.db import init_db
from billwise.common.ids import new_id
from billwise.common.logging import get_logger
from billwise.common.repositories import (
    CategorizationRepository,
    ReceiptRepository,
    ReviewRepository,
)
from billwise.common.schemas import (
    BoundingBox,
    CategoryPrediction,
    ExtractedField,
    LineItem,
    ReceiptRecord,
    ReviewChange,
)
from billwise.common.storage import ensure_directories


def main() -> None:
    logger = get_logger("billwise.demo.phase2")

    ensure_directories()
    init_db()

    receipt_id = new_id("rcpt")
    item_1 = new_id("item")
    item_2 = new_id("item")

    receipt = ReceiptRecord(
        receipt_id=receipt_id,
        image_path="data/raw/sample_receipt.jpg",
        source="local_upload",
        processing_status="extracted",
        review_status="pending",
        vendor_name="Stop & Shop",
        receipt_date="2026-04-19",
        receipt_time="14:45",
        subtotal=12.49,
        tax=0.62,
        total=13.11,
        payment_method="credit_card",
        card_last4="1234",
        receipt_number="A10025",
        extraction_method="hybrid",
        requires_review=False,
        fields=[
            ExtractedField(
                field_name="vendor_name",
                field_value="Stop & Shop",
                confidence=0.96,
                bbox=BoundingBox(x1=12, y1=20, x2=180, y2=60),
                source_model="hybrid-selected",
            ),
            ExtractedField(
                field_name="total",
                field_value="13.11",
                confidence=0.92,
                bbox=BoundingBox(x1=210, y1=820, x2=330, y2=860),
                source_model="hybrid-selected",
            ),
        ],
        items=[
            LineItem(
                item_id=item_1,
                raw_name="BANANA",
                normalized_name="banana",
                quantity=2,
                unit_price=0.79,
                item_total=1.58,
                item_confidence=0.88,
                item_source="hybrid-selected",
            ),
            LineItem(
                item_id=item_2,
                raw_name="MILK 2PCT",
                normalized_name="milk 2 percent",
                quantity=1,
                unit_price=4.29,
                item_total=4.29,
                item_confidence=0.84,
                item_source="hybrid-selected",
            ),
        ],
    )

    ReceiptRepository.upsert_receipt(receipt)
    artifact_path = save_processed_receipt_artifact(receipt)

    CategorizationRepository.replace_for_items(
        [
            CategoryPrediction(
                item_id=item_1,
                predicted_category="Produce",
                category_confidence=0.95,
                top_k_scores={"Produce": 0.95, "Snacks": 0.03, "Bakery": 0.02},
                categorizer_model="distilbert_receipt_classifier",
                needs_human_review=False,
                final_category="Produce",
            ),
            CategoryPrediction(
                item_id=item_2,
                predicted_category="Dairy",
                category_confidence=0.91,
                top_k_scores={"Dairy": 0.91, "Beverages": 0.05, "Frozen": 0.04},
                categorizer_model="distilbert_receipt_classifier",
                needs_human_review=False,
                final_category="Dairy",
            ),
        ]
    )

    ReviewRepository.log_change(
        ReviewChange(
            entity_type="receipt",
            entity_id=receipt_id,
            field_name="payment_method",
            old_value="credit_card",
            new_value="visa",
            review_source="phase2_demo",
        )
    )

    loaded = ReceiptRepository.get_receipt(receipt_id)

    logger.info("Saved receipt_id: %s", receipt_id)
    logger.info("Artifact path: %s", artifact_path)
    logger.info("Loaded vendor: %s", loaded.vendor_name if loaded else None)
    logger.info("Loaded item count: %s", len(loaded.items) if loaded else 0)


if __name__ == "__main__":
    main()