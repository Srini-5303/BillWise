from __future__ import annotations

from billwise.categorization.inference import categorize_text
from billwise.common.repositories import CategorizationRepository, ReceiptRepository
from billwise.common.schemas import CategoryPrediction


def categorize_receipt_items(receipt_id: str) -> list[CategoryPrediction]:
    receipt = ReceiptRepository.get_receipt(receipt_id)
    if receipt is None:
        raise ValueError(f"Receipt not found: {receipt_id}")

    predictions: list[CategoryPrediction] = []

    for item in receipt.items:
        result = categorize_text(item.raw_name)

        predictions.append(
            CategoryPrediction(
                item_id=item.item_id,
                predicted_category=result["predicted_label"],
                category_confidence=result["category_confidence"],
                top_k_scores=result["top_k_scores"],
                categorizer_model=result["categorizer_model"],
                needs_human_review=result["needs_human_review"],
                final_category=result["predicted_label"],
            )
        )

    CategorizationRepository.replace_for_items(predictions)

    requires_category_review = any(p.needs_human_review for p in predictions)
    ReceiptRepository.update_receipt_status(
        receipt_id=receipt_id,
        processing_status="categorized",
        review_status="pending" if requires_category_review else "approved",
        requires_review=requires_category_review or receipt.requires_review,
    )

    return predictions