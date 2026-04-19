from __future__ import annotations

import argparse

from billwise.categorization.service import categorize_receipt_items
from billwise.common.db import init_db
from billwise.common.logging import get_logger
from billwise.common.repositories import ReceiptRepository
from billwise.common.storage import ensure_directories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-id", required=True, help="Receipt ID already stored in BillWise DB")
    args = parser.parse_args()

    logger = get_logger("billwise.demo.phase4")

    ensure_directories()
    init_db()

    receipt = ReceiptRepository.get_receipt(args.receipt_id)
    if receipt is None:
        raise ValueError(f"Receipt not found: {args.receipt_id}")

    preds = categorize_receipt_items(args.receipt_id)

    logger.info("Receipt ID: %s", args.receipt_id)
    logger.info("Vendor: %s", receipt.vendor_name)
    logger.info("Item count: %s", len(receipt.items))
    logger.info("Prediction count: %s", len(preds))

    for pred in preds[:10]:
        logger.info(
            "item_id=%s | predicted=%s | confidence=%.4f | needs_review=%s",
            pred.item_id,
            pred.predicted_category,
            pred.category_confidence or 0.0,
            pred.needs_human_review,
        )


if __name__ == "__main__":
    main()