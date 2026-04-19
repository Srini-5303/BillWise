from __future__ import annotations

import argparse

from billwise.common.logging import get_logger
from billwise.pipeline.orchestrator import run_billwise_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a receipt image")
    args = parser.parse_args()

    logger = get_logger("billwise.run_pipeline")
    result = run_billwise_pipeline(args.image)

    logger.info("Receipt ID: %s", result.receipt_id)
    logger.info("Vendor: %s", result.vendor_name)
    logger.info("Item count: %s", result.item_count)
    logger.info("Category count: %s", result.category_count)
    logger.info("Requires review: %s", result.requires_review)
    logger.info("Processing status: %s", result.processing_status)
    logger.info("Review status: %s", result.review_status)


if __name__ == "__main__":
    main()