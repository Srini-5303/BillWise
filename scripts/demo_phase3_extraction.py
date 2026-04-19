from __future__ import annotations

import argparse

from billwise.common.db import init_db
from billwise.common.logging import get_logger
from billwise.common.storage import ensure_directories
from billwise.extraction.service import extract_receipt_with_hybrid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a receipt image")
    args = parser.parse_args()

    logger = get_logger("billwise.demo.phase3")

    ensure_directories()
    init_db()

    receipt, payload = extract_receipt_with_hybrid(args.image, persist=True)

    logger.info("Receipt ID: %s", receipt.receipt_id)
    logger.info("Stored image: %s", receipt.image_path)
    logger.info("Vendor: %s", receipt.vendor_name)
    logger.info("Total: %s", receipt.total)
    logger.info("Line items: %s", len(receipt.items))
    logger.info("Requires review: %s", receipt.requires_review)
    logger.info("Review reasons: %s", payload.get("review_reasons", []))


if __name__ == "__main__":
    main()