from __future__ import annotations

from billwise.common.logging import get_logger
from billwise.retraining.exports import export_retraining_feedback


def main() -> None:
    logger = get_logger("billwise.export_retraining_feedback")
    result = export_retraining_feedback()

    logger.info("Generated at: %s", result["generated_at"])
    logger.info("Category feedback count: %s", result["category_feedback_count"])
    logger.info("OCR feedback count: %s", result["ocr_feedback_count"])
    logger.info("Export dir: %s", result["export_dir"])

    if result["gcs_paths"]:
        for key, value in result["gcs_paths"].items():
            logger.info("%s -> %s", key, value)


if __name__ == "__main__":
    main()