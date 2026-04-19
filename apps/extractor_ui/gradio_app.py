from __future__ import annotations

import os

from billwise.common.config import get_config
from billwise.common.logging import get_logger
from billwise.extraction.legacy_runtime import load_gradio_app_factory


def main() -> None:
    logger = get_logger("billwise.extractor_ui")

    # This loads .env into the current process before the legacy app uses os.environ
    cfg = get_config()

    logger.info("GROQ_API_KEY present in config: %s", bool(cfg.groq_api_key))
    logger.info("GROQ_API_KEY present in os.environ: %s", "GROQ_API_KEY" in os.environ)

    create_app = load_gradio_app_factory()
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()