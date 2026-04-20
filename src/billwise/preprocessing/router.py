from __future__ import annotations

from pathlib import Path

import cv2

from billwise.common.config import get_config
from billwise.preprocessing.quality_assessor import assess_image
from billwise.preprocessing.schemas import PreprocessingResult
from billwise.preprocessing.scorer import score_variant
from billwise.preprocessing.variants import generate_variants


def run_preprocessing_router(image_path: str | Path) -> PreprocessingResult:
    cfg = get_config()
    image_path = Path(image_path)

    if not cfg.perform_preprocessing:
        return PreprocessingResult(
            original_path=str(image_path),
            quality=assess_image(cv2.imread(str(image_path))),
            variants=[],
            selected_ocr_variant="original",
            selected_vlm_variant="original",
            selected_ocr_path=str(image_path),
            selected_vlm_path=str(image_path),
            performed=False,
        )

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    quality = assess_image(image_bgr)

    variant_dir = image_path.parent / "preprocessed"
    variant_map = generate_variants(
        image_bgr=image_bgr,
        original_path=image_path,
        quality=quality,
        variant_dir=variant_dir,
    )

    scores = [score_variant(name, path) for name, path in variant_map.items()]
    scores_sorted = sorted(scores, key=lambda s: s.overall_score, reverse=True)

    best_ocr = scores_sorted[0]

    natural_candidates = [s for s in scores if s.name in {"original", "deskewed"}]
    best_vlm = max(natural_candidates, key=lambda s: s.overall_score) if natural_candidates else best_ocr

    return PreprocessingResult(
        original_path=str(image_path),
        quality=quality,
        variants=scores,
        selected_ocr_variant=best_ocr.name,
        selected_vlm_variant=best_vlm.name,
        selected_ocr_path=best_ocr.path,
        selected_vlm_path=best_vlm.path,
        performed=True,
    )