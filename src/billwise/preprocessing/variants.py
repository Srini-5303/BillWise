from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from billwise.preprocessing.geometric import deskew_image
from billwise.preprocessing.photometric import to_binarized, to_enhanced_gray
from billwise.preprocessing.schemas import QualityAssessment


def _save_variant(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return str(path)


def generate_variants(
    image_bgr: np.ndarray,
    original_path: Path,
    quality: QualityAssessment,
    variant_dir: Path,
) -> dict[str, str]:
    variants: dict[str, str] = {}

    variants["original"] = str(original_path)

    deskewed = deskew_image(image_bgr, -(quality.estimated_skew_deg or 0.0))
    variants["deskewed"] = _save_variant(variant_dir / f"{original_path.stem}_deskewed.jpg", deskewed)

    enhanced_gray = to_enhanced_gray(deskewed)
    variants["enhanced_gray"] = _save_variant(variant_dir / f"{original_path.stem}_enhanced_gray.jpg", enhanced_gray)

    binarized = to_binarized(deskewed)
    variants["binarized"] = _save_variant(variant_dir / f"{original_path.stem}_binarized.jpg", binarized)

    return variants