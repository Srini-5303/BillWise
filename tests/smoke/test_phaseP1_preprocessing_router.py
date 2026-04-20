from pathlib import Path

import cv2
import numpy as np

from billwise.preprocessing.quality_assessor import assess_image


def test_phaseP1_quality_assessor_runs(tmp_path):
    image = np.full((400, 300, 3), 255, dtype=np.uint8)
    cv2.putText(image, "TOTAL 12.99", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    img_path = tmp_path / "test_receipt.jpg"
    cv2.imwrite(str(img_path), image)

    loaded = cv2.imread(str(img_path))
    quality = assess_image(loaded)

    assert quality.blur_score is not None
    assert quality.contrast_score is not None
    assert quality.brightness_score is not None