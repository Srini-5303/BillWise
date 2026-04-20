from __future__ import annotations

import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def deskew_image(image: np.ndarray, estimated_skew_deg: float) -> np.ndarray:
    if abs(estimated_skew_deg) < 1.0:
        return image.copy()
    return rotate_image(image, estimated_skew_deg)