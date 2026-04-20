from __future__ import annotations

import cv2
import numpy as np

from billwise.preprocessing.schemas import QualityAssessment


def estimate_skew_deg(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=20)

    if lines is None:
        return 0.0

    angles = []
    for line in lines[:100]:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 <= angle <= 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def assess_image(image_bgr: np.ndarray) -> QualityAssessment:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast_score = float(gray.std())
    brightness_score = float(gray.mean())
    skew_deg = estimate_skew_deg(gray)

    issues = []

    if blur_score < 80:
        issues.append("blur")
    if contrast_score < 35:
        issues.append("low_contrast")
    if brightness_score < 90:
        issues.append("dark")
    elif brightness_score > 210:
        issues.append("overexposed")
    if abs(skew_deg) > 2.5:
        issues.append("skew")

    return QualityAssessment(
        blur_score=blur_score,
        contrast_score=contrast_score,
        brightness_score=brightness_score,
        estimated_skew_deg=skew_deg,
        issues=issues,
    )