from __future__ import annotations

from typing import List

from app.schemas import OCRToken


def normalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = bbox

    return [
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
        max(0, min(1000, int(1000 * x2 / width))),
        max(0, min(1000, int(1000 * y2 / height))),
    ]


def sort_tokens_reading_order(tokens: List[OCRToken]) -> List[OCRToken]:
    return sorted(tokens, key=lambda t: (t.bbox[1], t.bbox[0]))


def extract_words(tokens: List[OCRToken]) -> List[str]:
    return [t.text for t in tokens]


def extract_pixel_bboxes(tokens: List[OCRToken]) -> List[List[int]]:
    return [t.bbox for t in tokens]


def extract_normalized_bboxes(
    tokens: List[OCRToken], width: int, height: int
) -> List[List[int]]:
    return [normalize_bbox(t.bbox, width, height) for t in tokens]
