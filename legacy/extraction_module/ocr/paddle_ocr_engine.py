from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
from paddleocr import PaddleOCR

from app.config import (
    USE_DOC_ORIENTATION_CLASSIFY,
    USE_DOC_UNWARPING,
    USE_TEXTLINE_ORIENTATION,
)
from app.schemas import OCRToken


class PaddleOCREngine:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
            use_doc_unwarping=USE_DOC_UNWARPING,
            use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        )

    @staticmethod
    def _poly_to_bbox(poly: List[List[float]]) -> List[int]:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        return [min(xs), min(ys), max(xs), max(ys)]

    def load_image(self, image_path: str) -> Tuple[any, int, int]:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        h, w = img.shape[:2]
        return img, w, h

    def predict(self, image_path: str) -> Tuple[List[OCRToken], int, int]:
        image_path = str(Path(image_path))
        _, width, height = self.load_image(image_path)

        results = self.ocr.predict(image_path)

        tokens: List[OCRToken] = []
        token_id = 0

        for res in results:
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            rec_boxes = res.get("rec_boxes", [])
            dt_polys = res.get("dt_polys", [])
            dt_scores = res.get("dt_scores", [])

            n = min(len(rec_texts), len(rec_scores), len(rec_boxes))

            for i in range(n):
                text = str(rec_texts[i]).strip()
                if not text:
                    continue

                bbox = [int(v) for v in rec_boxes[i]]
                poly = None
                det_conf = None

                if i < len(dt_polys):
                    poly = [[int(p[0]), int(p[1])] for p in dt_polys[i]]
                    bbox = self._poly_to_bbox(poly)

                if i < len(dt_scores):
                    det_conf = float(dt_scores[i])

                tokens.append(
                    OCRToken(
                        id=token_id,
                        text=text,
                        bbox=bbox,
                        poly=poly,
                        ocr_confidence=float(rec_scores[i]),
                        det_confidence=det_conf,
                    )
                )
                token_id += 1

        return tokens, width, height