from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoProcessor

from app.config import (
    DEVICE,
    LAYOUTLM_MODEL_NAME,
    LAYOUTLM_WORD_CHUNK_SIZE,
    LAYOUTLM_WORD_OVERLAP,
)
from app.schemas import OCRToken
from ocr.ocr_utils import extract_normalized_bboxes, extract_words


class LayoutLMv3Inferencer:
    def __init__(self) -> None:
        self.device = torch.device(DEVICE)
        self.processor = AutoProcessor.from_pretrained(
            LAYOUTLM_MODEL_NAME,
            apply_ocr=False,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            LAYOUTLM_MODEL_NAME
        )
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @staticmethod
    def _build_chunks(
        tokens: List[OCRToken],
        chunk_size: int,
        overlap: int,
    ) -> List[Tuple[int, int, List[OCRToken]]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        chunks: List[Tuple[int, int, List[OCRToken]]] = []
        start = 0
        n = len(tokens)

        while start < n:
            end = min(start + chunk_size, n)
            chunks.append((start, end, tokens[start:end]))
            if end == n:
                break
            start = end - overlap

        return chunks

    def _predict_chunk(
        self,
        image,
        chunk_tokens: List[OCRToken],
        image_width: int,
        image_height: int,
    ) -> List[Dict[str, Any]]:
        words = extract_words(chunk_tokens)
        boxes = extract_normalized_bboxes(chunk_tokens, image_width, image_height)

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)

        model_inputs = {
            k: v.to(self.device) for k, v in encoding.items() if hasattr(v, "to")
        }

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)

        seen_word_ids = set()
        predictions: List[Dict[str, Any]] = []

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id in seen_word_ids:
                continue
            if word_id >= len(chunk_tokens):
                continue

            seen_word_ids.add(word_id)

            label_id = int(pred_ids[token_idx].item())
            label = self.id2label[label_id]
            confidence = float(probs[token_idx, label_id].item())

            ocr_token = chunk_tokens[word_id]

            predictions.append(
                {
                    "token_id": ocr_token.id,
                    "text": ocr_token.text,
                    "bbox": ocr_token.bbox,
                    "ocr_confidence": ocr_token.ocr_confidence,
                    "label": label,
                    "label_confidence": confidence,
                }
            )

        return predictions

    def predict(
        self,
        image_path: str,
        tokens: List[OCRToken],
        image_width: int,
        image_height: int,
    ) -> List[Dict[str, Any]]:
        image = Image.open(image_path).convert("RGB")

        chunks = self._build_chunks(
            tokens=tokens,
            chunk_size=LAYOUTLM_WORD_CHUNK_SIZE,
            overlap=LAYOUTLM_WORD_OVERLAP,
        )

        best_predictions: Dict[int, Dict[str, Any]] = {}

        for chunk_start, chunk_end, chunk_tokens in chunks:
            chunk_preds = self._predict_chunk(
                image=image,
                chunk_tokens=chunk_tokens,
                image_width=image_width,
                image_height=image_height,
            )

            for pred in chunk_preds:
                token_id = pred["token_id"]
                existing = best_predictions.get(token_id)

                if existing is None or pred["label_confidence"] > existing["label_confidence"]:
                    best_predictions[token_id] = pred

        ordered_predictions = [
            best_predictions[token.id]
            for token in tokens
            if token.id in best_predictions
        ]

        return ordered_predictions
