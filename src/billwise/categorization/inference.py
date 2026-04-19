from __future__ import annotations

import torch

from billwise.categorization.model import get_categorization_config, load_billwise_classifier
from billwise.categorization.normalize import normalize_item_text


def top_k_scores(score_map: dict[str, float], k: int) -> dict[str, float]:
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return {label: score for label, score in ranked}


def categorize_text(text: str) -> dict:
    cfg = get_categorization_config()
    bundle = load_billwise_classifier()

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    labels = bundle["labels"]
    max_length = bundle["max_length"]

    normalized_text, was_normalized = normalize_item_text(text)

    encoded = tokenizer(
        normalized_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

    score_map = {label: float(score) for label, score in zip(labels, probs)}
    predicted_label = max(score_map, key=score_map.get)
    confidence = score_map[predicted_label]
    top_k = top_k_scores(score_map, cfg["inference"]["top_k"])
    needs_human_review = confidence < float(cfg["inference"]["confidence_threshold"])

    return {
        "input_text": text,
        "normalized_text": normalized_text,
        "was_normalized": was_normalized,
        "predicted_label": predicted_label,
        "category_confidence": confidence,
        "top_k_scores": top_k,
        "needs_human_review": needs_human_review,
        "categorizer_model": cfg["model"]["name"],
    }