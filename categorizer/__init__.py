"""
BillWise — Categorizer Package
================================
Thin wrapper that loads the DistilBERT model and vocabulary once at startup,
then exposes a single categorize(item_name) -> str function for use in app.py.

Source files live in src/:
  src/Abbreviation_Normalization.py — normalization + vocabulary expansion
  src/Categorization.py             — DistilBERT inference wrapper

Environment variables:
  CATEGORIZER_MODEL_PATH    — path to full_ft_distilbert_unweighted_best.pt (required)
  CATEGORIZER_DATASET_PATH  — path to merged_labeled.csv (optional; enables abbrev expansion)
"""

import os
import sys
import logging

# Project root = parent of this package
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make src/ importable
_src_path = os.path.join(_ROOT, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# ---------------------------------------------------------------------------
# Config — env vars take precedence; local repo paths used as defaults
# ---------------------------------------------------------------------------

CONFIG = {
    "model_checkpoint":   os.environ.get(
        "CATEGORIZER_MODEL_PATH",
        os.path.join(_ROOT, "checkpoints", "full_ft_distilbert_unweighted_best.pt"),
    ),
    "base_model_name":    "distilbert-base-uncased",
    "max_length":         64,
    "high_threshold":     0.85,
    "low_threshold":      0.60,
    "dataset_path":       os.environ.get(
        "CATEGORIZER_DATASET_PATH",
        os.path.join(_ROOT, "data", "Processed_Datasets", "Labeled", "merged_labeled.csv"),
    ),
    "unresolved_log":     os.path.join(_ROOT, "logs", "unresolved_items.json"),
    "low_confidence_log": os.path.join(_ROOT, "logs", "low_confidence_items.json"),
    "human_review_log":   os.path.join(_ROOT, "logs", "human_review_items.json"),
    "gemini_model":       "gemini-2.0-flash-lite",
    "gemini_api_key":     os.environ.get("GEMINI_API_KEY"),
    "labels": [
        "Bakery & Flour", "Beverages", "Dairy", "Frozen / Processed",
        "Fruits", "Grains & Staples", "Herbs", "Meat", "Oils & Fats",
        "Poultry", "Pulses & Beans", "Sauces & Condiments", "Seafood",
        "Snacks & Ready-to-Eat", "Spices & Seasonings", "Vegetables",
    ],
}

# ---------------------------------------------------------------------------
# Singleton model state
# ---------------------------------------------------------------------------

_state = {
    "model":     None,
    "tokenizer": None,
    "device":    None,
    "ready":     False,
}


def init():
    """
    Load the DistilBERT model and (optionally) the vocabulary index.
    Call once at app startup — model stays resident for all requests.
    If the checkpoint file is missing, categorization is silently disabled.
    """
    if not os.path.exists(CONFIG["model_checkpoint"]):
        logging.warning(
            f"Categorizer: checkpoint not found at '{CONFIG['model_checkpoint']}' — "
            "Grocery_Category will be left blank."
        )
        return

    try:
        from Abbreviation_Normalization import init_pipeline
        from Categorization import load_classifier

        # Load vocabulary for abbreviation expansion (optional)
        dataset = CONFIG["dataset_path"]
        if dataset and os.path.exists(dataset):
            logging.info("Categorizer: loading vocabulary from dataset...")
            init_pipeline(dataset)
            logging.info("Categorizer: vocabulary ready.")
        else:
            logging.warning(
                "Categorizer: CATEGORIZER_DATASET_PATH not set or file missing — "
                "abbreviation expansion disabled."
            )

        # Load DistilBERT checkpoint
        logging.info(f"Categorizer: loading DistilBERT from {CONFIG['model_checkpoint']} ...")
        model, tokenizer, device = load_classifier(CONFIG)
        _state["model"]     = model
        _state["tokenizer"] = tokenizer
        _state["device"]    = device
        _state["ready"]     = True
        logging.info(f"Categorizer: ready on {device}.")

    except Exception as e:
        logging.error(f"Categorizer: failed to load — {e}. Categorization disabled.")


def categorize(item_name: str) -> str:
    """
    Return the predicted grocery category for item_name.

    Return values:
      - A category string (e.g. "Poultry") when confidence is sufficient.
      - "HUMAN_REVIEW_NEEDED" when neither DistilBERT nor Gemini could classify
        the item with enough confidence.  The item is also written to the human
        review log for future dashboard surfacing.
      - '' if the model was not loaded at startup (column stays blank in CSV).
    """
    if not _state["ready"]:
        return ""

    try:
        from Categorization import run_inference, HUMAN_REVIEW_NEEDED
        result = run_inference(
            item_name,
            _state["model"],
            _state["tokenizer"],
            _state["device"],
            None, None, None,   # inventory/vectorizer/tfidf_matrix not used by classifier
            CONFIG,
        )

        if result["needs_human_review"]:
            final = HUMAN_REVIEW_NEEDED
        else:
            final = result["final_label"]

        logging.getLogger("categorizer").info(
            "%-40s → %-25s (distilbert=%.4f, routing=%s, llm_used=%s, human_review=%s)",
            repr(item_name),
            final,
            result["confidence"],
            result["routing"],
            result["llm_used"],
            result["needs_human_review"],
        )
        return final
    except Exception as e:
        logging.error(f"Categorizer: error on '{item_name}' — {e}")
        return ""
