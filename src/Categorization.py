"""
BillWise — DistilBERT Inference Wrapper
========================================
End-to-end pipeline: Normalization → DistilBERT Classification → Routing

Integrates:
  - billwise_pipeline.py : normalization layer (smart_normalize, abbreviation expansion)
  - DistilBERT Full FT Unweighted checkpoint : production classification model

IMPORTANT: Do NOT modify billwise_pipeline.py — import only.
"""

import json
import os
import datetime
import pathlib

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from Abbreviation_Normalization import (
    smart_normalize,
    init_pipeline,
    _looks_like_abbreviation,
    _expand_token,
    _expansion_cache,  # noqa: F401 — imported for completeness / future inspection
    _vocab,            # noqa: F401 — imported for completeness / future inspection
)

# ---------------------------------------------------------------------------
# Configuration — all tuneable values live here, nowhere else
# ---------------------------------------------------------------------------

CONFIG = {
    # Model
    "model_checkpoint": "F:\\Study\\Sem 4\\AI Capstone\\checkpoints\\full_ft_distilbert_unweighted_best.pt",
    "base_model_name":  "distilbert-base-uncased",
    "max_length":       64,

    # Confidence thresholds (post-softmax)
    "high_threshold": 0.85,   # >= this → auto-assign
    "low_threshold":  0.60,   # < this  → human review

    # Paths
    "dataset_path":        "F:\Study\Sem 4\AI Capstone\data\Processed_Datasets\Labeled\merged_labeled.csv",
    "unresolved_log":      "logs/unresolved_items.json",
    "low_confidence_log":  "logs/low_confidence_items.json",

    # 16 canonical categories in label order (must match training label2id)
    "labels": [
        "Bakery & Flour", "Beverages", "Dairy", "Frozen / Processed",
        "Fruits", "Grains & Staples", "Herbs", "Meat", "Oils & Fats",
        "Poultry", "Pulses & Beans", "Sauces & Condiments", "Seafood",
        "Snacks & Ready-to-Eat", "Spices & Seasonings", "Vegetables",
    ],
}

# Routing decision constants
ROUTE_AUTO       = "auto_assign"             # high confidence  → direct to Google Sheets
ROUTE_LLM        = "llm_verification"        # medium confidence → LLM verification
ROUTE_HUMAN      = "human_review"            # low confidence   → human review queue
ROUTE_UNRESOLVED = "unresolved_abbreviation" # unrecognizable abbreviation → human verification


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_classifier(config: dict):
    """
    Load the DistilBERT classification model from checkpoint.

    Initialises the model architecture from HuggingFace Hub, then overlays
    the saved fine-tuned weights. Handles the common checkpoint wrapping
    patterns (bare state_dict, 'model_state_dict' key, 'state_dict' key).

    Returns:
        model     — model in eval mode on the appropriate device
        tokenizer — matching tokenizer
        device    — torch.device (cuda or cpu)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model architecture
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model_name"],
        num_labels=len(config["labels"]),
    )

    # Load checkpoint and extract state dict
    raw = torch.load(config["model_checkpoint"], map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = raw  # assume raw IS the state dict

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"])

    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Abbreviation Detection
# ---------------------------------------------------------------------------

def detect_unresolved_abbreviations(original_text: str, normalized_text: str) -> dict:
    """
    Inspect tokens in `original_text` for abbreviations that the normalization
    layer could not expand.

    Algorithm:
      1. Split original_text by whitespace.
      2. For each token (lowercased for vocab compatibility):
           - If _looks_like_abbreviation(token) is True:
               * Call _expand_token(token).
               * If the result equals the original token → expansion failed → UNRESOLVED.
               * Otherwise → record the resolved mapping.
      3. Return a structured dict.

    Returns:
        has_abbreviations — bool  : at least one abbreviation-like token detected
        has_unresolved    — bool  : at least one abbreviation failed to expand
        unresolved_tokens — list  : tokens that could not be resolved
        resolved_tokens   — dict  : {original_token: expanded_token}

    NOTE: normalized_text is accepted as a parameter for future use (e.g. diff-based
    detection) but the current implementation works from the original tokens directly.
    """
    has_abbreviations = False
    has_unresolved    = False
    unresolved_tokens: list[str] = []
    resolved_tokens:   dict[str, str] = {}

    for token in original_text.split():
        # Lowercase so _looks_like_abbreviation and _expand_token operate on
        # the same case as the vocabulary (which is all-lowercase).
        token_lower = token.lower()

        if _looks_like_abbreviation(token_lower):
            has_abbreviations = True
            expanded = _expand_token(token_lower)

            if expanded == token_lower:
                # _expand_token returned the token unchanged → could not resolve
                has_unresolved = True
                unresolved_tokens.append(token)
            else:
                resolved_tokens[token] = expanded

    return {
        "has_abbreviations": has_abbreviations,
        "has_unresolved":    has_unresolved,
        "unresolved_tokens": unresolved_tokens,
        "resolved_tokens":   resolved_tokens,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(text: str, model, tokenizer, device, config: dict) -> dict:
    """
    Run DistilBERT inference on `text` and return a routing decision.

    Routing logic (post-softmax confidence):
      >= high_threshold → ROUTE_AUTO   (auto-assign)
      >= low_threshold  → ROUTE_LLM    (LLM verification)
      <  low_threshold  → ROUTE_HUMAN  (human review)

    Returns:
        predicted_label — highest-confidence category string
        confidence      — max softmax score (float)
        all_scores      — {label: score} for all 16 classes
        routing         — one of ROUTE_AUTO / ROUTE_LLM / ROUTE_HUMAN
    """
    encoding = tokenizer(
        text,
        max_length=config["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    confidence, pred_idx = probs.max(dim=0)
    confidence  = confidence.item()
    pred_idx    = pred_idx.item()

    predicted_label = config["labels"][pred_idx]

    all_scores = {
        label: round(probs[i].item(), 6)
        for i, label in enumerate(config["labels"])
    }

    # Routing decision
    if confidence >= config["high_threshold"]:
        routing = ROUTE_AUTO
    elif confidence >= config["low_threshold"]:
        routing = ROUTE_LLM
    else:
        routing = ROUTE_HUMAN

    return {
        "predicted_label": predicted_label,
        "confidence":      confidence,
        "all_scores":      all_scores,
        "routing":         routing,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_item(filepath: str, entry: dict) -> None:
    """
    Append `entry` to the JSON array stored at `filepath`.

    - Creates the logs/ directory (and any parent directories) if absent.
    - Loads the existing JSON array or starts with an empty list on first write.
    - Guarantees every written entry has an ISO-format 'timestamp' field.
    - Writes the updated array back with 2-space indentation.
    """
    log_path = pathlib.Path(filepath)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []
    else:
        records = []

    # Ensure timestamp is always present
    if "timestamp" not in entry:
        entry["timestamp"] = datetime.datetime.now().isoformat()

    records.append(entry)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main Pipeline Function
# ---------------------------------------------------------------------------

def run_inference(
    item_text: str,
    model,
    tokenizer,
    device,
    inventory,
    vectorizer,
    tfidf_matrix,
    config: dict,
) -> dict:
    """
    Single entry point for processing one receipt item end-to-end.

    Steps:
      1. Normalize   — smart_normalize() from billwise_pipeline.py
      2. Detect      — check for unresolved abbreviations; log & flag if found,
                       but ALWAYS proceed to classification regardless
      3. Classify    — DistilBERT forward pass + softmax + routing decision
      4. Route & log — log low-confidence or unresolved items; emit console notice
      5. Return      — unified result dict

    Args:
        item_text      — raw receipt text (OCR / user input)
        model          — loaded DistilBERT model (eval mode)
        tokenizer      — matching HuggingFace tokenizer
        device         — torch.device
        inventory      — pd.DataFrame from init_pipeline
        vectorizer     — TF-IDF vectorizer from init_pipeline (passed through for
                         future similarity-augmentation; not used by classifier)
        tfidf_matrix   — TF-IDF matrix from init_pipeline (same note)
        config         — CONFIG dict

    Returns a dict with keys:
        input, normalized, was_normalized, abbreviation_info,
        predicted_label, confidence, all_scores, routing,
        flagged_unresolved, timestamp
    """
    timestamp = datetime.datetime.now().isoformat()

    # ── Step 1: Normalization ────────────────────────────────────────────────
    normalized_text, was_normalized = smart_normalize(item_text)

    # ── Step 2: Abbreviation detection ──────────────────────────────────────
    abbrev_info       = detect_unresolved_abbreviations(item_text, normalized_text)
    flagged_unresolved = abbrev_info["has_unresolved"]

    if flagged_unresolved:
        unresolved_tokens = abbrev_info["unresolved_tokens"]

        # Log to unresolved queue
        log_item(
            config["unresolved_log"],
            {
                "timestamp":        timestamp,
                "input":            item_text,
                "normalized":       normalized_text,
                "unresolved_tokens": unresolved_tokens,
                "routing":          ROUTE_UNRESOLVED,
                "status":           "pending_human_verification",
            },
        )

        print(
            f"\n[UNRESOLVED ABBREVIATION] '{item_text}' contains unrecognizable "
            f"tokens {unresolved_tokens}. Flagged for human verification."
        )
        # Pipeline continues — classification runs on best-effort normalized text

    # ── Step 3: Classification ───────────────────────────────────────────────
    classification = classify(normalized_text, model, tokenizer, device, config)
    routing        = classification["routing"]

    # ── Step 4: Routing & logging ────────────────────────────────────────────
    if routing == ROUTE_HUMAN or flagged_unresolved:
        log_item(
            config["low_confidence_log"],
            {
                "timestamp":         timestamp,
                "input":             item_text,
                "normalized":        normalized_text,
                "abbreviation_info": abbrev_info,
                "predicted_label":   classification["predicted_label"],
                "confidence":        classification["confidence"],
                "routing":           routing,
                "flagged_unresolved": flagged_unresolved,
            },
        )
        if routing == ROUTE_HUMAN:
            print(
                f"\n[LOW CONFIDENCE] '{item_text}' → '{classification['predicted_label']}' "
                f"(confidence={classification['confidence']:.4f}). "
                f"Sent to human review queue."
            )

    elif routing == ROUTE_LLM:
        print(
            f"\n[LLM VERIFICATION] '{item_text}' → '{classification['predicted_label']}' "
            f"(confidence={classification['confidence']:.4f}). "
            f"Queued for LLM verification."
        )
        # TODO: Call llm_fallback() from billwise_pipeline.py here
        #       Pass: original, normalized, and a mock match_result dict
        #       This will be wired in the next integration phase

    elif routing == ROUTE_AUTO:
        print(
            f"\n[AUTO-ASSIGN] '{item_text}' → '{classification['predicted_label']}' "
            f"(confidence={classification['confidence']:.4f}). Assigned automatically."
        )

    # ── Step 5: Unified result dict ──────────────────────────────────────────
    return {
        "input":             item_text,
        "normalized":        normalized_text,
        "was_normalized":    was_normalized,
        "abbreviation_info": abbrev_info,
        "predicted_label":   classification["predicted_label"],
        "confidence":        classification["confidence"],
        "all_scores":        classification["all_scores"],
        "routing":           routing,
        "flagged_unresolved": flagged_unresolved,
        "timestamp":         timestamp,
    }


# ---------------------------------------------------------------------------
# Pretty Printer
# ---------------------------------------------------------------------------

def print_inference_result(result: dict) -> None:
    """
    Print a clean, aligned summary of a single inference result.

    Top-3 scores are sorted by confidence descending (not alphabetically).
    """
    abbrev         = result["abbreviation_info"]
    unresolved_str = (
        f" {abbrev['unresolved_tokens']}" if abbrev["unresolved_tokens"] else ""
    )

    # Top 3 by score — descending
    top3 = sorted(result["all_scores"].items(), key=lambda kv: kv[1], reverse=True)[:3]

    print()
    print("════════════════════════════════════════════════")
    print(f"  INPUT          : {result['input']}")
    print(f"  NORMALIZED     : {result['normalized']}  (changed: {result['was_normalized']})")
    print("  ────────────────────────────────────────────")
    print(f"  ABBREVIATIONS  : {abbrev['has_abbreviations']}")
    print(f"  UNRESOLVED     : {abbrev['has_unresolved']}{unresolved_str}")
    print("  ────────────────────────────────────────────")
    print(f"  PREDICTION     : {result['predicted_label']}")
    print(f"  CONFIDENCE     : {result['confidence']:.4f}")
    print(f"  ROUTING        : {result['routing']}")
    print("  ────────────────────────────────────────────")
    print("  TOP 3 SCORES   :")
    for rank, (label, score) in enumerate(top3, 1):
        print(f"    {rank}. {label}: {score:.4f}")
    print("════════════════════════════════════════════════")


# ---------------------------------------------------------------------------
# Interactive Loop
# ---------------------------------------------------------------------------

def interactive_loop(
    model,
    tokenizer,
    device,
    inventory,
    vectorizer,
    tfidf_matrix,
    config: dict,
) -> None:
    """
    Interactive REPL for manual testing of the end-to-end inference pipeline.

    Commands:
        'quit' / 'exit'  — stop the loop
        'demo'           — run a built-in set of sample receipt items
    """
    demo_items = [
        # Clean items — no abbreviations
        "olive oil extra virgin",
        "fresh tomatoes",
        # Resolvable abbreviations — vocabulary expansion should succeed
        "chk brst bnls",
        "frzn grnd beef",
        # Unresolvable abbreviations — should trigger flag + log
        "xyz brst",
        "qwk frzn",
        # Ambiguous items — medium confidence expected
        "processed cheese slice",
        "frozen peas",
    ]

    print("\n─── BillWise Inference Engine ──────────────────────────")
    print("Type a receipt item text and press Enter.")
    print("Commands: 'quit' | 'exit' to stop, 'demo' to run sample items.\n")

    while True:
        try:
            text = input("Enter receipt item: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not text:
            continue

        if text.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            break

        if text.lower() == "demo":
            print(f"\nRunning {len(demo_items)} demo items...\n")
            for item in demo_items:
                result = run_inference(
                    item, model, tokenizer, device,
                    inventory, vectorizer, tfidf_matrix, config,
                )
                print_inference_result(result)
            continue

        result = run_inference(
            text, model, tokenizer, device,
            inventory, vectorizer, tfidf_matrix, config,
        )
        print_inference_result(result)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Guard: fail fast if the checkpoint is missing
    checkpoint_path = pathlib.Path(CONFIG["model_checkpoint"])
    if not checkpoint_path.exists():
        print(
            f"\n[ERROR] Model checkpoint not found: '{CONFIG['model_checkpoint']}'\n"
            f"        Resolved path : {checkpoint_path.resolve()}\n"
            f"        Place 'full_ft_distilbert_unweighted_best.pt' inside a "
            f"'checkpoints/' directory relative to this script.\n"
        )
        raise SystemExit(1)

    # Initialise normalization pipeline — loads inventory, builds TF-IDF, learns vocab
    inventory, vectorizer, tfidf_matrix = init_pipeline(CONFIG["dataset_path"])

    # Load DistilBERT classifier (done once; model stays resident for all items)
    print("Loading DistilBERT classifier...")
    model, tokenizer, device = load_classifier(CONFIG)
    print("  Classifier ready.\n")

    # Startup summary
    print("═══════════════════════════════════════════════════════")
    print("  BillWise Inference Engine — Ready")
    print("═══════════════════════════════════════════════════════")
    print(f"  Model checkpoint   : {CONFIG['model_checkpoint']}")
    print(f"  Device             : {device}")
    print(f"  High threshold     : {CONFIG['high_threshold']}   (>= → {ROUTE_AUTO})")
    print(f"  Low threshold      : {CONFIG['low_threshold']}   (<  → {ROUTE_HUMAN})")
    print(f"  Unresolved log     : {CONFIG['unresolved_log']}")
    print(f"  Low-confidence log : {CONFIG['low_confidence_log']}")
    print("═══════════════════════════════════════════════════════")

    interactive_loop(model, tokenizer, device, inventory, vectorizer, tfidf_matrix, CONFIG)
