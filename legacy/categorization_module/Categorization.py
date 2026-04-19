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
import logging
import os
import re
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
    "human_review_log":    "logs/human_review_items.json",

    # Gemini LLM fallback (medium-confidence tier)
    "gemini_model":        "gemini-2.0-flash-lite",
    "gemini_api_key":      None,   # reads GEMINI_API_KEY env var if None

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

# Sentinel returned by categorize() when an item must go to the review dashboard
HUMAN_REVIEW_NEEDED = "HUMAN_REVIEW_NEEDED"


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
# Gemini LLM Fallback (medium-confidence tier)
# ---------------------------------------------------------------------------

def llm_fallback_gemini(
    item_text: str,
    normalized_text: str,
    classification: dict,
    config: dict,
) -> dict:
    """
    Call a small Gemini model to verify/reclassify an item when DistilBERT
    confidence falls in the medium tier (ROUTE_LLM).

    The prompt gives Gemini the raw text, normalized text, DistilBERT's top
    prediction, and the top-5 softmax scores as context.  Gemini responds
    with a JSON payload containing its chosen label and its own confidence.

    Returns:
        final_label        — Gemini's validated category label
        confidence         — "low" | "medium" | "high" (Gemini-reported)
        reason             — one-sentence explanation
        needs_human_review — True when Gemini confidence is "low"
    """
    import google.generativeai as genai

    api_key = config.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.warning(
            "Categorizer: GEMINI_API_KEY not set — escalating '%s' to human review.",
            item_text,
        )
        return {
            "final_label":        classification["predicted_label"],
            "confidence":         "low",
            "reason":             "GEMINI_API_KEY not configured.",
            "needs_human_review": True,
        }

    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(config.get("gemini_model", "gemini-2.0-flash-lite"))

    # Top-5 DistilBERT scores as context for Gemini
    top5 = sorted(
        classification["all_scores"].items(), key=lambda kv: kv[1], reverse=True
    )[:5]
    candidates_text = "\n".join(
        f"  {i + 1}. {label} (score: {score:.4f})"
        for i, (label, score) in enumerate(top5)
    )
    allowed = ", ".join(config["labels"])

    prompt = f"""You are a grocery/food item categorization expert.

A receipt scanner returned this item text:
  Original  : "{item_text}"
  Normalized: "{normalized_text}"

A DistilBERT classifier predicted "{classification['predicted_label']}" with \
confidence {classification['confidence']:.4f} (medium — needs verification).

Top 5 candidate categories from the classifier:
{candidates_text}

Allowed categories: {allowed}

Choose the most appropriate category. If you are genuinely uncertain, set \
confidence to "low" so a human reviewer can be notified.

Respond with ONLY valid JSON (no markdown, no extra text):
{{
  "final_label": "<one of the allowed categories>",
  "confidence": "<low|medium|high>",
  "reason": "<one sentence>"
}}"""

    try:
        response = gemini.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)

        # Validate label — fall back to DistilBERT's prediction on bad output
        if result.get("final_label") not in config["labels"]:
            result["final_label"] = classification["predicted_label"]
            result["confidence"]  = "low"

        result["needs_human_review"] = (result.get("confidence") == "low")
        return result

    except Exception as exc:
        logging.error("Categorizer: Gemini call failed for '%s' — %s", item_text, exc)
        return {
            "final_label":        classification["predicted_label"],
            "confidence":         "low",
            "reason":             f"Gemini call failed: {exc}",
            "needs_human_review": True,
        }


# ---------------------------------------------------------------------------
# Human Review Placeholder
# ---------------------------------------------------------------------------

def flag_human_review(
    item_text: str,
    normalized_text: str,
    classification: dict,
    reason: str,
    config: dict,
) -> dict:
    """
    Placeholder: mark an item as requiring human review and log it.

    Writes a structured entry to the human review log so a future dashboard
    can surface it.  The returned dict contains a human-readable message and
    the DistilBERT best-guess label for pre-population of the review UI.

    Returns:
        needs_human_review   — always True
        human_review_message — descriptive message for display / dashboard
        suggested_label      — DistilBERT's best guess (unconfirmed)
    """
    entry = {
        "timestamp":       datetime.datetime.now().isoformat(),
        "input":           item_text,
        "normalized":      normalized_text,
        "suggested_label": classification["predicted_label"],
        "confidence":      classification["confidence"],
        "reason":          reason,
        "status":          "pending_human_review",
    }
    log_item(config.get("human_review_log", "logs/human_review_items.json"), entry)

    message = (
        f"Human review required for: '{item_text}' — "
        f"best guess '{classification['predicted_label']}' "
        f"(confidence={classification['confidence']:.4f}). "
        f"Reason: {reason}"
    )
    print(f"\n[HUMAN REVIEW NEEDED] {message}")

    return {
        "needs_human_review":  True,
        "human_review_message": message,
        "suggested_label":     classification["predicted_label"],
    }


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

    # ── Step 4: Routing, LLM fallback & logging ─────────────────────────────
    llm_result           = None
    llm_used             = False
    needs_human_review   = False
    human_review_message = None
    final_label          = classification["predicted_label"]

    if routing == ROUTE_AUTO:
        print(
            f"\n[AUTO-ASSIGN] '{item_text}' → '{final_label}' "
            f"(confidence={classification['confidence']:.4f}). Assigned automatically."
        )

    elif routing == ROUTE_LLM:
        # Medium confidence — call Gemini to verify
        print(
            f"\n[LLM VERIFICATION] '{item_text}' → '{final_label}' "
            f"(confidence={classification['confidence']:.4f}). Calling Gemini..."
        )
        llm_result = llm_fallback_gemini(item_text, normalized_text, classification, config)
        llm_used   = True

        if llm_result["needs_human_review"]:
            # Gemini also uncertain → escalate to human review
            hr = flag_human_review(
                item_text, normalized_text, classification,
                (
                    f"DistilBERT medium confidence ({classification['confidence']:.4f}); "
                    f"Gemini also uncertain. Gemini reason: {llm_result.get('reason', 'n/a')}"
                ),
                config,
            )
            needs_human_review   = True
            human_review_message = hr["human_review_message"]
            final_label          = hr["suggested_label"]
        else:
            final_label = llm_result["final_label"]
            print(
                f"  [GEMINI] Resolved → '{final_label}' "
                f"(confidence={llm_result.get('confidence', 'n/a')}). "
                f"Reason: {llm_result.get('reason', '')}"
            )
            log_item(
                config["low_confidence_log"],
                {
                    "timestamp":             timestamp,
                    "input":                 item_text,
                    "normalized":            normalized_text,
                    "distilbert_label":      classification["predicted_label"],
                    "distilbert_confidence": classification["confidence"],
                    "gemini_label":          final_label,
                    "gemini_confidence":     llm_result.get("confidence"),
                    "reason":                llm_result.get("reason"),
                    "routing":               routing,
                    "flagged_unresolved":    flagged_unresolved,
                },
            )

    elif routing == ROUTE_HUMAN:
        # Low DistilBERT confidence — flag directly for human review
        hr = flag_human_review(
            item_text, normalized_text, classification,
            f"DistilBERT low confidence ({classification['confidence']:.4f})",
            config,
        )
        needs_human_review   = True
        human_review_message = hr["human_review_message"]
        final_label          = hr["suggested_label"]

    # Unresolved abbreviations that sailed through auto-assign still get a
    # low-confidence log entry so they remain traceable.
    if flagged_unresolved and routing == ROUTE_AUTO:
        log_item(
            config["low_confidence_log"],
            {
                "timestamp":          timestamp,
                "input":              item_text,
                "normalized":         normalized_text,
                "abbreviation_info":  abbrev_info,
                "predicted_label":    final_label,
                "confidence":         classification["confidence"],
                "routing":            routing,
                "flagged_unresolved": flagged_unresolved,
            },
        )

    # ── Step 5: Unified result dict ──────────────────────────────────────────
    return {
        "input":                item_text,
        "normalized":           normalized_text,
        "was_normalized":       was_normalized,
        "abbreviation_info":    abbrev_info,
        "predicted_label":      classification["predicted_label"],
        "confidence":           classification["confidence"],
        "all_scores":           classification["all_scores"],
        "routing":              routing,
        "flagged_unresolved":   flagged_unresolved,
        "final_label":          final_label,
        "llm_used":             llm_used,
        "llm_result":           llm_result,
        "needs_human_review":   needs_human_review,
        "human_review_message": human_review_message,
        "timestamp":            timestamp,
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
    if result.get("llm_used"):
        llm = result.get("llm_result") or {}
        print(f"  GEMINI LABEL   : {llm.get('final_label', 'n/a')}")
        print(f"  GEMINI CONF    : {llm.get('confidence', 'n/a')}")
        print(f"  GEMINI REASON  : {llm.get('reason', 'n/a')}")
    print(f"  FINAL LABEL    : {result.get('final_label', result['predicted_label'])}")
    if result.get("needs_human_review"):
        print("  !! HUMAN REVIEW NEEDED !!")
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
