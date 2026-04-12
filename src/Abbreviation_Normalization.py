"""
BillWise — Receipt Item Categorization Pipeline
================================================
Architecture: OCR/VLM Output → Normalization → Similarity Matching → LLM Fallback → Final Label

Requires:
    pip install anthropic scikit-learn rapidfuzz pandas
    ANTHROPIC_API_KEY environment variable set.
"""

import os
import re
import json
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = "data\Processed_Datasets\merged_file.csv"

ALLOWED_LABELS = [
    "Vegetables", "Fruits", "Herbs", "Spices & Seasonings", "Oils & Fats",
    "Grains & Staples", "Pulses & Beans", "Dairy", "Poultry", "Meat",
    "Seafood", "Sauces & Condiments", "Bakery & Flour", "Snacks & Ready-to-Eat",
    "Beverages", "Frozen / Processed", "Other",
]

# Score thresholds for routing decisions
# score > 0.5 → accept (similarity label used directly)
# score ≤ 0.5 → fallback_llm (send to LLM)
ACCEPT_THRESHOLD = 0.50

# Weights for the hybrid score (must sum to 1.0)
TFIDF_WEIGHT   = 0.5
FUZZY_WEIGHT   = 0.5

# Small seed dictionary for unit stripping only.
# Abbreviation expansion is now handled automatically via vocabulary matching.
ABBREV_DICT = {
    "oz": "",   # strip unit
    "lb": "",   # strip unit
    "lbs": "",  # strip unit
    "kg": "",   # strip unit
    "g": "",    # strip unit
    "ml": "",   # strip unit
    "l": "",    # strip unit
}

# Minimum fuzzy score (0-100) to accept a vocabulary-based expansion
VOCAB_EXPAND_THRESHOLD = 72

# Learned expansion cache so we don't re-compute the same token twice
_expansion_cache: dict[str, str] = {}

# Inventory vocabulary — populated by build_vocab_index()
# Maps word → frequency count in the inventory
_vocab: list[str] = []
_word_freq: dict[str, int] = {}

# Units and noise tokens to strip
NOISE_TOKENS = {
    "oz", "lb", "lbs", "kg", "g", "ml", "l", "ct", "pk", "pkg",
    "ea", "each", "pc", "pcs", "piece", "pieces", "unit", "units",
}


# ---------------------------------------------------------------------------
# 1. Normalization Layer
# ---------------------------------------------------------------------------

def _looks_like_abbreviation(token: str) -> bool:
    """
    Heuristic: a token is likely an abbreviation if it is short (≤ 5 chars),
    alphabetic, and has at most one vowel.
    """
    if not token.isalpha():
        return False
    if len(token) > 5:
        return False
    vowels = sum(1 for c in token if c in "aeiou")
    return vowels <= 1


def build_vocab_index(inventory: pd.DataFrame) -> None:
    """
    Build a frequency-weighted vocabulary from the inventory.

    Only proper full words are kept (≥ 6 chars, alphabetic, ≥ 2 vowels).
    Words that appear fewer than 3 times are discarded — they are likely
    product-name noise, not genuine food vocabulary.

    The frequency count is used during expansion so common words like
    'chicken' (thousands of occurrences) always beat rare exotics like
    'chikoo' (1–2 occurrences) even when both pass the skeleton test.
    """
    global _vocab, _word_freq
    from collections import Counter
    import math

    freq: Counter = Counter()
    for name in inventory["ingredient"]:
        for word in str(name).split():
            if (len(word) >= 6 and word.isalpha()
                    and sum(1 for c in word if c in "aeiou") >= 2):
                freq[word] += 1

    # Keep only words that appear at least 3 times (filter out product-name noise)
    _word_freq = {w: c for w, c in freq.items() if c >= 3}
    _vocab = sorted(_word_freq)
    print(f"  Vocabulary index built: {len(_vocab):,} words (freq ≥ 3) learned from inventory.")


def _consonant_skeleton(word: str) -> str:
    """Strip vowels, returning only the consonant skeleton. e.g. 'chicken' → 'chckn'."""
    return "".join(c for c in word if c not in "aeiou")


def _skeleton_score(abbrev: str, word: str) -> float:
    """
    Score how well `abbrev` could abbreviate `word` using consonant skeleton
    subsequence matching.

    Steps:
      1. Reduce both to their consonant skeletons.
      2. Check that every consonant of the abbreviation appears in order
         inside the word's skeleton (subsequence check).
      3. Score = fraction of the word's skeleton covered by the abbreviation.
         Higher coverage → more confident match.
         Prefer shorter words (same coverage = shorter word wins).

    Returns 0.0 if the abbreviation is not a valid consonant subsequence.
    """
    abbrev_skel = _consonant_skeleton(abbrev)
    word_skel   = _consonant_skeleton(word)

    if not abbrev_skel or not word_skel:
        return 0.0

    # Subsequence check on consonant skeletons
    it = iter(word_skel)
    matched = sum(1 for c in abbrev_skel if c in it)
    if matched < len(abbrev_skel):
        return 0.0  # abbreviation consonants don't appear in order → not a match

    return len(abbrev_skel) / len(word_skel)  # higher = word skeleton is more covered


def _expand_token(token: str) -> str:
    """
    Self-learn abbreviation expansion from the inventory vocabulary.

    Algorithm:
      1. Pre-filter vocab to words starting with the same letter (O(n) → O(k)).
      2. Score each candidate via consonant skeleton subsequence matching.
      3. Pick the highest-coverage match; among ties, prefer the shortest word.
      4. Require coverage ≥ 0.40 to avoid spurious matches.
      5. Fall back to fuzzy ratio if no skeleton candidate qualifies.

    Results are cached so each token is resolved only once per session.

    Examples learned from the inventory automatically (no seed dict):
      chk  → chicken    (consonants: chk ⊆ chckn, chicken appears 1000s of times)
      brst → breast     (brst == brst, perfect skeleton match)
      bnls → boneless   (bnls ⊆ bnlss)
      frzn → frozen     (frzn == frzn, perfect)
      grnd → ground     (grnd == grnd, perfect)
      moz  → mozzarella (mz ⊆ mzzrll, very frequent in inventory)
      grl  → grilled    (grl ⊆ grlld, frequent)
    """
    if token in _expansion_cache:
        return _expansion_cache[token]

    if not _vocab:
        _expansion_cache[token] = token
        return token

    import math

    # Pre-filter: same starting letter
    candidates = [w for w in _vocab if w[0] == token[0]]

    # Combined score: skeleton coverage × log(frequency)
    # Frequency weighting ensures common food words beat obscure product names
    scored = []
    for w in candidates:
        skel = _skeleton_score(token, w)
        if skel >= 0.40:
            freq_bonus = math.log1p(_word_freq.get(w, 1))
            scored.append((w, skel * freq_bonus))

    if scored:
        best_word = max(scored, key=lambda x: x[1])[0]
    else:
        # Fuzzy fallback when skeleton matching yields nothing
        from rapidfuzz import process, fuzz as rfuzz
        result = process.extractOne(
            token, _vocab,
            scorer=rfuzz.token_sort_ratio,
            score_cutoff=VOCAB_EXPAND_THRESHOLD,
        )
        best_word = result[0] if result else token

    _expansion_cache[token] = best_word
    return best_word


def normalize_text(text: str, trace: bool = False) -> str:
    """
    Normalize a raw OCR/VLM receipt string:
      1. Lowercase + remove punctuation
      2. Strip unit/noise tokens
      3. For each remaining token:
           - if in seed ABBREV_DICT → apply mapping
           - elif looks like an abbreviation → auto-expand via vocabulary matching
           - else keep as-is
    """
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    tokens = text.split()
    expanded = []
    for token in tokens:
        if token in NOISE_TOKENS:
            continue

        # Seed dict (unit stripping)
        if token in ABBREV_DICT:
            replacement = ABBREV_DICT[token]
            if replacement:
                expanded.append(replacement)
            continue

        # Auto-expand unknown abbreviations via learned vocabulary
        if _looks_like_abbreviation(token):
            resolved = _expand_token(token)
            if trace and resolved != token:
                print(f"    [vocab expand] '{token}' → '{resolved}'")
            expanded.append(resolved)
        else:
            expanded.append(token)

    return " ".join(expanded).strip()


def smart_normalize(text: str, trace: bool = False) -> tuple[str, bool]:
    """
    Normalize the text. Always runs basic cleanup; auto-expansion fires
    only for tokens that look like abbreviations.
    Returns (normalized_text, was_changed).
    """
    normalized = normalize_text(text, trace=trace)
    return normalized, normalized != text.lower().strip()


# ---------------------------------------------------------------------------
# 2. Canonical Inventory & TF-IDF Index
# ---------------------------------------------------------------------------

def load_inventory(path: str) -> pd.DataFrame:
    """Load and deduplicate the labeled dataset."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["ingredient"] = df["ingredient"].str.lower().str.strip()
    df = df.dropna(subset=["ingredient", "predicted_label"])
    df = df.drop_duplicates(subset=["ingredient"])
    return df.reset_index(drop=True)


def build_tfidf_index(inventory: pd.DataFrame):
    """Fit a TF-IDF vectorizer over the canonical ingredient names."""
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",   # character n-grams handle OCR noise better
        ngram_range=(2, 4),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(inventory["ingredient"])
    return vectorizer, matrix


# ---------------------------------------------------------------------------
# 3. Hybrid Similarity Matching
# ---------------------------------------------------------------------------

def hybrid_similarity(
    query: str,
    inventory: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
    top_k: int = 5,
) -> list[dict]:
    """
    Compute a weighted hybrid score for each canonical item:
      hybrid = TFIDF_WEIGHT * tfidf_score + FUZZY_WEIGHT * fuzzy_score

    Returns a ranked list of the top_k candidates.
    """
    # TF-IDF cosine similarity
    query_vec = vectorizer.transform([query])
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Fuzzy token-sort ratio (normalised to [0, 1])
    fuzzy_scores = [
        fuzz.token_sort_ratio(query, item) / 100.0
        for item in inventory["ingredient"]
    ]

    hybrid_scores = [
        TFIDF_WEIGHT * c + FUZZY_WEIGHT * f
        for c, f in zip(cosine_scores, fuzzy_scores)
    ]

    # Build sorted candidate list
    indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)
    candidates = []
    for idx in indices[:top_k]:
        candidates.append({
            "item": inventory.iloc[idx]["ingredient"],
            "label": inventory.iloc[idx]["predicted_label"],
            "tfidf_score": round(float(cosine_scores[idx]), 4),
            "fuzzy_score": round(float(fuzzy_scores[idx]), 4),
            "hybrid_score": round(float(hybrid_scores[idx]), 4),
        })
    return candidates


def decide(score: float) -> str:
    """Map a hybrid score to a routing decision."""
    if score > ACCEPT_THRESHOLD:
        return "accept"
    else:
        return "fallback_llm"


def run_similarity_match(
    normalized: str,
    inventory: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
) -> dict:
    """Run hybrid match and return a structured result dict."""
    candidates = hybrid_similarity(normalized, inventory, vectorizer, tfidf_matrix)
    best = candidates[0]
    return {
        "normalized": normalized,
        "matched_item": best["item"],
        "similarity_label": best["label"],
        "hybrid_score": best["hybrid_score"],
        "tfidf_score": best["tfidf_score"],
        "fuzzy_score": best["fuzzy_score"],
        "decision": decide(best["hybrid_score"]),
        "top_candidates": candidates,
    }


# ---------------------------------------------------------------------------
# 4. LLM Fallback
# ---------------------------------------------------------------------------

_anthropic_client = None

def get_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    return _anthropic_client


def llm_fallback(
    original: str,
    normalized: str,
    match_result: dict,
) -> dict:
    """
    Call Claude to classify the item when similarity confidence is insufficient.
    Returns {"final_label": ..., "confidence": ..., "reason": ...}
    """
    top_5 = match_result["top_candidates"][:5]
    candidates_text = "\n".join(
        f"  {i+1}. \"{c['item']}\" → {c['label']} (score: {c['hybrid_score']})"
        for i, c in enumerate(top_5)
    )

    prompt = f"""You are a grocery/food item categorization expert for a restaurant supply system.

A receipt scanner returned this text:
  Original OCR text  : "{original}"
  Normalized text    : "{normalized}"
  Best similarity match: "{match_result['matched_item']}" (label: {match_result['similarity_label']}, score: {match_result['hybrid_score']})

Top 5 candidate matches from the inventory:
{candidates_text}

Allowed category labels:
  {", ".join(ALLOWED_LABELS)}

Based on the above, determine the most appropriate category for this item.

Respond with ONLY valid JSON (no markdown, no explanation outside the JSON):
{{
  "final_label": "<one of the allowed labels>",
  "confidence": "<low|medium|high>",
  "reason": "<one sentence explanation>"
}}"""

    client = get_client()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        # Validate the label
        if result.get("final_label") not in ALLOWED_LABELS:
            result["final_label"] = match_result["similarity_label"]  # safe fallback
            result["confidence"] = "low"
        return result
    except json.JSONDecodeError:
        return {
            "final_label": match_result["similarity_label"],
            "confidence": "low",
            "reason": f"LLM response parse error. Fell back to similarity label. Raw: {raw[:100]}",
        }


# ---------------------------------------------------------------------------
# 5. Final Pipeline Function
# ---------------------------------------------------------------------------

def categorize_receipt_item_with_fallback(
    item_text: str,
    inventory: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
) -> dict:
    """
    End-to-end pipeline for a single receipt item.

    Steps:
      1. Detect whether normalization is needed and normalize.
      2. Run hybrid similarity match.
      3. If score > 0.5, accept the similarity label directly.
      4. If score ≤ 0.5, call LLM fallback.
      5. Return a unified result dict.
    """
    original = item_text.strip()

    # Step 1 — Normalize
    normalized, was_normalized = smart_normalize(original)

    # Step 2 — Hybrid similarity match
    match = run_similarity_match(normalized, inventory, vectorizer, tfidf_matrix)

    result = {
        "input": original,
        "normalized": normalized,
        "was_normalized": was_normalized,
        "matched_item": match["matched_item"],
        "similarity_label": match["similarity_label"],
        "hybrid_score": match["hybrid_score"],
        "tfidf_score": match["tfidf_score"],
        "fuzzy_score": match["fuzzy_score"],
        "decision": match["decision"],
        "top_candidates": match["top_candidates"],
    }

    # Step 3 / 4 — Route based on decision
    if match["decision"] == "accept":
        result["final_label"] = match["similarity_label"]
        result["final_source"] = "similarity_pipeline"
        result["llm_response"] = None
    else:
        # Call LLM for review or fallback_llm
        llm_result = llm_fallback(original, normalized, match)
        result["final_label"] = llm_result["final_label"]
        result["final_source"] = "llm_fallback"
        result["llm_response"] = llm_result

    return result


# ---------------------------------------------------------------------------
# 6. Pretty Printer
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    """Pretty-print a pipeline result."""
    print("\n" + "=" * 60)
    print(f"  Input          : {result['input']}")
    print(f"  Normalized     : {result['normalized']}")
    print(f"  Matched item   : {result['matched_item']}")
    print(f"  Similarity lbl : {result['similarity_label']}")
    print(f"  Hybrid score   : {result['hybrid_score']:.4f}  "
          f"(TF-IDF={result['tfidf_score']:.4f}, Fuzzy={result['fuzzy_score']:.4f})")
    print(f"  Decision       : {result['decision']}")
    print(f"  ─── Final ─────────────────────────────────")
    print(f"  Final label    : {result['final_label']}")
    print(f"  Final source   : {result['final_source']}")
    if result.get("llm_response"):
        llm = result["llm_response"]
        print(f"  LLM confidence : {llm.get('confidence', 'n/a')}")
        print(f"  LLM reason     : {llm.get('reason', 'n/a')}")
    print(f"\n  Top candidates :")
    for i, c in enumerate(result["top_candidates"][:5], 1):
        print(f"    {i}. {c['item']:<40} {c['label']:<25} score={c['hybrid_score']:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 7. Initialise pipeline (load data & build index once)
# ---------------------------------------------------------------------------

def init_pipeline(dataset_path: str = DATASET_PATH):
    """Load inventory, build TF-IDF index, and learn vocabulary. Call once at startup."""
    print(f"Loading inventory from '{dataset_path}'...")
    inventory = load_inventory(dataset_path)
    print(f"  {len(inventory):,} unique items loaded.")
    print("Building TF-IDF index (char n-grams 2–4)...")
    vectorizer, tfidf_matrix = build_tfidf_index(inventory)
    print("  Index ready.")
    print("Learning vocabulary from inventory for auto abbreviation expansion...")
    build_vocab_index(inventory)
    print()
    return inventory, vectorizer, tfidf_matrix


# ---------------------------------------------------------------------------
# 8. Interactive Testing Loop
# ---------------------------------------------------------------------------

def interactive_loop(inventory, vectorizer, tfidf_matrix):
    """
    Interactive REPL: type a receipt item and inspect the full pipeline result.
    Type 'quit' or 'exit' to stop.
    """
    print("\n─── BillWise Interactive Tester ───────────────────────")
    print("Type a receipt item text and press Enter.")
    print("Commands: 'quit' | 'exit' to stop, 'demo' to run sample items.\n")

    demo_items = [
        "chk brst bnls",
        "moz chz shred",
        "tmto pste",
        "olive oil extra virgin",
        "frzn grnd beef 80/20",
        "sparkling mineral water",
        "grl chkn sandwich",
    ]

    while True:
        try:
            text = input("Enter receipt item text: ").strip()
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
                result = categorize_receipt_item_with_fallback(
                    item, inventory, vectorizer, tfidf_matrix
                )
                print_result(result)
            continue

        result = categorize_receipt_item_with_fallback(
            text, inventory, vectorizer, tfidf_matrix
        )
        print_result(result)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY not set. LLM fallback calls will fail.")
        print("         Export it with: export ANTHROPIC_API_KEY='sk-ant-...'")
        print("         Similarity-only results (decision='accept') will still work.\n")

    inventory, vectorizer, tfidf_matrix = init_pipeline()
    interactive_loop(inventory, vectorizer, tfidf_matrix)
