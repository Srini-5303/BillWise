# BillWise — Receipt Item Categorization

A two-stage ML pipeline that maps raw, noisy receipt item text (e.g. `"chk brst bnls"`) to one of 16 canonical food categories (e.g. `Poultry`). Built on top of a fine-tuned DistilBERT classifier backed by an intelligent normalization layer.

---

## Architecture

```
Raw OCR / receipt text
        │
        ▼
┌─────────────────────────────────────────────────┐
│       Abbreviation_Normalization.py             │
│                                                 │
│  1. Lowercase + strip punctuation               │
│  2. Remove unit/noise tokens (oz, lb, kg …)     │
│  3. Consonant-skeleton abbreviation expansion   │
│     chk → chicken   brst → breast   frzn → frozen│
│  4. TF-IDF char n-gram cosine similarity        │
│     + RapidFuzz token-sort ratio (50/50 hybrid) │
│  5. Score > 0.50 → accept similarity label      │
│     Score ≤ 0.50 → Claude LLM fallback          │
└─────────────────────────────────────────────────┘
        │
        ▼  (normalized text)
┌─────────────────────────────────────────────────┐
│           Categorization.py                     │
│                                                 │
│  1. Detect unresolved abbreviations             │
│  2. DistilBERT forward pass + softmax           │
│  3. Route by confidence tier:                   │
│     ≥ 0.85 → auto_assign                        │
│     ≥ 0.60 → llm_verification                   │
│     < 0.60 → human_review                       │
└─────────────────────────────────────────────────┘
```

---

## File Structure

```
AI Capstone/
├── src/
│   ├── Abbreviation_Normalization.py   # Normalization + similarity pipeline
│   ├── Categorization.py               # DistilBERT inference wrapper
│   └── Categorization_DataPrepTrain Final.ipynb  # Training notebook
│
├── data/
│   └── Processed_Datasets/
│       ├── Labeled/
│       │   └── merged_labeled.csv      # Master labeled dataset (~189k items)
│       └── Splits/
│           ├── train.csv               # 151,112 items
│           ├── val.csv                 # 18,889 items
│           └── test.csv                # 18,890 items
│
├── checkpoints/
│   └── full_ft_distilbert_unweighted_best.pt   # Production model (~256 MB)
│
├── results/
│   ├── full_ft_distilbert_unweighted_results.json
│   ├── full_ft_distilbert_weighted_results.json
│   ├── lora_distilbert_results.json
│   ├── lora_distilbert_weighted_results.json
│   ├── lora_roberta_results.json
│   ├── linear_probe_distilbert_results.json
│   ├── linear_probe_roberta_results.json
│   └── *_confusion.png                # Confusion matrices for each approach
│
├── readme_classification.md
└── requirements_classification.txt
```

---

## Setup

**Python 3.12+ required.**

```bash
pip install -r requirements_classification.txt
```

For GPU training (CUDA 12.8), install PyTorch from the dedicated index:

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

**Environment variable (required for LLM fallback):**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

The normalization pipeline works without an API key. The Anthropic client is only called when similarity confidence falls below 0.50.

---

## Usage

### Stage 1 — Normalization + Similarity Pipeline

`Abbreviation_Normalization.py` can be run standalone:

```bash
python src/Abbreviation_Normalization.py
```

**Interactive REPL:**
```
Enter receipt item text: chk brst bnls
  Input          : chk brst bnls
  Normalized     : chicken breast boneless
  Matched item   : chicken breast boneless skinless
  Similarity lbl : Poultry
  Hybrid score   : 0.8812  (TF-IDF=0.8901, Fuzzy=0.8724)
  Decision       : accept
  ─── Final ─────────────────────────────────
  Final label    : Poultry
  Final source   : similarity_pipeline
```

Type `demo` to run the built-in batch of 7 challenging examples. Type `quit` to exit.

**Programmatic usage:**

```python
from src.Abbreviation_Normalization import init_pipeline, categorize_receipt_item_with_fallback

inventory, vectorizer, tfidf_matrix = init_pipeline("data/Processed_Datasets/merged_labeled.csv")

result = categorize_receipt_item_with_fallback(
    "frzn grnd beef 80/20",
    inventory, vectorizer, tfidf_matrix,
)
print(result["final_label"])    # → Meat
print(result["final_source"])   # → similarity_pipeline
```

### Stage 2 — DistilBERT Inference

`Categorization.py` imports the normalization layer and adds DistilBERT classification on top:

```bash
python src/Categorization.py
```

**Interactive REPL:**
```
Enter receipt item: olive oil extra virgin

════════════════════════════════════════════════
  INPUT          : olive oil extra virgin
  NORMALIZED     : olive oil extra virgin  (changed: False)
  ────────────────────────────────────────────
  ABBREVIATIONS  : False
  UNRESOLVED     : False
  ────────────────────────────────────────────
  PREDICTION     : Oils & Fats
  CONFIDENCE     : 0.9741
  ROUTING        : auto_assign
  ────────────────────────────────────────────
  TOP 3 SCORES   :
    1. Oils & Fats: 0.9741
    2. Sauces & Condiments: 0.0089
    3. Vegetables: 0.0043
════════════════════════════════════════════════
```

Type `demo` to run 8 sample items (clean, abbreviations, unresolvable, ambiguous). Type `quit` to exit.

**Programmatic usage:**

```python
from src.Abbreviation_Normalization import init_pipeline
from src.Categorization import load_classifier, run_inference

# Load once at startup
inventory, vectorizer, tfidf_matrix = init_pipeline(
    "data/Processed_Datasets/Labeled/merged_labeled.csv"
)
model, tokenizer, device = load_classifier(CONFIG)

result = run_inference(
    "moz chz shred",
    model, tokenizer, device,
    inventory, vectorizer, tfidf_matrix,
    CONFIG,
)
print(result["predicted_label"])   # → Dairy
print(result["confidence"])        # → 0.9312
print(result["routing"])           # → auto_assign
```

**Routing decisions:**

| Decision | Trigger | Action |
|---|---|---|
| `auto_assign` | confidence ≥ 0.85 | Send directly to output |
| `llm_verification` | 0.60 ≤ confidence < 0.85 | Queue for LLM review |
| `human_review` | confidence < 0.60 | Flag for manual check |
| `unresolved_abbreviation` | abbreviation could not be expanded | Log to `logs/unresolved_items.json` |

---

## Categories

The classifier maps every receipt item to one of **16 canonical food categories**:

| # | Category | # | Category |
|---|---|---|---|
| 1 | Bakery & Flour | 9 | Oils & Fats |
| 2 | Beverages | 10 | Poultry |
| 3 | Dairy | 11 | Pulses & Beans |
| 4 | Frozen / Processed | 12 | Sauces & Condiments |
| 5 | Fruits | 13 | Seafood |
| 6 | Grains & Staples | 14 | Snacks & Ready-to-Eat |
| 7 | Herbs | 15 | Spices & Seasonings |
| 8 | Meat | 16 | Vegetables |

---

## Dataset

| Split | Items | File |
|---|---|---|
| Train | 151,112 | `data/Processed_Datasets/Splits/train.csv` |
| Validation | 18,889 | `data/Processed_Datasets/Splits/val.csv` |
| Test | 18,890 | `data/Processed_Datasets/Splits/test.csv` |
| Labeled master | ~189k | `data/Processed_Datasets/Labeled/merged_labeled.csv` |

**Schema:** `ingredient`, `predicted_label`, `confidence`

The dataset covers 16 food categories derived from restaurant supply and grocery inventory sources. A class-balanced downsampled split (`train_downsampled.csv`) is also available for experiments that require balanced training.

---

## Model Training

Training and evaluation are documented in `src/Categorization_DataPrepTrain Final.ipynb`.

Three fine-tuning strategies were explored across two base models (DistilBERT and RoBERTa):

- **Full Fine-Tune** — all transformer weights updated, with and without inverse-frequency class weighting
- **LoRA** — low-rank adapter fine-tuning via PEFT, with and without class weighting
- **Linear Probe** — frozen backbone, only the classification head trained

**Hyperparameters (full fine-tune):**
- Optimizer: AdamW, learning rate 1e-5 → 3e-5 (layer-wise)
- Epochs: 12 (unweighted), 9 (weighted, early stopped)
- Max sequence length: 64 tokens
- Backbone: `distilbert-base-uncased`

---

## Results

| Approach | Base Model | Test Accuracy | Macro F1 | Best Epoch |
|---|---|---|---|---|
| **Full FT — Unweighted** | DistilBERT | **87.19%** | **0.8656** | 9 |
| Full FT — Weighted | DistilBERT | 87.30% | 0.8646 | 6 |
| LoRA — Unweighted | DistilBERT | 86.00% | 0.8527 | 14 |
| LoRA — Weighted | DistilBERT | 85.59% | 0.8462 | 15 |
| LoRA — Unweighted | RoBERTa | 85.25% | 0.8478 | 13 |
| Linear Probe | DistilBERT | 68.16% | 0.6756 | 14 |
| Linear Probe | RoBERTa | 63.03% | 0.6062 | 12 |

**Production model:** Full FT DistilBERT Unweighted — chosen for its combination of highest macro F1 (0.8656) and best worst-class performance on the imbalanced `Frozen / Processed` category.

**Per-class F1 (production model):**

| Category | F1 | Category | F1 |
|---|---|---|---|
| Beverages | 0.9123 | Snacks & Ready-to-Eat | 0.8929 |
| Seafood | 0.9110 | Sauces & Condiments | 0.8880 |
| Oils & Fats | 0.9017 | Bakery & Flour | 0.8755 |
| Dairy | 0.8988 | Meat | 0.8763 |
| Grains & Staples | 0.8639 | Poultry | 0.8371 |
| Fruits | 0.8550 | Spices & Seasonings | 0.8456 |
| Pulses & Beans | 0.8548 | Herbs | 0.8213 |
| Vegetables | 0.8677 | Frozen / Processed | 0.7475 |

Confusion matrices for all approaches are in `results/`.

---

## Normalization: Abbreviation Expansion

The normalization layer requires no hand-crafted abbreviation dictionary. It learns expansions automatically from the ingredient inventory at startup.

**Algorithm:**
1. Build a frequency-weighted vocabulary of all full words (≥ 6 chars, ≥ 2 vowels, appearing ≥ 3 times) from the inventory.
2. For each abbreviation token, reduce both the token and each vocabulary candidate to their **consonant skeletons** (strip vowels).
3. Check if the token's consonant sequence is a **subsequence** of the candidate's skeleton.
4. Score = skeleton coverage × log(word frequency). Pick the highest-scoring candidate.
5. Fall back to fuzzy ratio if skeleton matching fails.
6. Cache every resolved token so each is computed only once per session.

**Examples learned automatically:**

| Input token | Resolved to | Method |
|---|---|---|
| `chk` | `chicken` | skeleton `chk` ⊆ `chckn`, high frequency |
| `brst` | `breast` | perfect skeleton match `brst == brst` |
| `bnls` | `boneless` | skeleton `bnls` ⊆ `bnlss` |
| `frzn` | `frozen` | perfect skeleton match `frzn == frzn` |
| `grnd` | `ground` | perfect skeleton match `grnd == grnd` |
| `moz` | `mozzarella` | `mz` ⊆ `mzzrll`, very frequent |
| `grl` | `grilled` | `grl` ⊆ `grlld`, frequent |
