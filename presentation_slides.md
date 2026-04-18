# BillWise — Presentation Slides Content

---

## Slide 1 — Title

**BillWise: AI-Powered Receipt Digitization & Grocery Intelligence**

- Team: [Your Team Name]
- Members: [Member Names]
- Course: AI Capstone Project

---

## Slide 2 — The Problem

**Managing grocery spending is tedious and opaque.**

- Paper receipts are physical, unstructured, and easily lost
- No easy way to track *what* was bought, *where*, and *how much* was spent over time
- Manually entering receipt data is time-consuming
- Existing apps require dedicated installs or manual photo uploads
- OCR alone gives raw text — not meaningful, queryable data

> *"You spent $312 last month, but on what? You have no idea."*

---

## Slide 3 — Goal

**Turn any grocery receipt photo into structured, queryable spending data — instantly, via SMS.**

1. **Digitize** — extract structured fields (store, date, total, items, payment) from a receipt image
2. **Categorize** — classify each line item into one of 16 food categories automatically
3. **Query** — let users ask natural language questions about their spending history

No app to install. No manual entry. Just text a photo.

---

## Slide 4 — Why Is This Interesting?

- **Zero-friction UX:** SMS is universally available — works on any phone, no app install
- **End-to-end AI pipeline:** OCR → NLP normalization → DistilBERT classification → LLM querying, all chained together
- **Real-world noise:** Receipt text is messy (abbreviations like `chk brst bnls`, varying formats, OCR errors) — a genuine NLP challenge
- **Conversational analytics:** Users can ask *"How much did I spend on dairy last month?"* and get an instant answer

---

## Slide 5 — Why Is This Important?

- **Financial awareness:** Grocery spending is one of the top household expense categories; structured data enables better budgeting
- **Accessibility:** SMS-based access means even users without smartphones or data plans can participate
- **Scalability:** The system is designed for any user — from individuals to families tracking shared expenses
- **Generalizability:** The categorization pipeline works on any CSV-structured data, not just receipts

---

## Slide 6 — What Makes It Hard?

| Challenge | Why It's Difficult |
|---|---|
| **OCR noise** | Receipt printers use compressed fonts; text like `CHK BRST BNLS 5.99` has no vowels, no spaces |
| **Abbreviation expansion** | `chk` → `chicken`, `brst` → `breast` — no standard dictionary; must be learned from data |
| **Duplicate detection** | Same receipt texted twice should be rejected; pixel-level and semantic similarity both needed |
| **Schema-agnostic chatbot** | Chatbot must work with any CSV — column names, types, and ranges are unknown at design time |
| **Model generalization** | 16 categories, class imbalance, and short noisy text strings challenge standard classifiers |
| **Low-latency inference** | SMS users expect a reply in seconds; model loading and inference must be fast |

---

## Slide 7 — Human–AI Interaction

**Three interaction modes:**

1. **SMS Image → AI Processing**
   - User texts a receipt photo
   - AI runs OCR, extracts fields, categorizes items, deduplicates
   - User receives confirmation: `✅ Bill #3 | Walmart | $47.32`

2. **SMS Query → Natural Language Answer**
   - User texts: *"What did I spend last week?"*
   - Gemini generates SQL → DuckDB executes → formatted answer returned via SMS

3. **Web Chat UI**
   - Same query pipeline, richer interface at `/chat`
   - Multi-turn conversation with context history

**Confidence-based routing adds a human-in-the-loop:**
- High confidence (≥ 0.85) → auto-assigned categy
- Medium confidence (0.60–0.85) → LLM verificaortion
- Low confidence (< 0.60) → flagged for human review

---

## Slide 8 — Related Work

| Area | Related Work |
|---|---|
| **Receipt OCR** | Google Cloud Vision API, Tesseract OCR, AWS Textract |
| **NER on receipts** | Donut (document understanding transformer), LayoutLM |
| **Text classification** | BERT, RoBERTa, DistilBERT for short-text classification |
| **Parameter-efficient fine-tuning** | LoRA (Hu et al., 2022), adapter layers |
| **Conversational data querying** | Text-to-SQL (DAIL-SQL, SQLCoder), LLM-based NL2SQL |
| **Expense tracking apps** | Expensify, Splitwise, Mint — require manual input or dedicated apps |

**Key differentiator:** BillWise combines all layers (OCR → abbreviation expansion → neural classification → conversational query) into a single zero-friction SMS pipeline.

---

## Slide 9 — System Architecture

```
SMS Image
    │
    ▼
Google Cloud Vision (OCR)
    │  extract: store, date, total, card, [(item_name, price), ...]
    ▼
Abbreviation Normalization
    │  "chk brst bnls" → "chicken breast boneless"
    │  TF-IDF char n-gram + RapidFuzz hybrid scoring
    ▼
DistilBERT Classifier (fine-tuned, 87.2% accuracy)
    │  → "Poultry"
    ▼
GCS CSV (one row per item)
    │  Serial_No | Store | Date | Total | Item_Name | Item_Price | Grocery_Category
    ▼
Gemini 2.5 Flash + DuckDB
    │  Natural language → SQL → formatted answer
    ▼
SMS / Web Chat Reply
```

**Deduplication (2 layers):**
- Layer 1: MD5 image hash — exact duplicate image rejected
- Layer 2: Fuzzy store name (LCS ≥ 0.85) + exact date + exact total — same bill, different photo rejected

---

## Slide 10 — Categorization Pipeline (Detail)

**Two-stage approach:**

**Stage 1 — Normalization (`Abbreviation_Normalization.py`)**
- Strip units (oz, lb, kg) and noise tokens
- Consonant-skeleton abbreviation expansion (self-learned from 189k-item inventory):
  - `chk` → `chicken` &nbsp; `brst` → `breast` &nbsp; `frzn` → `frozen`
- Hybrid similarity: 50% TF-IDF char n-gram cosine + 50% RapidFuzz token-sort ratio
- Score > 0.50 → accept similarity label; ≤ 0.50 → Claude LLM fallback

**Stage 2 — Classification (`Categorization.py`)**
- Fine-tuned `distilbert-base-uncased` on 151k training items
- 16 canonical food categories
- Softmax confidence-based routing (auto / llm_verification / human_review)

---

## Slide 11 — Hypothesis & Metrics

**Hypotheses:**
1. Full fine-tuning of DistilBERT will outperform parameter-efficient methods (LoRA, linear probe) on short, noisy receipt text
2. Consonant-skeleton abbreviation expansion will reduce OOV item text errors without a hand-crafted dictionary
3. A schema-agnostic LLM + DuckDB pipeline can answer natural language queries over any receipt CSV without pre-defined query templates

**Success Metrics:**

| Metric | Target | Achieved |
|---|---|---|
| Test Accuracy | ≥ 85% | **87.19%** ✅ |
| Macro F1 | ≥ 0.85 | **0.8656** ✅ |
| Worst-class F1 | ≥ 0.70 | **0.7475** ✅ |
| Duplicate rejection | 100% exact | ✅ |
| SMS round-trip | < 10s | ✅ |

---

## Slide 12 — Approaches Explored

| Approach | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| **Full FT DistilBERT (unweighted)** | **87.19%** | **0.8656** | ✅ Chosen for production |
| Full FT DistilBERT (weighted) | 87.30% | 0.8646 | Higher acc, lower macro F1 |
| LoRA DistilBERT (unweighted) | 86.00% | 0.8527 | Fewer trainable params |
| LoRA DistilBERT (weighted) | 85.59% | 0.8462 | |
| LoRA RoBERTa (unweighted) | 85.25% | 0.8478 | Larger model, similar results |
| Linear Probe DistilBERT | 68.16% | 0.6756 | Frozen encoder, poor performance |
| Linear Probe RoBERTa | 63.03% | 0.6062 | |

**Why Full FT Unweighted?**
- Best balance of accuracy and macro F1
- Most consistent per-class performance
- Weighted approach helps minority classes but hurts majority-class precision

---

## Slide 13 — Training Details

**Dataset:** 189,115 labeled receipt items across 16 food categories

| Split | Size |
|---|---|
| Train | 151,112 |
| Validation | 18,889 |
| Test | 18,890 |

**Key hyperparameters (production model):**
- Base model: `distilbert-base-uncased`
- Optimizer: AdamW with layer-wise LR (1e-5 lower, 3e-5 upper)
- Best epoch: 9 (early stopping)
- Max sequence length: 64 tokens

**Training curve:** Loss converged at epoch 9; continued training past epoch 9 shows overfitting (val loss increases while train loss keeps decreasing).

---

## Slide 14 — Per-Class Results

| Category | F1 | Category | F1 |
|---|---|---|---|
| Beverages | 0.9123 | Vegetables | 0.8677 |
| Seafood | 0.9110 | Bakery & Flour | 0.8755 |
| Oils & Fats | 0.9017 | Meat | 0.8763 |
| Dairy | 0.8988 | Poultry | 0.8371 |
| Snacks & Ready-to-Eat | 0.8929 | Spices & Seasonings | 0.8456 |
| Sauces & Condiments | 0.8880 | Herbs | 0.8213 |
| Grains & Staples | 0.8639 | **Frozen / Processed** | **0.7475** |
| Fruits | 0.8550 | Pulses & Beans | 0.8548 |

**Weakest class:** Frozen / Processed (0.7475) — likely due to class imbalance and overlap with other categories (e.g., frozen vegetables, frozen meat).

---

## Slide 15 — Data & Tools

| Component | Tool / Source |
|---|---|
| OCR | Google Cloud Vision API |
| SMS gateway | Twilio |
| Storage | Google Cloud Storage (CSV) |
| Normalization | scikit-learn TF-IDF, RapidFuzz |
| Classification model | HuggingFace Transformers (DistilBERT) |
| LLM fallback | Anthropic Claude API |
| Chatbot LLM | Google Gemini 2.5 Flash |
| Query engine | DuckDB (in-memory SQL) |
| Web framework | Flask + Gunicorn |
| Training data | 189k labeled grocery items |
| Deployment | Docker, Google Cloud Run |

---

## Slide 16 — Evaluation

**Quantitative:**
- Held-out test set: 18,890 items
- Metrics: accuracy, macro F1, per-class F1, confusion matrix
- Compared 7 model variants across 2 architectures (DistilBERT, RoBERTa) and 3 training strategies (full FT, LoRA, linear probe)

**Qualitative:**
- End-to-end SMS test: send real receipt photo → verify extracted fields + categories in GCS CSV
- Chatbot: ask natural language spending questions → verify SQL correctness and answer accuracy
- Abbreviation expansion: test known receipt abbreviations (`chk brst`, `frzn grnd beef`) → verify expansion

**Ablation insight:**
- Linear probe (frozen encoder) achieves only 68.16% — confirming that task-specific fine-tuning of all layers is essential for noisy short-text classification
- LoRA vs full FT: 1.19% accuracy gap justifies full fine-tuning for this task

---

## Slide 17 — Results Summary & Takeaways

**What worked well:**
- Full fine-tuning DistilBERT on grocery text achieves 87.19% accuracy with strong per-class balance
- Consonant-skeleton abbreviation expansion correctly resolves common receipt abbreviations with no hand-crafted rules
- Schema-agnostic chatbot handles arbitrary CSV structures — generalizes beyond grocery data
- Two-layer deduplication robustly rejects both exact and near-duplicate receipts

**Challenges encountered:**
- Class imbalance for Frozen / Processed category (lowest F1: 0.7475)
- Windows encoding issues with Unicode characters in torch/transformers logs
- Large model checkpoint (256MB) increases app startup time

**Future work:**
- Replace similarity pipeline with DistilBERT for all items (currently used for preprocessing only)
- Add spending visualizations to the web UI
- Migrate Gemini client to `google.genai` (current `google.generativeai` is deprecated)
- Export to Google Sheets integration

---
