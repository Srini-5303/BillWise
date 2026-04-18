# BillWise Dashboard 🧾

> An AI-powered restaurant supply expense tracker — scan receipts via WhatsApp, explore spending analytics, and ask questions about your data in plain English.

BillWise is a two-part system:
- **Bill Bot** (`BillWise-main/`) — a Flask + Twilio SMS bot that receives receipt photos, runs Google Cloud Vision OCR, and appends structured data to a GCS CSV.
- **Dashboard** (`BillWise-Dashboard/`) — a Streamlit analytics app that reads that CSV, categorises every line item, and lets you explore, query, and validate your spending data.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Dashboard Pages](#dashboard-pages)
3. [Text-to-SQL Engine](#text-to-sql-engine)
4. [Human Validation Queue](#human-validation-queue)
5. [Getting Started](#getting-started)
6. [Environment Variables](#environment-variables)
7. [Project Structure](#project-structure)
8. [Running the Evaluation Suite](#running-the-evaluation-suite)
9. [Data Pipeline](#data-pipeline)
10. [Tech Stack](#tech-stack)

---

## System Architecture

```
                        ┌─────────────────────────────────┐
  WhatsApp / SMS        │         Bill Bot (Flask)         │
  receipt photo  ──────▶│  Twilio → Cloud Vision OCR       │
                        │  → extract items, total, date    │
                        │  → dedup → append to GCS CSV     │
                        └──────────────┬──────────────────┘
                                       │  bills_output.csv
                                       ▼
                        ┌─────────────────────────────────┐
                        │     BillWise Dashboard           │
                        │     (Streamlit)                  │
                        │                                  │
                        │  data_loader  ──▶  3 DataFrames  │
                        │  analytics    ──▶  KPIs / charts │
                        │  text_to_sql  ──▶  NL → SQL      │
                        │  validation   ──▶  human review  │
                        └─────────────────────────────────┘
```

**Data source priority** (dashboard auto-detects):
1. Google Cloud Storage (`GCS_BUCKET_NAME` set) — live receipt data
2. Local `bills_output.csv` — offline copy
3. Seed demo data — always works, no credentials needed

---

## Dashboard Pages

### 1. 📊 Overview
- **KPI strip** — Total Spend, Receipt Count, Avg Basket, Unique Vendors (vs. prior period %)
- **Weekly spend trend** — Plotly line chart with 7-day rolling average
- **Recent receipts** — last 5 receipts with vendor, date, total, and confidence badge
- **High-spend alerts** — receipts that exceed 2× the rolling average
- **Anomaly detector** — flags items priced far outside their category norm

### 2. 🏷️ Categories
- Spend and quantity breakdown by the 17 grocery/supply categories
- Donut chart + ranked bar chart side-by-side
- Monthly category heatmap (row = category, column = month, cell = spend)
- Full category summary table with sortable columns

### 3. 🏪 Vendors
- Top vendors by total spend (bar chart + trend lines)
- Visit frequency vs. average spend-per-visit scatter
- Vendor month-over-month trend
- Searchable vendor table

### 4. 📦 Items
- Top items by total spend and by quantity purchased
- Monthly item price trend (useful for spotting supplier price changes)
- Raw OCR text vs. normalised canonical name comparison table

### 5. 🔍 Receipt Explorer
- Filter by vendor, date range, and spend threshold
- One-click drilldown into every line item on a receipt
- Duplicate receipt detector (flags similar store + date + total combos)
- Outlier spend flagging per receipt

### 6. 💬 Ask BillWise
Natural language query interface powered by Gemini 2.5 Flash.

```
You: "How much did we spend on dairy items last month?"

→ Gemini writes DuckDB SQL
→ DuckDB executes against in-memory DataFrames
→ Answer + chart returned in < 2 seconds
```

See [Text-to-SQL Engine](#text-to-sql-engine) for full details.

### 7. 🚨 Human Validation
Two-tier review queue for low-confidence categorisations and OCR errors.  
See [Human Validation Queue](#human-validation-queue) for full details.

---

## Text-to-SQL Engine

`text_to_sql.py` converts plain English questions into DuckDB SQL with three quality layers:

### Layer 1 — Few-Shot Prompting
Six hand-written Q→SQL pairs are injected into every prompt, covering the most common patterns (SUM, GROUP BY, WHERE filter, COUNT DISTINCT, date ranges, TOP-N). This alone improves Execution Accuracy by ~10% on BillWise-style queries.

### Layer 2 — Self-Correction Loop
If the generated SQL fails to execute, the error message is sent back to Gemini with the broken SQL and a request to fix it. Up to 3 retries per question.

```
Question → Gemini → SQL → DuckDB
                            │ error?
                            ▼
                    Gemini (fix it) → SQL → DuckDB  (repeat ≤ 3×)
```

### Layer 3 — LLM-as-Judge (optional, eval only)
`eval_text_to_sql.py --judge` uses Gemini 2.5 Flash Lite to score each answer on:
- **Correctness** (0–10) — does the answer match the question?
- **Faithfulness** (0–10) — is every claim supported by the data?
- **SQL Quality** (0–10) — is the SQL clean, readable, and efficient?

### Schema Context
The prompt includes an explicit column listing for all three tables (`receipts`, `line_items`, `joined`) with dtypes and sample values, minimising column hallucination.

### Database Tables (in-memory DuckDB)

| Table | Grain | Key columns |
|-------|-------|-------------|
| `receipts` | 1 row per receipt | `receipt_id`, `vendor_name`, `receipt_date`, `receipt_total`, `tax`, `card_used` |
| `line_items` | 1 row per item | `receipt_id`, `normalized_item_text`, `category`, `quantity`, `unit_price`, `line_total` |
| `joined` | line_items LEFT JOIN receipts | all columns from both — use this for most questions |

### Running the engine standalone

```bash
# Default 5 sample questions
python text_to_sql.py

# Custom questions as CLI args
python text_to_sql.py "How much did we spend on dairy?" "Top 3 vendors?"
```

---

## Human Validation Queue

`app.py` page 7 provides a structured review workflow for keeping categorisation accurate as receipt volume grows.

### Two confidence tiers

| Tier | Threshold | Badge |
|------|-----------|-------|
| **Urgent Review** | confidence < 60% | 🔴 red |
| **Needs Review** | 60% ≤ confidence < 75% | 🟡 yellow |
| **OK** | ≥ 75% | ✅ green |

### Per-item card
Each flagged item shows:
- Raw OCR text + current category
- Top-3 predicted categories with colour-coded confidence bars
- Dropdown to select the correct category
- Optional note field
- **Confirm** button — writes to `billwise.db → human_validations` table

### OCR Issues tab
Flags receipts with:
- Missing vendor name
- Zero or missing total
- Low extraction confidence (< 70%)
- Missing receipt date

### Validation Log
Full audit trail of every correction with timestamp, original category, and corrected category.

---

## Getting Started

### Prerequisites
- Python 3.9+
- A Google Gemini API key ([get one free](https://aistudio.google.com/app/apikey))
- (Optional) Google Cloud Storage bucket with `bills_output.csv`
- (Optional) GCP Application Default Credentials for GCS access

### 1. Clone and set up the virtual environment

```bash
git clone <repo-url>
cd BillWise-Dashboard

python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env   # or create .env manually
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here

# Optional — leave blank to use demo seed data
GCS_BUCKET_NAME=your-gcs-bucket-name
GCS_BILLS_BLOB=bills_output.csv
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar shows which data source is active (GCS / local CSV / demo).

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for Ask BillWise and the eval judge |
| `GCS_BUCKET_NAME` | No | GCS bucket name; leave blank to use local/demo data |
| `GCS_BILLS_BLOB` | No | GCS object name (default: `bills_output.csv`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | No | Path to GCP service account JSON (or use ADC) |

---

## Project Structure

```
BillWise-Dashboard/
├── app.py                  # Streamlit app — 7 pages, sidebar navigation, CSS
├── analytics.py            # Pure-pandas analytics functions (no LLM calls)
├── charts.py               # Plotly figure builders (shared colour palette)
├── data_loader.py          # GCS / local CSV / seed data → DataFrames + SQLite
├── text_to_sql.py          # Text-to-SQL engine (few-shot + self-correction)
├── eval_text_to_sql.py     # Evaluation harness (EX / answer accuracy / LLM judge)
├── utils.py                # Category keywords, item normalisation, date helpers
├── requirements.txt        # Python dependencies
├── .env                    # API keys (git-ignored)
└── billwise.db             # SQLite — validation history, OCR corrections
```

### Module responsibilities

```
data_loader.py
  └── load_all_data()       → (df_receipts, df_items, df_joined)
        ├── _load_from_gcs()          GCS path
        ├── _load_from_local_csv()    local path
        └── _seed_demo_data()         fallback

analytics.py
  ├── get_kpis()            → dict of KPI values + deltas
  ├── get_spend_trend()     → df ready for line chart
  ├── get_category_spend()  → df for bar / donut
  ├── get_flagged_items()   → (df_urgent, df_review)  ← Human Validation
  └── get_ocr_issues()      → df of problematic receipts

text_to_sql.py
  ├── build_schema_context()  → prompt string with schema + few-shot examples
  ├── generate_sql()          → (sql, raw_response)
  ├── correct_sql()           → fixed sql string
  ├── run_query()             → pd.DataFrame  (safe DuckDB execution)
  ├── format_result()         → (answer_text, chart)
  └── ask_billwise()          → SQLResult  ← called by app.py page 6
```

---

## Running the Evaluation Suite

`eval_text_to_sql.py` contains 21 test cases across 7 categories. Ground-truth answers are computed from pandas — no external oracle needed.

```bash
# Standard run (Execution Accuracy + Answer Accuracy)
python eval_text_to_sql.py

# Add LLM-as-judge scores (uses extra Gemini API calls)
python eval_text_to_sql.py --judge

# Self-consistency test — run each question 3 times
python eval_text_to_sql.py --runs 3

# Single category only
python eval_text_to_sql.py --category top_n

# Summary only (no per-test detail)
python eval_text_to_sql.py --quiet
```

### Test categories

| Category | Tests | What it checks |
|----------|-------|----------------|
| `aggregation` | 5 | SUM, COUNT, AVG across tables |
| `grouping` | 4 | GROUP BY + ORDER BY + LIMIT |
| `filtering` | 2 | WHERE clauses on category and date |
| `top_n` | 2 | TOP-N with correct ordering |
| `hallucination_trap` | 3 | Questions that could tempt the LLM to invent columns |
| `edge_case` | 2 | Threshold filters, single-value aggregations |
| `self_correction_stress` | 3 | Paraphrased questions testing robustness |

### Latest benchmark results (Gemini 2.5 Flash)

| Metric | Score |
|--------|-------|
| Execution Accuracy | **100%** |
| Answer Accuracy | **100%** |
| Hallucination Rate | **0%** |
| Self-correction needed | **0%** |
| Avg latency | **1.26 s / query** |

---

## Data Pipeline

### Bill Bot → GCS (BillWise-main)

```
SMS image received by Twilio
  → /webhook in app.py
  → Google Cloud Vision OCR
  → ocr_pipeline.py extracts:
      Store name, date, total, card, line items
  → csv_writer.py deduplicates:
      Layer 1: MD5 image hash
      Layer 2: LCS fuzzy match (store + date + total)
  → Appends row to GCS: bills_output.csv
```

### GCS CSV → Dashboard DataFrames

```
data_loader._load_from_gcs()
  → downloads bills_output.csv from GCS
  → auto-detects layout:
      per-item  (Item_Name column present)  ← current format
      per-receipt (legacy)
  → _gcs_per_item_layout():
      groups by (Serial_No, Bill_File) → unique receipts
      reads Grocery_Category from GCS if present (confidence 0.90)
      falls back to utils.categorize_item() if missing
  → returns (df_receipts, df_items, df_joined)
```

### GCS CSV Schema

| Column | Description |
|--------|-------------|
| `Serial_No` | Unique receipt sequence number |
| `Bill_File` | Original filename of the receipt image |
| `Store_Name` | Vendor / store name from OCR |
| `Invoice_Date` | Receipt date |
| `Total` | Receipt total |
| `Card_Used` | Payment card type + last 4 digits |
| `Item_Name` | Parsed line item description |
| `Item_Price` | Line item price |
| `Grocery_Category` | Category assigned by Bill Bot |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Dashboard framework | [Streamlit](https://streamlit.io/) |
| Data processing | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| SQL engine (in-memory) | [DuckDB](https://duckdb.org/) |
| Visualisations | [Plotly](https://plotly.com/python/) |
| LLM (Text-to-SQL + judge) | [Google Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) via `google-genai` |
| Cloud storage | [Google Cloud Storage](https://cloud.google.com/storage) |
| SMS / WhatsApp bot | [Flask](https://flask.palletsprojects.com/) + [Twilio](https://www.twilio.com/) |
| OCR | [Google Cloud Vision API](https://cloud.google.com/vision) |
| Validation persistence | SQLite (`billwise.db`) |
| Category classification | Keyword-scoring (`utils.py`) with 17 categories |

---

## Acknowledgements

- **Spider benchmark** (Yu et al. 2018) — defines Execution Accuracy as the primary Text-to-SQL metric
- **DIN-SQL** (Pourreza & Rafiei, NeurIPS 2023) — self-correction loop methodology
- **G-Eval** (Liu et al., EMNLP 2023) — LLM-as-judge evaluation framework
