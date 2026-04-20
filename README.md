# BillWise

BillWise is an end-to-end receipt intelligence system that supports receipt ingestion, preprocessing, information extraction, item categorization, dashboard analytics, GCS artifact storage, Twilio WhatsApp ingestion, and review-driven retraining exports.

---

## Core Capabilities

- **Hybrid receipt extraction**
  - Prototype: PaddleOCR + LayoutLMv3
  - VLM-assisted extraction
  - Hybrid field consolidation
- **Item categorization** using fine-tuned DistilBERT
- **Streamlit dashboard** for analytics and review
- **Twilio WhatsApp** receipt ingestion
- **GCS dual-write** for artifacts and exports
- **Review-driven retraining** feedback export
- **Optional preprocessing router** with variant selection

---

## Repository Structure

```text
BillWise/
├── apps/
│   ├── dashboard/
│   └── twilio_ingestion/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── reviewed/
│   ├── exports/
│   └── reference/
├── legacy/
│   ├── extraction_module/
│   └── categorization_module/
├── models/
│   └── categorization/
├── scripts/
├── src/
│   └── billwise/
├── tests/
└── README.md
```

---

## Main Pipeline

BillWise supports the following operational flow:

1. Receipt image arrives through local upload or Twilio WhatsApp ingestion
2. Optional preprocessing router evaluates image quality and generates variants
3. Hybrid extraction runs on the selected image
4. Extracted line items are categorized
5. Results are stored in SQLite and mirrored to GCS when enabled
6. Dashboard views are refreshed
7. Human validations can be exported as retraining feedback

---

## Environment Setup

### 1. Create and activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install BillWise

```bash
python -m pip install -e ".[extraction,categorization,dashboard,cloud,twilio_ingestion,dev]"
```

### 3. Additional OCR runtime note

If Paddle packages need manual reinstall on your machine:

```bash
pip install paddleocr paddlepaddle
```

---

## Required Local Assets

Place the following files before running the full system:

| Asset | Path |
|---|---|
| Categorization model | `models/categorization/distilbert_receipt_classifier.pt` |
| Categorization normalization CSV | `data/reference/merged_labeled.csv` |
| Legacy extraction module | `legacy/extraction_module/` |
| Legacy categorization module | `legacy/categorization_module/` |

---

## Environment Variables

Create a `.env` file in the repo root:

```env
GROQ_API_KEY=
GEMINI_API_KEY=

BILLWISE_STORAGE_BACKEND=hybrid
GCS_PROJECT_ID=
GCS_BUCKET_NAME=
GCS_PREFIX=billwise
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\gcp_credentials.json
GOOGLE_CLOUD_PROJECT=

PERFORM_PREPROCESSING=true

TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_NUMBER=
TWILIO_VERIFY_SIGNATURE=false
TWILIO_AUTO_EXPORT_VIEWS=true

PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

---

## Running the System

### Run tests

```bash
pytest
```

### Doctor check

```bash
python scripts\doctor.py --strict
```

### Run one receipt through the full pipeline

```bash
python scripts\run_billwise_pipeline.py --image "C:\path\to\receipt.jpg"
```

### Launch dashboard

```bash
python apps\dashboard\run_dashboard.py
```

### Run Twilio ingestion service

```bash
python scripts\run_twilio_ingestion.py
```

### Export dashboard views

```bash
python scripts\export_dashboard_views.py
```

### Export retraining feedback

```bash
python scripts\export_retraining_feedback.py
```

---

## Preprocessing Toggle

Preprocessing is controlled entirely through `.env`.

**Enable:**
```env
PERFORM_PREPROCESSING=true
```

**Disable:**
```env
PERFORM_PREPROCESSING=false
```

When enabled, BillWise:

- Assesses image quality
- Creates multiple variants
- Scores them with OCR-based heuristics
- Selects the best variant for extraction
- Stores preprocessing metadata locally and in GCS

---

## GCS Artifact Layout

When GCS hybrid mode is enabled, BillWise writes artifacts under a prefix like:

```text
billwise/
  raw/receipts/
  processed/receipts/
  preprocessing/receipts/
  pipeline_runs/YYYY/MM/DD/
  views/
  training/
```

---

## Dashboard Review Loop

The dashboard supports:

- Category validation
- OCR correction logging
- Analytics over receipts and line items
- **Ask BillWise** natural-language querying
- Retraining export generation from human review data

---

## Twilio WhatsApp Ingestion

Twilio ingestion accepts receipt images via WhatsApp sandbox or webhook-connected number.

**Flow:**

1. Twilio receives image
2. BillWise downloads media
3. Full receipt pipeline runs
4. Dashboard views auto-refresh
5. User receives a summary reply

---

## Retraining Feedback Exports

BillWise can export the following files:

- `category_feedback.jsonl`
- `category_feedback.csv`
- `ocr_feedback.jsonl`
- `ocr_feedback.csv`
- `manifest.json`

**Export destinations:**

- Local: `data/exports/retraining/<timestamp>/`
- GCS: `billwise/training/snapshots/<timestamp>/`
- GCS: `billwise/training/latest/`

---

## Security Notes

> **Do not commit** the following to version control:
> - `.env`
> - `gcp_credentials.json`
> - Any service account or secret files

Keep credentials outside the repo whenever possible.

---

## Current Status

BillWise currently includes:

- Hybrid extraction
- Categorization
- Dashboard
- Twilio ingestion
- GCS dual-write
- Preprocessing router
- Retraining feedback exports

---

## Future Improvements

- Stronger production deployment setup
- Automatic categorizer retraining job
- Extraction fine-tuning pipeline from reviewed OCR corrections
- Preprocessing analytics in dashboard