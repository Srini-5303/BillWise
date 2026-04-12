# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Bill Bot** — an SMS-based receipt digitization system. Users text receipt images to a Twilio number; the bot runs Google Cloud Vision OCR, extracts structured data (store, date, total, card, line items), deduplicates, and appends to a CSV in Google Cloud Storage. Users can also text questions to query their receipt history via a Gemini-backed chatbot.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (development)
flask run --port 8080

# Run with gunicorn (production)
gunicorn -w 2 -t 120 -b 0.0.0.0:8080 app:app

# Build and run via Docker
docker build -t bill-bot .
docker run -p 8080:8080 --env-file .env bill-bot
```

## Testing OCR

There is no automated test suite. The only test script is `vision_test.py` — place images in `raw_images/` and run it directly:

```bash
python vision_test.py
```

## Required Environment Variables

See `env.yaml` for the full template. Google Cloud credentials must also be available via `GOOGLE_APPLICATION_CREDENTIALS` or Application Default Credentials.

| Variable | Purpose |
|---|---|
| `TWILIO_ACCOUNT_SID` | Twilio account identifier |
| `TWILIO_AUTH_TOKEN` | Twilio authentication token |
| `GCS_BUCKET_NAME` | Google Cloud Storage bucket for CSV |
| `GEMINI_API_KEY` | Google Gemini API key for the chatbot |
| `GCS_BILLS_BLOB` | GCS blob name for the bills CSV (default: `bills_output.csv`) |
| `CATEGORIZER_MODEL_PATH` | Path to `full_ft_distilbert_unweighted_best.pt` — if unset, `Grocery_Category` is left blank |
| `CATEGORIZER_DATASET_PATH` | Path to `merged_labeled.csv` — optional; enables abbreviation expansion |
| `ANTHROPIC_API_KEY` | Anthropic API key — used by the categorizer's LLM fallback in `src/Abbreviation_Normalization.py` |

## Architecture

```
SMS image  → /webhook → OCR pipeline → categorizer → csv_writer → GCS CSV
SMS text   → /webhook → _is_question() → chatbot → Gemini → DuckDB → reply
Web UI     → /chat    → static/chat.js → /api/query → chatbot → Gemini → DuckDB
```

**Key files:**
- [app.py](app.py) — Flask webhook, Twilio integration, all routes; calls `categorizer.init()` at startup
- [ocr_pipeline.py](ocr_pipeline.py) — Receipt text extraction (dates, totals, items, card detection)
- [csv_writer.py](csv_writer.py) — GCS-backed CSV storage with thread-safe deduplication
- [chatbot/\_\_init\_\_.py](chatbot/__init__.py) — `handle_chat_message()` — the single entry point for all chat
- [chatbot/csv_loader.py](chatbot/csv_loader.py) — Loads any CSV from GCS or local into DuckDB
- [chatbot/schema_probe.py](chatbot/schema_probe.py) — Auto-discovers schema → context string for Gemini
- [chatbot/query_engine.py](chatbot/query_engine.py) — Gemini API call, extracts `<sql>` from response
- [chatbot/sql_runner.py](chatbot/sql_runner.py) — Safe DuckDB execution, formats result as markdown
- [chatbot/session_manager.py](chatbot/session_manager.py) — Per-user DuckDB connection + conversation history
- [categorizer/\_\_init\_\_.py](categorizer/__init__.py) — `init()` + `categorize(item_name) -> str` — public API, loads model once at startup
- [src/Categorization.py](src/Categorization.py) — DistilBERT inference: `load_classifier()`, `run_inference()`
- [src/Abbreviation_Normalization.py](src/Abbreviation_Normalization.py) — Normalization, vocabulary expansion, TF-IDF similarity, Claude LLM fallback

## Deduplication

Two independent layers in `csv_writer.py`. All CSV read/write operations are protected by a global `threading.Lock` — the entire read-append-write cycle is atomic.

1. **Image hash** — MD5 of raw bytes (computed before writing to disk); exact match → rejected
2. **Content fuzzy match** — LCS-based similarity ≥ 0.85 on store name **AND** exact date **AND** exact total → rejected

## CSV Schema

Stored in GCS as a single CSV — **one row per line item** (multiple rows share the same `Serial_No` for a single receipt):

`Serial_No, Bill_File, Store_Name, Invoice_Date, Total, Card_Used, Received_At, Sender, Image_Hash, Item_Name, Item_Price, Grocery_Category`

- `Serial_No` — bill-level ID (shared across all item rows for the same receipt)
- `Item_Name` / `Item_Price` — extracted from OCR by `extract_items()` in `ocr_pipeline.py`
- `Grocery_Category` — populated by the categorization pipeline (empty until categorized)

To reset the CSV in GCS (wipes all data, writes new headers):
```bash
python reset_csv.py
```

## Chatbot Pipeline

The `chatbot/` module is schema-agnostic — it works with any CSV, not just the bills schema.

**Flow per query:**
1. `csv_loader` downloads the CSV from GCS into a temp file, loads it into an in-memory DuckDB connection, registers it as table `data`
2. `schema_probe` runs `DESCRIBE data` + sample queries to build a compact context string (column names, types, ranges, sample values)
3. `query_engine` prepends the schema context + today's date to the system prompt, sends full conversation history to **Gemini 2.5 Flash** (temperature 0.1), extracts `<sql>...</sql>` from the response. The system prompt is prepended to the user's first message (not via `system_instruction`) due to a Gemini Flash API limitation.
4. `sql_runner` safety-checks the SQL against a blocklist (DROP, DELETE, INSERT, UPDATE, CREATE, ALTER, TRUNCATE, etc.), executes via DuckDB, and formats the result as markdown (tables capped at 30 rows)
5. `session_manager` caches the DuckDB connection per session (CSV loaded once, not per message). History is trimmed to the last 20 user+assistant pairs.

**Special SMS commands:** `reset`, `reload`, `refresh` are handled in `chatbot/__init__.py` before hitting Gemini.

**CSV cache invalidation:** `app.py` calls `reload_session(sender)` after every successful bill scan so the chatbot immediately sees newly added bills.

## Web Chat Routes

| Route | Method | Purpose |
|---|---|---|
| `/chat` | GET | Web chat UI |
| `/api/query` | POST | `{session_id, question, csv_source?}` → `{success, answer}` |
| `/api/reset` | POST | Clear history for a session |
| `/api/reload` | POST | Force CSV reload for a session |

## OCR Extraction Notes

- **Date:** 9 regex patterns covering MM/DD/YYYY, DD-MM-YYYY, text formats (JAN 01 2024, etc.); normalized to YYYY-MM-DD
- **Total:** Keyword-priority search (TOTAL, AMOUNT, BALANCE) then fallback to max detected amount; range-checked (0 < amount < 20000)
- **Items:** Lines with prices, filtered against exclusion keywords (TOTAL, TAX, DISCOUNT, SUBTOTAL, etc.)
- **Card:** Detects VISA/MASTERCARD/DEBIT/CREDIT + XXXX1234 pattern; defaults to "cash"
- **Store:** First non-date, non-amount line from the top 10 lines of the receipt
