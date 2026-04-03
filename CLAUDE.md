# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Bill Bot** — an SMS-based receipt digitization system. Users text receipt images to a Twilio number; the bot runs Google Cloud Vision OCR, extracts structured data (store, date, total, card, line items), deduplicates, and appends to a CSV in Google Cloud Storage.

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

```bash
# Test OCR on a specific image (place images in raw_images/ folder)
python vision_test.py
```

## Required Environment Variables

Set these before running (see `env.yaml` for template):

| Variable | Purpose |
|---|---|
| `TWILIO_ACCOUNT_SID` | Twilio account identifier |
| `TWILIO_AUTH_TOKEN` | Twilio authentication token |
| `GCS_BUCKET_NAME` | Google Cloud Storage bucket for CSV |

Google Cloud credentials must also be available (via `GOOGLE_APPLICATION_CREDENTIALS` or ADC).

| `GEMINI_API_KEY` | Google Gemini API key for the chatbot |
| `GCS_BILLS_BLOB` | GCS blob name for the bills CSV (default: `bills_output.csv`) |

## Architecture

```
SMS image  → /webhook → OCR pipeline → csv_writer → GCS CSV
SMS text   → /webhook → _is_question() → chatbot → Gemini → DuckDB → reply
Web UI     → /chat    → static/chat.js → /api/query → chatbot → Gemini → DuckDB
```

**Key files:**
- [app.py](app.py) — Flask webhook, Twilio integration, all routes
- [ocr_pipeline.py](ocr_pipeline.py) — Receipt text extraction (dates, totals, items, card detection)
- [csv_writer.py](csv_writer.py) — GCS-backed CSV storage with thread-safe deduplication
- [chatbot/\_\_init\_\_.py](chatbot/__init__.py) — `handle_chat_message()` — the single entry point for all chat
- [chatbot/csv_loader.py](chatbot/csv_loader.py) — Loads any CSV from GCS or local into DuckDB
- [chatbot/schema_probe.py](chatbot/schema_probe.py) — Auto-discovers schema → context string for Gemini
- [chatbot/query_engine.py](chatbot/query_engine.py) — Gemini API call, extracts `<sql>` from response
- [chatbot/sql_runner.py](chatbot/sql_runner.py) — Safe DuckDB execution, formats result as markdown
- [chatbot/session_manager.py](chatbot/session_manager.py) — Per-user DuckDB connection + conversation history

## Deduplication

Two independent layers in `csv_writer.py`:
1. **Image hash** — MD5 of raw bytes; exact match → rejected
2. **Content fuzzy match** — LCS-based similarity ≥ 0.85 on store name **AND** exact date **AND** exact total → rejected

## CSV Schema

Stored in GCS as a single CSV with columns:  
`Serial_No, Bill_File, Store_Name, Invoice_Date, Total, Card_Used, Received_At, Sender, Image_Hash, Items`

## Chatbot Pipeline

The `chatbot/` module is schema-agnostic — it works with any CSV, not just the bills schema.

**Flow per query:**
1. `csv_loader` downloads the CSV from GCS into a temp file, loads it into an in-memory DuckDB connection, registers it as table `data`
2. `schema_probe` runs `DESCRIBE data` + sample queries to build a compact context string (column names, types, ranges, sample values)
3. `query_engine` prepends the schema context + today's date to the Gemini system prompt, sends full conversation history, extracts `<sql>...</sql>` from the response
4. `sql_runner` safety-checks the SQL (blocks DROP/DELETE/INSERT/etc.), executes via DuckDB, formats scalar/table results as markdown
5. `session_manager` caches the DuckDB connection per session — CSV is loaded once, not on every message

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

- **Date:** 9 regex patterns covering MM/DD/YYYY, DD-MM-YYYY, text formats (JAN 01 2024, etc.)
- **Total:** Keyword-priority search (TOTAL, AMOUNT, BALANCE) then fallback to max detected amount
- **Items:** Lines with prices, filtered against exclusion keywords (TOTAL, TAX, DISCOUNT, SUBTOTAL, etc.)
- **Card:** Detects VISA/MASTERCARD/DEBIT/CREDIT + XXXX1234 pattern
- **Store:** First non-date, non-amount line from the top 10 lines of the receipt
