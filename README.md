# BillWise

Unified BillWise monorepo for:
- receipt extraction
- line item categorization
- analytics dashboard

## Phase 1
This phase sets up:
- repo scaffold
- configuration
- schemas
- SQLite bootstrap
- local artifact storage

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
python scripts/bootstrap_db.py
pytest


---

### 5) `configs/storage.yaml`

```yaml
project_name: BillWise

database:
  sqlite_path: data/billwise.db

paths:
  raw_dir: data/raw
  processed_dir: data/processed
  reviewed_dir: data/reviewed
  exports_dir: data/exports
  logs_dir: data/logs