"""
eval_text_to_sql.py — Evaluation harness for the BillWise Text-to-SQL engine.

Three evaluation layers:
  1. Execution Accuracy (EX)   — did the SQL run without error?
  2. Answer Accuracy           — does the result match pandas ground truth?
  3. LLM-as-Judge (G-Eval)    — Gemini scores each answer on 3 dimensions

Run:
  python eval_text_to_sql.py                  # standard run (layers 1+2)
  python eval_text_to_sql.py --judge          # add LLM-as-judge scores (layer 3)
  python eval_text_to_sql.py --runs 3         # self-consistency test
  python eval_text_to_sql.py --category top_n # single category only
  python eval_text_to_sql.py --quiet          # summary only

References
──────────
• Spider benchmark (Yu et al. 2018) — defines Execution Accuracy as primary metric
• DIN-SQL (Pourreza & Rafiei, NeurIPS 2023) — self-correction loop methodology
• G-Eval (Liu et al., EMNLP 2023) — LLM-as-judge evaluation framework
• MT-Bench (Zheng et al. 2023)    — multi-turn LLM judge scoring rubric
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
os.chdir(_HERE)

for _line in (_HERE / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, v = _line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

os.environ["STREAMLIT_RUNTIME_SECRETS_MANAGER"] = "false"

import pandas as pd
from data_loader import load_all_data
from text_to_sql import ask_billwise, run_query, build_schema_context, SQLResult

# ── Known schema ───────────────────────────────────────────────────────────────
_KNOWN_TABLES = {"receipts", "line_items", "joined"}
_KNOWN_COLUMNS: dict[str, set[str]] = {
    "receipts":   {"receipt_id","vendor_name","receipt_date","receipt_total",
                   "tax","card_used","sender","extraction_confidence","source"},
    "line_items": {"id","receipt_id","raw_item_text","normalized_item_text",
                   "matched_canonical_item","category","quantity","unit",
                   "unit_price","line_total","category_confidence"},
    "joined":     {"id","receipt_id","raw_item_text","normalized_item_text",
                   "matched_canonical_item","category","quantity","unit",
                   "unit_price","line_total","category_confidence",
                   "vendor_name","receipt_date","receipt_total","tax",
                   "card_used","extraction_confidence","month"},
}
_ALL_COLUMNS = set().union(*_KNOWN_COLUMNS.values())


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    id:               str
    question:         str
    expected_scalar:  float | int | str | None = None
    expected_rows:    int | None               = None
    expected_top:     str | None               = None
    tolerance:        float                    = 0.02
    category:         str                      = "general"
    notes:            str                      = ""


def build_test_cases(
    df_receipts: pd.DataFrame,
    df_items:    pd.DataFrame,
    df_joined:   pd.DataFrame,
) -> list[TestCase]:
    """All ground-truth values derived from pandas — no LLM involved."""
    total_receipt   = round(df_receipts["receipt_total"].sum(), 2)
    total_line      = round(df_joined["line_total"].sum(), 2)
    num_receipts    = len(df_receipts)
    num_items       = len(df_items)
    num_vendors     = df_receipts["vendor_name"].nunique()
    num_categories  = df_joined["category"].nunique()
    top_vendor      = (df_receipts.groupby("vendor_name")["receipt_total"]
                       .sum().sort_values(ascending=False).index[0])
    top_category    = (df_joined.groupby("category")["line_total"]
                       .sum().sort_values(ascending=False).index[0])
    max_line_total  = round(df_joined["line_total"].max(), 2)
    avg_receipt     = round(df_receipts["receipt_total"].mean(), 2)
    total_tax       = round(df_receipts["tax"].sum(), 2)
    avg_unit_price  = round(df_joined["unit_price"].mean(), 2)
    dairy_rows      = int((df_joined["category"] == "Dairy").sum())
    frozen_spend    = round(
        df_joined[df_joined["category"] == "Frozen / Processed"]["line_total"].sum(), 2)
    receipts_gt100  = int((df_receipts["receipt_total"] > 100).sum())
    total_line_spent = round(df_joined["line_total"].sum(), 2)

    return [
        # ── Aggregation ──────────────────────────────────────────────────────
        TestCase("AGG-01", "What is the total spend across all receipts?",
                 expected_scalar=total_receipt, category="aggregation",
                 notes="SUM(receipt_total)"),
        TestCase("AGG-02", "What is the total value of all line items?",
                 expected_scalar=total_line, category="aggregation",
                 notes="SUM(line_total)"),
        TestCase("AGG-03", "How many receipts are there?",
                 expected_scalar=num_receipts, tolerance=0.0, category="aggregation",
                 notes="COUNT(*)"),
        TestCase("AGG-04", "How many unique vendors do we have?",
                 expected_scalar=num_vendors, tolerance=0.0, category="aggregation",
                 notes="COUNT DISTINCT vendor_name"),
        TestCase("AGG-05", "How many line items are in the database?",
                 expected_scalar=num_items, tolerance=0.0, category="aggregation",
                 notes="COUNT(*) from line_items"),

        # ── Grouping ─────────────────────────────────────────────────────────
        TestCase("GRP-01", "Which vendor has the highest total spend?",
                 expected_top=top_vendor, category="grouping",
                 notes="GROUP BY vendor_name ORDER BY SUM DESC LIMIT 1"),
        TestCase("GRP-02", "What is the total spend per vendor?",
                 expected_rows=num_vendors, category="grouping",
                 notes="GROUP BY vendor_name"),
        TestCase("GRP-03", "Show total spend per category",
                 expected_rows=num_categories, category="grouping",
                 notes="GROUP BY category"),
        TestCase("GRP-04", "Which category had the highest total spend?",
                 expected_top=top_category, category="grouping",
                 notes="Top category by SUM(line_total)"),

        # ── Filtering ─────────────────────────────────────────────────────────
        TestCase("FLT-01", "List all items where category is Dairy",
                 expected_rows=dairy_rows, tolerance=0.0, category="filtering",
                 notes="WHERE category = 'Dairy'"),
        TestCase("FLT-02", "How much did we spend on Frozen / Processed items?",
                 expected_scalar=frozen_spend, category="filtering",
                 notes="SUM(line_total) WHERE category = 'Frozen / Processed'"),

        # ── Top-N ─────────────────────────────────────────────────────────────
        TestCase("TOP-01", "Show the top 3 most expensive items by line total",
                 expected_rows=3, tolerance=0.0, category="top_n",
                 notes="ORDER BY line_total DESC LIMIT 3"),
        TestCase("TOP-02", "What is the most expensive single item purchased?",
                 expected_scalar=max_line_total, category="top_n",
                 notes="MAX(line_total)"),

        # ── Hallucination traps ───────────────────────────────────────────────
        TestCase("HAL-01", "What is the average price per unit for all items?",
                 expected_scalar=avg_unit_price, category="hallucination_trap",
                 notes="AVG(unit_price) — column exists, don't invent 'avg_price'"),
        TestCase("HAL-02", "Show all receipts with their store name and date",
                 expected_rows=num_receipts, tolerance=0.0, category="hallucination_trap",
                 notes="'store name' = vendor_name — must not invent store_name column"),
        TestCase("HAL-03", "What is the total tax paid across all receipts?",
                 expected_scalar=total_tax, category="hallucination_trap",
                 notes="tax column exists in receipts"),

        # ── Edge cases ────────────────────────────────────────────────────────
        TestCase("EDG-01", "Show receipts where the total is greater than 100",
                 expected_rows=receipts_gt100, tolerance=0.0, category="edge_case",
                 notes="WHERE receipt_total > 100"),
        TestCase("EDG-02", "What is the average receipt total?",
                 expected_scalar=avg_receipt, category="edge_case",
                 notes="AVG(receipt_total)"),

        # ── Self-correction stress tests (intentionally tricky phrasing) ──────
        TestCase("COR-01",
                 "Tell me the sum of money spent on each supplier, sorted biggest first",
                 expected_rows=num_vendors, category="self_correction_stress",
                 notes="'supplier' = vendor_name; tests synonym understanding"),
        TestCase("COR-02",
                 "Which grocery category cost the most in total?",
                 expected_top=top_category, category="self_correction_stress",
                 notes="'grocery category' = category column"),
        TestCase("COR-03",
                 "Give me the 5 priciest ingredients we bought",
                 expected_rows=5, tolerance=0.0, category="self_correction_stress",
                 notes="'priciest ingredients' = ORDER BY line_total DESC LIMIT 5"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# HALLUCINATION DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def _extract_sql_identifiers(sql: str) -> tuple[set[str], set[str]]:
    sql_no_str = re.sub(r"'[^']*'", "''", sql.lower())

    # Aliases after AS — these are output labels, NOT schema references
    aliases = {m.group(1).lower()
               for m in re.finditer(r"\bas\s+([a-z_][a-z0-9_]*)", sql_no_str)}

    # Tables: identifiers after FROM / JOIN
    tables_used = {
        m.group(1).lower()
        for m in re.finditer(
            r"\b(?:from|join|into|update|table)\s+([a-z_][a-z0-9_]*)", sql_no_str
        )
    }

    _SQL_KW = {
        "select","from","where","group","by","order","having","limit","offset",
        "join","left","right","inner","outer","on","as","and","or","not","in",
        "is","null","like","between","case","when","then","else","end","distinct",
        "all","union","intersect","except","with","count","sum","avg","min","max",
        "desc","asc","true","false","interval","date","timestamp","current_date",
        "current_timestamp","strftime","date_trunc","date_part","coalesce","ilike",
        "round","cast","over","partition","rows","preceding","following","unbounded",
        "extract","epoch","year","month","day","hour","minute","second","if",
        "ifnull","nullif","exists","any","some","top","fetch","next","only","ties",
        "percent","recursive","filter","qualify","using",
    }

    columns_used = {
        m.group(1).lower()
        for m in re.finditer(r"\b([a-z_][a-z0-9_]*)\b", sql_no_str)
        if m.group(1).lower() not in _SQL_KW
        and m.group(1).lower() not in aliases
        and m.group(1).lower() not in tables_used
        and len(m.group(1)) > 1
        and not m.group(1).isdigit()
    }

    return tables_used, columns_used


def check_hallucinations(sql: str) -> dict:
    if not sql:
        return {"hallucinated_tables": set(), "hallucinated_columns": set(), "is_clean": True}
    tables_used, columns_used = _extract_sql_identifiers(sql)
    bad_tables  = tables_used  - _KNOWN_TABLES
    bad_columns = {c for c in (columns_used - _ALL_COLUMNS - _KNOWN_TABLES)
                   if len(c) > 1 and not c.isdigit()}
    return {
        "hallucinated_tables":  bad_tables,
        "hallucinated_columns": bad_columns,
        "is_clean":             not bad_tables and not bad_columns,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE  (G-Eval / MT-Bench pattern)
# ══════════════════════════════════════════════════════════════════════════════

_JUDGE_SYSTEM = """\
You are an expert SQL evaluator. You will be given:
  - A natural-language question about a restaurant supply database
  - A SQL query that was auto-generated to answer it
  - The result returned by running that SQL (first 5 rows shown)

Score the response on THREE dimensions, each 0–10:

  correctness  — Does the SQL logically answer the question?
                 Does the result make sense for the question asked?
                 10 = perfect answer, 0 = completely wrong.

  faithfulness — Does the SQL only reference real columns and tables?
                 Does it avoid making up column names that don't exist?
                 10 = no hallucinated schema, 0 = invented columns/tables.

  sql_quality  — Is the SQL well-formed, efficient, and readable?
                 Does it use appropriate aggregations, filters, and ordering?
                 10 = clean and efficient, 0 = messy or unnecessarily complex.

Return ONLY a JSON object, no commentary:
{"correctness": <0-10>, "faithfulness": <0-10>, "sql_quality": <0-10>, "reasoning": "<one sentence>"}
"""

_JUDGE_PROMPT = """\
Question: {question}

SQL generated:
{sql}

Result (first 5 rows):
{result_preview}

Known tables: receipts, line_items, joined
Known columns include: receipt_id, vendor_name, receipt_date, receipt_total, tax,
  card_used, extraction_confidence, raw_item_text, normalized_item_text,
  matched_canonical_item, category, quantity, unit, unit_price, line_total,
  category_confidence, month

Score this response:
"""


def llm_judge(
    question:   str,
    sql:        str,
    result_df:  pd.DataFrame | None,
    api_key:    str,
) -> dict:
    """
    Use Gemini as a judge to score the Text-to-SQL response.
    Returns dict with keys: correctness, faithfulness, sql_quality, reasoning.
    Returns None values on failure.
    """
    import google.generativeai as genai  # type: ignore

    if not sql:
        return {"correctness": 0, "faithfulness": 0, "sql_quality": 0,
                "reasoning": "No SQL generated."}

    # Build a compact result preview
    if result_df is not None and not result_df.empty:
        result_preview = result_df.head(5).to_string(index=False)
    else:
        result_preview = "(empty result)"

    prompt = _JUDGE_PROMPT.format(
        question=question, sql=sql, result_preview=result_preview
    )

    try:
        genai.configure(api_key=api_key)
        # Use gemini-2.5-flash-lite for the judge — lightweight, avoids thinking issues
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-lite",
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=300,
            ),
            system_instruction=_JUDGE_SYSTEM,
        )
        response = model.generate_content(prompt)

        # Safely extract text — 2.5-flash thinking models may need parts access
        raw = None
        try:
            raw = response.text.strip()
        except Exception:
            pass
        if not raw:
            # Walk parts, skip thought parts, take first text part
            for part in response.candidates[0].content.parts:
                part_text = getattr(part, "text", "")
                thought   = getattr(part, "thought", False)
                if part_text and not thought:
                    raw = part_text.strip()
                    break
        if not raw:
            raise ValueError("Empty response from judge model")

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)

        # Extract just the JSON object (Gemini sometimes adds prose around it)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)

        data = json.loads(raw)
        return {
            "correctness":  float(data.get("correctness", 0)),
            "faithfulness": float(data.get("faithfulness", 0)),
            "sql_quality":  float(data.get("sql_quality", 0)),
            "reasoning":    str(data.get("reasoning", "")),
        }
    except Exception as e:
        return {"correctness": None, "faithfulness": None,
                "sql_quality": None, "reasoning": f"Judge error: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# ANSWER COMPARATOR
# ══════════════════════════════════════════════════════════════════════════════

def _extract_scalar(df: pd.DataFrame) -> float | str | None:
    if df is None or df.empty:
        return None
    if df.shape == (1, 1):
        val = df.iloc[0, 0]
        return None if (isinstance(val, float) and math.isnan(val)) else val
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 1 and len(df) == 1:
        return df[num_cols[0]].iloc[0]
    return None


def compare_result(tc: TestCase, result_df: pd.DataFrame) -> tuple[bool, str]:
    if result_df is None or result_df.empty:
        return False, "Result is empty"

    if tc.expected_rows is not None:
        actual = len(result_df)
        return (actual == tc.expected_rows,
                f"Row count: expected {tc.expected_rows}, got {actual}")

    if tc.expected_top is not None:
        top_val = str(result_df.iloc[0, 0]).strip()
        match   = (tc.expected_top.lower() in top_val.lower()
                   or top_val.lower() in tc.expected_top.lower())
        return match, f"Top value: expected '{tc.expected_top}', got '{top_val}'"

    if tc.expected_scalar is not None:
        actual = _extract_scalar(result_df)
        if actual is None:
            return False, "Could not extract scalar"
        exp = tc.expected_scalar
        if isinstance(exp, str):
            return str(exp).lower() in str(actual).lower(), f"String: '{actual}'"
        try:
            a, e   = float(actual), float(exp)
            rel    = abs(a - e) / abs(e) if e != 0 else abs(a - e)
            passed = rel <= tc.tolerance
            return passed, f"Expected {e:,.2f}, got {a:,.2f} (err={rel*100:.2f}%)"
        except (TypeError, ValueError):
            return False, f"Cannot compare '{actual}' vs '{exp}'"

    return True, "Execution-only check — no expectation"


# ══════════════════════════════════════════════════════════════════════════════
# RUN RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    tc:             TestCase
    run_index:      int
    executed:       bool
    passed:         bool
    reason:         str
    sql:            str
    latency_s:      float
    attempts:       int        = 1
    was_corrected:  bool       = False
    hallucination:  dict       = field(default_factory=dict)
    judge_scores:   dict       = field(default_factory=dict)
    error:          str        = ""
    result_df:      Any        = None


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_test(
    tc:          TestCase,
    df_receipts: pd.DataFrame,
    df_items:    pd.DataFrame,
    df_joined:   pd.DataFrame,
    api_key:     str,
    run_index:   int  = 0,
    use_judge:   bool = False,
) -> RunResult:
    t0 = time.perf_counter()

    sql_result: SQLResult = ask_billwise(
        tc.question, df_receipts, df_items, df_joined, api_key=api_key
    )

    latency   = time.perf_counter() - t0
    executed  = sql_result.result_df is not None and sql_result.error == ""
    halluc    = check_hallucinations(sql_result.sql)

    if executed:
        passed, reason = compare_result(tc, sql_result.result_df)
    else:
        passed, reason = False, sql_result.error or sql_result.answer_text

    judge = {}
    if use_judge and sql_result.sql:
        judge = llm_judge(
            tc.question, sql_result.sql, sql_result.result_df, api_key
        )

    return RunResult(
        tc=tc, run_index=run_index,
        executed=executed, passed=passed, reason=reason,
        sql=sql_result.sql, latency_s=round(latency, 2),
        attempts=sql_result.attempts, was_corrected=sql_result.was_corrected,
        hallucination=halluc, judge_scores=judge,
        error=sql_result.error, result_df=sql_result.result_df,
    )


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(all_runs: list[RunResult], n_runs: int) -> dict:
    if not all_runs:
        return {}

    by_id: dict[str, list[RunResult]] = {}
    for r in all_runs:
        by_id.setdefault(r.tc.id, []).append(r)

    n_tests = len(by_id)
    total   = len(all_runs)

    exec_acc = sum(1 for runs in by_id.values() if any(r.executed for r in runs)) / n_tests
    ans_acc  = sum(1 for runs in by_id.values() if any(r.passed   for r in runs)) / n_tests

    halluc_runs     = [r for r in all_runs if not r.hallucination.get("is_clean", True)]
    halluc_rate     = len(halluc_runs) / total
    col_h_rate      = len([r for r in all_runs if r.hallucination.get("hallucinated_columns")]) / total
    tbl_h_rate      = len([r for r in all_runs if r.hallucination.get("hallucinated_tables")]) / total

    # Self-correction stats
    corrected_tests = sum(1 for runs in by_id.values() if any(r.was_corrected for r in runs))
    correction_rate = corrected_tests / n_tests
    avg_attempts    = sum(r.attempts for r in all_runs) / total

    # Self-consistency (runs > 1)
    consistency = None
    if n_runs > 1:
        scores = []
        for runs in by_id.values():
            executed_runs = [r for r in runs if r.executed]
            if not executed_runs:
                continue
            sigs = []
            for r in runs:
                if r.result_df is not None and not r.result_df.empty:
                    sigs.append(tuple(sorted(str(v) for v in r.result_df.iloc[:, 0].tolist())))
                else:
                    sigs.append(())
            scores.append(sum(1 for s in sigs if s == sigs[0]) / len(sigs))
        consistency = sum(scores) / len(scores) if scores else None

    # Latency
    lats    = [r.latency_s for r in all_runs]
    avg_lat = round(sum(lats) / len(lats), 2)

    # Judge scores (if present)
    judge_metrics: dict = {}
    judge_runs = [r for r in all_runs if r.judge_scores and r.judge_scores.get("correctness") is not None]
    if judge_runs:
        for dim in ("correctness", "faithfulness", "sql_quality"):
            vals = [r.judge_scores[dim] for r in judge_runs if r.judge_scores.get(dim) is not None]
            judge_metrics[f"judge_{dim}"] = round(sum(vals) / len(vals), 2) if vals else None
        judge_metrics["judge_n"] = len(judge_runs)

    # Category breakdown
    by_cat: dict[str, dict] = {}
    for runs in by_id.values():
        cat = runs[0].tc.category
        by_cat.setdefault(cat, {"total": 0, "passed": 0})
        by_cat[cat]["total"] += 1
        if any(r.passed for r in runs):
            by_cat[cat]["passed"] += 1

    return {
        "n_tests": n_tests, "n_runs": n_runs, "total_runs": total,
        "execution_acc": exec_acc, "answer_acc": ans_acc,
        "halluc_rate": halluc_rate, "col_halluc_rate": col_h_rate,
        "tbl_halluc_rate": tbl_h_rate,
        "correction_rate": correction_rate, "avg_attempts": round(avg_attempts, 2),
        "consistency": consistency,
        "avg_latency_s": avg_lat,
        "max_latency_s": round(max(lats), 2),
        "min_latency_s": round(min(lats), 2),
        "by_category": by_cat,
        **judge_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# REPORT PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def _bar(v: float | None, w: int = 20) -> str:
    if v is None:
        return "─" * w
    f = round(v * w)
    return "█" * f + "░" * (w - f)

def _pct(v: float | None) -> str:
    return f"{v*100:5.1f}%" if v is not None else "  N/A "

def _score(v: float | None) -> str:
    return f"{v:4.1f}/10" if v is not None else "  N/A  "


def print_report(all_runs: list[RunResult], metrics: dict, verbose: bool = True) -> None:
    W = 76
    print("\n" + "═" * W)
    print("  BILLWISE TEXT-TO-SQL EVALUATION REPORT")
    print("═" * W)

    by_id: dict[str, list[RunResult]] = {}
    for r in all_runs:
        by_id.setdefault(r.tc.id, []).append(r)

    # ── Per-test table ────────────────────────────────────────────────────────
    has_judge = any(r.judge_scores for r in all_runs)
    header = f"{'ID':<8} {'Category':<24} {'EX':>3} {'✓':>3} {'Halluc':>6} {'Tries':>5} {'ms':>6}"
    if has_judge:
        header += f"  {'Corr':>5} {'Faith':>5} {'SQL':>5}"
    print(f"\n{header}  Question")
    print("─" * W)

    for tid, runs in sorted(by_id.items()):
        tc      = runs[0].tc
        executed = any(r.executed  for r in runs)
        passed   = any(r.passed    for r in runs)
        halluc   = any(not r.hallucination.get("is_clean", True) for r in runs)
        corrected = any(r.was_corrected for r in runs)
        avg_ms   = round(sum(r.latency_s for r in runs) / len(runs) * 1000)
        max_tries = max(r.attempts for r in runs)

        ex_s   = "✓" if executed else "✗"
        pa_s   = "✓" if passed   else "✗"
        ha_s   = "⚠" if halluc   else "✓"
        try_s  = f"{max_tries}{'↺' if corrected else ''}"

        q_short = tc.question[:34] + ("…" if len(tc.question) > 34 else "")
        line = f"{tid:<8} {tc.category:<24} {ex_s:>3} {pa_s:>3} {ha_s:>6} {try_s:>5} {avg_ms:>5}ms"

        if has_judge:
            js = runs[0].judge_scores
            c  = f"{js['correctness']:.0f}"  if js and js.get("correctness")  is not None else "─"
            f_ = f"{js['faithfulness']:.0f}" if js and js.get("faithfulness") is not None else "─"
            sq = f"{js['sql_quality']:.0f}"  if js and js.get("sql_quality")  is not None else "─"
            line += f"  {c:>5} {f_:>5} {sq:>5}"

        print(f"{line}  {q_short}")

        if verbose:
            r0 = runs[0]
            if r0.sql:
                sql_p = r0.sql.replace("\n", " ")[:65]
                print(f"         SQL : {sql_p}" + ("…" if len(r0.sql) > 65 else ""))
            print(f"         Rslt: {r0.reason[:68]}")
            if r0.was_corrected:
                print(f"         ↺   : self-corrected in {r0.attempts} attempt(s)")
            if halluc:
                bc = runs[0].hallucination.get("hallucinated_columns", set())
                bt = runs[0].hallucination.get("hallucinated_tables",  set())
                if bc: print(f"         ⚠ columns : {bc}")
                if bt: print(f"         ⚠ tables  : {bt}")
            if has_judge and runs[0].judge_scores.get("reasoning"):
                print(f"         💬  : {runs[0].judge_scores['reasoning'][:72]}")
            print()

    # ── Metrics summary ───────────────────────────────────────────────────────
    m = metrics
    print("═" * W)
    print("  METRICS SUMMARY")
    print("═" * W)
    print(f"\n  Tests: {m['n_tests']} questions × {m['n_runs']} run(s) = {m['total_runs']} total\n")

    print(f"  Execution Accuracy    {_pct(m['execution_acc'])}  {_bar(m['execution_acc'])}")
    print(f"  Answer Accuracy       {_pct(m['answer_acc'])}  {_bar(m['answer_acc'])}")
    print(f"  Hallucination Rate    {_pct(m['halluc_rate'])}  {_bar(m['halluc_rate'])}  ← lower = better")
    print(f"    ↳ Column halluc     {_pct(m['col_halluc_rate'])}")
    print(f"    ↳ Table  halluc     {_pct(m['tbl_halluc_rate'])}")
    print(f"  Self-correction rate  {_pct(m['correction_rate'])}  avg {m['avg_attempts']} attempt(s)/query")

    if m.get("consistency") is not None:
        print(f"  Self-Consistency      {_pct(m['consistency'])}  {_bar(m['consistency'])}")

    print(f"\n  Latency avg/min/max   {m['avg_latency_s']}s / {m['min_latency_s']}s / {m['max_latency_s']}s")

    # Judge scores
    if m.get("judge_correctness") is not None:
        print(f"\n  ── LLM-as-Judge scores (G-Eval) ──────────────────────────────")
        print(f"  Correctness   {_score(m['judge_correctness'])}  {_bar(m['judge_correctness']/10)}")
        print(f"  Faithfulness  {_score(m['judge_faithfulness'])}  {_bar(m['judge_faithfulness']/10)}")
        print(f"  SQL Quality   {_score(m['judge_sql_quality'])}  {_bar(m['judge_sql_quality']/10)}")
        print(f"  (n={m['judge_n']} judge calls)")

    # Category breakdown
    print(f"\n  ── By category ────────────────────────────────────────────────")
    for cat, d in sorted(m["by_category"].items()):
        acc = d["passed"] / d["total"]
        print(f"  {cat:<28} {d['passed']}/{d['total']}  {_bar(acc, 12)}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print("\n" + "═" * W)
    print("  INTERPRETATION")
    print("═" * W)

    ex, ans, hal = m["execution_acc"], m["answer_acc"], m["halluc_rate"]
    cor = m.get("correction_rate", 0)

    print()
    if ex  >= 0.90: print("  ✅ Execution Accuracy ≥ 90%  — SQL is syntactically valid consistently.")
    elif ex >= 0.75: print("  🟡 Execution Accuracy 75-90% — occasional SQL errors; few-shot helps.")
    else:            print("  🔴 Execution Accuracy < 75%  — frequent errors; improve schema prompt.")

    if ans >= 0.85: print("  ✅ Answer Accuracy ≥ 85%     — factually correct for most questions.")
    elif ans >= 0.65: print("  🟡 Answer Accuracy 65-85%   — reasonable; self-correction can improve.")
    else:            print("  🔴 Answer Accuracy < 65%     — significant errors; review prompt.")

    if hal <= 0.05: print("  ✅ Hallucination Rate < 5%   — model respects the schema.")
    elif hal <= 0.20: print("  🟡 Hallucination Rate 5-20% — some invented names; enumerate columns.")
    else:            print("  🔴 Hallucination Rate > 20%  — high hallucination; add explicit column list.")

    if cor > 0:
        print(f"  ↺  Self-correction fired on {cor*100:.0f}% of tests — retry loop is working.")
    else:
        print("  ✅ Self-correction not needed — all queries executed on first attempt.")

    if m.get("judge_correctness") is not None:
        jc = m["judge_correctness"]
        if jc >= 8:   print(f"  ✅ Judge Correctness {jc:.1f}/10  — LLM agrees answers are correct.")
        elif jc >= 6: print(f"  🟡 Judge Correctness {jc:.1f}/10  — LLM sees partial correctness issues.")
        else:         print(f"  🔴 Judge Correctness {jc:.1f}/10  — LLM rates answers as largely incorrect.")

    # ── Improvement strategies ────────────────────────────────────────────────
    print("\n" + "═" * W)
    print("  IMPROVEMENT STRATEGIES (ranked by impact for this project)")
    print("═" * W)
    print("""
  Already implemented:
  ✅ Few-shot examples   — 6 hand-written Q→SQL pairs injected into every prompt.
                           Research shows +8-12% EX on Spider benchmark.
  ✅ Self-correction     — failed SQL + error sent back to Gemini (up to 3 retries).
                           DIN-SQL (NeurIPS 2023) shows +15% gain on hard queries.
  ✅ LLM-as-judge        — Gemini scores its own output on 3 dimensions (G-Eval).
                           Correlates with human eval at r=0.85 (Liu et al. 2023).

  Still available if accuracy degrades:
  ⬜ Schema-linking      — Ask Gemini "which columns are needed?" before SQL gen.
                           Reduces hallucination by ~40% (Pourreza & Rafiei 2023).
  ⬜ Self-consistency    — Run each query N times, take the majority answer.
                           +5-8% accuracy but 3× slower. Use --runs 3 to test.
  ⬜ Chain-of-thought    — Ask Gemini to reason step-by-step before writing SQL.
                           Improves complex multi-table queries specifically.
  ⬜ Fine-tuning         — Fine-tune a smaller model on your specific schema.
                           Most expensive but gives highest accuracy (>90%).
""")
    print("═" * W)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BillWise Text-to-SQL")
    parser.add_argument("--runs",     type=int,  default=1,     help="Runs per test (self-consistency)")
    parser.add_argument("--judge",    action="store_true",       help="Enable LLM-as-judge scoring")
    parser.add_argument("--quiet",    action="store_true",       help="Summary only")
    parser.add_argument("--category", type=str,  default=None,  help="Run one category only")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("⚠️  GEMINI_API_KEY not set in .env"); sys.exit(1)

    print("Loading data…")
    df_receipts, df_items, df_joined = load_all_data()
    print(f"  Receipts: {len(df_receipts)}  Items: {len(df_items)}\n")

    test_cases = build_test_cases(df_receipts, df_items, df_joined)
    if args.category:
        test_cases = [tc for tc in test_cases if tc.category == args.category]
        if not test_cases:
            print(f"No tests for category '{args.category}'"); sys.exit(1)

    judge_note = " + LLM-judge" if args.judge else ""
    print(f"Running {len(test_cases)} tests × {args.runs} run(s){judge_note}…\n")

    all_runs: list[RunResult] = []
    for tc in test_cases:
        for ri in range(args.runs):
            tag = f"[{tc.id} {ri+1}/{args.runs}]"
            print(f"  {tag:<18} {tc.question[:48]}…", end="", flush=True)
            rr = run_test(tc, df_receipts, df_items, df_joined,
                          api_key, ri, use_judge=args.judge)
            sym = "✓" if rr.passed else ("⚡" if rr.executed else "✗")
            retry = f" ↺×{rr.attempts-1}" if rr.was_corrected else ""
            judge_s = ""
            if args.judge and rr.judge_scores.get("correctness") is not None:
                judge_s = f" judge={rr.judge_scores['correctness']:.0f}/10"
            print(f"  {sym}{retry}{judge_s}  ({rr.latency_s}s)")
            all_runs.append(rr)

    metrics = compute_metrics(all_runs, args.runs)
    print_report(all_runs, metrics, verbose=not args.quiet)


if __name__ == "__main__":
    main()
