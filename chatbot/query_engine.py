"""
query_engine.py
Sends the user question + schema context to Gemini and extracts
a DuckDB-compatible SQL query from the response.
"""

import re
import os
from datetime import date

import google.generativeai as genai


# ── System prompt template ────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are a data analyst assistant for a restaurant expense tracking system.
The spending data is stored in a DuckDB table called `data`.

{schema}

Today's date is {today}.

RULES:
1. Always answer by writing a SQL SELECT query wrapped in <sql>...</sql> tags.
2. Use only DuckDB-compatible SQL syntax.
3. For relative date filters use INTERVAL, e.g.:
     WHERE Invoice_Date >= CURRENT_DATE - INTERVAL '1 month'
4. Column names are case-sensitive — use them exactly as listed above.
   Wrap column names in double-quotes if they contain spaces or special chars.
5. After the <sql> block write one short plain-English sentence explaining the result.
6. If the question cannot be answered from the available data, say so clearly and omit the SQL block.
7. Never generate DROP, DELETE, INSERT, UPDATE, CREATE, ALTER, ATTACH, COPY, or TRUNCATE statements.
"""


def _build_system_prompt(schema_context: str) -> str:
    return _SYSTEM_TEMPLATE.format(
        schema=schema_context,
        today=date.today().isoformat(),
    )


def _to_gemini_history(history: list[dict]) -> list[dict]:
    """Convert our internal history format to Gemini's expected format."""
    result = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        result.append({"role": role, "parts": [msg["content"]]})
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def get_model(api_key: str | None = None) -> genai.GenerativeModel:
    """
    Initialise and return a Gemini GenerativeModel.
    api_key defaults to env var GEMINI_API_KEY.
    """
    key = api_key or os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,   # low temp → more deterministic SQL
            max_output_tokens=1024,
        ),
    )


def ask(
    question: str,
    schema_context: str,
    history: list[dict],
    model: genai.GenerativeModel,
) -> tuple[str, str | None]:
    """
    Send the question to Gemini with full conversation history.

    Args:
        question       : The user's natural-language question.
        schema_context : Output of schema_probe.build_context_string().
        history        : List of {"role": "user"/"assistant", "content": str}.
                         Modified in-place — current turn appended.
        model          : A GenerativeModel instance from get_model().

    Returns:
        (reply_text, sql_or_None)
        reply_text — full Gemini response
        sql_or_None — extracted SQL string, or None if no <sql> block present
    """
    system_prompt = _build_system_prompt(schema_context)

    # Build history for Gemini (all prior turns, NOT including current question)
    gemini_history = _to_gemini_history(history)

    # Append current question to our internal history
    history.append({"role": "user", "content": question})

    # Send to Gemini
    chat = model.start_chat(
        history=gemini_history,
        # Gemini doesn't accept system_instruction in start_chat for flash,
        # so we prepend the system prompt to the first user message if history
        # is empty, otherwise it was already established.
    )

    # Prepend system prompt to the question on first turn so Gemini honours it
    if len(gemini_history) == 0:
        payload = f"{system_prompt}\n\n---\n\nUser question: {question}"
    else:
        payload = question

    response = chat.send_message(payload)
    reply = response.text

    # Append assistant reply to our internal history
    history.append({"role": "assistant", "content": reply})

    # Extract SQL from <sql>...</sql> tags
    match = re.search(r"<sql>(.*?)</sql>", reply, re.DOTALL | re.IGNORECASE)
    sql = match.group(1).strip() if match else None

    return reply, sql
