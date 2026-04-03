"""
chatbot/__init__.py
Public entry point used by app.py.

    from chatbot import handle_chat_message, reload_session

The only function app.py needs to call is handle_chat_message().
"""

import os

from .csv_loader     import get_connection
from .schema_probe   import probe_schema, build_context_string
from .query_engine   import get_model, ask
from .sql_runner     import run_sql
from .session_manager import SessionManager

# Module-level singletons (initialised lazily)
_sessions: SessionManager      = SessionManager()
_model                         = None   # genai.GenerativeModel


def _get_model():
    global _model
    if _model is None:
        _model = get_model()   # reads GEMINI_API_KEY from env
    return _model


def _ensure_loaded(session_id: str, csv_source: str | None):
    """Load the CSV into DuckDB for this session if not already cached."""
    if _sessions.has_connection(session_id):
        return

    source      = csv_source or os.environ.get("GCS_BILLS_BLOB", "bills_output.csv")
    source_type = "local" if os.path.exists(source) else "gcs"

    conn    = get_connection(source, source_type)
    schema  = build_context_string(probe_schema(conn))
    _sessions.set_connection(session_id, conn, schema)


# ── Public API ─────────────────────────────────────────────────────────────────

def handle_chat_message(
    session_id: str,
    message: str,
    csv_source: str | None = None,
) -> str:
    """
    Process one user message and return the assistant's reply as a string.

    Args:
        session_id : Unique identifier for this user/conversation.
                     Use sender phone number for SMS, UUID for web UI.
        message    : The user's natural-language question.
        csv_source : Optional override for the CSV file to query.
                     GCS blob name or local path.  Defaults to
                     GCS_BILLS_BLOB env var → "bills_output.csv".

    Returns:
        A plain-text / markdown string ready to send back to the user.
    """
    # Handle special control commands
    lower = message.strip().lower()
    if lower in ("reset", "clear", "start over"):
        _sessions.clear(session_id)
        return "Conversation reset. Ask me anything about your spending!"

    if lower in ("reload", "refresh", "refresh data"):
        _sessions.reload_csv(session_id)
        _ensure_loaded(session_id, csv_source)
        return "Data reloaded from storage. What would you like to know?"

    # Ensure CSV is loaded for this session
    try:
        _ensure_loaded(session_id, csv_source)
    except Exception as e:
        return f"Could not load the data file: {e}"

    conn          = _sessions.get_connection(session_id)
    schema        = _sessions.get_schema(session_id)
    history       = _sessions.get_history(session_id)

    # Ask Gemini — history is updated in-place inside ask()
    try:
        reply, sql = ask(message, schema, history, _get_model())
    except Exception as e:
        return f"I had trouble understanding that question: {e}"

    # If no SQL was generated, return the LLM's plain-text reply
    if sql is None:
        import re
        clean = re.sub(r"<sql>.*?</sql>", "", reply, flags=re.DOTALL | re.IGNORECASE).strip()
        return clean or reply

    # Execute the SQL
    success, result = run_sql(sql, conn)

    if not success:
        # Tell the user something went wrong; don't expose raw SQL errors in SMS
        return (
            "I couldn't compute that — the generated query had an issue.\n"
            f"Details: {result}"
        )

    # Strip the raw <sql> block from the reply and append the actual result
    import re
    explanation = re.sub(r"<sql>.*?</sql>", "", reply, flags=re.DOTALL | re.IGNORECASE).strip()
    return f"{explanation}\n\n{result}" if explanation else result


def reload_session(session_id: str):
    """Force CSV reload for a session (e.g. after new bills are added)."""
    _sessions.reload_csv(session_id)


def clear_session(session_id: str):
    """Wipe history and cached data for a session."""
    _sessions.clear(session_id)
