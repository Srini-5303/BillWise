"""
session_manager.py
Thread-safe, in-memory store for per-user state:
  - DuckDB connection (the loaded CSV)
  - Schema context string (built once on load)
  - Conversation history (list of {role, content} dicts)

session_id is the sender's phone number for SMS, or a UUID for web UI.
"""

import threading
import duckdb


_MAX_HISTORY_TURNS = 20   # keep last N user+assistant pairs


class SessionManager:
    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock  = threading.Lock()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ensure(self, session_id: str) -> dict:
        if session_id not in self._store:
            self._store[session_id] = {
                "conn":    None,   # duckdb.DuckDBPyConnection
                "schema":  None,   # str — context for LLM
                "history": [],     # list[dict]
            }
        return self._store[session_id]

    def _trim(self, session_id: str):
        """Keep history within MAX_HISTORY_TURNS user+assistant pairs."""
        h = self._store[session_id]["history"]
        max_msgs = _MAX_HISTORY_TURNS * 2
        if len(h) > max_msgs:
            self._store[session_id]["history"] = h[-max_msgs:]

    # ── Public API ─────────────────────────────────────────────────────────────

    def has_connection(self, session_id: str) -> bool:
        with self._lock:
            return (session_id in self._store and
                    self._store[session_id]["conn"] is not None)

    def set_connection(
        self,
        session_id: str,
        conn: duckdb.DuckDBPyConnection,
        schema: str,
    ):
        with self._lock:
            s = self._ensure(session_id)
            s["conn"]   = conn
            s["schema"] = schema

    def get_connection(self, session_id: str) -> duckdb.DuckDBPyConnection | None:
        with self._lock:
            return self._store.get(session_id, {}).get("conn")

    def get_schema(self, session_id: str) -> str | None:
        with self._lock:
            return self._store.get(session_id, {}).get("schema")

    def get_history(self, session_id: str) -> list[dict]:
        """Returns a reference to the history list — modifications are live."""
        with self._lock:
            return self._ensure(session_id)["history"]

    def append_message(self, session_id: str, role: str, content: str):
        with self._lock:
            s = self._ensure(session_id)
            s["history"].append({"role": role, "content": content})
            self._trim(session_id)

    def clear(self, session_id: str):
        """Reset history and drop the cached DuckDB connection."""
        with self._lock:
            if session_id in self._store:
                conn = self._store[session_id].get("conn")
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                del self._store[session_id]

    def reload_csv(self, session_id: str):
        """Force a CSV reload on next query by dropping the connection."""
        with self._lock:
            if session_id in self._store:
                self._store[session_id]["conn"]   = None
                self._store[session_id]["schema"] = None

    def active_sessions(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())
