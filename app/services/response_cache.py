"""Conversation-scoped, exact-question response cache backed by SQLite."""
import hashlib
import json
import os
from typing import List, Optional, Tuple

from app.database.db import get_connection

CACHE_TTL_DAYS = 7
CACHE_FORMAT_VERSION = "assistant-json-v1"


def normalize_question(message: str) -> str:
    """Normalize casing and whitespace while preserving meaningful punctuation."""
    return " ".join(message.casefold().split())


def _context_without_repeated_question(
    message: str, history: List[Tuple[str, str]]
) -> str:
    """Hash relevant history, omitting earlier copies of this same Q-and-A turn."""
    question = normalize_question(message)
    context = []
    skip_following_answer = False
    for role, content in history:
        if role == "user" and normalize_question(content) == question:
            skip_following_answer = True
            continue
        if skip_following_answer and role == "assistant":
            skip_following_answer = False
            continue
        if skip_following_answer:
            skip_following_answer = False
        context.append((role, content))
    # The model receives at most the latest ten messages. Reserving two slots for
    # the repeated question/answer keeps the pre- and post-answer context stable.
    context = context[-8:]
    serialized = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cache_key(conversation_id: str, route: str, message: str,
               history: List[Tuple[str, str]], source_version: str = "") -> str:
    key_data = {
        "conversation_id": conversation_id,
        "route": route,
        "question": normalize_question(message),
        "context": _context_without_repeated_question(message, history),
        "model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        "format": CACHE_FORMAT_VERSION,
        "source_version": source_version,
    }
    serialized = json.dumps(key_data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_cached_response(conversation_id: str, route: str, message: str,
                        history: List[Tuple[str, str]],
                        source_version: str = "") -> Optional[str]:
    """Return an unexpired matching response and increment its hit counter."""
    key = _cache_key(conversation_id, route, message, history, source_version)
    with get_connection() as db:
        row = db.execute(
            """SELECT answer FROM response_cache
               WHERE cache_key=? AND julianday(created_at) >= julianday('now', ?)""",
            (key, "-{} days".format(CACHE_TTL_DAYS)),
        ).fetchone()
        if row is None:
            return None
        db.execute(
            """UPDATE response_cache
               SET hit_count=hit_count+1,
                   last_hit_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
               WHERE cache_key=?""",
            (key,),
        )
        return row["answer"]


def save_cached_response(conversation_id: str, route: str, message: str,
                         history: List[Tuple[str, str]], answer: str,
                         source_version: str = "") -> None:
    """Persist a successful model answer and remove expired cache entries."""
    key = _cache_key(conversation_id, route, message, history, source_version)
    with get_connection() as db:
        db.execute(
            """INSERT INTO response_cache (cache_key, conversation_id, route, answer)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(cache_key) DO UPDATE SET
                   answer=excluded.answer,
                   created_at=strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                   last_hit_at=NULL,
                   hit_count=0""",
            (key, conversation_id, route, answer),
        )
        db.execute(
            "DELETE FROM response_cache WHERE julianday(created_at) < julianday('now', ?)",
            ("-{} days".format(CACHE_TTL_DAYS),),
        )


def cache_stats() -> dict:
    """Return small aggregate counters for the API demo/monitoring endpoint."""
    with get_connection() as db:
        row = db.execute(
            """SELECT COUNT(*) AS entries, COALESCE(SUM(hit_count), 0) AS hits
               FROM response_cache
               WHERE julianday(created_at) >= julianday('now', ?)""",
            ("-{} days".format(CACHE_TTL_DAYS),),
        ).fetchone()
    return {"entries": row["entries"], "hits": row["hits"], "ttl_days": CACHE_TTL_DAYS}
