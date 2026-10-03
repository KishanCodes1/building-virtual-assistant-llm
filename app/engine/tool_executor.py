"""Execute exact note, to-do, and reminder actions in SQLite."""
from datetime import datetime, timezone
from typing import Tuple

from app.database.db import get_connection
from app.engine.intent_classifier import Intent


def _insert(table: str, fields: Tuple[str, ...], values: Tuple[object, ...]) -> dict:
    # Identifiers are internal constants, never user-provided values.
    with get_connection() as db:
        cursor = db.execute(
            "INSERT INTO {} ({}) VALUES ({})".format(
                table, ", ".join(fields), ", ".join("?" for _ in values)
            ), values,
        )
        return dict(db.execute(
            "SELECT * FROM {} WHERE id=?".format(table), (cursor.lastrowid,)
        ).fetchone())


def execute_tool(intent: Intent) -> Tuple[dict, str]:
    if intent.name == "save_note":
        if "|" not in intent.payload:
            raise ValueError("Use: note: title | note content")
        title, content = (part.strip() for part in intent.payload.split("|", 1))
        if not title or not content:
            raise ValueError("A note needs both a title and content.")
        return _insert("notes", ("title", "content"), (title, content)), "save_note"

    if intent.name == "add_todo":
        title = intent.payload.strip()
        if not title:
            raise ValueError("Use: todo: task description")
        return _insert("todos", ("title",), (title,)), "add_todo"

    if intent.name == "create_reminder":
        if "|" not in intent.payload:
            raise ValueError("Use: reminder: title | 2026-10-01T18:00:00+05:30")
        title, time_text = (part.strip() for part in intent.payload.split("|", 1))
        if not title:
            raise ValueError("A reminder needs a title.")
        try:
            remind_at = datetime.fromisoformat(time_text.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("Use an ISO 8601 date and time after the | symbol.")
        if remind_at.tzinfo is None:
            remind_at = remind_at.replace(tzinfo=timezone.utc)
        stored_time = remind_at.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        return _insert("reminders", ("title", "remind_at"), (title, stored_time)), "create_reminder"

    raise ValueError("This request does not match an available Python tool.")
