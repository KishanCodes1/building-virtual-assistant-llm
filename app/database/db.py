"""SQLite connection and schema for the virtual assistant."""

from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "assistant.db"


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    details TEXT,
    due_at TEXT,
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'completed', 'cancelled')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    remind_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'sent', 'cancelled')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS tool_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    route TEXT NOT NULL,
    tool_name TEXT,
    input_summary TEXT,
    output_summary TEXT,
    status TEXT NOT NULL CHECK (status IN ('success', 'error', 'needs_clarification')),
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS response_cache (
    cache_key TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    route TEXT NOT NULL CHECK (route IN ('llm_chat', 'document_qa')),
    answer TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    last_hit_at TEXT,
    hit_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_response_cache_conversation
    ON response_cache (conversation_id, created_at);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS parent_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    parent_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    UNIQUE(document_id, page_number, parent_index)
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_chunk_id INTEGER REFERENCES parent_chunks(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
    ON messages (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_todos_status_due
    ON todos (status, due_at);
CREATE INDEX IF NOT EXISTS idx_reminders_status_time
    ON reminders (status, remind_at);
CREATE INDEX IF NOT EXISTS idx_tool_traces_conversation_created
    ON tool_traces (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document
    ON document_chunks (document_id, page_number);
"""


def get_connection() -> sqlite3.Connection:
    """Open the app database, creating its parent directory if needed."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    """Create the tables and indexes if they do not already exist."""
    with get_connection() as connection:
        connection.executescript(SCHEMA)
        # Existing MVP databases have flat chunks. Add this nullable link in place
        # so their current data remains readable while new uploads use parent/child.
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(document_chunks)")}
        if "parent_chunk_id" not in columns:
            connection.execute(
                "ALTER TABLE document_chunks ADD COLUMN parent_chunk_id "
                "INTEGER REFERENCES parent_chunks(id) ON DELETE CASCADE"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_chunks_parent "
            "ON document_chunks (parent_chunk_id)"
        )


if __name__ == "__main__":
    initialize_database()
    print(f"Database ready: {DATABASE_PATH}")
