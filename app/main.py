"""SQLite-backed FastAPI backend for the virtual assistant MVP."""
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.database.db import get_connection, initialize_database
from app.engine.errors import AssistantError
from app.engine.hybrid_engine import run_hybrid_engine
from app.rag.vector_store import split_parent_child, store_documents
from app.services.response_cache import cache_stats

app = FastAPI(title="Hybrid Virtual Assistant API", version="0.1.0")
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str = Field("default", min_length=1, max_length=100)


class NoteCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=10000)


class TodoCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    details: Optional[str] = None
    due_at: Optional[datetime] = None


class ReminderCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    remind_at: datetime


def utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def recent_messages(conversation_id: str, limit: int = 10) -> list:
    with get_connection() as db:
        rows = db.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
    return [(row["role"], row["content"]) for row in reversed(rows)]


def save_message(conversation_id: str, role: str, content: str) -> None:
    with get_connection() as db:
        db.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )


def save_trace(conversation_id: str, route: str, tool: Optional[str],
               source: str, result: str, status: str = "success") -> None:
    with get_connection() as db:
        db.execute(
            """INSERT INTO tool_traces
               (conversation_id, route, tool_name, input_summary, output_summary, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (conversation_id, route, tool, source[:500], result[:500], status),
        )


def create_item(table: str, fields: tuple, values: tuple) -> dict:
    # Table names are selected only by the fixed endpoint code, never by user input.
    with get_connection() as db:
        cursor = db.execute(
            "INSERT INTO {} ({}) VALUES ({})".format(table, ", ".join(fields), ", ".join("?" for _ in values)),
            values,
        )
        row = db.execute("SELECT * FROM {} WHERE id = ?".format(table), (cursor.lastrowid,)).fetchone()
        return dict(row)


@app.on_event("startup")
def startup() -> None:
    initialize_database()


@app.get("/")
def home() -> dict:
    return {"message": "Hybrid Virtual Assistant API is running", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/cache/stats")
def response_cache_stats() -> dict:
    """Show cache entry and hit counts for the project demo."""
    return cache_stats()


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Pass each message through the hybrid engine and persist its trace."""
    message = request.message.strip()
    cid = request.conversation_id
    history = recent_messages(cid)
    save_message(cid, "user", message)
    try:
        result = run_hybrid_engine(message, history, conversation_id=cid)
        route, tool, reply = result.route, result.tool, result.reply
        save_trace(cid, route, tool, result.input_summary, reply, result.status)
    except ValueError as error:
        route = "needs_clarification"
        tool = None
        reply = str(error)
        save_trace(cid, route, None, message, reply, "needs_clarification")
    except AssistantError as error:
        # Each layer reports its own failure, so the reply names the real stage
        # instead of guessing from the intent. Detail goes to the terminal only.
        logger.warning("Chat failed at stage '%s': %s", error.route, error.detail)
        route, tool = error.route, error.tool
        reply = error.user_message
        save_trace(cid, route, tool, message, reply, "error")
    except Exception:
        logger.exception("Unexpected chat failure")
        route, tool = "internal_error", None
        reply = ("Something went wrong inside the assistant. Your saved notes, "
                 "to-dos and documents are unchanged. Check the API terminal for details.")
        save_trace(cid, route, None, message, reply, "error")

    save_message(cid, "assistant", reply)
    return {"conversation_id": cid, "route": route, "reply": reply}


@app.post("/notes", status_code=201)
def create_note(note: NoteCreate) -> dict:
    return create_item("notes", ("title", "content"), (note.title, note.content))


@app.get("/notes")
def list_notes() -> list:
    with get_connection() as db:
        return [dict(row) for row in db.execute("SELECT * FROM notes ORDER BY id DESC")]


@app.post("/todos", status_code=201)
def create_todo(todo: TodoCreate) -> dict:
    due_at = utc_text(todo.due_at) if todo.due_at else None
    return create_item("todos", ("title", "details", "due_at"), (todo.title, todo.details, due_at))


@app.get("/todos")
def list_todos() -> list:
    with get_connection() as db:
        return [dict(row) for row in db.execute("SELECT * FROM todos ORDER BY id DESC")]


@app.patch("/todos/{todo_id}/complete")
def complete_todo(todo_id: int) -> dict:
    with get_connection() as db:
        result = db.execute(
            "UPDATE todos SET status='completed', completed_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
            "WHERE id=? AND status='open'", (todo_id,)
        )
        if not result.rowcount:
            raise HTTPException(status_code=404, detail="Open to-do not found")
        return dict(db.execute("SELECT * FROM todos WHERE id=?", (todo_id,)).fetchone())


@app.post("/reminders", status_code=201)
def create_reminder(reminder: ReminderCreate) -> dict:
    return create_item("reminders", ("title", "remind_at"),
                       (reminder.title, utc_text(reminder.remind_at)))


@app.get("/reminders")
def list_reminders() -> list:
    with get_connection() as db:
        return [dict(row) for row in db.execute("SELECT * FROM reminders ORDER BY remind_at")]


def _derive_title(first_user_message: Optional[str], conversation_id: str,
                  limit: int = 45) -> str:
    """Build a readable conversation title from its first user message."""
    text = " ".join((first_user_message or "").split())
    if not text:
        return conversation_id
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


@app.get("/conversations")
def list_conversations() -> list:
    """Derive the conversation list from the messages table.

    There is no conversations table: the title comes from the first user message
    and the timestamps come from the message rows themselves, so no schema
    change is needed to list conversations.
    """
    with get_connection() as db:
        rows = db.execute(
            """SELECT m.conversation_id,
                      COUNT(*) AS message_count,
                      MIN(m.created_at) AS started_at,
                      MAX(m.created_at) AS last_activity_at,
                      (SELECT content FROM messages
                        WHERE conversation_id = m.conversation_id AND role = 'user'
                        ORDER BY id LIMIT 1) AS first_user_message
               FROM messages m
               GROUP BY m.conversation_id
               ORDER BY MAX(m.id) DESC"""
        ).fetchall()
    conversations = []
    for row in rows:
        item = dict(row)
        item["title"] = _derive_title(item.pop("first_user_message"), item["conversation_id"])
        conversations.append(item)
    return conversations


@app.get("/conversations/{conversation_id}/messages")
def conversation_history(conversation_id: str) -> list:
    with get_connection() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,)
        )]


@app.post("/documents", status_code=201)
async def upload_documents(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
) -> dict:
    """Ingest one or many PDFs; the singular `file` field remains supported."""
    uploads = files or ([file] if file is not None else [])
    if not uploads:
        raise HTTPException(status_code=400, detail="Choose at least one PDF file.")

    prepared = []
    errors = []
    seen_names = set()
    for uploaded in uploads:
        filename = (uploaded.filename or "document.pdf").replace("\\", "/").split("/")[-1][:255]
        if filename in seen_names:
            errors.append({"filename": filename, "error": "This filename was selected more than once."})
            continue
        seen_names.add(filename)
        if not filename.lower().endswith(".pdf"):
            errors.append({"filename": filename, "error": "Upload a PDF file."})
            continue
        content = await uploaded.read(15 * 1024 * 1024 + 1)
        if len(content) > 15 * 1024 * 1024:
            errors.append({"filename": filename, "error": "PDF must be 15 MB or smaller."})
            continue
        try:
            from io import BytesIO
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                errors.append({"filename": filename, "error": "Password-protected PDFs are not supported."})
                continue
            page_parents = []
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                for parent_index, (parent, children) in enumerate(split_parent_child(page_text)):
                    page_parents.append((page_number, parent_index, parent, children))
            if not page_parents:
                errors.append({
                    "filename": filename,
                    "error": "No selectable text found; scanned PDFs need OCR, which is not included.",
                })
                continue
            prepared.append((filename, page_parents))
        except Exception as error:
            errors.append({"filename": filename, "error": "Could not read PDF: {}".format(error)})

    try:
        indexed = store_documents(prepared)
    except Exception:
        logger.exception("Document index build failed")
        raise HTTPException(
            status_code=500,
            detail="Could not build the document index. Check the API terminal for details.",
        )

    child_count = sum(item["chunks"] for item in indexed)
    parent_count = sum(item["parent_chunks"] for item in indexed)
    return {
        "indexed": len(indexed),
        "failed": len(errors),
        "documents": indexed,
        "errors": errors,
        "message": "Indexed {} PDF(s), {} child chunks, and {} parent sections.".format(
            len(indexed), child_count, parent_count
        ),
    }


@app.get("/documents")
def list_documents() -> list:
    with get_connection() as db:
        return [dict(row) for row in db.execute(
            """SELECT d.id, d.filename, d.page_count, d.chunk_count, d.created_at,
                      (SELECT COUNT(*) FROM parent_chunks p WHERE p.document_id=d.id)
                        AS parent_chunk_count
               FROM documents d ORDER BY d.id DESC"""
        )]
