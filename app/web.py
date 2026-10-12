"""Flask browser interface that talks to the FastAPI service."""
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
API_BASE = os.getenv("ASSISTANT_API_URL", "http://127.0.0.1:8000")
# Existing history lives under this id, so it stays the landing conversation.
DEFAULT_CONVERSATION_ID = "flask-demo"
# Ids are interpolated into API paths, so restrict them to URL-safe characters.
SAFE_CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
IST = timezone(timedelta(hours=5, minutes=30))


def to_ist(value):
    """Format a stored UTC timestamp as readable India Standard Time."""
    if not value:
        return ""
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return value
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(IST).strftime("%d %b %Y, %I:%M %p")


app.jinja_env.filters["ist"] = to_ist


def current_conversation_id():
    """Resolve the active conversation from the POST form or the query string.

    Both forms post a hidden conversation_id, so a sent message or an uploaded
    PDF always belongs to the conversation on screen rather than the default.
    """
    for candidate in (request.form.get("conversation_id"), request.args.get("c")):
        value = (candidate or "").strip()
        if SAFE_CONVERSATION_ID.match(value):
            return value
    return DEFAULT_CONVERSATION_ID


def api_get(path):
    req = Request(API_BASE + path, headers={"Accept": "application/json"})
    with urlopen(req, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def api_post(path, payload):
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        API_BASE + path,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def with_active_conversation(conversations, conversation_id):
    """Show a brand-new chat in the list before its first message is saved.

    An empty chat is never written to SQLite, so the active conversation is
    added to the displayed list in memory only.
    """
    if any(item.get("conversation_id") == conversation_id for item in conversations):
        return conversations
    placeholder = {
        "conversation_id": conversation_id,
        "title": "New chat",
        "message_count": 0,
        "started_at": None,
        "last_activity_at": None,
    }
    return [placeholder] + conversations


def page_data(conversation_id):
    error = None
    conversations = []
    try:
        messages = api_get("/conversations/{}/messages".format(conversation_id))
        todos = api_get("/todos")
        documents = api_get("/documents")
        conversations = api_get("/conversations")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        messages, todos, documents = [], [], []
        error = "Could not reach the API at {}. Start the FastAPI service first. ({})".format(
            API_BASE, exc
        )
    else:
        conversations = with_active_conversation(conversations, conversation_id)
    return messages, todos, documents, conversations, error


def api_upload(files):
    boundary = "----HybridAssistantPdfUploadBoundary"
    parts = []
    for filename, content in files:
        parts.append(
            ("--{}\r\n"
             'Content-Disposition: form-data; name="files"; filename="{}"\r\n'
             "Content-Type: application/pdf\r\n\r\n").format(boundary, filename).encode("utf-8")
            + content + b"\r\n"
        )
    body = b"".join(parts) + ("--{}--\r\n".format(boundary)).encode("ascii")
    req = Request(
        API_BASE + "/documents", data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary,
                 "Accept": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


@app.route("/new")
def new_chat():
    """Start an empty conversation. Nothing is stored until a message is sent."""
    return redirect(url_for("chat_page", c="chat-" + uuid.uuid4().hex[:10]))


@app.route("/", methods=["GET", "POST"])
def chat_page():
    conversation_id = current_conversation_id()
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if not message:
            return redirect(url_for("chat_page", c=conversation_id,
                                    error="Type a message before sending."))
        try:
            api_post("/chat", {"message": message, "conversation_id": conversation_id})
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            return redirect(url_for("chat_page", c=conversation_id,
                                    error="The API request failed: {}".format(exc)))
        # Redirect after POST so the URL keeps ?c= and a refresh does not resend.
        return redirect(url_for("chat_page", c=conversation_id))

    messages, todos, documents, conversations, api_error = page_data(conversation_id)
    return render_template("chat.html", messages=messages, todos=todos, documents=documents,
                           conversations=conversations, conversation_id=conversation_id,
                           error=request.args.get("error") or api_error,
                           notice=request.args.get("notice"))


@app.route("/upload", methods=["POST"])
def upload_pdf():
    conversation_id = current_conversation_id()
    uploaded_files = request.files.getlist("files") or request.files.getlist("file")
    files = [
        (secure_filename(uploaded.filename), uploaded.read())
        for uploaded in uploaded_files if uploaded.filename
    ]
    if not files:
        return redirect(url_for("chat_page", c=conversation_id,
                                notice="Choose at least one PDF file."))
    try:
        result = api_upload(files)
        notice = result.get("message", "PDF upload finished.")
        failures = result.get("errors", [])
        if failures:
            examples = "; ".join(
                "{}: {}".format(item["filename"], item["error"])
                for item in failures[:2]
            )
            extra = "; and {} more".format(len(failures) - 2) if len(failures) > 2 else ""
            notice += " Skipped {} file(s): {}{}".format(len(failures), examples, extra)
        if result.get("indexed", 0):
            notice += " Ask about them with 'document: your question'."
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail", str(exc))
        except (ValueError, OSError):
            detail = str(exc)
        notice = "PDF upload failed: {}".format(detail)
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        notice = "Could not upload the PDF: {}".format(exc)
    return redirect(url_for("chat_page", c=conversation_id, notice=notice))
