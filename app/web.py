"""Flask browser interface that talks to the FastAPI service."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
API_BASE = os.getenv("ASSISTANT_API_URL", "http://127.0.0.1:8000")
CONVERSATION_ID = "flask-demo"


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


def page_data():
    error = None
    try:
        messages = api_get("/conversations/{}/messages".format(CONVERSATION_ID))
        todos = api_get("/todos")
        documents = api_get("/documents")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        messages, todos, documents = [], [], []
        error = "Could not reach the API at {}. Start the FastAPI service first. ({})".format(
            API_BASE, exc
        )
    return messages, todos, documents, error


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


@app.route("/", methods=["GET", "POST"])
def chat_page():
    error = None
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if not message:
            error = "Type a message before sending."
        else:
            try:
                api_post("/chat", {"message": message, "conversation_id": CONVERSATION_ID})
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                error = "The API request failed: {}".format(exc)

    messages, todos, documents, api_error = page_data()
    return render_template("chat.html", messages=messages, todos=todos, documents=documents,
                           error=error or api_error, notice=request.args.get("notice"))


@app.route("/upload", methods=["POST"])
def upload_pdf():
    uploaded_files = request.files.getlist("files") or request.files.getlist("file")
    files = [
        (secure_filename(uploaded.filename), uploaded.read())
        for uploaded in uploaded_files if uploaded.filename
    ]
    if not files:
        return redirect(url_for("chat_page", notice="Choose at least one PDF file."))
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
    return redirect(url_for("chat_page", notice=notice))
