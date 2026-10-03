# Hybrid LLM Virtual Assistant

## Hybrid approach

FastAPI passes every chat message to the hybrid engine. The engine has four explicit components:

1. **Intent classifier** labels command-style input as a note, to-do, reminder, document question, or general chat.
2. **Rule router** sends exact actions to Python tools, PDF questions to RAG, and open-ended questions to Gemini.
3. **Tool executor** writes notes, to-dos, and reminders directly to SQLite; these changes do not depend on a model response.
4. **Output validator** rejects empty replies and checks tool confirmations before the API returns them.

### Conversation response cache

Before a general or document question reaches Gemini, the hybrid engine checks a persistent SQLite response cache. The key includes the normalized question, conversation ID, relevant chat context, model, output-format version, and (for RAG) the indexed-document revision. A matching response younger than seven days is returned directly and recorded as a `cache_hit`; repeated tool commands still run normally so they cannot accidentally create duplicate or skipped actions. New PDF uploads automatically produce a different RAG cache key. `GET /cache/stats` reports live entries and cache hits. This is an application-level answer cache, separate from Gemini's provider-side prompt caching.

### Document ingestion and parent-child RAG

On PDF upload, the ingestion pipeline extracts selectable text page by page. Each page is split into larger **parent chunks** (up to about 1,800 characters), then smaller overlapping **child chunks** (about 600 characters). SQLite stores the parent/child relationship and source page. Sentence Transformers creates embeddings for child chunks, and FAISS indexes those vectors.

For a `document:` question, the RAG layer searches with both **BM25** and FAISS. BM25 ranks child chunks that share important query terms; FAISS finds semantically similar chunks even when wording differs. Reciprocal-rank fusion combines the two ranked lists. The system then uses the linked parent passages as context and injects them, with filename and page metadata, into Gemini's prompt. Gemini is instructed to answer from that retrieved context and cite the PDF page. The child makes retrieval specific; the parent gives the answer more surrounding context. Scanned image-only PDFs still need OCR, which is not part of this MVP.

### Gemini structured JSON output

Gemini's general-chat and document-answer paths request **structured output mode** (`application/json`) with a Pydantic schema. The model returns an `answer` and a `citations` list; Python validates the JSON, checks document citations against retrieved filename/page pairs, and then displays readable text in the chat. This makes the model-to-backend format predictable and avoids hand-parsing free-form replies. It does not by itself reduce Gemini token usage or guarantee faster responses. Exact notes, to-dos, and reminders still use deterministic Python routes and SQLite.

### Project architecture

```text
Flask presentation (chat, task panel, PDF upload)
                    ↓
FastAPI API (/chat, /documents, notes, to-dos, reminders)
                    ↓
Hybrid engine: intent classifier → rule router → SQLite response cache → tool executor / RAG / Gemini → output validator
                    ↓
Python/SQLite tools | Gemini structured JSON chat | RAG: BM25 + FAISS → parent passage injection → Gemini structured JSON
                    ↓
SQLite (messages, notes, tasks, reminders, traces, chunks) + FAISS (child embeddings)
```

## Run the project

Install dependencies from the project folder:

```bash
python3 -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Google AI Studio key as `GEMINI_API_KEY`. Keep `.env` private; it is ignored by Git.

Start the API in **Terminal 1**:

```bash
python3 -m uvicorn app.main:app --reload
```

Start the Flask website in **Terminal 2**:

```bash
python3 -m flask --app app.web run --debug
```

Open the Flask chat at `http://127.0.0.1:5000`. API documentation is at `http://127.0.0.1:8000/docs`.

To demonstrate the response cache, ask the same standalone question twice in the same conversation. The first request should use Gemini; the repeat should return the cached answer. View `http://127.0.0.1:8000/cache/stats` to see the entry and hit counters. The cache expires after seven days.

## Use document Q&A

Select one or more selectable-text PDFs in the Flask upload form; each file must be 15 MB or smaller. They are ingested as a batch and the FAISS index is rebuilt once after the batch. The first indexing request downloads the Sentence Transformers model once. Ask about the uploaded content with a message such as `document: What are the project objectives?`. Replies should cite their source as `[filename, p. N]`. Image-only scanned PDFs need OCR, which is not included in this MVP.

## Tool commands

- `note: title | note content`
- `todo: task description`
- `reminder: title | 2026-10-01T18:00:00+05:30`

Other messages go to Gemini. SQLite and the FAISS vector index are separate stores under `data/`. BM25 reads the stored child chunks from SQLite. Existing flat-chunk records remain searchable, but re-upload a PDF to create its parent-child hierarchy.
