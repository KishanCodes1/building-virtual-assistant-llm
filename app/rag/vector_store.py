"""PDF text chunking and local semantic search with Sentence Transformers + FAISS."""
from pathlib import Path
from collections import Counter
import math
import os
import re
import threading
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.database.db import DATABASE_PATH, get_connection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "data" / "documents.faiss"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MIN_SIMILARITY = 0.32
BM25_K1 = 1.5
BM25_B = 0.75
RRF_CONSTANT = 60
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def split_text(text: str, chunk_size: int = 900, overlap: int = 140) -> List[str]:
    """Split text into overlapping chunks, preferring word boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def split_parent_child(text: str, parent_size: int = 1800,
                       child_size: int = 600, child_overlap: int = 100) -> List[Tuple[str, List[str]]]:
    """Create larger context parents and smaller searchable child chunks."""
    parents = split_text(text, chunk_size=parent_size, overlap=0)
    hierarchy = []
    for parent in parents:
        children = split_text(parent, chunk_size=child_size, overlap=child_overlap)
        if children:
            hierarchy.append((parent, children))
    return hierarchy


def tokenize(text: str) -> List[str]:
    """Lowercase tokenization shared by BM25 query and chunk text."""
    return TOKEN_PATTERN.findall(text.lower())


class LocalVectorStore:
    def __init__(self):
        self.model = None
        self.index = None
        self.lock = threading.Lock()

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        return self.model

    def _rows(self):
        with get_connection() as db:
            return db.execute(
                "SELECT id, document_id, page_number, content FROM document_chunks ORDER BY id"
            ).fetchall()

    def _build_index(self, rows):
        model = self._load_model()
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        if rows:
            vectors = model.encode(
                [row["content"] for row in rows],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            ids = np.asarray([row["id"] for row in rows], dtype=np.int64)
            index.add_with_ids(vectors, ids)
        temp_path = INDEX_PATH.with_suffix(".tmp")
        faiss.write_index(index, str(temp_path))
        os.replace(str(temp_path), str(INDEX_PATH))
        self.index = index

    def _ensure_index(self):
        rows = self._rows()
        if self.index is not None and self.index.ntotal == len(rows):
            return rows
        if INDEX_PATH.exists():
            try:
                loaded = faiss.read_index(str(INDEX_PATH))
                if loaded.ntotal == len(rows):
                    self.index = loaded
                    return rows
            except (RuntimeError, OSError):
                pass
        self._build_index(rows)
        return rows

    def rebuild(self):
        with self.lock:
            self._build_index(self._rows())

    def search(self, query: str, top_k: int = 4) -> List[dict]:
        with self.lock:
            rows = self._ensure_index()
            if not rows:
                return []
            model = self._load_model()
            vector = model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            candidate_k = min(max(top_k * 4, 12), self.index.ntotal)
            dense_scores, dense_ids = self.index.search(vector, candidate_k)

            # BM25 scores exact keyword overlap; FAISS scores semantic similarity.
            # Reciprocal-rank fusion combines their result lists without assuming
            # that their raw scores share a scale.
            dense_rank = {}
            for rank, (chunk_id, score) in enumerate(zip(dense_ids[0], dense_scores[0]), start=1):
                chunk_id = int(chunk_id)
                if chunk_id >= 0 and float(score) >= MIN_SIMILARITY:
                    dense_rank[chunk_id] = rank

            query_terms = set(tokenize(query))
            documents = [tokenize(row["content"]) for row in rows]
            document_frequency = Counter(
                term for terms in documents for term in set(terms)
            )
            average_length = (sum(len(terms) for terms in documents) / float(len(documents))) or 1.0
            bm25_scores = []
            if query_terms:
                for row, terms in zip(rows, documents):
                    term_counts = Counter(terms)
                    score = 0.0
                    for term in query_terms:
                        frequency = term_counts.get(term, 0)
                        if not frequency:
                            continue
                        doc_frequency = document_frequency[term]
                        inverse_frequency = math.log(
                            1.0 + (len(documents) - doc_frequency + 0.5) / (doc_frequency + 0.5)
                        )
                        length_norm = BM25_K1 * (
                            1.0 - BM25_B + BM25_B * len(terms) / average_length
                        )
                        score += inverse_frequency * (
                            frequency * (BM25_K1 + 1.0) / (frequency + length_norm)
                        )
                    if score > 0.0:
                        bm25_scores.append((int(row["id"]), score))
            bm25_scores.sort(key=lambda item: item[1], reverse=True)
            bm25_scores = bm25_scores[:candidate_k]
            lexical_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25_scores, start=1)}

            fused_scores = {}
            for chunk_id, rank in dense_rank.items():
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (RRF_CONSTANT + rank)
            for chunk_id, rank in lexical_rank.items():
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (RRF_CONSTANT + rank)
            ranked_ids = [chunk_id for chunk_id, _ in sorted(
                fused_scores.items(), key=lambda item: item[1], reverse=True
            )]
            if not ranked_ids:
                return []
            placeholders = ",".join("?" for _ in ranked_ids)
            with get_connection() as db:
                found = db.execute(
                    """SELECT c.id, c.parent_chunk_id, c.page_number AS child_page,
                              c.content AS child_content,
                              COALESCE(p.content, c.content) AS content,
                              COALESCE(p.page_number, c.page_number) AS page_number,
                              d.filename
                       FROM document_chunks c JOIN documents d ON d.id = c.document_id
                       LEFT JOIN parent_chunks p ON p.id = c.parent_chunk_id
                       WHERE c.id IN ({})""".format(placeholders),
                    ranked_ids,
                ).fetchall()
            by_id = {row["id"]: dict(row) for row in found}
            results = []
            seen_parents = set()
            for chunk_id in ranked_ids:
                if chunk_id not in by_id:
                    continue
                item = by_id[chunk_id]
                parent_key = item["parent_chunk_id"] or ("legacy", chunk_id)
                if parent_key in seen_parents:
                    continue
                seen_parents.add(parent_key)
                item["score"] = float(fused_scores[chunk_id])
                item["retrieval_methods"] = [
                    name for name, ranks in (("FAISS", dense_rank), ("BM25", lexical_rank))
                    if chunk_id in ranks
                ]
                results.append(item)
                if len(results) >= top_k:
                    break
            return results


vector_store = LocalVectorStore()


def document_corpus_version() -> str:
    """Return a cheap revision token that changes when PDFs are re-ingested."""
    with get_connection() as db:
        row = db.execute(
            """SELECT COUNT(*) AS documents, COALESCE(MAX(id), 0) AS latest_id,
                      COALESCE(SUM(chunk_count), 0) AS chunks
               FROM documents"""
        ).fetchone()
    return "{}:{}:{}".format(row["documents"], row["latest_id"], row["chunks"])


def store_document(filename: str, page_parents: List[Tuple[int, int, str, List[str]]]) -> dict:
    """Save one PDF through the shared batch-ingestion path."""
    return store_documents([(filename, page_parents)])[0]


def store_documents(documents: List[Tuple[str, List[Tuple[int, int, str, List[str]]]]]) -> List[dict]:
    """Save a batch of PDFs and rebuild the persisted FAISS index only once."""
    if not documents:
        return []
    results = []
    with get_connection() as db:
        for filename, page_parents in documents:
            child_count = sum(len(children) for _, _, _, children in page_parents)
            # Re-uploading a filename replaces its previous extracted records.
            db.execute("DELETE FROM documents WHERE filename=?", (filename,))
            cursor = db.execute(
                "INSERT INTO documents (filename, page_count, chunk_count) VALUES (?, ?, ?)",
                (filename, len(set(page for page, _, _, _ in page_parents)), child_count),
            )
            document_id = cursor.lastrowid
            child_index = 0
            for page, parent_index, parent_content, children in page_parents:
                parent_cursor = db.execute(
                    """INSERT INTO parent_chunks (document_id, page_number, parent_index, content)
                       VALUES (?, ?, ?, ?)""",
                    (document_id, page, parent_index, parent_content),
                )
                parent_id = parent_cursor.lastrowid
                for child_content in children:
                    db.execute(
                        """INSERT INTO document_chunks
                           (document_id, parent_chunk_id, page_number, chunk_index, content)
                           VALUES (?, ?, ?, ?, ?)""",
                        (document_id, parent_id, page, child_index, child_content),
                    )
                    child_index += 1
            results.append({
                "id": document_id,
                "filename": filename,
                "chunks": child_count,
                "parent_chunks": len(page_parents),
            })
    vector_store.rebuild()
    return results


def search_documents(query: str) -> List[dict]:
    return vector_store.search(query)
