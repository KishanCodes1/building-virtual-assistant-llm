"""Coordinate the hybrid rule, tool, RAG, and LLM execution paths."""
from typing import List, Optional, Tuple

from app.engine.intent_classifier import Intent, classify_intent
from app.engine.output_validator import validate_output, validate_tool_result
from app.engine.rule_router import rule_router
from app.engine.tool_executor import execute_tool
from app.llm.gemini_client import generate_reply
from app.rag.vector_store import document_corpus_version, search_documents
from app.services.response_cache import get_cached_response, save_cached_response


class EngineResult:
    def __init__(self, route: str, tool: Optional[str], reply: str,
                 status: str = "success", input_summary: str = ""):
        self.route = route
        self.tool = tool
        self.reply = reply
        self.status = status
        self.input_summary = input_summary


def run_hybrid_engine(message: str, history: List[Tuple[str, str]],
                      conversation_id: str = "default") -> EngineResult:
    """Route one message through tools, parent-child RAG, or Gemini chat."""
    intent = classify_intent(message)
    route = rule_router(intent)

    if route == "python_tool":
        record, tool = execute_tool(intent)
        record = validate_tool_result(intent.name, record)
        if intent.name == "save_note":
            reply = "Saved note '{}'. (ID {})".format(record["title"], record["id"])
        elif intent.name == "add_todo":
            reply = "Added to-do '{}'. (ID {})".format(record["title"], record["id"])
        else:
            reply = "Created reminder '{}'. (ID {})".format(record["title"], record["id"])
        return EngineResult(intent.name, tool, validate_output(route, reply),
                            input_summary=message)

    if route == "rag":
        if not intent.payload:
            raise ValueError("Use: document: your question about the uploaded PDF")
        source_version = document_corpus_version()
        cached = get_cached_response(
            conversation_id, "document_qa", intent.payload, history, source_version
        )
        if cached is not None:
            return EngineResult("cache_hit", "sqlite_response_cache", cached,
                                input_summary="document_qa response cache hit")
        sources = search_documents(intent.payload)
        if sources:
            # Search matches child chunks; linked parent passages are injected into
            # Gemini's prompt so the answer receives useful surrounding context.
            reply = generate_reply(intent.payload, history, sources=sources)
        else:
            reply = "I could not find an answer to that in the uploaded documents."
        save_cached_response(
            conversation_id, "document_qa", intent.payload, history, reply, source_version
        )
        return EngineResult("document_qa", "parent_child_bm25_faiss_retrieval",
                            validate_output(route, reply), input_summary=intent.payload)

    cached = get_cached_response(conversation_id, "llm_chat", message, history)
    if cached is not None:
        return EngineResult("cache_hit", "sqlite_response_cache", cached,
                            input_summary="llm_chat response cache hit")
    reply = generate_reply(message, history)
    save_cached_response(conversation_id, "llm_chat", message, history, reply)
    return EngineResult("llm_chat", "gemini", validate_output(route, reply),
                        input_summary=message)
