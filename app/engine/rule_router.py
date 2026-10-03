"""Deterministically map classified intents to execution layers."""
from app.engine.intent_classifier import Intent


def rule_router(intent: Intent) -> str:
    return {
        "save_note": "python_tool",
        "add_todo": "python_tool",
        "create_reminder": "python_tool",
        "document_qa": "rag",
        "general_chat": "llm",
    }.get(intent.name, "needs_clarification")
