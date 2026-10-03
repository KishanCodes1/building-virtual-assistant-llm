"""Classify chat input into deterministic tool, RAG, or LLM intents."""
from dataclasses import dataclass
import re


@dataclass
class Intent:
    name: str
    payload: str


def classify_intent(message: str) -> Intent:
    """Recognize explicit commands plus unambiguous command-style phrases."""
    text = message.strip()
    command = re.match(r"^(note|todo|reminder|document)\s*:\s*(.*)$", text, flags=re.IGNORECASE)
    if command:
        name = {
            "note": "save_note",
            "todo": "add_todo",
            "reminder": "create_reminder",
            "document": "document_qa",
        }[command.group(1).lower()]
        return Intent(name, command.group(2).strip())

    patterns = (
        (r"^(?:add|create)\s+(?:a\s+)?(?:to-?do|task)\s*[:\-]\s*(.+)$", "add_todo"),
        (r"^(?:save|create)\s+(?:a\s+)?note\s*[:\-]\s*(.+)$", "save_note"),
        (r"^(?:ask|search)\s+(?:the\s+)?(?:uploaded\s+)?document\s*[:\-]\s*(.+)$", "document_qa"),
        (r"^create\s+(?:a\s+)?reminder\s*[:\-]\s*(.+)$", "create_reminder"),
    )
    for pattern, name in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return Intent(name, match.group(1).strip())
    return Intent("general_chat", text)
