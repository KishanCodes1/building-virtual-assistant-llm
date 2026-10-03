"""Check route outputs before the API returns or stores them."""


def validate_tool_result(intent_name: str, record: dict) -> dict:
    """Ensure a database tool returned the fields its confirmation relies on."""
    if not isinstance(record, dict) or not isinstance(record.get("id"), int):
        raise ValueError("The tool did not return a valid saved record.")
    required = {
        "save_note": ("title", "content"),
        "add_todo": ("title",),
        "create_reminder": ("title", "remind_at"),
    }.get(intent_name)
    if required is None or any(not record.get(field) for field in required):
        raise ValueError("The saved record is missing required fields.")
    return record


def validate_output(route: str, reply: str) -> str:
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("The selected route did not produce a usable reply.")
    cleaned = reply.strip()
    if route == "python_tool" and len(cleaned) > 1000:
        raise ValueError("The tool confirmation was unexpectedly long.")
    return cleaned
