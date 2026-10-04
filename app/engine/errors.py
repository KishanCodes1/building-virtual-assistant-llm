"""Typed failures so the API can report which stage failed instead of guessing.

Each class carries a fixed ``user_message`` that is safe to show in the browser.
Technical detail goes to ``detail``, which is written to the API terminal log only.
Keeping ``user_message`` a constant means no stack trace, provider text, or API
key can ever reach the chat page.
"""


class AssistantError(Exception):
    """Base class for failures the chat endpoint can describe accurately."""

    route = "error"
    tool = None
    user_message = "Something went wrong. Please try again."

    def __init__(self, detail: str = ""):
        super().__init__(detail or self.user_message)
        self.detail = detail


class RetrievalError(AssistantError):
    """Document search itself failed: index, embedding model, or database."""

    route = "document_error"
    tool = "parent_child_bm25_faiss_retrieval"
    user_message = (
        "I couldn't search your uploaded documents. The document index or the "
        "embedding model could not be loaded. Your PDFs and saved data are unchanged."
    )


class LLMConfigError(AssistantError):
    """No Gemini API key is configured."""

    route = "llm_config_error"
    tool = "gemini"
    user_message = (
        "Gemini is not configured. Add GEMINI_API_KEY to your local .env file "
        "and restart the API."
    )


class LLMUnavailableError(AssistantError):
    """HTTP 503/500/502/504 - the provider is overloaded. Temporary."""

    route = "llm_unavailable"
    tool = "gemini"
    user_message = (
        "Gemini is temporarily unavailable and did not answer (server error 500, "
        "502, 503, or 504). This is a problem at Google's end, not with your "
        "question, your API key, or your documents. Please try again in a moment."
    )


class LLMQuotaError(AssistantError):
    """HTTP 429 - rate limit or quota exhausted. Not the same as a 503."""

    route = "llm_quota"
    tool = "gemini"
    user_message = (
        "Gemini refused the request because the rate limit or quota for this API "
        "project is used up (429). Wait for the quota window to reset. A second "
        "key in the same Google project shares the same quota."
    )


class LLMTimeoutError(AssistantError):
    """Gemini did not answer within the configured timeout."""

    route = "llm_timeout"
    tool = "gemini"
    user_message = "Gemini did not respond within the time limit. Please try again."


class LLMResponseError(AssistantError):
    """The model replied, but the reply failed Pydantic schema validation."""

    route = "llm_invalid_output"
    tool = "gemini"
    user_message = (
        "Gemini's reply did not match the required JSON format, so it was "
        "rejected instead of being shown to you. Please try again."
    )


class LLMServiceError(AssistantError):
    """Any other Gemini failure that is not overload, quota, or bad output."""

    route = "llm_error"
    tool = "gemini"
    user_message = (
        "Gemini could not complete the request. Check the API terminal for the "
        "technical detail."
    )
