"""Token counting utilities using tiktoken."""

from __future__ import annotations

from functools import lru_cache

import tiktoken


# Model -> encoding mapping
_MODEL_ENCODINGS: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    # Claude models use cl100k_base as an approximation.
    # Anthropic doesn't publish their tokenizer publicly,
    # so this gives a reasonable estimate (~5-10% variance).
}

# Context limits by model (tokens)
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4": 8_192,
    "gpt-4-turbo": 128_000,
    "gpt-4-turbo-preview": 128_000,
    "gpt-3.5-turbo": 16_385,
    "gpt-3.5-turbo-16k": 16_385,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-20250514": 200_000,
}

DEFAULT_CONTEXT_LIMIT = 128_000


def get_context_limit(model: str) -> int:
    """Get context window size for a model."""
    if model in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model]
    # Fuzzy match for claude models
    if "claude" in model.lower():
        return 200_000
    return DEFAULT_CONTEXT_LIMIT


@lru_cache(maxsize=16)
def _get_encoding(model: str) -> tiktoken.Encoding:
    """Get tiktoken encoding for a model."""
    # Direct match
    if model in _MODEL_ENCODINGS:
        return tiktoken.get_encoding(_MODEL_ENCODINGS[model])

    # Claude approximation
    if model.lower().startswith("claude"):
        return tiktoken.get_encoding("cl100k_base")

    # Try tiktoken's model lookup
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a text string for a given model.

    Falls back to len(text)/4 estimate if tiktoken fails.
    """
    if not text:
        return 0
    try:
        enc = _get_encoding(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def count_messages(
    messages: list[dict], model: str = "gpt-4o"
) -> dict:
    """Count tokens across a message list.

    Returns:
        {
            "total": int,
            "by_role": {"system": int, "user": int, "assistant": int, "tool": int},
            "by_segment": [{"role": str, "tokens": int, "index": int}, ...]
        }
    """
    by_role: dict[str, int] = {"system": 0, "user": 0, "assistant": 0, "tool": 0}
    by_segment: list[dict] = []
    total = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    text_parts.append(block.get("text", str(block)))
                else:
                    text_parts.append(str(block))
            content = "\n".join(text_parts)

        tokens = count_tokens(content, model)
        # Add overhead per message (~4 tokens for role/formatting)
        tokens += 4
        total += tokens
        by_role[role] = by_role.get(role, 0) + tokens
        by_segment.append({"role": role, "tokens": tokens, "index": i})

    # Add 2 tokens for priming
    total += 2

    return {"total": total, "by_role": by_role, "by_segment": by_segment}


def classify_segment(role: str, content: str, position: int, total_messages: int) -> str:
    """Classify a message into a semantic segment type.

    Heuristics:
    - system role at position 0 -> system_prompt
    - tool role -> tool_result
    - user role -> human
    - assistant role at last position -> output
    - assistant role mid-conversation -> history
    - content mentioning memory/recall keywords -> memory
    """
    content_lower = (content or "").lower()[:500]

    if role == "system":
        return "system_prompt"
    if role == "tool":
        return "tool_result"

    # Check for memory/RAG content markers
    memory_keywords = ["<memory>", "[memory]", "recalled", "retrieved context", "<context>", "[context]", "rag", "knowledge base"]
    if any(kw in content_lower for kw in memory_keywords):
        return "memory"

    if role == "assistant":
        if position == total_messages - 1:
            return "output"
        return "history"

    if role == "user":
        return "human"

    return "human"
