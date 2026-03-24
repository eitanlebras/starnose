"""Tests for starnose token counting."""

import pytest

from starnose.tokens import count_tokens, count_messages, classify_segment, get_context_limit


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("", "gpt-4o") == 0

    def test_known_string(self):
        # "Hello, world!" is typically 4 tokens with cl100k_base
        result = count_tokens("Hello, world!", "gpt-4o")
        assert 3 <= result <= 5

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        result = count_tokens(text, "gpt-4o")
        assert result > 50

    def test_claude_model_fallback(self):
        # Claude models should still work with cl100k_base approximation
        result = count_tokens("Hello world", "claude-3-opus-20240229")
        assert result > 0

    def test_unknown_model_fallback(self):
        result = count_tokens("Hello world", "some-unknown-model-v9")
        assert result > 0

    def test_single_token(self):
        result = count_tokens("a", "gpt-4o")
        assert result == 1


class TestCountMessages:
    def test_basic_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = count_messages(messages, "gpt-4o")
        assert result["total"] > 0
        assert result["by_role"]["system"] > 0
        assert result["by_role"]["user"] > 0
        assert len(result["by_segment"]) == 2

    def test_empty_messages(self):
        result = count_messages([], "gpt-4o")
        assert result["total"] == 2  # just the priming tokens

    def test_all_roles(self):
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Assistant"},
            {"role": "tool", "content": "Tool"},
        ]
        result = count_messages(messages, "gpt-4o")
        assert all(result["by_role"][r] > 0 for r in ["system", "user", "assistant", "tool"])

    def test_content_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            }
        ]
        result = count_messages(messages, "gpt-4o")
        assert result["total"] > 0


class TestClassifySegment:
    def test_system_at_start(self):
        assert classify_segment("system", "You are a bot", 0, 3) == "system_prompt"

    def test_tool_role(self):
        assert classify_segment("tool", "search results", 2, 5) == "tool_result"

    def test_user_is_human(self):
        assert classify_segment("user", "Hello", 1, 3) == "human"

    def test_assistant_last_is_output(self):
        assert classify_segment("assistant", "Response", 2, 3) == "output"

    def test_assistant_mid_is_history(self):
        assert classify_segment("assistant", "Earlier response", 1, 5) == "history"

    def test_memory_keywords(self):
        assert classify_segment("user", "<memory>recalled facts</memory>", 1, 3) == "memory"
        assert classify_segment("user", "[context] retrieved data", 1, 3) == "memory"


class TestGetContextLimit:
    def test_known_models(self):
        assert get_context_limit("gpt-4o") == 128_000
        assert get_context_limit("gpt-4") == 8_192

    def test_claude_models(self):
        assert get_context_limit("claude-3-opus-20240229") == 200_000
        assert get_context_limit("claude-something-new") == 200_000

    def test_unknown_model(self):
        assert get_context_limit("unknown-model") == 128_000
