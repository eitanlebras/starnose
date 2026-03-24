"""Tests for starnose proxy server."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi.testclient import TestClient

from starnose.db import Database
from starnose.proxy import create_proxy_app, find_free_port


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def run_id(db):
    run = db.create_run(name="test-proxy")
    return run.id


@pytest.fixture
def app(db, run_id):
    return create_proxy_app(db, run_id)


@pytest.fixture
def client(app):
    return TestClient(app)


class TestProxyOpenAI:
    def test_chat_completions_non_stream(self, client, db, run_id):
        """Test non-streaming chat completion interception."""
        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.content = json.dumps(mock_response).encode()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_resp.json.return_value = mock_response
            mock_client.post = AsyncMock(return_value=mock_resp)

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                headers={"Authorization": "Bearer test-key"},
            )

        assert response.status_code == 200

        # Verify call was recorded
        run = db.get_run(run_id)
        assert len(run.calls) == 1
        assert run.calls[0].model == "gpt-4o"


class TestProxyAnthropic:
    def test_messages_non_stream(self, client, db, run_id):
        """Test Anthropic messages API interception."""
        mock_response = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 15, "output_tokens": 5},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.content = json.dumps(mock_response).encode()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_resp.json.return_value = mock_response
            mock_client.post = AsyncMock(return_value=mock_resp)

            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-opus-20240229",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "system": "You are helpful.",
                    "max_tokens": 100,
                },
                headers={
                    "x-api-key": "test-key",
                    "anthropic-version": "2023-06-01",
                },
            )

        assert response.status_code == 200

        run = db.get_run(run_id)
        assert len(run.calls) == 1


class TestFindFreePort:
    def test_returns_valid_port(self):
        port = find_free_port()
        assert 1024 <= port <= 65535

    def test_returns_different_ports(self):
        ports = {find_free_port() for _ in range(5)}
        # At least some should be different
        assert len(ports) >= 2


class TestProxyFallthrough:
    def test_unknown_path_passes_through(self, client):
        """Non-chat-completion paths should pass through."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.content = b'{"models": []}'
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/json"}
            mock_client.request = AsyncMock(return_value=mock_resp)

            response = client.get("/v1/models")

        assert response.status_code == 200
