"""Local OpenAI/Anthropic-compatible proxy server for intercepting LLM calls."""

from __future__ import annotations

import json
import os
import time
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from starnose.db import Database


def create_proxy_app(db: Database, run_id: str) -> FastAPI:
    """Create a FastAPI proxy app that intercepts and records LLM calls."""
    app = FastAPI(title="starnose-proxy", docs_url=None, redoc_url=None)

    # Determine upstream URLs
    openai_upstream = os.environ.get(
        "STARNOSE_UPSTREAM",
        os.environ.get("STARNOSE_OPENAI_UPSTREAM", "https://api.openai.com"),
    )
    anthropic_upstream = os.environ.get(
        "STARNOSE_ANTHROPIC_UPSTREAM", "https://api.anthropic.com"
    )

    @app.api_route("/v1/chat/completions", methods=["POST"])
    async def proxy_openai_chat(request: Request) -> Response:
        """Proxy OpenAI chat completions — the main interception point."""
        body = await request.body()
        start_ms = _now_ms()

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return await _passthrough_openai(request, body, openai_upstream)

        messages = data.get("messages", [])
        model = data.get("model", "unknown")
        stream = data.get("stream", False)
        params = {
            k: v
            for k, v in data.items()
            if k not in ("messages", "model", "stream")
        }

        # Build upstream headers — forward auth
        headers = _build_openai_headers(request)

        if stream:
            return await _handle_openai_stream(
                openai_upstream, headers, data, messages, model, params,
                start_ms, db, run_id,
            )
        else:
            return await _handle_openai_nonstream(
                openai_upstream, headers, data, messages, model, params,
                start_ms, db, run_id,
            )

    @app.api_route("/v1/messages", methods=["POST"])
    async def proxy_anthropic_messages(request: Request) -> Response:
        """Proxy Anthropic messages API."""
        body = await request.body()
        start_ms = _now_ms()

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return await _passthrough_anthropic(request, body, anthropic_upstream)

        model = data.get("model", "unknown")
        stream = data.get("stream", False)

        # Convert Anthropic messages to internal format
        messages = _anthropic_to_messages(data)
        params = {
            k: v
            for k, v in data.items()
            if k not in ("messages", "model", "stream", "system")
        }

        headers = _build_anthropic_headers(request)

        if stream:
            return await _handle_anthropic_stream(
                anthropic_upstream, headers, data, messages, model, params,
                start_ms, db, run_id,
            )
        else:
            return await _handle_anthropic_nonstream(
                anthropic_upstream, headers, data, messages, model, params,
                start_ms, db, run_id,
            )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_fallback(request: Request, path: str) -> Response:
        """Pass through any other requests untouched."""
        body = await request.body()
        # Determine upstream based on headers
        if request.headers.get("x-api-key") or request.headers.get("anthropic-version"):
            upstream = anthropic_upstream
        else:
            upstream = openai_upstream

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.request(
                method=request.method,
                url=f"{upstream}/{path}",
                headers={
                    k: v for k, v in request.headers.items()
                    if k.lower() not in ("host", "content-length")
                },
                content=body,
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    return app


# ── Helpers ──────────────────────────────────────────────────────────────────


def _now_ms() -> int:
    return int(time.time() * 1000)


def _build_openai_headers(request: Request) -> dict[str, str]:
    headers = {}
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    for key in ("openai-organization", "openai-project"):
        val = request.headers.get(key)
        if val:
            headers[key] = val
    headers["Content-Type"] = "application/json"
    return headers


def _build_anthropic_headers(request: Request) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = request.headers.get("x-api-key")
    if api_key:
        headers["x-api-key"] = api_key
    version = request.headers.get("anthropic-version")
    if version:
        headers["anthropic-version"] = version
    for key in request.headers:
        if key.lower().startswith("anthropic-"):
            headers[key] = request.headers[key]
    return headers


def _anthropic_to_messages(data: dict) -> list[dict]:
    """Convert Anthropic API format to normalized messages list."""
    messages = []
    # Anthropic system prompt is a top-level field
    system = data.get("system", "")
    if system:
        if isinstance(system, list):
            text_parts = [
                b.get("text", "") for b in system if isinstance(b, dict)
            ]
            system = "\n".join(text_parts)
        messages.append({"role": "system", "content": system})

    for msg in data.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_result":
                        text_parts.append(
                            block.get("content", str(block))
                            if isinstance(block.get("content"), str)
                            else str(block)
                        )
                    else:
                        text_parts.append(block.get("text", str(block)))
                else:
                    text_parts.append(str(block))
            content = "\n".join(text_parts)
        messages.append({"role": msg.get("role", "user"), "content": content})
    return messages


def _record_call(
    db: Database,
    run_id: str,
    model: str,
    params: dict,
    messages: list[dict],
    response: dict,
    latency_ms: int,
) -> None:
    """Record a call to the database, swallowing errors."""
    try:
        db.add_call(run_id, model, params, messages, response, latency_ms)
    except Exception:
        pass  # Fail open — never break the agent


async def _passthrough_openai(
    request: Request, body: bytes, upstream: str
) -> Response:
    """Pass request through without interception."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{upstream}/v1/chat/completions",
            headers={
                k: v for k, v in request.headers.items()
                if k.lower() not in ("host", "content-length")
            },
            content=body,
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


async def _passthrough_anthropic(
    request: Request, body: bytes, upstream: str
) -> Response:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{upstream}/v1/messages",
            headers={
                k: v for k, v in request.headers.items()
                if k.lower() not in ("host", "content-length")
            },
            content=body,
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


async def _handle_openai_nonstream(
    upstream: str,
    headers: dict,
    data: dict,
    messages: list[dict],
    model: str,
    params: dict,
    start_ms: int,
    db: Database,
    run_id: str,
) -> Response:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{upstream}/v1/chat/completions",
                headers=headers,
                json=data,
            )
        latency = _now_ms() - start_ms
        try:
            response_data = resp.json()
        except Exception:
            response_data = {}

        _record_call(db, run_id, model, params, messages, response_data, latency)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except Exception:
        # Fail open
        return Response(
            content=json.dumps({"error": "proxy error"}).encode(),
            status_code=502,
        )


async def _handle_openai_stream(
    upstream: str,
    headers: dict,
    data: dict,
    messages: list[dict],
    model: str,
    params: dict,
    start_ms: int,
    db: Database,
    run_id: str,
) -> StreamingResponse:
    collected_content = []
    finish_reason = None
    usage_data = {}

    async def stream_and_capture() -> AsyncIterator[bytes]:
        nonlocal finish_reason, usage_data
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{upstream}/v1/chat/completions",
                    headers=headers,
                    json=data,
                ) as resp:
                    async for line in resp.aiter_lines():
                        yield (line + "\n").encode()
                        # Parse SSE data
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                chunk = json.loads(line[6:])
                                delta = (
                                    chunk.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if delta:
                                    collected_content.append(delta)
                                fr = (
                                    chunk.get("choices", [{}])[0]
                                    .get("finish_reason")
                                )
                                if fr:
                                    finish_reason = fr
                                if chunk.get("usage"):
                                    usage_data = chunk["usage"]
                            except (json.JSONDecodeError, IndexError):
                                pass
        except Exception:
            pass

        # Record after stream completes
        latency = _now_ms() - start_ms
        full_content = "".join(collected_content)
        response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": full_content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage_data,
        }
        _record_call(db, run_id, model, params, messages, response_data, latency)

    return StreamingResponse(
        stream_and_capture(),
        media_type="text/event-stream",
    )


async def _handle_anthropic_nonstream(
    upstream: str,
    headers: dict,
    data: dict,
    messages: list[dict],
    model: str,
    params: dict,
    start_ms: int,
    db: Database,
    run_id: str,
) -> Response:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{upstream}/v1/messages",
                headers=headers,
                json=data,
            )
        latency = _now_ms() - start_ms
        try:
            response_data = resp.json()
        except Exception:
            response_data = {}

        _record_call(db, run_id, model, params, messages, response_data, latency)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except Exception:
        return Response(
            content=json.dumps({"error": "proxy error"}).encode(),
            status_code=502,
        )


async def _handle_anthropic_stream(
    upstream: str,
    headers: dict,
    data: dict,
    messages: list[dict],
    model: str,
    params: dict,
    start_ms: int,
    db: Database,
    run_id: str,
) -> StreamingResponse:
    collected_content = []
    usage_data = {}
    stop_reason = None

    async def stream_and_capture() -> AsyncIterator[bytes]:
        nonlocal stop_reason, usage_data
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{upstream}/v1/messages",
                    headers=headers,
                    json=data,
                ) as resp:
                    async for line in resp.aiter_lines():
                        yield (line + "\n").encode()
                        if line.startswith("data: "):
                            try:
                                chunk = json.loads(line[6:])
                                if chunk.get("type") == "content_block_delta":
                                    delta = chunk.get("delta", {}).get("text", "")
                                    if delta:
                                        collected_content.append(delta)
                                elif chunk.get("type") == "message_delta":
                                    stop_reason = chunk.get("delta", {}).get("stop_reason")
                                    usage_data = chunk.get("usage", usage_data)
                                elif chunk.get("type") == "message_start":
                                    msg = chunk.get("message", {})
                                    usage_data = msg.get("usage", usage_data)
                            except (json.JSONDecodeError, KeyError):
                                pass
        except Exception:
            pass

        latency = _now_ms() - start_ms
        full_content = "".join(collected_content)
        response_data = {
            "content": [{"type": "text", "text": full_content}],
            "stop_reason": stop_reason,
            "usage": usage_data,
        }
        _record_call(db, run_id, model, params, messages, response_data, latency)

    return StreamingResponse(
        stream_and_capture(),
        media_type="text/event-stream",
    )


def find_free_port() -> int:
    """Find a free port on localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_proxy_server(db: Database, run_id: str, port: int) -> None:
    """Run the proxy server (blocking). Intended to be run in a thread."""
    app = create_proxy_app(db, run_id)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()
