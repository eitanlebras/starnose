"""Monkeypatch Anthropic client to record all message calls."""

from __future__ import annotations

import os
import time
import functools
from typing import Any

from starnose.db import Database


_patched = False


def patch_anthropic() -> None:
    """Monkeypatch anthropic.Anthropic to intercept messages.create calls."""
    global _patched
    if _patched:
        return
    _patched = True

    try:
        import anthropic
    except ImportError:
        return

    db = Database()
    run_id = os.environ.get("STARNOSE_RUN_ID")

    if not run_id:
        run = db.create_run(name="anthropic-patch")
        run_id = run.id

    # Patch sync client
    _original_create = anthropic.resources.messages.Messages.create

    @functools.wraps(_original_create)
    def _patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        start = time.time()

        result = _original_create(self, *args, **kwargs)

        latency_ms = int((time.time() - start) * 1000)

        # Convert Anthropic format to internal
        messages = _extract_messages(kwargs)
        params = {
            k: v for k, v in kwargs.items()
            if k not in ("messages", "model", "stream", "system")
        }

        try:
            response_data = result.model_dump() if hasattr(result, "model_dump") else {}
            db.add_call(run_id, model, params, messages, response_data, latency_ms)
        except Exception:
            pass

        return result

    anthropic.resources.messages.Messages.create = _patched_create

    # Patch async client
    try:
        _original_async_create = anthropic.resources.messages.AsyncMessages.create

        @functools.wraps(_original_async_create)
        async def _patched_async_create(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "unknown")
            start = time.time()

            result = await _original_async_create(self, *args, **kwargs)

            latency_ms = int((time.time() - start) * 1000)
            messages = _extract_messages(kwargs)
            params = {
                k: v for k, v in kwargs.items()
                if k not in ("messages", "model", "stream", "system")
            }

            try:
                response_data = result.model_dump() if hasattr(result, "model_dump") else {}
                db.add_call(run_id, model, params, messages, response_data, latency_ms)
            except Exception:
                pass

            return result

        anthropic.resources.messages.AsyncMessages.create = _patched_async_create
    except AttributeError:
        pass


def _extract_messages(kwargs: dict) -> list[dict]:
    """Convert Anthropic kwargs to normalized messages list."""
    messages = []
    system = kwargs.get("system", "")
    if system:
        if isinstance(system, list):
            text = "\n".join(
                b.get("text", "") for b in system if isinstance(b, dict)
            )
        else:
            text = str(system)
        messages.append({"role": "system", "content": text})

    for msg in kwargs.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", str(block)))
                else:
                    parts.append(str(block))
            content = "\n".join(parts)
        messages.append({"role": msg.get("role", "user"), "content": content})

    return messages
