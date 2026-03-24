"""Monkeypatch OpenAI client to record all chat completion calls."""

from __future__ import annotations

import os
import time
import functools
from typing import Any

from starnose.db import Database


_patched = False


def patch_openai() -> None:
    """Monkeypatch openai.OpenAI and openai.AsyncOpenAI to intercept chat completions."""
    global _patched
    if _patched:
        return
    _patched = True

    try:
        import openai
    except ImportError:
        return

    db = Database()
    run_id = os.environ.get("STARNOSE_RUN_ID")

    if not run_id:
        run = db.create_run(name="openai-patch")
        run_id = run.id

    # Patch sync client
    _original_create = openai.resources.chat.completions.Completions.create

    @functools.wraps(_original_create)
    def _patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", args[0] if args else [])
        model = kwargs.get("model", "unknown")
        start = time.time()

        result = _original_create(self, *args, **kwargs)

        latency_ms = int((time.time() - start) * 1000)
        params = {
            k: v for k, v in kwargs.items()
            if k not in ("messages", "model", "stream")
        }

        try:
            response_data = result.model_dump() if hasattr(result, "model_dump") else {}
            db.add_call(run_id, model, params, messages, response_data, latency_ms)
        except Exception:
            pass

        return result

    openai.resources.chat.completions.Completions.create = _patched_create

    # Patch async client
    try:
        _original_async_create = openai.resources.chat.completions.AsyncCompletions.create

        @functools.wraps(_original_async_create)
        async def _patched_async_create(self: Any, *args: Any, **kwargs: Any) -> Any:
            messages = kwargs.get("messages", args[0] if args else [])
            model = kwargs.get("model", "unknown")
            start = time.time()

            result = await _original_async_create(self, *args, **kwargs)

            latency_ms = int((time.time() - start) * 1000)
            params = {
                k: v for k, v in kwargs.items()
                if k not in ("messages", "model", "stream")
            }

            try:
                response_data = result.model_dump() if hasattr(result, "model_dump") else {}
                db.add_call(run_id, model, params, messages, response_data, latency_ms)
            except Exception:
                pass

            return result

        openai.resources.chat.completions.AsyncCompletions.create = _patched_async_create
    except AttributeError:
        pass
