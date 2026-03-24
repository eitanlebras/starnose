"""LangChain callback handler for starnose."""

from __future__ import annotations

import os
import time
from typing import Any
from uuid import UUID

from starnose.db import Database


class LangChainTracer:
    """LangChain BaseCallbackHandler that records LLM and tool calls to starnose.

    Usage:
        from starnose.integrations import LangChainTracer
        agent = AgentExecutor(..., callbacks=[LangChainTracer()])
    """

    def __init__(self, name: str | None = None, tags: list[str] | None = None):
        self.db = Database()
        run_id = os.environ.get("STARNOSE_RUN_ID")
        if run_id:
            self.run = self.db.get_run(run_id)
            if not self.run:
                self.run = self.db.create_run(name=name or "langchain", tags=tags)
        else:
            self.run = self.db.create_run(name=name or "langchain", tags=tags)

        self._call_starts: dict[str, float] = {}
        self._call_messages: dict[str, list[dict]] = {}
        self._tool_starts: dict[str, float] = {}

    # ── LLM callbacks ────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        self._call_starts[key] = time.time()
        messages = [{"role": "user", "content": p} for p in prompts]
        self._call_messages[key] = messages

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        self._call_starts[key] = time.time()

        flat_messages = []
        for msg_list in messages:
            for msg in msg_list:
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    role_map = {
                        "human": "user",
                        "ai": "assistant",
                        "system": "system",
                        "tool": "tool",
                    }
                    role = role_map.get(msg.type, "user")
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    flat_messages.append({"role": role, "content": content})
                elif isinstance(msg, dict):
                    flat_messages.append(msg)
        self._call_messages[key] = flat_messages

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        start = self._call_starts.pop(key, time.time())
        messages = self._call_messages.pop(key, [])
        latency_ms = int((time.time() - start) * 1000)

        # Extract response content
        output_text = ""
        model = "unknown"
        if hasattr(response, "generations") and response.generations:
            gen = response.generations[0]
            if gen:
                output_text = gen[0].text if hasattr(gen[0], "text") else str(gen[0])
        if hasattr(response, "llm_output") and response.llm_output:
            model = response.llm_output.get("model_name", model)

        response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

        try:
            self.db.add_call(
                self.run.id, model, {}, messages, response_data, latency_ms
            )
        except Exception:
            pass

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        self._call_starts.pop(key, None)
        self._call_messages.pop(key, None)

    # ── Tool callbacks ───────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._tool_starts[str(run_id)] = time.time()

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._tool_starts.pop(str(run_id), None)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._tool_starts.pop(str(run_id), None)
