"""Python SDK — trace decorator and snapshot context manager."""

from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, Generator

from starnose.db import Database, Run


class TraceContext:
    """Active trace context for recording snapshots and metadata."""

    def __init__(self, db: Database, run: Run):
        self.db = db
        self.run = run
        self.run_id = run.id
        self._snapshot_count = 0

    def snapshot(self, label: str, messages: list[dict], model: str = "gpt-4o") -> None:
        """Record a snapshot of the current message state."""
        self._snapshot_count += 1
        params = {"snapshot_label": label, "snapshot_index": self._snapshot_count}
        response: dict[str, Any] = {"choices": [], "usage": {}}
        self.db.add_call(
            run_id=self.run_id,
            model=model,
            params=params,
            messages=messages,
            response=response,
            latency_ms=0,
        )

    def set_metadata(self, **kwargs: Any) -> None:
        """Attach metadata to the current run."""
        self.db.update_run_metadata(self.run_id, kwargs)


@contextmanager
def trace(
    name: str | None = None,
    tags: list[str] | None = None,
) -> Generator[TraceContext, None, None]:
    """Context manager for tracing agent runs.

    Usage:
        with trace("my-run", tags=["prod"]) as t:
            result = agent.run(query)
            t.snapshot("after-retrieval", messages)
    """
    # If inside a snose run, attach to existing run
    existing_run_id = os.environ.get("STARNOSE_RUN_ID")
    db = Database()

    if existing_run_id:
        run = db.get_run(existing_run_id)
        if not run:
            run = db.create_run(name=name, tags=tags)
    else:
        effective_name = name or os.environ.get("STARNOSE_RUN_NAME")
        run = db.create_run(name=effective_name, tags=tags)

    ctx = TraceContext(db, run)
    try:
        yield ctx
        if not existing_run_id:
            db.update_run_status(run.id, "success")
    except Exception:
        if not existing_run_id:
            db.update_run_status(run.id, "failed")
        raise


def snapshot(label: str, messages: list[dict], model: str = "gpt-4o") -> None:
    """Standalone snapshot — records to the active run or creates one."""
    run_id = os.environ.get("STARNOSE_RUN_ID")
    db = Database()

    if run_id:
        run = db.get_run(run_id)
    else:
        run = db.get_last_run()

    if not run:
        run = db.create_run(name="snapshot")

    ctx = TraceContext(db, run)
    ctx.snapshot(label, messages, model)
