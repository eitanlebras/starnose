"""SQLite storage layer for starnose runs, calls, and segments."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    ForeignKey,
    create_engine,
    desc,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    relationship,
    sessionmaker,
)

from starnose.tokens import count_tokens, classify_segment


def _default_db_path() -> Path:
    p = Path(os.environ.get("STARNOSE_DB", "~/.starnose/runs.db")).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _gen_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


# ── ORM models ───────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class RunRow(Base):
    __tablename__ = "runs"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    tags = Column(Text, default="[]")
    status = Column(String, default="running")
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    finished_at = Column(DateTime, nullable=True)
    metadata_ = Column("metadata", Text, default="{}")

    calls = relationship("CallRow", back_populates="run", order_by="CallRow.sequence")


class CallRow(Base):
    __tablename__ = "calls"

    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("runs.id"))
    sequence = Column(Integer)
    model = Column(String, nullable=True)
    params = Column(Text, default="{}")
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    finish_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    run = relationship("RunRow", back_populates="calls")
    segments = relationship("SegmentRow", back_populates="call", order_by="SegmentRow.position")


class SegmentRow(Base):
    __tablename__ = "segments"

    id = Column(String, primary_key=True)
    call_id = Column(String, ForeignKey("calls.id"))
    role = Column(String)
    seg_type = Column(String)
    content = Column(Text, default="")
    token_count = Column(Integer, default=0)
    position = Column(Integer, default=0)

    call = relationship("CallRow", back_populates="segments")


# ── Data classes for API results ─────────────────────────────────────────────


@dataclass
class Segment:
    id: str
    call_id: str
    role: str
    seg_type: str
    content: str
    token_count: int
    position: int


@dataclass
class Call:
    id: str
    run_id: str
    sequence: int
    model: str
    params: dict
    input_tokens: int
    output_tokens: int
    latency_ms: int
    finish_reason: str | None
    created_at: datetime
    segments: list[Segment] = field(default_factory=list)


@dataclass
class Run:
    id: str
    name: str | None
    tags: list[str]
    status: str
    started_at: datetime
    finished_at: datetime | None
    metadata: dict
    calls: list[Call] = field(default_factory=list)


@dataclass
class StatsResult:
    total_runs: int
    success_count: int
    failed_count: int
    avg_tokens: float
    avg_latency_ms: float
    avg_input_tokens: float
    avg_output_tokens: float
    segment_breakdown: dict[str, int]
    runs: list[Run]


# ── Database handle ──────────────────────────────────────────────────────────


class Database:
    """Synchronous SQLite database handle."""

    def __init__(self, db_path: str | Path | None = None):
        self.path = Path(db_path) if db_path else _default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.path}",
            echo=False,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        # Enable WAL mode for better concurrent read/write
        from sqlalchemy import event as sa_event

        @sa_event.listens_for(self.engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    def session(self) -> Session:
        return self._Session()

    # ── Run operations ───────────────────────────────────────────────────

    def create_run(
        self,
        name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Run:
        run_id = _gen_id("run")
        now = datetime.now(timezone.utc)
        row = RunRow(
            id=run_id,
            name=name,
            tags=json.dumps(tags or []),
            status="running",
            started_at=now,
            metadata_=json.dumps(metadata or {}),
        )
        with self.session() as s:
            s.add(row)
            s.commit()
        return Run(
            id=run_id,
            name=name,
            tags=tags or [],
            status="running",
            started_at=now,
            finished_at=None,
            metadata=metadata or {},
            calls=[],
        )

    def update_run_status(self, run_id: str, status: str) -> None:
        with self.session() as s:
            row = s.query(RunRow).filter_by(id=run_id).first()
            if row:
                row.status = status
                if status in ("success", "failed"):
                    row.finished_at = datetime.now(timezone.utc)
                s.commit()

    def update_run_metadata(self, run_id: str, metadata: dict) -> None:
        with self.session() as s:
            row = s.query(RunRow).filter_by(id=run_id).first()
            if row:
                existing = json.loads(row.metadata_ or "{}")
                existing.update(metadata)
                row.metadata_ = json.dumps(existing)
                s.commit()

    # ── Call operations ──────────────────────────────────────────────────

    def add_call(
        self,
        run_id: str,
        model: str,
        params: dict,
        messages: list[dict],
        response: dict,
        latency_ms: int,
    ) -> Call:
        call_id = _gen_id("call")

        # Count existing calls for sequencing
        with self.session() as s:
            seq = s.query(CallRow).filter_by(run_id=run_id).count()

        # Parse response for token counts and finish reason
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

        finish_reason = None
        choices = response.get("choices", [])
        if choices:
            finish_reason = choices[0].get("finish_reason")
        elif response.get("stop_reason"):
            finish_reason = response.get("stop_reason")

        # Build segments from messages
        segments: list[Segment] = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle content blocks (e.g. Anthropic format)
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        text_parts.append(block.get("text", str(block)))
                    else:
                        text_parts.append(str(block))
                content = "\n".join(text_parts)

            seg_type = classify_segment(role, content, i, len(messages))
            tok_count = count_tokens(content, model) if content else 0

            seg = Segment(
                id=_gen_id("seg"),
                call_id=call_id,
                role=role,
                seg_type=seg_type,
                content=content,
                token_count=tok_count,
                position=i,
            )
            segments.append(seg)

        # Add output segment from response
        output_text = ""
        if choices:
            output_text = choices[0].get("message", {}).get("content", "") or ""
        elif response.get("content"):
            resp_content = response["content"]
            if isinstance(resp_content, list):
                output_text = "\n".join(
                    b.get("text", "") for b in resp_content if isinstance(b, dict)
                )
            else:
                output_text = str(resp_content)

        if output_text:
            segments.append(
                Segment(
                    id=_gen_id("seg"),
                    call_id=call_id,
                    role="assistant",
                    seg_type="output",
                    content=output_text,
                    token_count=count_tokens(output_text, model) if output_text else 0,
                    position=len(messages),
                )
            )

        # If API didn't return token counts, compute from segments
        if not input_tokens:
            input_tokens = sum(
                seg.token_count for seg in segments if seg.seg_type != "output"
            )
        if not output_tokens:
            output_tokens = sum(
                seg.token_count for seg in segments if seg.seg_type == "output"
            )

        now = datetime.now(timezone.utc)
        call_row = CallRow(
            id=call_id,
            run_id=run_id,
            sequence=seq,
            model=model,
            params=json.dumps(params),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            created_at=now,
        )

        seg_rows = [
            SegmentRow(
                id=seg.id,
                call_id=call_id,
                role=seg.role,
                seg_type=seg.seg_type,
                content=seg.content,
                token_count=seg.token_count,
                position=seg.position,
            )
            for seg in segments
        ]

        with self.session() as s:
            s.add(call_row)
            s.add_all(seg_rows)
            s.commit()

        return Call(
            id=call_id,
            run_id=run_id,
            sequence=seq,
            model=model,
            params=params,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            created_at=now,
            segments=segments,
        )

    # ── Query operations ─────────────────────────────────────────────────

    def _row_to_run(self, row: RunRow, include_calls: bool = False) -> Run:
        run = Run(
            id=row.id,
            name=row.name,
            tags=json.loads(row.tags or "[]"),
            status=row.status,
            started_at=row.started_at,
            finished_at=row.finished_at,
            metadata=json.loads(row.metadata_ or "{}"),
        )
        if include_calls:
            for crow in row.calls:
                segs = [
                    Segment(
                        id=sr.id,
                        call_id=sr.call_id,
                        role=sr.role,
                        seg_type=sr.seg_type,
                        content=sr.content,
                        token_count=sr.token_count,
                        position=sr.position,
                    )
                    for sr in crow.segments
                ]
                run.calls.append(
                    Call(
                        id=crow.id,
                        run_id=crow.run_id,
                        sequence=crow.sequence,
                        model=crow.model,
                        params=json.loads(crow.params or "{}"),
                        input_tokens=crow.input_tokens,
                        output_tokens=crow.output_tokens,
                        latency_ms=crow.latency_ms,
                        finish_reason=crow.finish_reason,
                        created_at=crow.created_at,
                        segments=segs,
                    )
                )
        return run

    def get_run(self, run_id: str) -> Run | None:
        with self.session() as s:
            row = s.query(RunRow).filter_by(id=run_id).first()
            if not row:
                return None
            return self._row_to_run(row, include_calls=True)

    def list_runs(
        self,
        limit: int = 20,
        tags: list[str] | None = None,
        since: datetime | None = None,
    ) -> list[Run]:
        with self.session() as s:
            q = s.query(RunRow).order_by(desc(RunRow.started_at))
            if since:
                q = q.filter(RunRow.started_at >= since)
            if tags:
                for tag in tags:
                    q = q.filter(RunRow.tags.contains(f'"{tag}"'))
            q = q.limit(limit)
            return [self._row_to_run(r, include_calls=True) for r in q.all()]

    def get_last_run(self) -> Run | None:
        runs = self.list_runs(limit=1)
        return runs[0] if runs else None

    def get_last_n_runs(self, n: int = 2) -> list[Run]:
        return self.list_runs(limit=n)

    def get_stats(self, run_ids: list[str] | None = None, limit: int = 20) -> StatsResult:
        if run_ids:
            runs = []
            for rid in run_ids:
                r = self.get_run(rid)
                if r:
                    runs.append(r)
        else:
            runs = self.list_runs(limit=limit)

        if not runs:
            return StatsResult(
                total_runs=0,
                success_count=0,
                failed_count=0,
                avg_tokens=0,
                avg_latency_ms=0,
                avg_input_tokens=0,
                avg_output_tokens=0,
                segment_breakdown={},
                runs=[],
            )

        total_runs = len(runs)
        success_count = sum(1 for r in runs if r.status == "success")
        failed_count = sum(1 for r in runs if r.status == "failed")

        all_input = []
        all_output = []
        all_latency = []
        seg_breakdown: dict[str, int] = {}

        for run in runs:
            for call in run.calls:
                all_input.append(call.input_tokens)
                all_output.append(call.output_tokens)
                all_latency.append(call.latency_ms)
                for seg in call.segments:
                    seg_breakdown[seg.seg_type] = (
                        seg_breakdown.get(seg.seg_type, 0) + seg.token_count
                    )

        avg_in = sum(all_input) / len(all_input) if all_input else 0
        avg_out = sum(all_output) / len(all_output) if all_output else 0
        avg_lat = sum(all_latency) / len(all_latency) if all_latency else 0
        avg_tok = (sum(all_input) + sum(all_output)) / total_runs if total_runs else 0

        return StatsResult(
            total_runs=total_runs,
            success_count=success_count,
            failed_count=failed_count,
            avg_tokens=avg_tok,
            avg_latency_ms=avg_lat,
            avg_input_tokens=avg_in,
            avg_output_tokens=avg_out,
            segment_breakdown=seg_breakdown,
            runs=runs,
        )

    def get_running_run(self) -> Run | None:
        """Get the most recent running run."""
        with self.session() as s:
            row = (
                s.query(RunRow)
                .filter_by(status="running")
                .order_by(desc(RunRow.started_at))
                .first()
            )
            if not row:
                return None
            return self._row_to_run(row, include_calls=True)

    def get_latest_calls(self, run_id: str, after_sequence: int = -1) -> list[Call]:
        """Get calls added after a given sequence number."""
        with self.session() as s:
            rows = (
                s.query(CallRow)
                .filter(CallRow.run_id == run_id, CallRow.sequence > after_sequence)
                .order_by(CallRow.sequence)
                .all()
            )
            calls = []
            for crow in rows:
                segs = [
                    Segment(
                        id=sr.id,
                        call_id=sr.call_id,
                        role=sr.role,
                        seg_type=sr.seg_type,
                        content=sr.content,
                        token_count=sr.token_count,
                        position=sr.position,
                    )
                    for sr in crow.segments
                ]
                calls.append(
                    Call(
                        id=crow.id,
                        run_id=crow.run_id,
                        sequence=crow.sequence,
                        model=crow.model,
                        params=json.loads(crow.params or "{}"),
                        input_tokens=crow.input_tokens,
                        output_tokens=crow.output_tokens,
                        latency_ms=crow.latency_ms,
                        finish_reason=crow.finish_reason,
                        created_at=crow.created_at,
                        segments=segs,
                    )
                )
            return calls
