"""Tests for starnose hypothesis engine."""

from datetime import datetime, timezone

import pytest

from starnose.db import Call, Run, Segment
from starnose.hypotheses import generate_delta_table, generate_hypotheses


def _make_run(
    run_id: str = "run_test",
    status: str = "success",
    segments: list[tuple[str, str, int]] | None = None,
    latency_ms: int = 1000,
) -> Run:
    """Helper to create a run with segments.

    segments: list of (role, seg_type, token_count) tuples
    """
    segs = []
    total_input = 0
    for i, (role, seg_type, tok) in enumerate(segments or []):
        segs.append(
            Segment(
                id=f"seg_{i}",
                call_id="call_0",
                role=role,
                seg_type=seg_type,
                content="x" * tok,
                token_count=tok,
                position=i,
            )
        )
        if seg_type != "output":
            total_input += tok

    output_tokens = sum(s.token_count for s in segs if s.seg_type == "output")

    call = Call(
        id="call_0",
        run_id=run_id,
        sequence=0,
        model="gpt-4o",
        params={},
        input_tokens=total_input,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        finish_reason="stop",
        created_at=datetime.now(timezone.utc),
        segments=segs,
    )

    return Run(
        id=run_id,
        name=None,
        tags=[],
        status=status,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        metadata={},
        calls=[call],
    )


class TestToolBloatHypothesis:
    def test_triggers_on_large_tool_delta(self):
        run_a = _make_run(
            "run_a",
            segments=[
                ("system", "system_prompt", 2000),
                ("user", "human", 500),
                ("tool", "tool_result", 1000),
            ],
        )
        run_b = _make_run(
            "run_b",
            segments=[
                ("system", "system_prompt", 2000),
                ("user", "human", 500),
                ("tool", "tool_result", 60000),
            ],
        )
        hyps = generate_hypotheses(run_a, run_b)
        assert any(h.category == "tool_bloat" for h in hyps)
        assert any("high" == h.confidence for h in hyps if h.category == "tool_bloat")


class TestSystemChangeHypothesis:
    def test_triggers_on_system_prompt_change(self):
        run_a = _make_run(
            "run_a",
            segments=[("system", "system_prompt", 1000), ("user", "human", 500)],
        )
        run_b = _make_run(
            "run_b",
            segments=[("system", "system_prompt", 3000), ("user", "human", 500)],
        )
        hyps = generate_hypotheses(run_a, run_b)
        assert any(h.category == "system_change" for h in hyps)


class TestBudgetHypothesis:
    def test_triggers_when_budget_exceeded(self):
        # gpt-4o has 128k limit
        run_a = _make_run(
            "run_a",
            segments=[("user", "human", 50000)],  # ~39%
        )
        run_b = _make_run(
            "run_b",
            status="failed",
            segments=[("user", "human", 110000)],  # ~86%
        )
        hyps = generate_hypotheses(run_a, run_b)
        assert any(h.category == "budget" for h in hyps)


class TestLatencyHypothesis:
    def test_triggers_on_proportional_increase(self):
        run_a = _make_run(
            "run_a",
            segments=[("user", "human", 5000)],
            latency_ms=5000,
        )
        run_b = _make_run(
            "run_b",
            segments=[("user", "human", 50000)],
            latency_ms=50000,
        )
        hyps = generate_hypotheses(run_a, run_b)
        assert any(h.category == "latency" for h in hyps)


class TestFallbackHypothesis:
    def test_fallback_when_no_pattern(self):
        run_a = _make_run(
            "run_a",
            segments=[("user", "human", 1000)],
        )
        run_b = _make_run(
            "run_b",
            segments=[("user", "human", 1000)],
        )
        hyps = generate_hypotheses(run_a, run_b)
        assert any(h.category == "unknown" for h in hyps)


class TestDeltaTable:
    def test_generates_delta_rows(self):
        run_a = _make_run(
            "run_a",
            segments=[
                ("system", "system_prompt", 2000),
                ("tool", "tool_result", 3000),
            ],
        )
        run_b = _make_run(
            "run_b",
            segments=[
                ("system", "system_prompt", 2000),
                ("tool", "tool_result", 8000),
            ],
        )
        rows = generate_delta_table(run_a, run_b)

        # Should have segment rows + TOTAL
        assert any(r["segment"] == "TOTAL" for r in rows)
        assert any(r["segment"] == "tool_result" for r in rows)

        tool_row = next(r for r in rows if r["segment"] == "tool_result")
        assert tool_row["delta"] == 5000
        assert not tool_row["identical"]

        sys_row = next(r for r in rows if r["segment"] == "system_prompt")
        assert sys_row["identical"]
