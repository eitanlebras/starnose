"""Tests for starnose database layer."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from starnose.db import Database


@pytest.fixture
def db(tmp_path):
    """Create a fresh database for each test."""
    db_path = tmp_path / "test.db"
    return Database(db_path)


class TestCreateRun:
    def test_create_run_basic(self, db):
        run = db.create_run()
        assert run.id.startswith("run_")
        assert run.status == "running"
        assert run.tags == []
        assert run.name is None

    def test_create_run_with_name_and_tags(self, db):
        run = db.create_run(name="test-run", tags=["prod", "v2"])
        assert run.name == "test-run"
        assert run.tags == ["prod", "v2"]

    def test_create_run_with_metadata(self, db):
        run = db.create_run(metadata={"key": "value"})
        assert run.metadata == {"key": "value"}


class TestUpdateRunStatus:
    def test_update_to_success(self, db):
        run = db.create_run()
        db.update_run_status(run.id, "success")
        updated = db.get_run(run.id)
        assert updated.status == "success"
        assert updated.finished_at is not None

    def test_update_to_failed(self, db):
        run = db.create_run()
        db.update_run_status(run.id, "failed")
        updated = db.get_run(run.id)
        assert updated.status == "failed"


class TestAddCall:
    def test_add_call_basic(self, db):
        run = db.create_run()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        }
        call = db.add_call(run.id, "gpt-4o", {}, messages, response, 150)
        assert call.id.startswith("call_")
        assert call.sequence == 0
        assert call.model == "gpt-4o"
        assert call.input_tokens == 20
        assert call.output_tokens == 5
        assert call.latency_ms == 150
        assert len(call.segments) == 3  # system + user + output

    def test_add_call_segments_classified(self, db):
        run = db.create_run()
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question"},
            {"role": "tool", "content": "Tool output data"},
        ]
        response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
        call = db.add_call(run.id, "gpt-4o", {}, messages, response, 100)
        types = [s.seg_type for s in call.segments]
        assert "system_prompt" in types
        assert "human" in types
        assert "tool_result" in types
        assert "output" in types

    def test_sequential_calls_increment_sequence(self, db):
        run = db.create_run()
        msgs = [{"role": "user", "content": "Hi"}]
        resp = {"choices": [{"message": {"content": "Hey"}, "finish_reason": "stop"}], "usage": {}}

        c1 = db.add_call(run.id, "gpt-4o", {}, msgs, resp, 50)
        c2 = db.add_call(run.id, "gpt-4o", {}, msgs, resp, 60)
        assert c1.sequence == 0
        assert c2.sequence == 1


class TestListRuns:
    def test_list_runs_ordered(self, db):
        db.create_run(name="first")
        db.create_run(name="second")
        db.create_run(name="third")
        runs = db.list_runs()
        assert len(runs) == 3
        assert runs[0].name == "third"  # most recent first

    def test_list_runs_with_limit(self, db):
        for i in range(5):
            db.create_run(name=f"run-{i}")
        runs = db.list_runs(limit=3)
        assert len(runs) == 3

    def test_list_runs_by_tag(self, db):
        db.create_run(name="tagged", tags=["prod"])
        db.create_run(name="untagged")
        runs = db.list_runs(tags=["prod"])
        assert len(runs) == 1
        assert runs[0].name == "tagged"


class TestGetStats:
    def test_stats_basic(self, db):
        run = db.create_run()
        msgs = [{"role": "user", "content": "Hello world"}]
        resp = {
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        db.add_call(run.id, "gpt-4o", {}, msgs, resp, 100)
        db.update_run_status(run.id, "success")

        stats = db.get_stats()
        assert stats.total_runs == 1
        assert stats.success_count == 1
        assert stats.avg_tokens > 0

    def test_stats_empty(self, db):
        stats = db.get_stats()
        assert stats.total_runs == 0
        assert stats.avg_tokens == 0


class TestGetRun:
    def test_get_existing_run(self, db):
        run = db.create_run(name="find-me")
        found = db.get_run(run.id)
        assert found is not None
        assert found.name == "find-me"

    def test_get_nonexistent_run(self, db):
        assert db.get_run("run_nonexistent") is None

    def test_get_run_includes_calls(self, db):
        run = db.create_run()
        msgs = [{"role": "user", "content": "Hi"}]
        resp = {"choices": [{"message": {"content": "Hey"}, "finish_reason": "stop"}], "usage": {}}
        db.add_call(run.id, "gpt-4o", {}, msgs, resp, 50)

        found = db.get_run(run.id)
        assert len(found.calls) == 1
        assert len(found.calls[0].segments) > 0
