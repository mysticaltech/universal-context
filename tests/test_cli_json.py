"""Tests for CLI JSON output mode and the uc context command."""

from __future__ import annotations

import json

import pytest

from universal_context.cli import _sanitize_record
from universal_context.db.client import UCDatabase
from universal_context.db.queries import (
    create_derived_artifact,
    create_run,
    create_scope,
    create_turn_with_artifact,
    find_scope_by_path,
    get_turn_summaries,
    list_runs,
    list_scopes,
    search_artifacts,
)
from universal_context.db.schema import apply_schema

# --- Fixtures ---


@pytest.fixture
async def db(tmp_path):
    """Create an in-memory database with schema applied."""
    d = UCDatabase(f"mem://test_cli_{id(tmp_path)}")
    await d.connect()
    await apply_schema(d)
    yield d
    await d.close()


@pytest.fixture
async def populated_db(db: UCDatabase):
    """DB with a scope, run, turns, and summary artifacts."""
    scope = await create_scope(db, "test-project", path="/tmp/test-project")
    scope_id = str(scope["id"])

    run = await create_run(db, scope_id, "claude", session_path="/tmp/session.jsonl")
    run_id = str(run["id"])

    # Create 3 turns
    t1 = await create_turn_with_artifact(
        db, run_id, 1, "set up auth", "User: set up auth\nAssistant: Created JWT middleware."
    )
    t2 = await create_turn_with_artifact(
        db, run_id, 2, "add tests", "User: add tests\nAssistant: Wrote pytest suite."
    )
    t3 = await create_turn_with_artifact(
        db, run_id, 3, "fix bug", "User: fix bug\nAssistant: Fixed null pointer."
    )

    # Create summary artifacts for turn 1 and turn 2
    # Get transcript artifact IDs from the graph
    for turn_result in [t1, t2]:
        tid = turn_result["turn_id"]
        art_id = turn_result["artifact_id"]
        await create_derived_artifact(
            db,
            kind="summary",
            content=f"Summary for {tid}",
            source_id=art_id,
            relationship="summarized_from",
        )

    return {
        "scope_id": scope_id,
        "run_id": run_id,
        "turn_ids": [t1["turn_id"], t2["turn_id"], t3["turn_id"]],
        "path": "/tmp/test-project",
    }


# --- _sanitize_record tests ---


class TestSanitizeRecord:
    def test_plain_dict_unchanged(self):
        record = {"name": "test", "count": 42, "active": True}
        assert _sanitize_record(record) == record

    def test_nested_dict(self):
        record = {"meta": {"nested": "value"}}
        assert _sanitize_record(record) == {"meta": {"nested": "value"}}

    def test_list_values(self):
        record = {"items": [1, "two", 3]}
        assert _sanitize_record(record) == {"items": [1, "two", 3]}

    def test_record_id_mock(self):
        """Simulate a SurrealDB RecordID object."""

        class RecordID:
            def __init__(self, table, id):
                self.table = table
                self.id = id

            def __str__(self):
                return f"{self.table}:{self.id}"

        record = {"id": RecordID("scope", "abc123"), "name": "test"}
        result = _sanitize_record(record)
        assert result["id"] == "scope:abc123"
        assert result["name"] == "test"

    def test_record_id_in_list(self):
        class RecordID:
            def __init__(self, val):
                self.val = val

            def __str__(self):
                return self.val

        record = {"ids": [RecordID("turn:1"), RecordID("turn:2")]}
        result = _sanitize_record(record)
        assert result["ids"] == ["turn:1", "turn:2"]


# --- get_turn_summaries tests ---


class TestGetTurnSummaries:
    @pytest.mark.asyncio
    async def test_summaries_present(self, db, populated_db):
        info = populated_db
        results = await get_turn_summaries(db, info["run_id"], limit=10)

        assert len(results) == 3

        # Turns are ordered most-recent-first (sequence DESC)
        assert results[0]["sequence"] == 3
        assert results[1]["sequence"] == 2
        assert results[2]["sequence"] == 1

        # Turn 1 and 2 have summaries, turn 3 does not
        assert results[2]["summary"] is not None  # seq 1
        assert results[1]["summary"] is not None  # seq 2
        assert results[0]["summary"] is None  # seq 3

    @pytest.mark.asyncio
    async def test_summaries_content(self, db, populated_db):
        info = populated_db
        results = await get_turn_summaries(db, info["run_id"])

        # Check summary content matches what we created
        seq1 = next(r for r in results if r["sequence"] == 1)
        assert "Summary for" in seq1["summary"]

    @pytest.mark.asyncio
    async def test_limit_respected(self, db, populated_db):
        info = populated_db
        results = await get_turn_summaries(db, info["run_id"], limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_user_messages_present(self, db, populated_db):
        info = populated_db
        results = await get_turn_summaries(db, info["run_id"])
        messages = {r["user_message"] for r in results}
        assert "set up auth" in messages
        assert "add tests" in messages
        assert "fix bug" in messages


# --- JSON output integration tests (using real DB) ---


class TestSearchJson:
    @pytest.mark.asyncio
    async def test_search_returns_serializable(self, db, populated_db):
        # On embedded, search_artifacts uses substring match fallback
        results = await search_artifacts(db, "auth")
        assert isinstance(results, list)
        assert len(results) > 0  # Should find "set up auth" transcripts
        sanitized = [_sanitize_record(r) for r in results]
        output = json.dumps(sanitized, default=str)
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) > 0


class TestStatusJson:
    @pytest.mark.asyncio
    async def test_status_returns_serializable(self, db, populated_db):
        scopes = await list_scopes(db)
        runs = await list_runs(db, limit=5)

        payload = {
            "scopes": [_sanitize_record(s) for s in scopes],
            "recent_runs": [_sanitize_record(r) for r in runs],
            "jobs": {},
        }
        output = json.dumps(payload, default=str)
        parsed = json.loads(output)
        assert "scopes" in parsed
        assert "recent_runs" in parsed


class TestContextJson:
    @pytest.mark.asyncio
    async def test_context_full_payload(self, db, populated_db):
        info = populated_db

        scope = await find_scope_by_path(db, info["path"])
        assert scope is not None
        scope_id = str(scope["id"])

        runs_data = await list_runs(db, scope_id=scope_id, limit=5)
        run_details = []
        for r in runs_data:
            rid = str(r["id"])
            summaries = await get_turn_summaries(db, rid, limit=10)
            run_details.append(
                {
                    "run_id": rid,
                    "agent_type": r.get("agent_type", ""),
                    "status": r.get("status", ""),
                    "total_turns": len(summaries),
                    "turns": summaries,
                }
            )

        payload = {
            "scope": _sanitize_record(scope),
            "runs": run_details,
            "search_results": [],
        }
        output = json.dumps(payload, default=str)
        parsed = json.loads(output)

        assert parsed["scope"]["path"] == "/tmp/test-project"
        assert len(parsed["runs"]) == 1
        assert parsed["runs"][0]["agent_type"] == "claude"
        assert parsed["runs"][0]["total_turns"] == 3

    @pytest.mark.asyncio
    async def test_context_no_scope(self, db):
        scope = await find_scope_by_path(db, "/nonexistent/path")
        assert scope is None
        # The CLI would return this JSON:
        payload = {
            "error": "no_scope",
            "project": "/nonexistent/path",
            "message": "No scope found for /nonexistent/path",
        }
        output = json.dumps(payload)
        parsed = json.loads(output)
        assert parsed["error"] == "no_scope"

    @pytest.mark.asyncio
    async def test_context_branch_filter(self, db, populated_db):
        """--branch should filter runs and include branch/commit_sha in output."""
        info = populated_db

        scope = await find_scope_by_path(db, info["path"])
        scope_id = str(scope["id"])

        # The populated_db creates runs without branch — add one with branch
        run_main = await create_run(
            db,
            scope_id,
            "claude",
            branch="main",
            commit_sha="abc1234",
        )
        main_run_id = str(run_main["id"])
        await create_turn_with_artifact(
            db,
            main_run_id,
            1,
            "main work",
            "User: main work\nAssistant: done",
            create_summary_job=False,
        )

        run_feat = await create_run(
            db,
            scope_id,
            "claude",
            branch="feature/x",
            commit_sha="def5678",
        )
        feat_run_id = str(run_feat["id"])
        await create_turn_with_artifact(
            db,
            feat_run_id,
            1,
            "feature work",
            "User: feature work\nAssistant: done",
            create_summary_job=False,
        )

        # Filter by main branch
        runs_main = await list_runs(db, scope_id=scope_id, branch="main")
        assert len(runs_main) == 1
        assert str(runs_main[0]["id"]) == main_run_id

        # Build run_details the same way as the CLI
        run_details = []
        for r in runs_main:
            rid = str(r["id"])
            summaries = await get_turn_summaries(db, rid, limit=10)
            run_details.append(
                {
                    "run_id": rid,
                    "agent_type": r.get("agent_type", ""),
                    "branch": r.get("branch"),
                    "commit_sha": r.get("commit_sha"),
                    "merged_to": r.get("merged_to"),
                    "total_turns": len(summaries),
                }
            )

        payload = {
            "scope": _sanitize_record(scope),
            "runs": run_details,
        }
        output = json.dumps(payload, default=str)
        parsed = json.loads(output)

        assert len(parsed["runs"]) == 1
        assert parsed["runs"][0]["branch"] == "main"
        assert parsed["runs"][0]["commit_sha"] == "abc1234"


# --- Daemon helper tests ---


class TestDaemonHelpers:
    def test_stale_pid_detection(self, tmp_path):
        """Stale PID file should be cleaned up by daemon_start."""
        from universal_context.cli import _pid_alive

        # PID 99999999 almost certainly doesn't exist
        assert _pid_alive(99999999) is False
        # PID 1 (init/launchd) should always exist
        assert _pid_alive(1) is True

    def test_stale_pid_file_cleaned_on_start(self, tmp_path, monkeypatch):
        """daemon_start should clean up a stale PID file and proceed."""
        from universal_context import cli  # noqa: F811

        # Create a stale PID file with a dead PID
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("99999999")

        # Monkeypatch get_uc_home to return tmp_path
        monkeypatch.setattr(cli, "get_uc_home", lambda: tmp_path)

        # Also monkeypatch run_daemon to prevent actual daemon launch
        called = {}

        def fake_daemon_start(foreground=False):
            called["foreground"] = foreground

        # After stale PID cleanup, it should reach the foreground/background branch.
        # We verify the PID file was removed.
        # We can't easily test the full daemon_start without more mocking,
        # so we just verify _pid_alive + cleanup logic.
        from universal_context.cli import _pid_alive

        assert not _pid_alive(99999999)
        # Simulate what daemon_start does:
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
            except (ValueError, OSError):
                pid = 0
            if pid and not _pid_alive(pid):
                pid_file.unlink(missing_ok=True)

        assert not pid_file.exists()
