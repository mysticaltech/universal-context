"""Tests for the daemon — watcher, worker, and processors."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from universal_context.daemon.processors.base import BaseProcessor
from universal_context.daemon.processors.summarizer import TurnSummarizer
from universal_context.daemon.watcher import Watcher
from universal_context.daemon.worker import Worker
from universal_context.db.client import UCDatabase
from universal_context.db.queries import (
    backfill_canonical_ids,
    claim_next_job,
    create_job,
    create_run,
    create_scope,
    create_turn_with_artifact,
    detect_merged_runs,
    end_run,
    get_run,
    list_jobs,
    list_runs,
)
from universal_context.db.schema import apply_schema


@pytest.fixture
async def db():
    """In-memory database with schema."""
    database = UCDatabase.in_memory()
    await database.connect()
    await apply_schema(database)
    yield database
    await database.close()


# ============================================================
# WATCHER TESTS
# ============================================================


class TestWatcher:
    async def test_recover_interrupted_runs(self, db: UCDatabase):
        """Watcher should mark active runs as crashed on startup."""
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])
        run = await create_run(db, scope_id, "claude")
        run_id = str(run["id"])

        watcher = Watcher(db=db, poll_interval=1.0)
        await watcher.recover_interrupted_runs()

        updated = await get_run(db, run_id)
        assert updated["status"] == "crashed"

    async def test_recover_interrupted_runs_processes_all_active_runs(
        self, db: UCDatabase,
    ):
        """Recovery should not be limited by default list_runs pagination."""
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])
        for i in range(60):
            await create_run(db, scope_id, "claude", session_path=f"/tmp/sessions/{i}")

        watcher = Watcher(db=db, poll_interval=1.0)
        await watcher.recover_interrupted_runs()

        active_count = await db.query(
            'SELECT count() FROM run WHERE status = "active" GROUP ALL'
        )
        crashed_count = await db.query(
            'SELECT count() FROM run WHERE status = "crashed" GROUP ALL'
        )
        assert active_count[0]["count"] == 0
        assert crashed_count[0]["count"] == 60

    async def test_ensure_scope_reuse(self, db: UCDatabase):
        """Watcher should reuse existing scope for same path."""
        watcher = Watcher(db=db, poll_interval=1.0)
        id1 = await watcher._ensure_scope(Path("/tmp/proj"))
        id2 = await watcher._ensure_scope(Path("/tmp/proj"))
        assert id1 == id2

    async def test_ensure_scope_creates_new(self, db: UCDatabase):
        """Watcher should create new scope for new path."""
        watcher = Watcher(db=db, poll_interval=1.0)
        id1 = await watcher._ensure_scope(Path("/tmp/proj1"))
        id2 = await watcher._ensure_scope(Path("/tmp/proj2"))
        assert id1 != id2


# ============================================================
# WORKER TESTS
# ============================================================


class TestWorker:
    async def test_claim_and_process(self, db: UCDatabase):
        """Worker should claim a job and process it."""

        class DummyProcessor(BaseProcessor):
            async def process(self, db, job):
                return {"status": "processed"}

        worker = Worker(db=db, poll_interval=0.1)
        worker.register_processor("test_job", DummyProcessor())

        # Create a job
        await create_job(db, "test_job", "target:1")

        # Process one job
        processed = await worker._claim_and_process()
        assert processed is True

        # Job should be completed
        jobs = await list_jobs(db, status="completed")
        assert len(jobs) == 1
        assert jobs[0]["result"]["status"] == "processed"

    async def test_no_jobs_returns_false(self, db: UCDatabase):
        worker = Worker(db=db, poll_interval=0.1)
        processed = await worker._claim_and_process()
        assert processed is False

    async def test_unknown_job_type_fails(self, db: UCDatabase):
        """Jobs with no registered processor should fail."""
        worker = Worker(db=db, poll_interval=0.1)
        await create_job(db, "unknown_type", "target:1")
        await worker._claim_and_process()

        jobs = await list_jobs(db, status="pending")
        # Should be back to pending (retry)
        assert len(jobs) == 1
        assert "No processor" in (jobs[0].get("error") or "")

    async def test_processor_exception_fails_job(self, db: UCDatabase):
        """If a processor raises, the job should be marked failed."""

        class FailProcessor(BaseProcessor):
            async def process(self, db, job):
                raise RuntimeError("LLM timeout")

        worker = Worker(db=db, poll_interval=0.1)
        worker.register_processor("fail_job", FailProcessor())
        await create_job(db, "fail_job", "target:1")
        await worker._claim_and_process()

        jobs = await list_jobs(db, status="pending")
        assert len(jobs) == 1
        assert "LLM timeout" in (jobs[0].get("error") or "")

    async def test_worker_loop_stops(self, db: UCDatabase):
        """Worker should stop when stop() is called."""
        worker = Worker(db=db, poll_interval=0.05)

        async def stop_after_delay():
            await asyncio.sleep(0.15)
            worker.stop()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(worker.run())
            tg.create_task(stop_after_delay())

        assert not worker._running


# ============================================================
# SUMMARIZER PROCESSOR TESTS
# ============================================================


class TestTurnSummarizer:
    async def _setup_turn(self, db: UCDatabase) -> tuple[str, str]:
        """Create a scope→run→turn+artifact chain. Returns (turn_id, artifact_id)."""
        scope = await create_scope(db, "proj")
        scope_id = str(scope["id"])
        run = await create_run(db, scope_id, "claude")
        run_id = str(run["id"])
        result = await create_turn_with_artifact(
            db,
            run_id,
            sequence=1,
            user_message="explain binary search",
            raw_content=(
                "user: explain binary search\n"
                "assistant: Binary search is an efficient algorithm that finds "
                "the position of a target value within a sorted array. It works "
                "by repeatedly dividing the search interval in half."
            ),
            create_summary_job=True,
        )
        return result["turn_id"], result["artifact_id"]

    async def test_extractive_summary(self, db: UCDatabase):
        """Without LLM, summarizer should produce extractive summary."""
        turn_id, artifact_id = await self._setup_turn(db)

        # Claim the auto-created job
        job = await claim_next_job(db)
        assert job is not None

        summarizer = TurnSummarizer(max_chars=100)
        result = await summarizer.process(db, job)

        assert result["method"] == "extractive"
        assert result["summary_id"].startswith("artifact:")
        assert result["length"] > 0

    async def test_llm_summary(self, db: UCDatabase):
        """With LLM function, summarizer should use it."""
        turn_id, artifact_id = await self._setup_turn(db)
        job = await claim_next_job(db)

        llm_fn = AsyncMock(return_value="Binary search divides and conquers.")
        summarizer = TurnSummarizer(llm_fn=llm_fn)
        result = await summarizer.process(db, job)

        assert result["method"] == "llm"
        llm_fn.assert_awaited_once()

    async def test_llm_fallback(self, db: UCDatabase):
        """If LLM fails, should fall back to extractive."""
        turn_id, artifact_id = await self._setup_turn(db)
        job = await claim_next_job(db)

        llm_fn = AsyncMock(side_effect=RuntimeError("API error"))
        summarizer = TurnSummarizer(llm_fn=llm_fn, max_chars=50)
        result = await summarizer.process(db, job)

        assert result["method"] == "extractive_fallback"

    async def test_extractive_truncation(self):
        """Extractive summary should truncate at word boundaries."""
        summarizer = TurnSummarizer(max_chars=20)
        result = summarizer._extractive_summary("Hello world this is a test")
        assert result.endswith("...")
        assert len(result) <= 25  # 20 + "..."

    async def test_short_content_no_truncation(self):
        """Short content should not be truncated."""
        summarizer = TurnSummarizer(max_chars=100)
        result = summarizer._extractive_summary("Short text")
        assert result == "Short text"


# ============================================================
# BACKFILL ON STARTUP
# ============================================================


class TestBackfillOnStartup:
    async def test_backfill_canonical_ids_sets_id(self, db: UCDatabase):
        """Scopes without canonical_id should get one after backfill."""
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])

        # Verify it starts without canonical_id
        assert scope.get("canonical_id") is None

        with patch(
            "universal_context.git.resolve_canonical_id",
            return_value="github.com/user/proj",
        ):
            count = await backfill_canonical_ids(db)

        assert count == 1

        # Verify it was set
        from universal_context.db.queries import get_scope
        updated = await get_scope(db, scope_id)
        assert updated["canonical_id"] == "github.com/user/proj"

    async def test_backfill_skips_existing(self, db: UCDatabase):
        """Scopes with canonical_id should be skipped."""
        await create_scope(db, "proj", "/tmp/proj", canonical_id="github.com/user/proj")

        with patch(
            "universal_context.git.resolve_canonical_id",
            return_value="github.com/user/proj",
        ):
            count = await backfill_canonical_ids(db)

        assert count == 0

    async def test_backfill_merges_duplicates(self, db: UCDatabase):
        """Two scopes resolving to same canonical_id should merge."""
        await create_scope(db, "proj1", "/tmp/proj1", canonical_id="github.com/user/repo")
        s2 = await create_scope(db, "proj2", "/tmp/proj2")  # no canonical_id

        with patch(
            "universal_context.git.resolve_canonical_id",
            return_value="github.com/user/repo",
        ):
            count = await backfill_canonical_ids(db)

        assert count == 1  # s2 merged into s1

        from universal_context.db.queries import get_scope
        # s2 should be gone (merged)
        deleted = await get_scope(db, str(s2["id"]))
        assert deleted is None


# ============================================================
# MERGE DETECTION
# ============================================================


class TestMergeDetection:
    async def test_detect_merged_runs(self, db: UCDatabase):
        """Runs from branches merged into main should get tagged."""
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])

        # Create a run on feature branch
        run = await create_run(
            db, scope_id, "claude", branch="feature/auth", commit_sha="abc1234",
        )
        run_id = str(run["id"])

        # Create a run on main (current)
        await create_run(db, scope_id, "claude", branch="main", commit_sha="def5678")

        with patch(
            "universal_context.git.is_ancestor",
            return_value=True,
        ):
            count = await detect_merged_runs(
                db, scope_id, "main", Path("/tmp/proj"),
            )

        assert count == 1

        # Verify merged_to was set
        updated = await get_run(db, run_id)
        assert updated["merged_to"] == "main"

    async def test_detect_no_merged_runs(self, db: UCDatabase):
        """Unmerged feature branch runs should not be tagged."""
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])

        await create_run(
            db, scope_id, "claude", branch="feature/wip", commit_sha="abc1234",
        )

        with patch(
            "universal_context.git.is_ancestor",
            return_value=False,
        ):
            count = await detect_merged_runs(
                db, scope_id, "main", Path("/tmp/proj"),
            )

        assert count == 0

    async def test_detect_skips_same_branch(self, db: UCDatabase):
        """Runs already on main should not be checked."""
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])

        await create_run(db, scope_id, "claude", branch="main", commit_sha="abc1234")

        with patch(
            "universal_context.git.is_ancestor",
        ) as mock_ancestor:
            count = await detect_merged_runs(
                db, scope_id, "main", Path("/tmp/proj"),
            )

        assert count == 0
        mock_ancestor.assert_not_called()

    async def test_detect_skips_already_tagged(self, db: UCDatabase):
        """Runs already tagged with merged_to should be skipped."""
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])

        run = await create_run(
            db, scope_id, "claude", branch="feature/old", commit_sha="abc1234",
        )
        # Manually tag it
        await db.query(f"UPDATE {str(run['id'])} SET merged_to = 'main'")

        with patch(
            "universal_context.git.is_ancestor",
        ) as mock_ancestor:
            count = await detect_merged_runs(
                db, scope_id, "main", Path("/tmp/proj"),
            )

        assert count == 0
        mock_ancestor.assert_not_called()


# ============================================================
# WATCHER SESSION DEDUP
# ============================================================


class TestWatcherSessionDedup:
    """Test that _register_session deduplicates against existing DB runs."""

    async def _make_adapter(self, name: str = "claude", turn_count: int = 3):
        """Create a mock adapter for testing."""
        from unittest.mock import MagicMock

        adapter = MagicMock()
        adapter.name = name
        adapter.extract_project_path.return_value = None
        adapter.count_turns.return_value = turn_count
        adapter.extract_turn_info.return_value = None
        return adapter

    async def test_resume_existing_completed_run(self, db: UCDatabase):
        """If a completed run exists for the session, reuse it — no new run created."""
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])
        session_path = "/tmp/sessions/session_abc"

        # Create a completed run with 3 turns already ingested
        run = await create_run(db, scope_id, "claude", session_path=session_path)
        run_id = str(run["id"])
        await end_run(db, run_id, "completed")
        for seq in range(1, 4):
            await create_turn_with_artifact(
                db, run_id, seq, f"msg{seq}", f"content{seq}", create_summary_job=False,
            )

        adapter = await self._make_adapter(turn_count=3)
        watcher = Watcher(db=db, poll_interval=1.0)
        await watcher._register_session(Path(session_path), adapter)

        # Should reuse the existing run
        state = watcher._sessions[session_path]
        assert state.run_id == run_id
        assert state.last_turn_count == 3

        # No new run should have been created
        runs = await list_runs(db, scope_id=scope_id)
        assert len(runs) == 1

    async def test_skip_turns_from_crashed_run(self, db: UCDatabase):
        """Crashed run → new run, but start from crashed run's turn count."""
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])
        session_path = "/tmp/sessions/session_def"

        # Create a crashed run with 2 turns
        run = await create_run(db, scope_id, "claude", session_path=session_path)
        old_run_id = str(run["id"])
        await end_run(db, old_run_id, "crashed")
        for seq in range(1, 3):
            await create_turn_with_artifact(
                db, old_run_id, seq, f"msg{seq}", f"content{seq}",
                create_summary_job=False,
            )

        adapter = await self._make_adapter(turn_count=2)
        watcher = Watcher(db=db, poll_interval=1.0)
        await watcher._register_session(Path(session_path), adapter)

        state = watcher._sessions[session_path]
        # Should have created a NEW run (not reused the crashed one)
        assert state.run_id != old_run_id
        # But should skip the 2 already-ingested turns
        assert state.last_turn_count == 2

        # Two runs should exist total: old crashed + new active
        # (new run may be under a different scope since mock adapter has no project_path)
        all_runs = await list_runs(db)
        assert len(all_runs) == 2

    async def test_skip_uses_max_turns_across_multiple_crashed_runs(
        self, db: UCDatabase,
    ):
        """When several crashed runs exist, dedup should use the max captured turns."""
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])
        session_path = "/tmp/sessions/session_multi_crash"

        # Older crashed run with 3 turns captured
        older = await create_run(db, scope_id, "claude", session_path=session_path)
        older_id = str(older["id"])
        for seq in range(1, 4):
            await create_turn_with_artifact(
                db, older_id, seq, f"msg{seq}", f"content{seq}",
                create_summary_job=False,
            )
        await end_run(db, older_id, "crashed")

        # Newer crashed run with fewer turns
        newer = await create_run(db, scope_id, "claude", session_path=session_path)
        newer_id = str(newer["id"])
        await create_turn_with_artifact(
            db, newer_id, 1, "msg1", "content1", create_summary_job=False,
        )
        await end_run(db, newer_id, "crashed")

        # Set deterministic ordering without sleeping.
        await db.query(
            f"UPDATE {older_id} SET started_at = $started_at",
            {"started_at": datetime(2024, 1, 1, tzinfo=UTC)},
        )
        await db.query(
            f"UPDATE {newer_id} SET started_at = $started_at",
            {"started_at": datetime(2024, 1, 2, tzinfo=UTC)},
        )

        adapter = await self._make_adapter(turn_count=3)
        watcher = Watcher(db=db, poll_interval=1.0)
        await watcher._register_session(Path(session_path), adapter)

        state = watcher._sessions[session_path]
        assert state.run_id not in {older_id, newer_id}
        # Must skip all already-ingested turns, not just those in the latest crashed run
        assert state.last_turn_count == 3

    async def test_first_time_session_creates_fresh_run(self, db: UCDatabase):
        """No existing run → creates from scratch with last_turn_count=0."""
        session_path = "/tmp/sessions/session_new"

        adapter = await self._make_adapter(turn_count=0)
        watcher = Watcher(db=db, poll_interval=1.0)
        await watcher._register_session(Path(session_path), adapter)

        state = watcher._sessions[session_path]
        assert state.last_turn_count == 0
        assert state.run_id  # Should have a run ID
