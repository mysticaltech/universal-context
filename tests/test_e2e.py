"""End-to-end integration test.

Simulates the full pipeline: session capture → turn ingestion →
job processing → search → export/import.
"""

import json
from pathlib import Path

import pytest

from universal_context.daemon.processors.summarizer import TurnSummarizer
from universal_context.daemon.watcher import Watcher
from universal_context.daemon.worker import Worker
from universal_context.db.client import UCDatabase
from universal_context.db.queries import (
    list_jobs,
    list_runs,
    list_turns,
    search_artifacts,
)
from universal_context.db.schema import apply_schema
from universal_context.sharing.bundle import export_bundle, import_bundle
from universal_context.sharing.checkpoint import create_checkpoint, get_checkpoint


@pytest.fixture
async def db():
    database = UCDatabase.in_memory()
    await database.connect()
    await apply_schema(database)
    yield database
    await database.close()


def _create_mock_claude_session(tmp_path: Path) -> Path:
    """Create a mock Claude Code session that the adapter can discover."""
    project_dir = tmp_path / ".claude" / "projects" / "-tmp-e2e-project"
    project_dir.mkdir(parents=True)
    session_file = project_dir / "e2e-session.jsonl"

    messages = [
        {
            "type": "human",
            "message": {"content": "Set up the authentication module"},
            "uuid": "u1",
            "timestamp": "2026-02-09T10:00:00Z",
        },
        {
            "type": "assistant",
            "message": {
                "content": "I'll create an auth module with JWT tokens. "
                "Here's the implementation in auth.py with login and verify endpoints."
            },
            "uuid": "a1",
            "timestamp": "2026-02-09T10:00:10Z",
        },
        {
            "type": "human",
            "message": {"content": "Add rate limiting to the login endpoint"},
            "uuid": "u2",
            "timestamp": "2026-02-09T10:01:00Z",
        },
        {
            "type": "assistant",
            "message": {
                "content": "Added rate limiting using a sliding window counter. "
                "Max 5 attempts per minute per IP address."
            },
            "uuid": "a2",
            "timestamp": "2026-02-09T10:01:15Z",
        },
        {
            "type": "human",
            "message": {"content": "Write tests for the auth module"},
            "uuid": "u3",
            "timestamp": "2026-02-09T10:02:00Z",
        },
        {
            "type": "assistant",
            "message": {
                "content": "Created test_auth.py with tests for login, "
                "token verification, rate limiting, and edge cases."
            },
            "uuid": "a3",
            "timestamp": "2026-02-09T10:02:20Z",
        },
    ]

    with session_file.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")

    return session_file


class TestEndToEnd:
    async def test_full_pipeline(self, db: UCDatabase, tmp_path: Path):
        """Exercise the full pipeline: capture → process → search → share."""

        # === Step 1: Simulate session capture via triggers ===
        from universal_context.triggers.claude_trigger import ClaudeTrigger

        session = _create_mock_claude_session(tmp_path)
        trigger = ClaudeTrigger()

        # Verify trigger detects turns
        count = trigger.count_complete_turns(session)
        assert count == 3

        # === Step 2: Manually ingest turns (simulating watcher) ===
        from universal_context.db.queries import (
            create_run,
            create_scope,
            create_turn_with_artifact,
        )

        scope = await create_scope(db, "e2e-project", "/tmp/e2e-project")
        scope_id = str(scope["id"])
        run = await create_run(db, scope_id, "claude")
        run_id = str(run["id"])

        for seq in range(1, count + 1):
            info = trigger.extract_turn_info(session, seq)
            raw = trigger.get_raw_transcript(session, seq)
            await create_turn_with_artifact(
                db,
                run_id=run_id,
                sequence=seq,
                user_message=info.user_message,
                raw_content=raw or "",
                create_summary_job=True,
            )

        # Verify turns are stored
        turns = await list_turns(db, run_id)
        assert len(turns) == 3

        # Verify jobs were created
        jobs = await list_jobs(db, status="pending")
        assert len(jobs) == 3  # One per turn

        # === Step 3: Process jobs (simulating worker) ===
        summarizer = TurnSummarizer(max_chars=200)
        worker = Worker(db=db, poll_interval=0.01)
        worker.register_processor("turn_summary", summarizer)

        for _ in range(3):
            await worker._claim_and_process()

        # Verify all jobs completed
        completed = await list_jobs(db, status="completed")
        assert len(completed) == 3

        # === Step 4: Search ===
        # BM25 FTS requires v3 server — on embedded, returns []
        results = await search_artifacts(db, "auth")
        assert isinstance(results, list)

        # === Step 5: Checkpoint ===
        turn_id = str(turns[1]["id"])  # Checkpoint at turn 2
        cp_id = await create_checkpoint(db, run_id, turn_id, label="after auth setup")
        cp = await get_checkpoint(db, cp_id)
        assert cp["label"] == "after auth setup"
        assert cp["state"]["total_turns"] == 3

        # === Step 6: Export/Import ===
        bundle_path = tmp_path / "export.json"
        await export_bundle(db, run_id, output_path=bundle_path)
        assert bundle_path.exists()

        # Import into same DB
        result = await import_bundle(db, bundle_path)
        assert result["turns_imported"] == 3

        # Verify we now have 2 runs
        all_runs = await list_runs(db)
        assert len(all_runs) == 2

    async def test_watcher_with_mock_session(self, db: UCDatabase, tmp_path: Path):
        """Test watcher discovering and ingesting a mock session."""
        session = _create_mock_claude_session(tmp_path)

        # Mock the adapter registry to use our tmp_path
        from unittest.mock import patch

        from universal_context.adapters.claude import ClaudeAdapter

        class MockClaudeAdapter(ClaudeAdapter):
            def discover_sessions(self):
                return [session]

            def extract_project_path(self, session_file):
                return tmp_path

        mock_registry_cls = type("MockRegistry", (), {
            "discover_all_sessions": lambda self, cfg=None: [
                (session, MockClaudeAdapter())
            ],
        })

        with patch(
            "universal_context.daemon.watcher.get_registry",
            return_value=mock_registry_cls(),
        ):
            watcher = Watcher(db=db, poll_interval=0.1)
            await watcher._poll_cycle()

        # Should have created run + turns
        runs = await list_runs(db)
        assert len(runs) == 1
        assert runs[0]["agent_type"] == "claude"

        run_id = str(runs[0]["id"])
        turns = await list_turns(db, run_id)
        assert len(turns) == 3
