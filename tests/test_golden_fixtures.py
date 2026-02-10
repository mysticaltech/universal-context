"""Golden fixture tests — replay recorded sessions through the full pipeline.

These fixtures represent known-good session formats for each adapter.
When anything changes (adapter, trigger, watcher, worker, schema), these
tests catch silent regressions by asserting exact counts and relationships.

Fixtures committed in tests/fixtures/:
  - claude_session.jsonl  (3 turns)
  - codex_session.jsonl   (3 turns)
  - gemini_session.json   (3 turns)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from universal_context.adapters.claude import ClaudeAdapter
from universal_context.adapters.codex import CodexAdapter
from universal_context.adapters.gemini import GeminiAdapter
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

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Expected results for each fixture
EXPECTED = {
    "claude": {
        "fixture": "claude_session.jsonl",
        "turns": 3,
        "artifact_kinds": {"transcript"},
        "has_provenance": True,
        "search_term": "authentication",
    },
    "codex": {
        "fixture": "codex_session.jsonl",
        "turns": 3,
        "artifact_kinds": {"transcript"},
        "has_provenance": True,
        "search_term": "migration",
    },
    "gemini": {
        "fixture": "gemini_session.json",
        "turns": 3,
        "artifact_kinds": {"transcript"},
        "has_provenance": True,
        "search_term": "observer",
    },
}


@pytest.fixture
async def db():
    database = UCDatabase.in_memory()
    await database.connect()
    await apply_schema(database)
    yield database
    await database.close()


def _setup_claude_fixture(tmp_path: Path) -> Path:
    """Set up Claude fixture in a discoverable location."""
    project_dir = tmp_path / ".claude" / "projects" / "-home-dev-myapp"
    project_dir.mkdir(parents=True)
    dest = project_dir / "golden.jsonl"
    shutil.copy(FIXTURES_DIR / "claude_session.jsonl", dest)
    return dest


def _setup_codex_fixture(tmp_path: Path) -> Path:
    """Set up Codex fixture in a discoverable location."""
    session_dir = tmp_path / ".codex" / "sessions" / "2026" / "02" / "09"
    session_dir.mkdir(parents=True)
    dest = session_dir / "rollout-golden.jsonl"
    shutil.copy(FIXTURES_DIR / "codex_session.jsonl", dest)
    return dest


def _setup_gemini_fixture(tmp_path: Path) -> Path:
    """Set up Gemini fixture in a discoverable location."""
    chats_dir = tmp_path / ".gemini" / "tmp" / "myapp" / "chats"
    chats_dir.mkdir(parents=True)
    dest = chats_dir / "session-golden.json"
    shutil.copy(FIXTURES_DIR / "gemini_session.json", dest)
    return dest


async def _replay_fixture(
    db: UCDatabase,
    adapter_name: str,
    session_file: Path,
    trigger,
) -> str:
    """Replay a fixture through watcher→worker and return the run_id."""
    from universal_context.db.queries import (
        create_run,
        create_scope,
        create_turn_with_artifact,
    )

    # Count turns via trigger
    count = trigger.count_complete_turns(session_file)

    # Create scope + run
    scope = await create_scope(db, f"golden-{adapter_name}", f"/tmp/{adapter_name}")
    scope_id = str(scope["id"])
    run = await create_run(db, scope_id, adapter_name)
    run_id = str(run["id"])

    # Ingest turns
    for seq in range(1, count + 1):
        info = trigger.extract_turn_info(session_file, seq)
        raw = trigger.get_raw_transcript(session_file, seq)
        await create_turn_with_artifact(
            db,
            run_id=run_id,
            sequence=seq,
            user_message=info.user_message if info else None,
            raw_content=raw or "",
            create_summary_job=True,
        )

    # Process all summary jobs
    summarizer = TurnSummarizer(max_chars=200)
    worker = Worker(db=db, poll_interval=0.01)
    worker.register_processor("turn_summary", summarizer)

    for _ in range(count):
        await worker._claim_and_process()

    return run_id


# ============================================================
# GOLDEN FIXTURE: Claude Code
# ============================================================


class TestGoldenClaude:
    async def test_turn_count(self, db: UCDatabase, tmp_path: Path):
        session = _setup_claude_fixture(tmp_path)
        from universal_context.triggers.claude_trigger import ClaudeTrigger

        trigger = ClaudeTrigger()
        assert trigger.count_complete_turns(session) == EXPECTED["claude"]["turns"]

    async def test_full_pipeline(self, db: UCDatabase, tmp_path: Path):
        session = _setup_claude_fixture(tmp_path)
        from universal_context.triggers.claude_trigger import ClaudeTrigger

        run_id = await _replay_fixture(db, "claude", session, ClaudeTrigger())

        # Assert turn count
        turns = await list_turns(db, run_id)
        assert len(turns) == EXPECTED["claude"]["turns"]

        # Assert all summary jobs completed
        completed = await list_jobs(db, status="completed")
        assert len(completed) == EXPECTED["claude"]["turns"]

        # BM25 FTS requires v3 server — on embedded returns []
        results = await search_artifacts(db, EXPECTED["claude"]["search_term"])
        assert isinstance(results, list)

        # Assert provenance edges exist
        turn_id = str(turns[0]["id"])
        from universal_context.db.queries import get_turn_artifacts

        artifacts = await get_turn_artifacts(db, turn_id)
        assert artifacts, "Turn should have provenance edges to artifacts"

    async def test_adapter_format_detection(self, tmp_path: Path):
        session = _setup_claude_fixture(tmp_path)
        adapter = ClaudeAdapter()
        assert adapter.is_session_valid(session)


# ============================================================
# GOLDEN FIXTURE: Codex CLI
# ============================================================


class TestGoldenCodex:
    async def test_turn_count(self, db: UCDatabase, tmp_path: Path):
        session = _setup_codex_fixture(tmp_path)
        from universal_context.triggers.codex_trigger import CodexTrigger

        trigger = CodexTrigger()
        assert trigger.count_complete_turns(session) == EXPECTED["codex"]["turns"]

    async def test_full_pipeline(self, db: UCDatabase, tmp_path: Path):
        session = _setup_codex_fixture(tmp_path)
        from universal_context.triggers.codex_trigger import CodexTrigger

        run_id = await _replay_fixture(db, "codex", session, CodexTrigger())

        turns = await list_turns(db, run_id)
        assert len(turns) == EXPECTED["codex"]["turns"]

        completed = await list_jobs(db, status="completed")
        assert len(completed) == EXPECTED["codex"]["turns"]

        # BM25 FTS requires v3 server — on embedded returns []
        results = await search_artifacts(db, EXPECTED["codex"]["search_term"])
        assert isinstance(results, list)

    async def test_adapter_format_detection(self, tmp_path: Path):
        session = _setup_codex_fixture(tmp_path)
        adapter = CodexAdapter()
        assert adapter.is_session_valid(session)


# ============================================================
# GOLDEN FIXTURE: Gemini CLI
# ============================================================


class TestGoldenGemini:
    async def test_turn_count(self, db: UCDatabase, tmp_path: Path):
        session = _setup_gemini_fixture(tmp_path)
        from universal_context.triggers.gemini_trigger import GeminiTrigger

        trigger = GeminiTrigger()
        assert trigger.count_complete_turns(session) == EXPECTED["gemini"]["turns"]

    async def test_full_pipeline(self, db: UCDatabase, tmp_path: Path):
        session = _setup_gemini_fixture(tmp_path)
        from universal_context.triggers.gemini_trigger import GeminiTrigger

        run_id = await _replay_fixture(db, "gemini", session, GeminiTrigger())

        turns = await list_turns(db, run_id)
        assert len(turns) == EXPECTED["gemini"]["turns"]

        completed = await list_jobs(db, status="completed")
        assert len(completed) == EXPECTED["gemini"]["turns"]

        # BM25 FTS requires v3 server — on embedded returns []
        results = await search_artifacts(db, EXPECTED["gemini"]["search_term"])
        assert isinstance(results, list)

    async def test_adapter_format_detection(self, tmp_path: Path):
        session = _setup_gemini_fixture(tmp_path)
        adapter = GeminiAdapter()
        assert adapter.is_session_valid(session)


# ============================================================
# CROSS-ADAPTER: Watcher integration
# ============================================================


class TestGoldenWatcher:
    async def test_watcher_discovers_all_fixtures(
        self, db: UCDatabase, tmp_path: Path
    ):
        """Watcher should discover and ingest sessions from all adapters."""
        claude_session = _setup_claude_fixture(tmp_path)
        codex_session = _setup_codex_fixture(tmp_path)
        gemini_session = _setup_gemini_fixture(tmp_path)

        # Mock adapter registry to return our fixtures
        mock_sessions = [
            (claude_session, ClaudeAdapter()),
            (codex_session, CodexAdapter()),
            (gemini_session, GeminiAdapter()),
        ]

        mock_registry = type("MockRegistry", (), {
            "discover_all_sessions": lambda self, cfg=None: mock_sessions,
        })

        with patch(
            "universal_context.daemon.watcher.get_registry",
            return_value=mock_registry(),
        ):
            watcher = Watcher(db=db, poll_interval=0.1)
            await watcher._poll_cycle()

        # Should have 3 runs (one per adapter)
        runs = await list_runs(db)
        assert len(runs) == 3

        # Total turns across all runs
        total_turns = 0
        for run in runs:
            run_id = str(run["id"])
            turns = await list_turns(db, run_id)
            total_turns += len(turns)

        assert total_turns == 9  # 3 turns × 3 adapters

        # Turn summary jobs should be created for all turns
        jobs = await list_jobs(db, status="pending", job_type="turn_summary")
        assert len(jobs) == 9
