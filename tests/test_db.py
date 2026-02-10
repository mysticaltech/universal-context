"""Tests for SurrealDB storage layer — client, schema, queries."""

import pytest

from universal_context.db import queries
from universal_context.db.client import UCDatabase
from universal_context.db.schema import apply_schema


@pytest.fixture
async def db():
    """Create an in-memory database with schema applied."""
    database = UCDatabase.in_memory()
    await database.connect()
    await apply_schema(database)
    yield database
    await database.close()


# ============================================================
# CLIENT
# ============================================================


class TestClient:
    async def test_connect_and_health(self, db: UCDatabase):
        assert await db.health() is True

    async def test_query_basic(self, db: UCDatabase):
        result = await db.query("SELECT * FROM scope")
        assert isinstance(result, list)

    async def test_context_manager(self):
        async with UCDatabase.in_memory() as database:
            assert await database.health() is True
        # Should be closed now


# ============================================================
# SCHEMA
# ============================================================


class TestSchema:
    async def test_schema_creates_tables(self, db: UCDatabase):
        result = await db.query("INFO FOR DB")
        assert result  # Should have table info

    async def test_schema_idempotent(self, db: UCDatabase):
        # Applying schema twice should not error
        await apply_schema(db)
        assert await db.health() is True

    async def test_meta_schema_version(self, db: UCDatabase):
        result = await db.query("SELECT * FROM meta:schema")
        assert result
        assert result[0]["version"] == 3


# ============================================================
# SCOPE QUERIES
# ============================================================


class TestScopeQueries:
    async def test_create_scope(self, db: UCDatabase):
        scope = await queries.create_scope(db, "my-project", "/home/user/proj")
        assert scope["name"] == "my-project"
        assert scope["path"] == "/home/user/proj"

    async def test_list_scopes(self, db: UCDatabase):
        await queries.create_scope(db, "proj-a")
        await queries.create_scope(db, "proj-b")
        scopes = await queries.list_scopes(db)
        assert len(scopes) == 2

    async def test_find_scope_by_path(self, db: UCDatabase):
        await queries.create_scope(db, "proj", "/some/path")
        found = await queries.find_scope_by_path(db, "/some/path")
        assert found is not None
        assert found["name"] == "proj"

    async def test_find_scope_by_path_not_found(self, db: UCDatabase):
        found = await queries.find_scope_by_path(db, "/nonexistent")
        assert found is None


# ============================================================
# RUN QUERIES
# ============================================================


class TestRunQueries:
    async def test_create_run(self, db: UCDatabase):
        scope = await queries.create_scope(db, "proj")
        scope_id = str(scope["id"])
        run = await queries.create_run(db, scope_id, "claude")
        assert run  # Transaction returns first statement result

    async def test_end_run(self, db: UCDatabase):
        scope = await queries.create_scope(db, "proj")
        scope_id = str(scope["id"])
        run = await queries.create_run(db, scope_id, "claude")
        run_id = str(run["id"])
        await queries.end_run(db, run_id, "completed")
        updated = await queries.get_run(db, run_id)
        assert updated["status"] == "completed"
        assert updated["ended_at"] is not None

    async def test_list_runs_by_status(self, db: UCDatabase):
        scope = await queries.create_scope(db, "proj")
        scope_id = str(scope["id"])
        await queries.create_run(db, scope_id, "claude")
        active_runs = await queries.list_runs(db, status="active")
        assert len(active_runs) == 1


# ============================================================
# TURN + ARTIFACT (atomic creation)
# ============================================================


class TestTurnQueries:
    async def _setup_run(self, db: UCDatabase) -> str:
        scope = await queries.create_scope(db, "proj")
        scope_id = str(scope["id"])
        run = await queries.create_run(db, scope_id, "claude")
        return str(run["id"])

    async def test_create_turn_with_artifact(self, db: UCDatabase):
        run_id = await self._setup_run(db)
        result = await queries.create_turn_with_artifact(
            db, run_id, sequence=1, user_message="fix the bug",
            raw_content="user: fix the bug\nassistant: I'll look into it"
        )
        assert "turn_id" in result
        assert "artifact_id" in result

    async def test_turn_creates_job(self, db: UCDatabase):
        run_id = await self._setup_run(db)
        await queries.create_turn_with_artifact(
            db, run_id, sequence=1, user_message="hello",
            raw_content="test content", create_summary_job=True,
        )
        jobs = await queries.list_jobs(db, status="pending")
        assert len(jobs) == 1
        assert jobs[0]["job_type"] == "turn_summary"

    async def test_count_turns(self, db: UCDatabase):
        run_id = await self._setup_run(db)
        await queries.create_turn_with_artifact(
            db, run_id, 1, "msg1", "content1", create_summary_job=False
        )
        await queries.create_turn_with_artifact(
            db, run_id, 2, "msg2", "content2", create_summary_job=False
        )
        count = await queries.count_turns(db, run_id)
        assert count == 2

    async def test_list_turns_ordered(self, db: UCDatabase):
        run_id = await self._setup_run(db)
        await queries.create_turn_with_artifact(
            db, run_id, 2, "second", "c2", create_summary_job=False
        )
        await queries.create_turn_with_artifact(
            db, run_id, 1, "first", "c1", create_summary_job=False
        )
        turns = await queries.list_turns(db, run_id)
        assert len(turns) == 2
        assert turns[0]["sequence"] == 1
        assert turns[1]["sequence"] == 2


# ============================================================
# ARTIFACT QUERIES
# ============================================================


class TestArtifactQueries:
    async def test_create_artifact(self, db: UCDatabase):
        aid = await queries.create_artifact(db, "note", content="some note")
        assert aid.startswith("artifact:")

    async def test_create_derived_artifact(self, db: UCDatabase):
        source_id = await queries.create_artifact(db, "transcript", content="raw text")
        derived_id = await queries.create_derived_artifact(
            db, "summary", "summarized text", source_id
        )
        assert derived_id.startswith("artifact:")

        # Verify lineage
        lineage = await queries.get_artifact_lineage(db, derived_id)
        assert lineage  # Should have depends_on edge

    async def test_search_artifacts_fts(self, db: UCDatabase):
        """On embedded, search falls back to substring match."""
        await queries.create_artifact(db, "transcript", content="debugging the auth module")
        await queries.create_artifact(db, "transcript", content="fixing CSS styles")
        results = await queries.search_artifacts(db, "auth")
        # On embedded (mem://), uses substring match fallback
        assert isinstance(results, list)
        assert len(results) == 1
        assert "auth" in results[0]["content"]


# ============================================================
# PROVENANCE
# ============================================================


class TestProvenance:
    async def test_full_provenance_chain(self, db: UCDatabase):
        scope = await queries.create_scope(db, "proj", "/path")
        scope_id = str(scope["id"])
        run = await queries.create_run(db, scope_id, "claude")
        run_id = str(run["id"])
        result = await queries.create_turn_with_artifact(
            db, run_id, 1, "hello", "transcript content", create_summary_job=False
        )
        # Traverse from artifact back to scope
        chain = await queries.get_provenance_chain(db, result["artifact_id"])
        assert chain  # Should have provenance data

    async def test_turn_artifacts(self, db: UCDatabase):
        scope = await queries.create_scope(db, "proj")
        scope_id = str(scope["id"])
        run = await queries.create_run(db, scope_id, "codex")
        run_id = str(run["id"])
        result = await queries.create_turn_with_artifact(
            db, run_id, 1, "msg", "content", create_summary_job=False
        )
        artifacts = await queries.get_turn_artifacts(db, result["turn_id"])
        assert artifacts  # Should have at least the transcript


# ============================================================
# JOB QUEUE
# ============================================================


class TestJobQueue:
    async def test_create_and_claim_job(self, db: UCDatabase):
        await queries.create_job(db, "turn_summary", "turn:abc")
        claimed = await queries.claim_next_job(db)
        assert claimed is not None
        assert claimed["status"] == "running"
        assert claimed["job_type"] == "turn_summary"

    async def test_claim_returns_none_when_empty(self, db: UCDatabase):
        claimed = await queries.claim_next_job(db)
        assert claimed is None

    async def test_complete_job(self, db: UCDatabase):
        await queries.create_job(db, "turn_summary", "turn:abc")
        claimed = await queries.claim_next_job(db)
        job_id_str = str(claimed["id"])
        await queries.complete_job(db, job_id_str, {"summary": "done"})
        jobs = await queries.list_jobs(db, status="completed")
        assert len(jobs) == 1

    async def test_fail_job_with_retry(self, db: UCDatabase):
        await queries.create_job(db, "turn_summary", "turn:abc")
        claimed = await queries.claim_next_job(db)
        job_id = str(claimed["id"])
        await queries.fail_job(db, job_id, "LLM timeout")
        # Should be back to pending (attempts < max_attempts)
        jobs = await queries.list_jobs(db, status="pending")
        assert len(jobs) == 1
        assert jobs[0]["attempts"] == 1
        assert jobs[0]["error"] == "LLM timeout"

    async def test_count_jobs_by_status(self, db: UCDatabase):
        await queries.create_job(db, "turn_summary", "turn:1")
        await queries.create_job(db, "embedding", "artifact:1")
        await queries.claim_next_job(db)
        counts = await queries.count_jobs_by_status(db)
        assert "running" in counts or "pending" in counts

    async def test_priority_ordering(self, db: UCDatabase):
        await queries.create_job(db, "embedding", "artifact:1", priority=0)
        await queries.create_job(db, "turn_summary", "turn:1", priority=10)
        claimed = await queries.claim_next_job(db)
        assert claimed["job_type"] == "turn_summary"  # Higher priority first
