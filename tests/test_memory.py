"""Tests for the working memory system — processor, queries, and CLI."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from universal_context.daemon.processors.memory import (
    WorkingMemoryProcessor,
    _format_summaries,
    build_distillation_prompt,
)
from universal_context.db.client import UCDatabase
from universal_context.db.queries import (
    create_derived_artifact,
    create_run,
    create_scope,
    create_turn_with_artifact,
    get_scope_summaries_for_distillation,
    get_working_memory,
    get_working_memory_history,
    set_working_memory_reasoning_metadata,
    upsert_working_memory,
)
from universal_context.db.schema import apply_schema
from universal_context.models.types import ArtifactKind, JobType


@pytest.fixture
async def db():
    """In-memory database with schema."""
    database = UCDatabase.in_memory()
    await database.connect()
    await apply_schema(database)
    yield database
    await database.close()


async def _setup_scope_with_turns(db: UCDatabase, n_turns: int = 3) -> tuple[str, str]:
    """Create a scope with a run and N turns+summaries. Returns (scope_id, run_id)."""
    scope = await create_scope(db, "test-project", "/tmp/test-project")
    scope_id = str(scope["id"])
    run = await create_run(db, scope_id, "claude")
    run_id = str(run["id"])

    for i in range(1, n_turns + 1):
        result = await create_turn_with_artifact(
            db,
            run_id,
            sequence=i,
            user_message=f"User question {i}",
            raw_content=f"Turn {i} transcript content",
            create_summary_job=False,
        )
        # Create a summary artifact derived from the transcript
        await create_derived_artifact(
            db,
            kind="summary",
            content=f"Summary of turn {i}: did thing {i}",
            source_id=result["artifact_id"],
            relationship="summarized_from",
        )

    return scope_id, run_id


# ============================================================
# ENUM TESTS
# ============================================================


class TestEnums:
    def test_working_memory_artifact_kind(self):
        assert ArtifactKind.WORKING_MEMORY == "working_memory"

    def test_memory_update_job_type(self):
        assert JobType.MEMORY_UPDATE == "memory_update"


# ============================================================
# QUERY TESTS
# ============================================================


class TestWorkingMemoryQueries:
    async def test_upsert_creates_artifact(self, db: UCDatabase):
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])

        aid = await upsert_working_memory(db, scope_id, "# Memory\n- point 1")
        assert aid.startswith("artifact:")

        memory = await get_working_memory(db, scope_id)
        assert memory is not None
        assert memory["content"] == "# Memory\n- point 1"
        assert memory["kind"] == "working_memory"
        assert memory["metadata"]["scope_id"] == scope_id
        assert memory["metadata"]["method"] == "llm"

    async def test_upsert_supersedes_previous(self, db: UCDatabase):
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])

        await upsert_working_memory(db, scope_id, "v1")
        await upsert_working_memory(db, scope_id, "v2")

        # Latest should be v2
        memory = await get_working_memory(db, scope_id)
        assert memory["content"] == "v2"

        # Both should exist in history
        history = await get_working_memory_history(db, scope_id)
        assert len(history) == 2
        assert history[0]["content"] == "v2"
        assert history[1]["content"] == "v1"

    async def test_get_working_memory_none(self, db: UCDatabase):
        scope = await create_scope(db, "empty", "/tmp/empty")
        memory = await get_working_memory(db, str(scope["id"]))
        assert memory is None

    async def test_history_respects_limit(self, db: UCDatabase):
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])

        for i in range(5):
            await upsert_working_memory(db, scope_id, f"v{i}")

        history = await get_working_memory_history(db, scope_id, limit=3)
        assert len(history) == 3

    async def test_set_reasoning_metadata_on_latest_memory(self, db: UCDatabase):
        scope = await create_scope(db, "proj", "/tmp/proj")
        scope_id = str(scope["id"])
        await upsert_working_memory(db, scope_id, "v1")

        updated_id = await set_working_memory_reasoning_metadata(
            db,
            scope_id,
            {
                "facts": ["Fact 1"],
                "decisions": ["Decision 1"],
                "open_questions": [],
                "evidence_ids": ["artifact:abc123"],
            },
        )
        assert updated_id is not None

        memory = await get_working_memory(db, scope_id)
        assert memory is not None
        last = memory.get("metadata", {}).get("last_reasoning", {})
        assert last.get("facts") == ["Fact 1"]
        assert last.get("decisions") == ["Decision 1"]
        assert last.get("evidence_ids") == ["artifact:abc123"]

    async def test_get_scope_summaries_for_distillation(self, db: UCDatabase):
        scope_id, run_id = await _setup_scope_with_turns(db, n_turns=3)

        summaries = await get_scope_summaries_for_distillation(db, scope_id)
        assert len(summaries) == 3
        for s in summaries:
            assert s["run_id"] == run_id
            assert s["agent_type"] == "claude"
            assert s["user_message"] is not None

    async def test_summaries_respects_limit(self, db: UCDatabase):
        scope_id, _ = await _setup_scope_with_turns(db, n_turns=5)

        summaries = await get_scope_summaries_for_distillation(db, scope_id, limit=2)
        assert len(summaries) == 2

    async def test_summaries_empty_scope(self, db: UCDatabase):
        scope = await create_scope(db, "empty", "/tmp/empty")
        summaries = await get_scope_summaries_for_distillation(db, str(scope["id"]))
        assert summaries == []


# ============================================================
# PROMPT BUILDING TESTS
# ============================================================


class TestPromptBuilding:
    def test_format_summaries_empty(self):
        assert _format_summaries([]) == "(no summaries available)"

    def test_format_summaries_basic(self):
        summaries = [{
            "agent_type": "claude",
            "run_id": "run:abc",
            "sequence": 1,
            "user_message": "fix the bug",
            "summary": "Fixed auth bug in middleware",
            "started_at": "2024-01-15T10:00:00",
        }]
        result = _format_summaries(summaries)
        assert "claude" in result
        assert "fix the bug" in result
        assert "Fixed auth bug" in result

    def test_build_prompt_first_distillation(self):
        prompt = build_distillation_prompt("MyProject", [], None)
        assert "MyProject" in prompt
        assert "first distillation" in prompt.lower()

    def test_build_prompt_with_previous_memory(self):
        previous = "# Project Memory: MyProject\n## Key Decisions\n- Use PostgreSQL"
        prompt = build_distillation_prompt("MyProject", [], previous)
        assert "PostgreSQL" in prompt
        assert "MyProject" in prompt

    def test_build_prompt_includes_summaries(self):
        summaries = [{
            "agent_type": "codex",
            "run_id": "run:xyz",
            "sequence": 2,
            "user_message": "add caching",
            "summary": "Added Redis caching layer",
            "started_at": "2024-02-01T14:00:00",
        }]
        prompt = build_distillation_prompt("MyProject", summaries, None)
        assert "Redis caching" in prompt
        assert "add caching" in prompt


# ============================================================
# PROCESSOR TESTS
# ============================================================


class TestWorkingMemoryProcessor:
    async def test_process_creates_memory(self, db: UCDatabase):
        scope_id, _ = await _setup_scope_with_turns(db, n_turns=3)

        llm_fn = AsyncMock(
            return_value="# Project Memory: test-project\n## Key Decisions\n- Use async"
        )
        processor = WorkingMemoryProcessor(llm_fn=llm_fn)

        job = {"target": scope_id}
        result = await processor.process(db, job)

        assert result["method"] == "llm"
        assert result["summaries_used"] == 3
        assert result["had_previous"] is False
        assert result["artifact_id"].startswith("artifact:")

        # Verify stored
        memory = await get_working_memory(db, scope_id)
        assert memory is not None
        assert "async" in memory["content"]

    async def test_process_evolves_memory(self, db: UCDatabase):
        scope_id, _ = await _setup_scope_with_turns(db, n_turns=2)

        # Create initial memory
        await upsert_working_memory(db, scope_id, "# v1 memory")

        llm_fn = AsyncMock(return_value="# v2 memory with new info")
        processor = WorkingMemoryProcessor(llm_fn=llm_fn)

        job = {"target": scope_id}
        result = await processor.process(db, job)

        assert result["had_previous"] is True
        # LLM should have received the previous memory
        call_args = llm_fn.call_args[0][0]
        assert "v1 memory" in call_args

    async def test_process_skips_empty_scope(self, db: UCDatabase):
        scope = await create_scope(db, "empty", "/tmp/empty")
        scope_id = str(scope["id"])

        llm_fn = AsyncMock()
        processor = WorkingMemoryProcessor(llm_fn=llm_fn)

        result = await processor.process(db, {"target": scope_id})
        assert result["status"] == "skipped"
        assert result["reason"] == "no_summaries"
        llm_fn.assert_not_awaited()

    async def test_process_embeds_when_provider_available(self, db: UCDatabase):
        scope_id, _ = await _setup_scope_with_turns(db, n_turns=1)

        llm_fn = AsyncMock(return_value="# Memory")
        embed_fn = AsyncMock()
        embed_fn.embed_document = AsyncMock(return_value=[0.1] * 768)

        processor = WorkingMemoryProcessor(llm_fn=llm_fn, embed_fn=embed_fn)
        result = await processor.process(db, {"target": scope_id})

        assert result["embedded"] is True
        embed_fn.embed_document.assert_awaited_once()

    async def test_process_no_target_raises(self, db: UCDatabase):
        processor = WorkingMemoryProcessor(llm_fn=AsyncMock())
        with pytest.raises(ValueError, match="no target"):
            await processor.process(db, {"target": ""})

    async def test_process_missing_scope_raises(self, db: UCDatabase):
        processor = WorkingMemoryProcessor(llm_fn=AsyncMock())
        with pytest.raises(ValueError, match="Scope not found"):
            await processor.process(db, {"target": "scope:nonexistent"})

    async def test_process_empty_llm_response_raises(self, db: UCDatabase):
        scope_id, _ = await _setup_scope_with_turns(db, n_turns=1)

        llm_fn = AsyncMock(return_value="")
        processor = WorkingMemoryProcessor(llm_fn=llm_fn)

        with pytest.raises(ValueError, match="empty working memory"):
            await processor.process(db, {"target": scope_id})


# ============================================================
# CLI INJECT/EJECT TESTS (unit-level, filesystem only)
# ============================================================


class TestMemoryInjection:
    """Tests for the sentinel-marker injection logic (no DB needed)."""

    def test_sentinel_markers_in_cli(self):
        from universal_context.cli import _MEMORY_END, _MEMORY_START

        assert "UC:MEMORY:START" in _MEMORY_START
        assert "UC:MEMORY:END" in _MEMORY_END

    def test_inject_into_empty_file(self, tmp_path: Path):
        from universal_context.cli import _MEMORY_END, _MEMORY_START

        target = tmp_path / "AGENTS.md"
        content = "# Project Memory\n- point 1"
        block = f"{_MEMORY_START}\n{content}\n{_MEMORY_END}"

        target.write_text(block + "\n", encoding="utf-8")
        text = target.read_text(encoding="utf-8")
        assert _MEMORY_START in text
        assert "point 1" in text
        assert _MEMORY_END in text

    def test_inject_replaces_existing_section(self, tmp_path: Path):
        from universal_context.cli import _MEMORY_END, _MEMORY_START

        target = tmp_path / "AGENTS.md"
        old_block = f"{_MEMORY_START}\nold content\n{_MEMORY_END}"
        target.write_text(f"# Header\n\n{old_block}\n\n# Footer\n", encoding="utf-8")

        new_block = f"{_MEMORY_START}\nnew content\n{_MEMORY_END}"
        existing = target.read_text(encoding="utf-8")
        pattern = re.escape(_MEMORY_START) + r".*?" + re.escape(_MEMORY_END)
        updated = re.sub(pattern, new_block, existing, flags=re.DOTALL)
        target.write_text(updated, encoding="utf-8")

        text = target.read_text(encoding="utf-8")
        assert "new content" in text
        assert "old content" not in text
        assert "# Header" in text
        assert "# Footer" in text

    def test_eject_removes_section(self, tmp_path: Path):
        from universal_context.cli import _MEMORY_END, _MEMORY_START

        target = tmp_path / "AGENTS.md"
        target.write_text(
            f"# Header\n\n{_MEMORY_START}\nmemory stuff\n{_MEMORY_END}\n\n# Footer\n",
            encoding="utf-8",
        )

        existing = target.read_text(encoding="utf-8")
        pattern = r"\n*" + re.escape(_MEMORY_START) + r".*?" + re.escape(_MEMORY_END) + r"\n*"
        cleaned = re.sub(pattern, "\n", existing, flags=re.DOTALL).strip()
        target.write_text(cleaned + "\n", encoding="utf-8")

        text = target.read_text(encoding="utf-8")
        assert "memory stuff" not in text
        assert _MEMORY_START not in text
        assert "# Header" in text
        assert "# Footer" in text
