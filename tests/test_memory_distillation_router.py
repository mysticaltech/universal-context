"""Tests for distillation routing into durable memory section files."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from universal_context.daemon.processors.memory import (
    MAX_SUMMARY_EVIDENCE_ROWS,
    WorkingMemoryProcessor,
    _build_summary_evidence,
)
from universal_context.db.client import UCDatabase
from universal_context.db.queries import (
    create_derived_artifact,
    create_run,
    create_scope,
    create_turn_with_artifact,
)
from universal_context.db.schema import apply_schema
from universal_context.memory_repo import list_scope_sections


@pytest.fixture
async def db():
    database = UCDatabase.in_memory()
    await database.connect()
    await apply_schema(database)
    yield database
    await database.close()


async def _seed_scope_with_summary(
    db: UCDatabase,
    project_path: str,
    canonical_id: str,
) -> str:
    scope = await create_scope(
        db,
        "router-project",
        project_path,
        canonical_id=canonical_id,
    )
    scope_id = str(scope["id"])
    run = await create_run(db, scope_id, "claude")
    run_id = str(run["id"])

    turn = await create_turn_with_artifact(
        db,
        run_id,
        sequence=1,
        user_message="ship routing",
        raw_content="User asked for routing changes",
        create_summary_job=False,
    )
    await create_derived_artifact(
        db,
        kind="summary",
        content="Implemented distillation router for memory files.",
        source_id=turn["artifact_id"],
        relationship="summarized_from",
    )
    return scope_id


@pytest.mark.asyncio
async def test_distillation_routes_to_multiple_sections(tmp_path, monkeypatch, db: UCDatabase):
    uc_home = tmp_path / ".uc"
    monkeypatch.setattr("universal_context.memory_repo.get_uc_home", lambda: uc_home)

    canonical_id = "github.com/acme/router-project"
    project_path = str(tmp_path / "router-project")
    scope_id = await _seed_scope_with_summary(db, project_path, canonical_id)

    llm_output = """# Project Memory: router-project
## Architecture & Key Decisions
- Durable memory is canonical.

## Current State
- Distillation writes memory files first.
"""
    processor = WorkingMemoryProcessor(llm_fn=AsyncMock(return_value=llm_output))
    result = await processor.process(db, {"target": scope_id})
    assert result["method"] == "llm"

    sections = list_scope_sections(
        canonical_id=canonical_id,
        display_name="router-project",
        root=uc_home / "memory",
        scope_path=project_path,
    )
    assert sections["architecture"]
    assert sections["state"]
    assert "Durable memory is canonical." in sections["architecture"][0]["content"]
    assert "Distillation writes memory files first." in sections["state"][0]["content"]


@pytest.mark.asyncio
async def test_distillation_falls_back_to_state_section(tmp_path, monkeypatch, db: UCDatabase):
    uc_home = tmp_path / ".uc"
    monkeypatch.setattr("universal_context.memory_repo.get_uc_home", lambda: uc_home)

    canonical_id = "github.com/acme/router-fallback"
    project_path = str(tmp_path / "router-fallback")
    scope_id = await _seed_scope_with_summary(db, project_path, canonical_id)

    llm_output = "No headings, one block of memory content."
    processor = WorkingMemoryProcessor(llm_fn=AsyncMock(return_value=llm_output))
    await processor.process(db, {"target": scope_id})

    sections = list_scope_sections(
        canonical_id=canonical_id,
        display_name="router-project",
        root=uc_home / "memory",
        scope_path=project_path,
    )
    assert sections["state"]
    assert "No headings, one block of memory content." in sections["state"][0]["content"]


def test_summary_evidence_is_bounded():
    summaries = [
        {"run_id": f"run:{idx}", "turn_id": f"turn:{idx}"}
        for idx in range(100)
    ]
    evidence = _build_summary_evidence(summaries)
    assert len(evidence) <= MAX_SUMMARY_EVIDENCE_ROWS
