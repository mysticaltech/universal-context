"""Tests for admin DB rebuild durability contract."""

from __future__ import annotations

import pytest

from universal_context.config import UCConfig
from universal_context.db.client import UCDatabase
from universal_context.db.queries import find_scope_by_canonical_id, get_working_memory
from universal_context.db.schema import apply_schema
from universal_context.memory_repo import (
    append_section_entry,
    bootstrap_memory_repo,
    get_scope_map_path,
    get_section_file,
)
from universal_context.rebuild import rebuild_derived_db


@pytest.fixture
async def db():
    database = UCDatabase.in_memory()
    await database.connect()
    await apply_schema(database)
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_rebuild_rehydrates_memory_only_scope_without_mutating_memory(
    tmp_path,
    monkeypatch,
    db: UCDatabase,
):
    uc_home = tmp_path / ".uc"
    monkeypatch.setattr("universal_context.memory_repo.get_uc_home", lambda: uc_home)

    memory_root = bootstrap_memory_repo(uc_home / "memory", init_git=False)
    canonical_id = "github.com/acme/memory-only"

    append_section_entry(
        canonical_id=canonical_id,
        section="state",
        content="Survives DB loss",
        display_name="memory-only",
        memory_type="durable_fact",
        manual=True,
        source="remember",
        evidence=[{"artifact_id": "artifact:abc123"}],
        root=memory_root,
        scope_path=str(tmp_path / "memory-only"),
    )

    scope_map = get_scope_map_path(memory_root)
    state_file = get_section_file(
        canonical_id=canonical_id,
        section="state",
        display_name="memory-only",
        root=memory_root,
        create=False,
    )
    scope_before = scope_map.read_text(encoding="utf-8")
    state_before = state_file.read_text(encoding="utf-8")

    class FakeRegistry:
        def discover_all_sessions(self, _config):
            return []

    monkeypatch.setattr("universal_context.rebuild.get_registry", lambda: FakeRegistry())

    result = await rebuild_derived_db(db, UCConfig(use_llm=False))

    assert result["mode"] == "full"
    assert result["scopes_from_registry"] >= 1
    assert result["working_memory_rehydrated"] >= 1

    scope_after = scope_map.read_text(encoding="utf-8")
    state_after = state_file.read_text(encoding="utf-8")
    assert scope_after == scope_before
    assert state_after == state_before

    scope = await find_scope_by_canonical_id(db, canonical_id)
    assert scope is not None

    wm = await get_working_memory(db, str(scope["id"]))
    assert wm is not None
    assert wm["metadata"]["method"] == "memory_repo"
    assert "Survives DB loss" in wm["content"]


@pytest.mark.asyncio
async def test_rebuild_skips_unsupported_adapter_sessions(tmp_path, monkeypatch, db: UCDatabase):
    uc_home = tmp_path / ".uc"
    monkeypatch.setattr("universal_context.memory_repo.get_uc_home", lambda: uc_home)
    bootstrap_memory_repo(uc_home / "memory", init_git=False)

    class UnsupportedAdapter:
        name = "unsupported"

    class FakeRegistry:
        def discover_all_sessions(self, _config):
            return [(tmp_path / "dummy.session", UnsupportedAdapter())]

    monkeypatch.setattr("universal_context.rebuild.get_registry", lambda: FakeRegistry())
    result = await rebuild_derived_db(db, UCConfig(use_llm=False))
    assert result["skipped_sessions_unsupported"] == 1
