"""Tests for `uc memory migrate-db` command."""

from __future__ import annotations

import asyncio
import json

from typer.testing import CliRunner

import universal_context.cli as cli_mod
from universal_context.db.client import UCDatabase
from universal_context.db.queries import create_scope, upsert_working_memory
from universal_context.db.schema import apply_schema
from universal_context.memory_repo import (
    get_memory_migrations_root,
    list_scope_sections,
)

runner = CliRunner()


def test_memory_migrate_db_is_idempotent(tmp_path, monkeypatch):
    uc_home = tmp_path / ".uc"
    project_path = tmp_path / "proj"
    project_path.mkdir()
    canonical_id = "github.com/acme/migrate-proj"

    monkeypatch.setattr("universal_context.memory_repo.get_uc_home", lambda: uc_home)
    monkeypatch.setattr("universal_context.git.resolve_canonical_id", lambda _p: canonical_id)

    db = UCDatabase.from_path(tmp_path / "uc-migrate.db")

    async def _seed() -> str:
        await db.connect()
        await apply_schema(db)
        scope = await create_scope(
            db,
            project_path.name,
            str(project_path),
            canonical_id=canonical_id,
        )
        scope_id = str(scope["id"])
        await upsert_working_memory(
            db,
            scope_id,
            "# Project Memory\n## Current State\n- migration keeps content",
            method="llm",
        )
        await db.close()
        return scope_id

    scope_id = asyncio.run(_seed())
    monkeypatch.setattr(cli_mod, "_get_db", lambda: db)

    first = runner.invoke(
        cli_mod.app,
        ["memory", "migrate-db", "--project", str(project_path), "--json"],
    )
    assert first.exit_code == 0
    payload = json.loads(first.stdout)
    assert payload["ok"] is True
    assert scope_id in payload["migrated"]

    sections = list_scope_sections(
        canonical_id=canonical_id,
        display_name=project_path.name,
        root=uc_home / "memory",
        scope_path=str(project_path),
    )
    assert sections["state"]
    assert any(
        "migration keeps content" in str(entry.get("content", ""))
        for entry in sections["state"]
    )
    assert all(entry.get("source") == "migrated_db" for entry in sections["state"])

    marker_files = list(get_memory_migrations_root(uc_home / "memory").glob("*.json"))
    assert marker_files

    second = runner.invoke(
        cli_mod.app,
        ["memory", "migrate-db", "--project", str(project_path), "--json"],
    )
    assert second.exit_code == 0
    payload2 = json.loads(second.stdout)
    assert payload2["ok"] is True
    assert scope_id in payload2["skipped"]
    assert not payload2["failed"]
