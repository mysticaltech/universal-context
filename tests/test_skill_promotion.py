"""Tests for skill promotion from durable procedure memories."""

from __future__ import annotations

import asyncio

from typer.testing import CliRunner

import universal_context.cli as cli_mod
from universal_context.db.client import UCDatabase
from universal_context.db.queries import create_scope
from universal_context.db.schema import apply_schema
from universal_context.memory_repo import (
    append_section_entry,
    bootstrap_memory_repo,
    get_memory_skills_root,
)

runner = CliRunner()


def test_skill_promotion_emits_skill_files(tmp_path, monkeypatch):
    uc_home = tmp_path / ".uc"
    project_path = tmp_path / "proj"
    project_path.mkdir()
    canonical_id = "github.com/acme/proj"

    monkeypatch.setattr("universal_context.memory_repo.get_uc_home", lambda: uc_home)
    monkeypatch.setattr("universal_context.git.resolve_canonical_id", lambda _p: canonical_id)

    memory_root = bootstrap_memory_repo(uc_home / "memory", init_git=False)
    for idx in range(2):
        append_section_entry(
            canonical_id=canonical_id,
            section="procedures",
            content="Run `ruff check .` before opening a PR.",
            display_name=project_path.name,
            memory_type="procedure",
            confidence=0.95,
            manual=False,
            source="distilled",
            evidence=[{"run_id": f"run:{idx}"}],
            root=memory_root,
            scope_path=str(project_path),
        )

    db = UCDatabase.from_path(tmp_path / "uc-skill-promotion.db")

    async def _seed_db() -> None:
        await db.connect()
        await apply_schema(db)
        await create_scope(
            db,
            project_path.name,
            str(project_path),
            canonical_id=canonical_id,
        )
        await db.close()

    asyncio.run(_seed_db())

    monkeypatch.setattr(cli_mod, "_get_db", lambda: db)

    result = runner.invoke(
        cli_mod.app,
        [
            "admin",
            "promote-skills",
            "--project",
            str(project_path),
            "--min-occurrences",
            "2",
            "--min-confidence",
            "0.8",
            "--min-evidence",
            "1",
        ],
    )
    assert result.exit_code == 0

    skills_root = get_memory_skills_root(uc_home / "memory")
    skill_docs = list(skills_root.glob("*/SKILL.md"))
    assert skill_docs
    skill_doc = skill_docs[0].read_text(encoding="utf-8")
    metadata = (skill_docs[0].parent / "metadata.yaml").read_text(encoding="utf-8")

    assert "Run `ruff check .` before opening a PR." in skill_doc
    assert canonical_id in metadata
