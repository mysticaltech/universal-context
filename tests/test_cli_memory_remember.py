"""Tests for `uc remember` / `uc memory remember` durable writes."""

from __future__ import annotations

from typer.testing import CliRunner

import universal_context.cli as cli_mod
from universal_context.memory_repo import list_scope_sections

runner = CliRunner()


def test_remember_requires_project_and_type():
    missing_project = runner.invoke(cli_mod.app, ["remember", "hello", "--type", "decision"])
    assert missing_project.exit_code != 0

    missing_type = runner.invoke(cli_mod.app, ["remember", "hello", "--project", "."])
    assert missing_type.exit_code != 0


def test_top_level_remember_writes_manual_entry(tmp_path, monkeypatch):
    uc_home = tmp_path / ".uc"
    project_path = tmp_path / "proj"
    project_path.mkdir()
    canonical_id = "github.com/acme/proj"

    monkeypatch.setattr("universal_context.memory_repo.get_uc_home", lambda: uc_home)
    monkeypatch.setattr("universal_context.git.resolve_canonical_id", lambda _p: canonical_id)

    result = runner.invoke(
        cli_mod.app,
        [
            "remember",
            "Prefer graph traversal for provenance checks",
            "--project",
            str(project_path),
            "--type",
            "decision",
        ],
    )
    assert result.exit_code == 0

    sections = list_scope_sections(
        canonical_id=canonical_id,
        display_name=project_path.name,
        root=uc_home / "memory",
        scope_path=str(project_path),
    )
    assert sections["preferences"]
    entry = sections["preferences"][0]
    assert entry["manual"] is True
    assert entry["type"] == "decision"
    assert entry["source"] == "remember"


def test_memory_subcommand_remember_supports_evidence(tmp_path, monkeypatch):
    uc_home = tmp_path / ".uc"
    project_path = tmp_path / "proj"
    project_path.mkdir()
    canonical_id = "github.com/acme/proj"

    monkeypatch.setattr("universal_context.memory_repo.get_uc_home", lambda: uc_home)
    monkeypatch.setattr("universal_context.git.resolve_canonical_id", lambda _p: canonical_id)

    result = runner.invoke(
        cli_mod.app,
        [
            "memory",
            "remember",
            "Run uc admin db rebuild after DB corruption",
            "--project",
            str(project_path),
            "--type",
            "procedure",
            "--evidence",
            "run:run123",
            "--evidence",
            "turn:turn456",
        ],
    )
    assert result.exit_code == 0

    sections = list_scope_sections(
        canonical_id=canonical_id,
        display_name=project_path.name,
        root=uc_home / "memory",
        scope_path=str(project_path),
    )
    assert sections["procedures"]
    evidence = sections["procedures"][0]["evidence"]
    assert {"run_id": "run123"} in evidence
    assert {"turn_id": "turn456"} in evidence
