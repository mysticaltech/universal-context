"""Tests for durable file-first memory read precedence."""

from __future__ import annotations

import json

from typer.testing import CliRunner

import universal_context.cli as cli_mod
from universal_context.memory_repo import append_section_entry, bootstrap_memory_repo

runner = CliRunner()


def _seed_memory_file(uc_home, canonical_id: str, project_name: str, project_path: str) -> None:
    memory_root = bootstrap_memory_repo(uc_home / "memory", init_git=False)
    append_section_entry(
        canonical_id=canonical_id,
        section="state",
        content="Durable file memory entry",
        display_name=project_name,
        memory_type="durable_fact",
        manual=True,
        source="remember",
        root=memory_root,
        scope_path=project_path,
    )


def test_memory_show_prefers_files_without_db(tmp_path, monkeypatch):
    uc_home = tmp_path / ".uc"
    project_path = tmp_path / "proj"
    project_path.mkdir()
    canonical_id = "github.com/acme/prefer-files"

    _seed_memory_file(uc_home, canonical_id, project_path.name, str(project_path))

    monkeypatch.setattr("universal_context.memory_repo.get_uc_home", lambda: uc_home)
    monkeypatch.setattr("universal_context.git.resolve_canonical_id", lambda _p: canonical_id)
    monkeypatch.setattr(
        cli_mod,
        "_get_db",
        lambda: (_ for _ in ()).throw(AssertionError("DB should not be used")),
    )

    result = runner.invoke(
        cli_mod.app,
        ["memory", "show", "--project", str(project_path), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["metadata"]["method"] == "memory_repo"
    assert "Durable file memory entry" in payload["content"]


def test_ask_uses_files_when_db_unavailable(tmp_path, monkeypatch):
    uc_home = tmp_path / ".uc"
    project_path = tmp_path / "proj"
    project_path.mkdir()
    canonical_id = "github.com/acme/ask-files"

    _seed_memory_file(uc_home, canonical_id, project_path.name, str(project_path))

    monkeypatch.setattr("universal_context.memory_repo.get_uc_home", lambda: uc_home)
    monkeypatch.setattr("universal_context.git.resolve_canonical_id", lambda _p: canonical_id)

    def broken_db():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(cli_mod, "_get_db", broken_db)

    async def fake_create_llm_fn(*_args, **_kwargs):
        async def _fake_llm(_prompt: str) -> str:
            return "Answer from durable memory only"

        return _fake_llm

    monkeypatch.setattr("universal_context.llm.create_llm_fn", fake_create_llm_fn)

    result = runner.invoke(
        cli_mod.app,
        ["ask", "what is current state?", "--project", str(project_path), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "shallow"
    assert payload["answer"] == "Answer from durable memory only"
    assert payload["sources"] == 0
