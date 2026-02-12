"""Alpha CLI surface and routing tests."""

from __future__ import annotations

import json

from typer.testing import CliRunner

import universal_context.cli as cli_mod

runner = CliRunner()


def test_top_level_help_is_minimal_surface():
    result = runner.invoke(cli_mod.app, ["--help"])
    assert result.exit_code == 0
    output = result.stdout
    assert "ask" in output
    assert "find" in output
    assert "memory" in output
    assert "admin" in output
    assert "│ search " not in output
    assert "│ context " not in output
    assert "│ reason " not in output


def test_admin_help_contains_operator_commands_not_checkpoint():
    result = runner.invoke(cli_mod.app, ["admin", "--help"])
    assert result.exit_code == 0
    output = result.stdout
    assert "context" in output
    assert "reason" in output
    assert "scope" in output
    assert "checkpoint" not in output


def test_memory_help_includes_sync():
    result = runner.invoke(cli_mod.app, ["memory", "--help"])
    assert result.exit_code == 0
    assert "sync" in result.stdout


def test_sanitize_search_result_drops_embedding():
    out = cli_mod._sanitize_search_result(
        {"id": "artifact:abc", "content": "x", "embedding": [0.1, 0.2]}
    )
    assert out["id"] == "artifact:abc"
    assert "embedding" not in out


def test_memory_sync_calls_refresh_and_inject(monkeypatch):
    calls: list[tuple[str, object, object]] = []

    def fake_refresh(project=None, json_output=False):
        calls.append(("refresh", project, json_output))

    def fake_inject(project=None, target="AGENTS.md"):
        calls.append(("inject", project, target))

    monkeypatch.setattr(cli_mod, "memory_refresh", fake_refresh)
    monkeypatch.setattr(cli_mod, "memory_inject", fake_inject)

    cli_mod.memory_sync(project=None, target="AGENTS.md")

    assert calls == [
        ("refresh", None, False),
        ("inject", None, "AGENTS.md"),
    ]


def test_ask_deep_json_returns_structured_payload(monkeypatch):
    async def fake_run_reasoning(*args, **kwargs):
        return {
            "answer": "Deep answer",
            "facts": ["F1"],
            "decisions": ["D1"],
            "open_questions": [],
            "evidence_ids": ["artifact:abc123"],
            "trajectory": [],
            "iterations": 2,
            "scope": "scope:test",
        }

    async def fake_persist(project_path: str, result: dict[str, object]):
        return "artifact:wm123"

    monkeypatch.setattr(cli_mod, "_run_reasoning", fake_run_reasoning)
    monkeypatch.setattr(cli_mod, "_persist_reasoning_snapshot", fake_persist)

    result = runner.invoke(cli_mod.app, ["ask", "question", "--deep", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "deep"
    assert payload["answer"] == "Deep answer"
    assert payload["facts"] == ["F1"]
    assert payload["decisions"] == ["D1"]
    assert payload["persisted_working_memory"] == "artifact:wm123"


def test_ask_auto_deep_when_shallow_context_missing(monkeypatch):
    class FakeDB:
        async def connect(self):
            return None

        async def close(self):
            return None

    async def fake_apply_schema(db):
        return None

    async def fake_resolve_scope(db, project_path: str):
        return None

    async def fake_search_artifacts(*args, **kwargs):
        return []

    async def fake_semantic_search(*args, **kwargs):
        return []

    async def fake_run_reasoning(*args, **kwargs):
        return {
            "answer": "Auto deep answer",
            "facts": [],
            "decisions": [],
            "open_questions": [],
            "evidence_ids": ["artifact:auto1"],
            "trajectory": [],
            "iterations": 1,
            "scope": None,
        }

    async def fake_persist(*args, **kwargs):
        return None

    monkeypatch.setattr(cli_mod, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(cli_mod, "_resolve_scope", fake_resolve_scope)
    monkeypatch.setattr(cli_mod, "_semantic_search", fake_semantic_search)
    monkeypatch.setattr(cli_mod, "_run_reasoning", fake_run_reasoning)
    monkeypatch.setattr(cli_mod, "_persist_reasoning_snapshot", fake_persist)
    monkeypatch.setattr("universal_context.db.schema.apply_schema", fake_apply_schema)
    monkeypatch.setattr("universal_context.db.queries.search_artifacts", fake_search_artifacts)

    result = runner.invoke(cli_mod.app, ["ask", "question", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "deep_auto"
    assert payload["answer"] == "Auto deep answer"
    assert payload["evidence_ids"] == ["artifact:auto1"]


def test_find_auto_mode_prefers_semantic_results(monkeypatch):
    class FakeDB:
        async def connect(self):
            return None

        async def close(self):
            return None

    async def fake_apply_schema(db):
        return None

    async def fake_resolve_scope(db, project_path: str):
        return {"id": "scope:test"}

    async def fake_search_artifacts(*args, **kwargs):
        return [{"id": "artifact:kw1", "kind": "summary", "content": "keyword match"}]

    async def fake_semantic_search(*args, **kwargs):
        return [
            {
                "id": "artifact:sem1",
                "kind": "summary",
                "content": "semantic match",
                "score": 0.99,
            }
        ]

    monkeypatch.setattr(cli_mod, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(cli_mod, "_resolve_scope", fake_resolve_scope)
    monkeypatch.setattr(cli_mod, "_semantic_search", fake_semantic_search)
    monkeypatch.setattr("universal_context.db.schema.apply_schema", fake_apply_schema)
    monkeypatch.setattr("universal_context.db.queries.search_artifacts", fake_search_artifacts)

    result = runner.invoke(
        cli_mod.app,
        ["find", "auth", "--mode", "auto", "--project", ".", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["scope"] == "scope:test"
    assert payload["results"][0]["id"] == "artifact:sem1"
    assert payload["keyword_results"][0]["id"] == "artifact:kw1"
    assert payload["semantic_results"][0]["id"] == "artifact:sem1"


def test_find_invalid_mode_fails_fast():
    result = runner.invoke(cli_mod.app, ["find", "auth", "--mode", "bad-mode"])
    assert result.exit_code != 0
    assert "Invalid mode" in result.stdout
