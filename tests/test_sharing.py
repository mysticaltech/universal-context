"""Tests for sharing & checkpointing."""

from pathlib import Path

import pytest

from universal_context.db.client import UCDatabase
from universal_context.db.queries import (
    create_run,
    create_scope,
    create_turn_with_artifact,
    list_turns,
)
from universal_context.db.schema import apply_schema
from universal_context.sharing.bundle import export_bundle, import_bundle
from universal_context.sharing.checkpoint import (
    create_checkpoint,
    get_checkpoint,
    list_checkpoints,
)


@pytest.fixture
async def db():
    """In-memory database with schema."""
    database = UCDatabase.in_memory()
    await database.connect()
    await apply_schema(database)
    yield database
    await database.close()


async def _make_run_with_turns(db: UCDatabase) -> tuple[str, str, str]:
    """Create a scope→run with 2 turns. Returns (scope_id, run_id, turn_id)."""
    scope = await create_scope(db, "proj", "/tmp/proj")
    scope_id = str(scope["id"])
    run = await create_run(db, scope_id, "claude")
    run_id = str(run["id"])

    result1 = await create_turn_with_artifact(
        db, run_id, 1, "fix the bug",
        "user: fix the bug\nassistant: I found the issue in auth.py",
        create_summary_job=False,
    )
    await create_turn_with_artifact(
        db, run_id, 2, "add tests",
        "user: add tests\nassistant: Done, added test_auth.py",
        create_summary_job=False,
    )
    return scope_id, run_id, result1["turn_id"]


# ============================================================
# CHECKPOINT TESTS
# ============================================================


class TestCheckpoint:
    async def test_create_checkpoint(self, db: UCDatabase):
        _, run_id, turn_id = await _make_run_with_turns(db)
        cid = await create_checkpoint(db, run_id, turn_id, label="before refactor")
        assert cid.startswith("checkpoint:")

    async def test_get_checkpoint(self, db: UCDatabase):
        _, run_id, turn_id = await _make_run_with_turns(db)
        cid = await create_checkpoint(db, run_id, turn_id, label="v1")
        cp = await get_checkpoint(db, cid)
        assert cp is not None
        assert cp["label"] == "v1"
        assert cp["state"]["total_turns"] == 2

    async def test_list_checkpoints(self, db: UCDatabase):
        _, run_id, turn_id = await _make_run_with_turns(db)
        await create_checkpoint(db, run_id, turn_id, label="cp1")
        await create_checkpoint(db, run_id, turn_id, label="cp2")
        cps = await list_checkpoints(db, run_id)
        assert len(cps) == 2

    async def test_list_all_checkpoints(self, db: UCDatabase):
        _, run_id, turn_id = await _make_run_with_turns(db)
        await create_checkpoint(db, run_id, turn_id)
        cps = await list_checkpoints(db)
        assert len(cps) >= 1

    async def test_invalid_run_raises(self, db: UCDatabase):
        with pytest.raises(ValueError, match="Run not found"):
            await create_checkpoint(db, "run:nonexistent", "turn:x")

    async def test_invalid_turn_raises(self, db: UCDatabase):
        _, run_id, _ = await _make_run_with_turns(db)
        with pytest.raises(ValueError, match="Turn not found"):
            await create_checkpoint(db, run_id, "turn:nonexistent")


# ============================================================
# BUNDLE EXPORT/IMPORT TESTS
# ============================================================


class TestBundle:
    async def test_export_bundle(self, db: UCDatabase, tmp_path: Path):
        _, run_id, _ = await _make_run_with_turns(db)
        out = tmp_path / "bundle.json"
        result = await export_bundle(db, run_id, output_path=out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    async def test_roundtrip_export_import(self, db: UCDatabase, tmp_path: Path):
        _, run_id, _ = await _make_run_with_turns(db)
        out = tmp_path / "bundle.json"
        await export_bundle(db, run_id, output_path=out)

        # Import into same DB (different scope/run)
        result = await import_bundle(db, out)
        assert result["turns_imported"] == 2
        assert result["run_id"].startswith("run:")

        # Verify imported turns
        imported_turns = await list_turns(db, result["run_id"])
        assert len(imported_turns) == 2

    async def test_encrypted_roundtrip(self, db: UCDatabase, tmp_path: Path):
        _, run_id, _ = await _make_run_with_turns(db)
        out = tmp_path / "bundle.enc"
        await export_bundle(db, run_id, output_path=out, passphrase="secret123")

        # Content should be encrypted (not valid JSON)
        content = out.read_text()
        assert not content.startswith("{")

        # Import with correct passphrase
        result = await import_bundle(db, out, passphrase="secret123")
        assert result["turns_imported"] == 2

    async def test_encrypted_wrong_passphrase(self, db: UCDatabase, tmp_path: Path):
        _, run_id, _ = await _make_run_with_turns(db)
        out = tmp_path / "bundle.enc"
        await export_bundle(db, run_id, output_path=out, passphrase="secret123")

        with pytest.raises(Exception):
            await import_bundle(db, out, passphrase="wrong")

    async def test_export_nonexistent_run(self, db: UCDatabase, tmp_path: Path):
        with pytest.raises(ValueError, match="Run not found"):
            await export_bundle(db, "run:ghost", output_path=tmp_path / "x.json")

    async def test_default_output_path(self, db: UCDatabase, tmp_path: Path):
        import os
        _, run_id, _ = await _make_run_with_turns(db)
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = await export_bundle(db, run_id)
            assert result.exists()
            assert "uc-bundle" in result.name
        finally:
            os.chdir(original_cwd)

    async def test_export_v2_has_scope_metadata(self, db: UCDatabase, tmp_path: Path):
        """v2 bundles should include scope metadata."""
        _, run_id, _ = await _make_run_with_turns(db)
        out = tmp_path / "bundle.json"
        await export_bundle(db, run_id, output_path=out)

        import json
        bundle = json.loads(out.read_text())
        assert bundle["version"] == 2
        assert bundle.get("scope") is not None
        assert bundle["scope"].get("name") == "proj"

    async def test_import_v2_matches_scope_by_canonical_id(
        self, db: UCDatabase, tmp_path: Path,
    ):
        """v2 import should match existing scope by canonical_id."""
        from universal_context.db.queries import update_scope

        scope_id, run_id, _ = await _make_run_with_turns(db)
        # Set a canonical_id on the scope
        await update_scope(db, scope_id, canonical_id="github.com/user/proj")

        # Export
        out = tmp_path / "bundle.json"
        await export_bundle(db, run_id, output_path=out)

        # Import without explicit target — should match by canonical_id
        result = await import_bundle(db, out)
        assert result["scope_id"] == scope_id
        assert result["turns_imported"] == 2

    async def test_import_with_explicit_target_scope(
        self, db: UCDatabase, tmp_path: Path,
    ):
        """Import with explicit target_scope_id should use that scope."""
        from universal_context.db.queries import create_scope as cs

        scope_id, run_id, _ = await _make_run_with_turns(db)
        out = tmp_path / "bundle.json"
        await export_bundle(db, run_id, output_path=out)

        # Create a different target scope
        target = await cs(db, "target-proj", "/tmp/target")
        target_id = str(target["id"])

        result = await import_bundle(db, out, target_scope_id=target_id)
        assert result["scope_id"] == target_id

    async def test_import_v1_compat(self, db: UCDatabase, tmp_path: Path):
        """v1 bundles (no scope metadata) should create throwaway scope."""
        import json

        # Create a v1 bundle manually
        v1_bundle = {
            "version": 1,
            "exported_at": "2025-01-01T00:00:00",
            "run": {
                "id": "run:old123",
                "agent_type": "claude",
                "scope": "scope:old",
                "status": "completed",
            },
            "turns": [
                {"id": "turn:t1", "sequence": 1, "user_message": "hello"},
            ],
            "artifacts": [
                {"id": "artifact:a1", "kind": "transcript", "content": "hello world"},
            ],
        }
        out = tmp_path / "v1-bundle.json"
        out.write_text(json.dumps(v1_bundle))

        result = await import_bundle(db, out)
        assert result["turns_imported"] == 1
        # Should have created a throwaway import-xxx scope
        assert result["scope_id"].startswith("scope:")

    async def test_import_unsupported_version_raises(
        self, db: UCDatabase, tmp_path: Path,
    ):
        """Bundles with unsupported version should raise."""
        import json

        bad_bundle = {"version": 99}
        out = tmp_path / "bad.json"
        out.write_text(json.dumps(bad_bundle))

        with pytest.raises(ValueError, match="Unsupported bundle version"):
            await import_bundle(db, out)
