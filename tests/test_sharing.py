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
