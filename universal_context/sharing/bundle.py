"""Share bundle export and import.

A share bundle is a JSON file containing a run with all its turns,
artifacts, and provenance edges — portable across UC instances.
Optionally encrypted with a passphrase using Fernet symmetric encryption.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..db.client import UCDatabase
from ..db.queries import get_run, list_turns


async def export_bundle(
    db: UCDatabase,
    run_id: str,
    output_path: Path | None = None,
    passphrase: str | None = None,
) -> Path:
    """Export a run as a portable v2 share bundle with scope metadata.

    Returns the path to the created bundle file.
    """
    from ..db.queries import get_scope

    run = await get_run(db, run_id)
    if not run:
        raise ValueError(f"Run not found: {run_id}")

    # Collect turns
    turns = await list_turns(db, run_id)

    # Collect artifacts for each turn
    artifacts: list[dict[str, Any]] = []
    for turn in turns:
        turn_id = str(turn["id"])
        turn_artifacts = await db.query(
            f"SELECT ->produced->artifact.* FROM {turn_id}"
        )
        if turn_artifacts:
            for ta in turn_artifacts:
                produced = ta.get("->produced", {})
                art_list = produced.get("->artifact", [])
                for art in art_list:
                    if isinstance(art, dict):
                        artifacts.append(_serialize_record(art))

    # Resolve scope metadata for cross-machine sharing
    run_scope = run.get("scope")
    scope_data = None
    if run_scope:
        scope_record = await get_scope(db, str(run_scope))
        if scope_record:
            scope_data = _serialize_record(scope_record)

    # Build v2 bundle with scope metadata
    bundle = {
        "version": 2,
        "exported_at": datetime.now().isoformat(),
        "run": _serialize_record(run),
        "turns": [_serialize_record(t) for t in turns],
        "artifacts": artifacts,
        "scope": scope_data,
    }

    payload = json.dumps(bundle, indent=2, default=str)

    if passphrase:
        payload = _encrypt(payload, passphrase)

    # Write file
    if output_path is None:
        safe_id = run_id.replace(":", "-")
        output_path = Path(f"uc-bundle-{safe_id}.json")

    output_path.write_text(payload, encoding="utf-8")
    return output_path


async def import_bundle(
    db: UCDatabase,
    bundle_path: Path,
    passphrase: str | None = None,
    target_scope_id: str | None = None,
) -> dict[str, Any]:
    """Import a share bundle into the local database.

    Smart scope resolution:
    1. If target_scope_id provided → use it (explicit --project)
    2. If bundle v2 has scope.canonical_id → look up by canonical_id
    3. Fallback → create import-{id} scope (v1 backwards compat)

    Returns summary of imported records.
    """
    payload = bundle_path.read_text(encoding="utf-8")

    if passphrase:
        payload = _decrypt(payload, passphrase)

    bundle = json.loads(payload)

    version = bundle.get("version")
    if version not in (1, 2):
        raise ValueError(f"Unsupported bundle version: {version}")

    run_data = bundle["run"]

    from ..db.queries import (
        _gen_id,
        create_run,
        create_scope,
        create_turn_with_artifact,
        find_scope_by_canonical_id,
        get_scope,
    )

    # Resolve target scope
    scope_id = None

    # 1. Explicit target
    if target_scope_id:
        scope = await get_scope(db, target_scope_id)
        if scope:
            scope_id = str(scope["id"])

    # 2. Bundle v2 scope metadata — match by canonical_id
    if scope_id is None and version == 2:
        bundle_scope = bundle.get("scope")
        if bundle_scope and bundle_scope.get("canonical_id"):
            existing = await find_scope_by_canonical_id(
                db, bundle_scope["canonical_id"],
            )
            if existing:
                scope_id = str(existing["id"])

    # 3. Fallback — create throwaway scope
    if scope_id is None:
        new_id = _gen_id()
        scope = await create_scope(db, f"import-{new_id[:6]}")
        scope_id = str(scope["id"])

    run = await create_run(
        db,
        scope_id,
        run_data.get("agent_type", "unknown"),
        branch=run_data.get("branch"),
        commit_sha=run_data.get("commit_sha"),
    )
    imported_run_id = str(run["id"])

    # Import turns + artifacts
    imported_turns = 0

    for turn_data in bundle.get("turns", []):
        raw_content = ""
        # Find matching artifact
        for art in bundle.get("artifacts", []):
            if art.get("kind") == "transcript":
                raw_content = art.get("content", "")
                break

        await create_turn_with_artifact(
            db,
            run_id=imported_run_id,
            sequence=turn_data.get("sequence", imported_turns + 1),
            user_message=turn_data.get("user_message"),
            raw_content=raw_content,
            create_summary_job=False,
        )
        imported_turns += 1

    imported_artifacts = len(bundle.get("artifacts", []))

    return {
        "run_id": imported_run_id,
        "scope_id": scope_id,
        "turns_imported": imported_turns,
        "artifacts_imported": imported_artifacts,
    }


def _serialize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a SurrealDB record to JSON-serializable dict."""
    result = {}
    for key, value in record.items():
        result[key] = str(value) if hasattr(value, "table_name") else value
    return result


def _encrypt(payload: str, passphrase: str) -> str:
    """Encrypt payload with Fernet using passphrase-derived key."""
    from cryptography.fernet import Fernet

    key = _derive_key(passphrase)
    f = Fernet(key)
    encrypted = f.encrypt(payload.encode("utf-8"))
    return base64.urlsafe_b64encode(encrypted).decode("ascii")


def _decrypt(payload: str, passphrase: str) -> str:
    """Decrypt payload with Fernet using passphrase-derived key."""
    from cryptography.fernet import Fernet

    key = _derive_key(passphrase)
    f = Fernet(key)
    encrypted = base64.urlsafe_b64decode(payload.encode("ascii"))
    return f.decrypt(encrypted).decode("utf-8")


def _derive_key(passphrase: str) -> bytes:
    """Derive a Fernet-compatible key from a passphrase."""
    # PBKDF2 with SHA256
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        b"uc-share-salt-v1",
        iterations=100_000,
        dklen=32,
    )
    return base64.urlsafe_b64encode(dk)
