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
    """Export a run as a portable share bundle.

    Returns the path to the created bundle file.
    """
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

    # Build bundle
    bundle = {
        "version": 1,
        "exported_at": datetime.now().isoformat(),
        "run": _serialize_record(run),
        "turns": [_serialize_record(t) for t in turns],
        "artifacts": artifacts,
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
) -> dict[str, Any]:
    """Import a share bundle into the local database.

    Returns summary of imported records.
    """
    payload = bundle_path.read_text(encoding="utf-8")

    if passphrase:
        payload = _decrypt(payload, passphrase)

    bundle = json.loads(payload)

    if bundle.get("version") != 1:
        raise ValueError(f"Unsupported bundle version: {bundle.get('version')}")

    # Import run
    run_data = bundle["run"]
    from ..db.queries import _gen_id

    new_run_id = _gen_id()
    run_data.get("scope", "imported")

    # Create scope for imported run
    from ..db.queries import create_scope

    scope = await create_scope(db, f"import-{new_run_id[:6]}")
    scope_id = str(scope["id"])

    from ..db.queries import create_run

    run = await create_run(
        db,
        scope_id,
        run_data.get("agent_type", "unknown"),
    )
    imported_run_id = str(run["id"])

    # Import turns + artifacts
    imported_turns = 0
    imported_artifacts = 0

    for turn_data in bundle.get("turns", []):
        from ..db.queries import create_turn_with_artifact

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
