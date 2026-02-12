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

    turns = await list_turns(db, run_id)

    artifacts_by_id: dict[str, dict[str, Any]] = {}
    turn_transcripts: dict[str, dict[str, str]] = {}
    produced: list[dict[str, str]] = []
    seen_produced: set[tuple[str, str]] = set()

    for turn in turns:
        turn_id = str(turn["id"])
        sequence = turn.get("sequence")
        produced_edges = await db.query(
            f"SELECT in, out FROM produced WHERE in = {turn_id}"
        )
        for edge in produced_edges if isinstance(produced_edges, list) else []:
            artifact_ref = edge.get("out") if isinstance(edge, dict) else None
            if not artifact_ref:
                continue
            artifact_id = str(artifact_ref)
            key = (turn_id, artifact_id)
            if key not in seen_produced:
                seen_produced.add(key)
                produced.append({"in": turn_id, "out": artifact_id})

            details = await db.query(f"SELECT * FROM {artifact_id}")
            if not details:
                continue
            art = details[0]
            serialized = _serialize_record(art)
            artifacts_by_id[artifact_id] = serialized
            if art.get("kind") == "transcript":
                info = {
                    "artifact_id": artifact_id,
                    "content": str(art.get("content", "") or ""),
                }
                if sequence is not None:
                    turn_transcripts[str(sequence)] = info
                turn_transcripts[turn_id] = info

    # Include artifacts that depend on turn-produced artifacts (e.g. summaries)
    queue = list(artifacts_by_id.keys())
    while queue:
        source_id = queue.pop(0)
        dependents = await db.query(f"SELECT in FROM depends_on WHERE out = {source_id}")
        for dep in dependents if isinstance(dependents, list) else []:
            dep_ref = dep.get("in") if isinstance(dep, dict) else None
            if not dep_ref:
                continue
            dep_id = str(dep_ref)
            if dep_id in artifacts_by_id:
                continue
            details = await db.query(f"SELECT * FROM {dep_id}")
            if not details:
                continue
            artifacts_by_id[dep_id] = _serialize_record(details[0])
            queue.append(dep_id)

    artifact_ids = set(artifacts_by_id.keys())
    depends_on: list[dict[str, Any]] = []
    seen_depends_on: set[tuple[str, str, str]] = set()
    for in_id in artifact_ids:
        edges = await db.query(
            f"SELECT in, out, relationship FROM depends_on WHERE in = {in_id}"
        )
        for edge in edges if isinstance(edges, list) else []:
            edge_in = str(edge.get("in")) if isinstance(edge, dict) and edge.get("in") else ""
            edge_out = str(edge.get("out")) if isinstance(edge, dict) and edge.get("out") else ""
            if not edge_in or not edge_out or edge_out not in artifact_ids:
                continue
            rel = (
                str(edge.get("relationship", "derived_from"))
                if isinstance(edge, dict)
                else "derived_from"
            )
            key = (edge_in, edge_out, rel)
            if key in seen_depends_on:
                continue
            seen_depends_on.add(key)
            depends_on.append({"in": edge_in, "out": edge_out, "relationship": rel})

    checkpoints_raw = await db.query(
        f"SELECT * FROM checkpoint WHERE run = {run_id} ORDER BY created_at ASC"
    )
    checkpoints = [
        _serialize_record(cp) for cp in checkpoints_raw if isinstance(cp, dict)
    ]

    checkpoint_at: list[dict[str, str]] = []
    seen_checkpoint_at: set[tuple[str, str]] = set()
    for cp in checkpoints:
        cp_id = str(cp.get("id", ""))
        if not cp_id:
            continue
        edges = await db.query(f"SELECT in, out FROM checkpoint_at WHERE out = {cp_id}")
        for edge in edges if isinstance(edges, list) else []:
            edge_in = str(edge.get("in")) if isinstance(edge, dict) and edge.get("in") else ""
            edge_out = str(edge.get("out")) if isinstance(edge, dict) and edge.get("out") else ""
            if not edge_in or not edge_out:
                continue
            key = (edge_in, edge_out)
            if key in seen_checkpoint_at:
                continue
            seen_checkpoint_at.add(key)
            checkpoint_at.append({"in": edge_in, "out": edge_out})

    # Resolve scope metadata for cross-machine sharing
    run_scope = run.get("scope")
    scope_data = None
    if run_scope:
        scope_record = await get_scope(db, str(run_scope))
        if scope_record:
            scope_data = _serialize_record(scope_record)

    bundle = {
        "version": 2,
        "exported_at": datetime.now().isoformat(),
        "run": _serialize_record(run),
        "turns": [_serialize_record(t) for t in turns],
        "artifacts": list(artifacts_by_id.values()),
        "produced": produced,
        "depends_on": depends_on,
        "turn_transcripts": turn_transcripts,
        "checkpoints": checkpoints,
        "checkpoint_at": checkpoint_at,
        "scope": scope_data,
    }

    payload = json.dumps(bundle, indent=2, default=str)

    if passphrase:
        payload = _encrypt(payload, passphrase)

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

    Supports v2 bundles (current) and v1 bundles (legacy).
    """
    payload = bundle_path.read_text(encoding="utf-8")

    if passphrase:
        payload = _decrypt(payload, passphrase)

    bundle = json.loads(payload)

    version = bundle.get("version")
    if version not in (1, 2):
        raise ValueError(f"Unsupported bundle version: {version}")

    run_data = bundle.get("run")
    if not isinstance(run_data, dict):
        raise ValueError("Invalid bundle: missing run record")

    from ..db.queries import (
        _gen_id,
        create_artifact,
        create_run,
        create_scope,
        create_turn_with_artifact,
        find_scope_by_canonical_id,
        get_scope,
    )

    scope_id = None
    if target_scope_id:
        scope = await get_scope(db, target_scope_id)
        if scope:
            scope_id = str(scope["id"])

    bundle_scope = bundle.get("scope") if isinstance(bundle.get("scope"), dict) else None
    canonical_id = (
        str(bundle_scope.get("canonical_id"))
        if bundle_scope and bundle_scope.get("canonical_id")
        else None
    )
    if scope_id is None and canonical_id:
        existing = await find_scope_by_canonical_id(db, canonical_id)
        if existing:
            scope_id = str(existing["id"])

    if scope_id is None:
        scope_name = (
            str(bundle_scope.get("name"))
            if bundle_scope and bundle_scope.get("name")
            else f"import-{_gen_id()[:6]}"
        )
        scope_path = (
            str(bundle_scope.get("path"))
            if bundle_scope and bundle_scope.get("path")
            else None
        )
        scope = await create_scope(
            db,
            scope_name,
            scope_path,
            canonical_id=canonical_id,
        )
        scope_id = str(scope["id"])

    run = await create_run(
        db,
        scope_id,
        str(run_data.get("agent_type", "unknown")),
        session_path=run_data.get("session_path"),
        branch=run_data.get("branch"),
        commit_sha=run_data.get("commit_sha"),
    )
    imported_run_id = str(run["id"])
    await _update_run_record(db, imported_run_id, run_data)

    turns = [t for t in bundle.get("turns", []) if isinstance(t, dict)]
    artifacts = [a for a in bundle.get("artifacts", []) if isinstance(a, dict)]
    artifacts_by_old_id: dict[str, dict[str, Any]] = {}
    for art in artifacts:
        old_id = str(art.get("id", ""))
        if old_id:
            artifacts_by_old_id[old_id] = art

    turn_transcripts = bundle.get("turn_transcripts")
    if not isinstance(turn_transcripts, dict):
        turn_transcripts = _derive_turn_transcripts(turns, artifacts_by_old_id)

    imported_turns = 0
    imported_artifacts = 0
    imported_checkpoints = 0
    turn_id_map: dict[str, str] = {}
    artifact_id_map: dict[str, str] = {}
    old_run_id = str(run_data.get("id", ""))

    for turn_data in turns:
        sequence = _coerce_sequence(turn_data.get("sequence"), imported_turns + 1)
        transcript_info = _resolve_transcript_info(turn_data, sequence, turn_transcripts)
        raw_content = (
            str(transcript_info.get("content", ""))
            if isinstance(transcript_info, dict)
            else ""
        )

        created = await create_turn_with_artifact(
            db,
            run_id=imported_run_id,
            sequence=sequence,
            user_message=turn_data.get("user_message"),
            raw_content=raw_content,
            create_summary_job=False,
        )
        new_turn_id = created["turn_id"]

        old_turn_id = str(turn_data.get("id", ""))
        if old_turn_id:
            turn_id_map[old_turn_id] = new_turn_id
        await _update_turn_record(db, new_turn_id, turn_data)

        transcript_old_id = ""
        if isinstance(transcript_info, dict) and transcript_info.get("artifact_id"):
            transcript_old_id = str(transcript_info["artifact_id"])
        if transcript_old_id:
            artifact_id_map[transcript_old_id] = created["artifact_id"]
            transcript_record = artifacts_by_old_id.get(transcript_old_id)
            if transcript_record:
                await _update_artifact_record(
                    db,
                    created["artifact_id"],
                    transcript_record,
                )

        imported_turns += 1
        imported_artifacts += 1

    # Import remaining artifacts (including unmapped transcripts from legacy bundles).
    for art in artifacts:
        old_id = str(art.get("id", ""))
        if not old_id or old_id in artifact_id_map:
            continue

        new_id = await create_artifact(
            db,
            kind=str(art.get("kind", "note")),
            content=art.get("content"),
            blob_path=art.get("blob_path"),
            metadata=art.get("metadata") if isinstance(art.get("metadata"), dict) else {},
            scope_id=scope_id,
        )
        artifact_id_map[old_id] = new_id
        await _update_artifact_record(db, new_id, art)
        imported_artifacts += 1

    # Restore produced edges for non-transcript artifacts when exported.
    produced_edges = bundle.get("produced")
    if isinstance(produced_edges, list):
        for edge in produced_edges:
            if not isinstance(edge, dict):
                continue
            old_turn_ref = str(edge.get("in", ""))
            old_art_ref = str(edge.get("out", ""))
            new_turn_ref = turn_id_map.get(old_turn_ref)
            new_art_ref = artifact_id_map.get(old_art_ref)
            if not new_turn_ref or not new_art_ref:
                continue
            await _ensure_edge(db, "produced", new_turn_ref, new_art_ref)

    depends_on_edges = bundle.get("depends_on")
    if isinstance(depends_on_edges, list):
        for edge in depends_on_edges:
            if not isinstance(edge, dict):
                continue
            new_in = artifact_id_map.get(str(edge.get("in", "")))
            new_out = artifact_id_map.get(str(edge.get("out", "")))
            if not new_in or not new_out:
                continue
            relationship = str(edge.get("relationship", "derived_from"))
            existing = await db.query(
                f"SELECT id FROM depends_on WHERE in = {new_in} AND out = {new_out} LIMIT 1"
            )
            if not existing:
                await db.query(
                    f"RELATE {new_in}->depends_on->{new_out} SET relationship = $rel",
                    {"rel": relationship},
                )

    checkpoint_id_map: dict[str, str] = {}
    checkpoints = bundle.get("checkpoints")
    if isinstance(checkpoints, list):
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            old_cp_id = str(checkpoint.get("id", ""))
            old_turn_ref = str(checkpoint.get("turn", ""))
            new_turn_ref = turn_id_map.get(old_turn_ref)
            if not new_turn_ref:
                continue
            cid = _gen_id()
            new_cp_id = f"checkpoint:{cid}"
            state = checkpoint.get("state")
            state_dict = state if isinstance(state, dict) else {}
            remapped_state = _remap_checkpoint_state(
                state_dict,
                old_run_id=old_run_id,
                new_run_id=imported_run_id,
                turn_id_map=turn_id_map,
            )
            await db.query(
                f"CREATE checkpoint:{cid} SET run = {imported_run_id}, turn = {new_turn_ref}, "
                "label = $label, state = $state",
                {"label": checkpoint.get("label"), "state": remapped_state},
            )
            created_at = _parse_datetime(checkpoint.get("created_at"))
            if created_at is not None:
                await db.query(
                    f"UPDATE checkpoint:{cid} SET created_at = $created_at",
                    {"created_at": created_at},
                )
            if old_cp_id:
                checkpoint_id_map[old_cp_id] = new_cp_id
            imported_checkpoints += 1

    restored_checkpoint_edge = False
    checkpoint_edges = bundle.get("checkpoint_at")
    if isinstance(checkpoint_edges, list):
        for edge in checkpoint_edges:
            if not isinstance(edge, dict):
                continue
            old_out = str(edge.get("out", ""))
            new_out = checkpoint_id_map.get(old_out)
            if not new_out:
                continue
            new_in = _remap_record_id(
                old_id=str(edge.get("in", "")),
                old_run_id=old_run_id,
                new_run_id=imported_run_id,
                turn_id_map=turn_id_map,
                checkpoint_id_map=checkpoint_id_map,
            )
            if not new_in:
                continue
            await _ensure_edge(db, "checkpoint_at", new_in, new_out)
            restored_checkpoint_edge = True

    # Ensure checkpoints remain reachable from the run even if edge data is absent.
    if imported_checkpoints > 0 and not restored_checkpoint_edge:
        for new_cp_id in checkpoint_id_map.values():
            await _ensure_edge(db, "checkpoint_at", imported_run_id, new_cp_id)

    return {
        "run_id": imported_run_id,
        "scope_id": scope_id,
        "turns_imported": imported_turns,
        "artifacts_imported": imported_artifacts,
        "checkpoints_imported": imported_checkpoints,
    }


async def _update_run_record(db: UCDatabase, run_id: str, run_data: dict[str, Any]) -> None:
    set_parts: list[str] = []
    params: dict[str, Any] = {}

    for field in ("status", "session_path", "merged_to"):
        if field in run_data:
            set_parts.append(f"{field} = ${field}")
            params[field] = run_data.get(field)

    if isinstance(run_data.get("metadata"), dict):
        set_parts.append("metadata = $metadata")
        params["metadata"] = run_data["metadata"]

    started_at = _parse_datetime(run_data.get("started_at"))
    if started_at is not None:
        set_parts.append("started_at = $started_at")
        params["started_at"] = started_at

    ended_at = _parse_datetime(run_data.get("ended_at"))
    if ended_at is not None:
        set_parts.append("ended_at = $ended_at")
        params["ended_at"] = ended_at

    if set_parts:
        await db.query(f"UPDATE {run_id} SET {', '.join(set_parts)}", params)


async def _update_turn_record(db: UCDatabase, turn_id: str, turn_data: dict[str, Any]) -> None:
    set_parts: list[str] = []
    params: dict[str, Any] = {}

    if "user_message" in turn_data:
        set_parts.append("user_message = $user_message")
        params["user_message"] = turn_data.get("user_message")

    if isinstance(turn_data.get("metadata"), dict):
        set_parts.append("metadata = $metadata")
        params["metadata"] = turn_data["metadata"]

    started_at = _parse_datetime(turn_data.get("started_at"))
    if started_at is not None:
        set_parts.append("started_at = $started_at")
        params["started_at"] = started_at

    ended_at = _parse_datetime(turn_data.get("ended_at"))
    if ended_at is not None:
        set_parts.append("ended_at = $ended_at")
        params["ended_at"] = ended_at

    if set_parts:
        await db.query(f"UPDATE {turn_id} SET {', '.join(set_parts)}", params)


async def _update_artifact_record(
    db: UCDatabase,
    artifact_id: str,
    artifact_data: dict[str, Any],
) -> None:
    set_parts: list[str] = []
    params: dict[str, Any] = {}

    for field in ("kind", "content", "content_hash", "blob_path"):
        if field in artifact_data:
            set_parts.append(f"{field} = ${field}")
            params[field] = artifact_data.get(field)

    if isinstance(artifact_data.get("metadata"), dict):
        set_parts.append("metadata = $metadata")
        params["metadata"] = artifact_data["metadata"]

    embedding = artifact_data.get("embedding")
    if isinstance(embedding, list):
        set_parts.append("embedding = $embedding")
        params["embedding"] = embedding

    created_at = _parse_datetime(artifact_data.get("created_at"))
    if created_at is not None:
        set_parts.append("created_at = $created_at")
        params["created_at"] = created_at

    if set_parts:
        await db.query(f"UPDATE {artifact_id} SET {', '.join(set_parts)}", params)


async def _ensure_edge(
    db: UCDatabase,
    table: str,
    edge_in: str,
    edge_out: str,
) -> None:
    existing = await db.query(
        f"SELECT id FROM {table} WHERE in = {edge_in} AND out = {edge_out} LIMIT 1"
    )
    if not existing:
        await db.query(f"RELATE {edge_in}->{table}->{edge_out}")


def _derive_turn_transcripts(
    turns: list[dict[str, Any]],
    artifacts_by_old_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    transcripts = [
        art for art in artifacts_by_old_id.values() if art.get("kind") == "transcript"
    ]
    transcripts.sort(
        key=lambda art: (
            str(art.get("created_at", "")),
            str(art.get("id", "")),
        )
    )

    mapping: dict[str, dict[str, str]] = {}
    if not transcripts:
        return mapping

    for idx, turn in enumerate(turns):
        transcript = transcripts[idx] if idx < len(transcripts) else transcripts[0]
        info = {
            "artifact_id": str(transcript.get("id", "")),
            "content": str(transcript.get("content", "") or ""),
        }
        sequence = _coerce_sequence(turn.get("sequence"), idx + 1)
        mapping[str(sequence)] = info
        turn_id = str(turn.get("id", ""))
        if turn_id:
            mapping[turn_id] = info

    return mapping


def _resolve_transcript_info(
    turn_data: dict[str, Any],
    sequence: int,
    turn_transcripts: dict[str, Any],
) -> dict[str, Any]:
    turn_id = str(turn_data.get("id", ""))
    by_turn = turn_transcripts.get(turn_id)
    if isinstance(by_turn, dict):
        return by_turn
    by_sequence = turn_transcripts.get(str(sequence))
    if isinstance(by_sequence, dict):
        return by_sequence
    return {}


def _coerce_sequence(value: Any, fallback: int) -> int:
    if isinstance(value, int):
        return value
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _remap_checkpoint_state(
    state: dict[str, Any],
    old_run_id: str,
    new_run_id: str,
    turn_id_map: dict[str, str],
) -> dict[str, Any]:
    remapped = dict(state)
    run_id = remapped.get("run_id")
    if isinstance(run_id, str) and old_run_id and run_id == old_run_id:
        remapped["run_id"] = new_run_id
    turn_id = remapped.get("turn_id")
    if isinstance(turn_id, str):
        remapped["turn_id"] = turn_id_map.get(turn_id, turn_id)
    return remapped


def _remap_record_id(
    old_id: str,
    old_run_id: str,
    new_run_id: str,
    turn_id_map: dict[str, str],
    checkpoint_id_map: dict[str, str],
) -> str | None:
    if not old_id:
        return None
    if old_run_id and old_id == old_run_id:
        return new_run_id
    if old_id in turn_id_map:
        return turn_id_map[old_id]
    if old_id in checkpoint_id_map:
        return checkpoint_id_map[old_id]
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _serialize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a SurrealDB record to JSON-serializable dict."""
    return {key: _serialize_value(value) for key, value in record.items()}


def _serialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "table_name"):
        return str(value)
    return value


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
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        b"uc-share-salt-v1",
        iterations=100_000,
        dklen=32,
    )
    return base64.urlsafe_b64encode(dk)
