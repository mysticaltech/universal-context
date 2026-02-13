"""Derived index rebuild orchestration.

Replays raw-session captures into a fresh or incremental SurrealDB graph and
rehydrates query-friendly projection records.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapters.registry import get_registry
from .config import UCConfig
from .daemon.processors.summarizer import TurnSummarizer
from .db.client import UCDatabase
from .db.queries import (
    clear_derived_data,
    complete_job,
    count_turns,
    create_run,
    create_scope,
    create_turn_with_artifact,
    fail_job,
    find_runs_by_session_path,
    find_scope_by_canonical_id,
    find_scope_by_path,
    list_scopes,
    store_embedding,
    update_scope,
    upsert_working_memory,
)
from .db.schema import rebuild_hnsw_index
from .embed import create_embed_provider
from .git import get_current_branch, get_head_sha, resolve_canonical_id
from .llm import create_llm_fn
from .memory_repo import (
    MEMORY_SECTIONS,
    list_scope_registry,
    list_scope_sections,
    render_section_text,
)
from .redact import redact_secrets

logger = logging.getLogger(__name__)


def _repo_text_from_sections(sections: dict[str, list[dict[str, Any]]]) -> str:
    """Render canonical section files into a memory-project markdown blob."""
    blocks: list[str] = []
    for section in MEMORY_SECTIONS:
        entries = sections.get(section, [])
        body = render_section_text(entries)
        if not body:
            continue
        blocks.append(f"# {section.replace('_', ' ').title()}\n\n{body}")
    return "\n\n".join(blocks)


def _scope_display_name(scope: dict[str, Any]) -> str:
    return str(scope.get("name") or "project")


async def _ensure_registry_scope_rows(
    db: UCDatabase,
) -> int:
    """Ensure every durable memory registry scope has a DB scope row."""
    created = 0
    for record in list_scope_registry():
        canonical_id = str(record.canonical_id or "").strip()
        if not canonical_id:
            continue
        existing = await find_scope_by_canonical_id(db, canonical_id)
        if existing:
            if not existing.get("path") and record.path:
                await update_scope(db, str(existing["id"]), path=record.path)
            continue

        display_name = record.scope_name or record.display_name or "project"
        scope_path = record.path or None
        await create_scope(
            db,
            display_name,
            path=scope_path,
            canonical_id=canonical_id,
        )
        created += 1
    return created


async def _resolve_scope_for_session(
    db: UCDatabase,
    project_path: Path,
    display_name: str | None = None,
) -> str:
    """Get or create a canonical scope row for a session path."""
    canonical_id = await asyncio.to_thread(resolve_canonical_id, project_path)
    if not canonical_id:
        # Should not happen, but keep existing behavior deterministic.
        canonical_id = f"path://{project_path}"

    existing = await find_scope_by_canonical_id(db, canonical_id)
    if existing:
        current = str(existing["id"])
        if existing.get("path") != str(project_path):
            await update_scope(db, current, path=str(project_path))
        return current

    existing_by_path = await find_scope_by_path(db, str(project_path))
    if existing_by_path:
        existing_id = str(existing_by_path["id"])
        if not existing_by_path.get("canonical_id"):
            await update_scope(db, existing_id, canonical_id=canonical_id)
        return existing_id

    resolved_name = display_name or project_path.name
    scope = await create_scope(db, resolved_name, str(project_path), canonical_id=canonical_id)
    return str(scope["id"])


async def _replay_session(
    db: UCDatabase,
    config: UCConfig,
    session_path: Path,
    adapter: Any,
    *,
    skip_before_ts: float | None = None,
) -> tuple[int, list[str], int]:
    """Replay one adapter session file into DB.

    Returns:
        (runs_created, turn_ids, turns_replayed)
    """
    if skip_before_ts is not None:
        try:
            if session_path.stat().st_mtime < skip_before_ts:
                return 0, [], 0
        except OSError:
            return 0, [], 0

    turn_count = adapter.count_turns(session_path)
    if turn_count <= 0:
        return 0, [], 0

    project_path = adapter.extract_project_path(session_path) or Path.home()
    scope_id = await _resolve_scope_for_session(
        db,
        project_path=project_path,
        display_name=project_path.name,
    )

    existing_runs = await find_runs_by_session_path(db, str(session_path))
    last_turn_count = 0
    reusable_run_id: str | None = None
    for run in existing_runs:
        rid = str(run["id"])
        seen = await count_turns(db, rid)
        if seen > last_turn_count:
            last_turn_count = seen
        if run.get("status") != "crashed" and reusable_run_id is None:
            reusable_run_id = rid

    branch = None
    commit_sha = None
    if project_path:
        branch = await asyncio.to_thread(get_current_branch, project_path)
        commit_sha = await asyncio.to_thread(get_head_sha, project_path)

    run_id = reusable_run_id
    runs_created = 0
    if run_id is None:
        run = await create_run(
            db,
            scope_id,
            adapter.name,
            session_path=str(session_path),
            branch=branch,
            commit_sha=commit_sha,
        )
        run_id = str(run["id"])
        runs_created = 1

    if not run_id:
        return 0, [], 0

    new_turn_ids: list[str] = []
    turns_replayed = 0
    for seq in range(last_turn_count + 1, turn_count + 1):
        info = adapter.extract_turn_info(session_path, seq)
        if info is None:
            continue

        raw = adapter.get_raw_transcript(session_path, seq)
        if raw is None:
            raw = info.raw_content or ""

        if config.redact_secrets:
            raw = redact_secrets(raw)

        artifact = await create_turn_with_artifact(
            db,
            run_id=run_id,
            sequence=seq,
            user_message=info.user_message,
            raw_content=raw,
            create_summary_job=True,
        )
        turn_id = artifact.get("turn_id")
        if turn_id:
            new_turn_ids.append(str(turn_id))
        turns_replayed += 1

    return runs_created, new_turn_ids, turns_replayed


def _adapter_supports_rebuild(adapter: Any) -> bool:
    required_methods = (
        "count_turns",
        "extract_project_path",
        "extract_turn_info",
        "get_raw_transcript",
    )
    return all(callable(getattr(adapter, name, None)) for name in required_methods)


async def _process_summary_jobs(
    db: UCDatabase,
    turn_ids: list[str],
    llm_fn: Any | None,
    embed_fn: Any | None,
    use_cache: bool,
    use_llm_on_cache_miss: bool,
) -> tuple[int, int]:
    """Process queued turn_summary jobs for the provided turn IDs."""
    if not turn_ids:
        return 0, 0

    # Drain only turn_summary jobs for turns generated in this rebuild run.
    jobs = await db.query(
        'SELECT * FROM job WHERE status = "pending" '
        'AND job_type = "turn_summary" AND target IN $targets '
        "ORDER BY created_at ASC",
        {"targets": turn_ids},
    )
    if not isinstance(jobs, list):
        return 0, 0

    target_set = set(turn_ids)
    processor = TurnSummarizer(
        llm_fn=llm_fn,
        embed_fn=embed_fn,
        use_cache=use_cache,
        use_llm_on_cache_miss=use_llm_on_cache_miss,
    )

    processed = 0
    failed = 0
    for job in jobs:
        target = str(job.get("target", ""))
        if target and target not in target_set:
            continue

        job_id = str(job.get("id", ""))
        if not job_id:
            continue
        try:
            result = await processor.process(db, job)
            await complete_job(db, job_id, result)
            processed += 1
        except Exception as exc:
            await fail_job(db, job_id, str(exc))
            failed += 1

    return processed, failed


def _normalize_since(since: str | None) -> datetime | None:
    """Parse ISO8601-like date/time and normalize timezone."""
    if not since:
        return None

    value = since.strip()
    if not value:
        return None

    normalized_input = value.replace(" ", "T")
    candidates = [normalized_input]
    if "T" not in normalized_input:
        candidates.append(f"{value}T00:00:00")
    if normalized_input.endswith("Z"):
        candidates.append(normalized_input[:-1] + "+00:00")

    parsed = None
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            break
        except ValueError:
            continue

    if parsed is None:
        raise ValueError(
            "Invalid --since value; expected ISO8601 like "
            "'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS[Z]'."
        )

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


async def rebuild_derived_db(
    db: UCDatabase,
    config: UCConfig,
    *,
    since: str | None = None,
    llm_missing: bool = False,
    no_cache: bool = False,
) -> dict[str, Any]:
    """Rebuild derived index from raw session files and memory repo."""
    since_dt = _normalize_since(since)
    skip_before_ts = since_dt.timestamp() if since_dt else None

    if since is None:
        cleared = await clear_derived_data(db, include_scopes=True)
    else:
        cleared = {"mode": "incremental"}

    embed_provider = await create_embed_provider(config)
    llm_fn: Any | None = await create_llm_fn(config) if llm_missing else None

    registry = get_registry()
    sessions = registry.discover_all_sessions(asdict(config))

    runs_created = 0
    turns_replayed = 0
    turn_ids: list[str] = []
    skipped_sessions_unsupported = 0

    for session_path, adapter in sessions:
        if not _adapter_supports_rebuild(adapter):
            skipped_sessions_unsupported += 1
            logger.warning(
                "Skipping session %s for adapter %s: missing rebuild methods",
                session_path,
                getattr(adapter, "name", type(adapter).__name__),
            )
            continue
        r_created, touched_turns, t_replayed = await _replay_session(
            db,
            config,
            session_path,
            adapter,
            skip_before_ts=skip_before_ts,
        )
        runs_created += r_created
        turns_replayed += t_replayed
        turn_ids.extend(touched_turns)

    summary_processed, summary_failed = await _process_summary_jobs(
        db,
        turn_ids,
        llm_fn=llm_fn,
        embed_fn=embed_provider,
        use_cache=not no_cache,
        use_llm_on_cache_miss=llm_missing,
    )

    scopes_from_registry = await _ensure_registry_scope_rows(db)

    # Rehydrate working memory projections from canonical files.
    scopes = await list_scopes(db)
    rehydrated = 0
    for scope in scopes:
        scope_id = str(scope["id"])
        canonical_id = scope.get("canonical_id") or scope.get("path")
        if not canonical_id:
            continue
        scope_path = scope.get("path")
        sections = list_scope_sections(
            canonical_id=canonical_id,
            display_name=scope.get("name", "project"),
            scope_path=scope_path,
        )
        content = _repo_text_from_sections(sections)
        if not content:
            continue

        wm_id = await upsert_working_memory(db, scope_id, content, method="memory_repo")
        if embed_provider is not None:
            try:
                embedding = await embed_provider.embed_document(content)
                await store_embedding(db, wm_id, embedding)
            except Exception:
                pass
        rehydrated += 1

    hnsw_rebuilt = False
    if embed_provider is not None:
        hnsw_rebuilt = await rebuild_hnsw_index(
            db,
            embedding_dim=embed_provider.dim,
        )

    return {
        "mode": "full" if since is None else "incremental",
        "since": since,
        "cleared": cleared,
        "runs_created": runs_created,
        "turns_replayed": turns_replayed,
        "skipped_sessions_unsupported": skipped_sessions_unsupported,
        "summary_jobs_processed": summary_processed,
        "summary_jobs_failed": summary_failed,
        "scopes_from_registry": scopes_from_registry,
        "working_memory_rehydrated": rehydrated,
        "hnsw_rebuilt": hnsw_rebuilt,
    }
