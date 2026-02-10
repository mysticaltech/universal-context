"""Session watcher — polls adapters and captures new turns.

The watcher discovers active sessions, tracks which turns have been
ingested, and creates turn records + artifacts for new ones.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ..adapters.registry import get_registry
from ..db.client import UCDatabase
from ..db.queries import (
    create_job,
    create_run,
    create_scope,
    create_turn_with_artifact,
    end_run,
    find_scope_by_canonical_id,
    find_scope_by_path,
    list_jobs,
    list_runs,
    list_scopes,
    update_scope,
)
from ..git import get_current_branch, get_head_sha, resolve_canonical_id

logger = logging.getLogger(__name__)


class SessionState:
    """Tracks the state of a watched session."""

    def __init__(
        self,
        session_path: Path,
        adapter_name: str,
        scope_id: str,
        run_id: str,
        last_turn_count: int = 0,
    ) -> None:
        self.session_path = session_path
        self.adapter_name = adapter_name
        self.scope_id = scope_id
        self.run_id = run_id
        self.last_turn_count = last_turn_count


class Watcher:
    """Polls adapters to discover sessions and ingest new turns."""

    def __init__(
        self,
        db: UCDatabase,
        poll_interval: float = 2.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._db = db
        self._poll_interval = poll_interval
        self._config = config or {}
        self._sessions: dict[str, SessionState] = {}  # keyed by session path string
        self._running = False
        # Track turns ingested per scope since last memory update
        self._scope_turn_counts: dict[str, int] = {}

    async def run(self) -> None:
        """Run the watcher loop until cancelled."""
        self._running = True
        logger.info("Watcher started (poll_interval=%.1fs)", self._poll_interval)
        try:
            while self._running:
                await self._poll_cycle()
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            logger.info("Watcher stopping (cancelled)")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    async def _poll_cycle(self) -> None:
        """One cycle: discover sessions, check for new turns."""
        registry = get_registry()

        # Collect known scope paths for Gemini hash resolution
        scopes = await list_scopes(self._db)
        known_paths = [s.get("path", "") for s in scopes if s.get("path")]
        config_with_paths = {**self._config, "known_paths": known_paths}
        all_sessions = registry.discover_all_sessions(config_with_paths)

        seen_paths: set[str] = set()
        for session_path, adapter in all_sessions:
            path_key = str(session_path)
            seen_paths.add(path_key)

            if path_key in self._sessions:
                # Known session — check for new turns
                state = self._sessions[path_key]
                await self._check_new_turns(state, adapter)
            else:
                # New session — register it
                await self._register_session(session_path, adapter)

        # Detect ended sessions (no longer discovered)
        ended = set(self._sessions.keys()) - seen_paths
        for path_key in ended:
            await self._handle_session_end(path_key)

    async def _register_session(self, session_path: Path, adapter: Any) -> None:
        """Register a newly discovered session."""
        path_key = str(session_path)
        project_path = adapter.extract_project_path(session_path)

        # Ensure scope exists — don't create garbage scopes from session dirs
        if project_path:
            scope_id = await self._ensure_scope(project_path)
        else:
            scope_id = await self._ensure_scope(
                Path.home(), name_override="unknown"
            )

        # Capture git context for provenance
        branch = None
        commit_sha = None
        if project_path:
            branch = await asyncio.to_thread(get_current_branch, project_path)
            commit_sha = await asyncio.to_thread(get_head_sha, project_path)

        # Create a run
        run = await create_run(
            self._db,
            scope_id,
            adapter.name,
            session_path=str(session_path),
            branch=branch,
            commit_sha=commit_sha,
        )
        run_id = str(run["id"])

        state = SessionState(
            session_path=session_path,
            adapter_name=adapter.name,
            scope_id=scope_id,
            run_id=run_id,
        )
        self._sessions[path_key] = state
        logger.info(
            "Registered session: %s (%s) -> run %s",
            session_path.name,
            adapter.name,
            run_id,
        )

        # Detect runs from other branches that have been merged into current branch
        if branch and project_path:
            from ..db.queries import detect_merged_runs

            try:
                merged = await detect_merged_runs(
                    self._db, scope_id, branch, project_path,
                )
                if merged:
                    logger.info("Detected %d merged run(s) into %s", merged, branch)
            except Exception as e:
                logger.debug("Merge detection failed: %s", e)

        # Ingest any existing turns
        await self._check_new_turns(state, adapter)

    async def _check_new_turns(self, state: SessionState, adapter: Any) -> None:
        """Check for and ingest new turns in a session."""
        current_count = adapter.count_turns(state.session_path)
        if current_count <= state.last_turn_count:
            return

        new_turns_ingested = 0
        for seq in range(state.last_turn_count + 1, current_count + 1):
            info = adapter.extract_turn_info(state.session_path, seq)
            if info is None:
                continue

            raw = adapter.get_raw_transcript(state.session_path, seq)
            if raw is None:
                raw = info.raw_content or ""

            # Redact secrets if configured
            if self._config.get("redact_secrets", True):
                from ..redact import redact_secrets

                raw = redact_secrets(raw)

            await create_turn_with_artifact(
                self._db,
                run_id=state.run_id,
                sequence=seq,
                user_message=info.user_message,
                raw_content=raw,
                create_summary_job=True,
            )
            logger.debug("Ingested turn %d for run %s", seq, state.run_id)
            new_turns_ingested += 1

        state.last_turn_count = current_count

        # Schedule working memory update if threshold reached
        if new_turns_ingested > 0 and self._config.get("memory_enabled", True):
            threshold = self._config.get("memory_update_threshold", 5)
            scope_id = state.scope_id
            self._scope_turn_counts[scope_id] = (
                self._scope_turn_counts.get(scope_id, 0) + new_turns_ingested
            )
            if self._scope_turn_counts[scope_id] >= threshold:
                await self._maybe_schedule_memory_update(scope_id)
                self._scope_turn_counts[scope_id] = 0

    async def _handle_session_end(self, path_key: str) -> None:
        """Handle a session that's no longer active."""
        state = self._sessions.pop(path_key, None)
        if state is None:
            return
        await end_run(self._db, state.run_id, "completed")
        logger.info("Session ended: %s -> run %s", state.session_path.name, state.run_id)

    async def _ensure_scope(
        self, path: Path, name_override: str | None = None,
    ) -> str:
        """Get or create a scope for the given path.

        Uses git-aware canonical identity: same git remote = same scope,
        even across worktrees or different checkout paths.
        """
        path_str = str(path.resolve())
        canonical_id = await asyncio.to_thread(resolve_canonical_id, path)

        # 1. Primary: look up by canonical_id
        existing = await find_scope_by_canonical_id(self._db, canonical_id)
        if existing:
            return str(existing["id"])

        # 2. Backwards compat: look up by path (pre-migration scopes without canonical_id)
        existing = await find_scope_by_path(self._db, path_str)
        if existing:
            if not existing.get("canonical_id"):
                await update_scope(
                    self._db, str(existing["id"]), canonical_id=canonical_id,
                )
            return str(existing["id"])

        # 3. Create new scope with canonical_id
        name = name_override or path.name
        scope = await create_scope(
            self._db, name, path_str, canonical_id=canonical_id,
        )
        return str(scope["id"])

    async def _maybe_schedule_memory_update(self, scope_id: str) -> None:
        """Schedule a memory_update job if one isn't already pending/running."""
        existing = await list_jobs(
            self._db, status="pending", job_type="memory_update",
        )
        for job in existing:
            if job.get("target") == scope_id:
                logger.debug("Memory update already pending for %s", scope_id)
                return

        running = await list_jobs(
            self._db, status="running", job_type="memory_update",
        )
        for job in running:
            if job.get("target") == scope_id:
                logger.debug("Memory update already running for %s", scope_id)
                return

        await create_job(self._db, "memory_update", scope_id, priority=1)
        logger.info("Scheduled memory_update job for scope %s", scope_id)

    async def recover_interrupted_runs(self) -> None:
        """On startup, mark any active runs as crashed."""
        active_runs = await list_runs(self._db, status="active")
        for run in active_runs:
            run_id = str(run["id"])
            await end_run(self._db, run_id, "crashed")
            logger.warning("Marked interrupted run as crashed: %s", run_id)
