"""Codex CLI session adapter.

Sessions live at ~/.codex/sessions/{YYYY}/{MM}/{DD}/rollout-*.jsonl.
Discovery is date-bounded (default: last 2 days) to avoid scanning
the full history.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..triggers.codex_trigger import CodexTrigger
from .base import SessionAdapter

logger = logging.getLogger(__name__)

CODEX_HOME = Path.home() / ".codex" / "sessions"
DEFAULT_DISCOVERY_DAYS = 365  # Overridable via config.codex_discovery_days


class CodexAdapter(SessionAdapter):
    """Session discovery for Codex CLI."""

    name = "codex"
    trigger_class = CodexTrigger

    @property
    def _discovery_days(self) -> int:
        return int(self.config.get("codex_discovery_days", DEFAULT_DISCOVERY_DAYS))

    def _session_roots(self) -> list[Path]:
        """Return all possible session root directories."""
        roots = [CODEX_HOME]
        # Could add aline-managed roots here in the future
        return [r for r in roots if r.is_dir()]

    def discover_sessions(self) -> list[Path]:
        sessions: list[Path] = []
        now = datetime.now(tz=UTC)

        for root in self._session_roots():
            for days_ago in range(self._discovery_days + 1):
                dt = now - timedelta(days=days_ago)
                date_path = (
                    root / str(dt.year) / f"{dt.month:02d}" / f"{dt.day:02d}"
                )
                if date_path.is_dir():
                    sessions.extend(date_path.glob("rollout-*.jsonl"))

        sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return sessions

    def discover_sessions_for_project(
        self, project_path: Path
    ) -> list[Path]:
        resolved = project_path.resolve()
        matching: list[Path] = []
        for session in self.discover_sessions():
            cwd = self._extract_cwd(session)
            if cwd and cwd.resolve() == resolved:
                matching.append(session)
        return matching

    def extract_project_path(self, session_file: Path) -> Path | None:
        return self._extract_cwd(session_file)

    @staticmethod
    def _extract_cwd(session_file: Path) -> Path | None:
        """Read project path from session_meta event."""
        try:
            with session_file.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "session_meta":
                        payload = data.get("payload", {})
                        cwd = payload.get("cwd")
                        if cwd:
                            return Path(cwd)
        except OSError:
            pass
        return None
