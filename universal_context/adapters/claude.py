"""Claude Code session adapter.

Sessions live at ~/.claude/projects/{encoded-path}/{uuid}.jsonl.
The directory name encodes the project path: /Users/alice/MyProject
becomes -Users-alice-MyProject.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..triggers.claude_trigger import ClaudeTrigger
from .base import SessionAdapter

logger = logging.getLogger(__name__)

CLAUDE_HOME = Path.home() / ".claude" / "projects"


def _decode_project_path(dir_name: str) -> Path | None:
    """Decode a Claude project directory name to a filesystem path.

    Claude encodes /Users/alice/MyProject as -Users-alice-MyProject.
    """
    if not dir_name.startswith("-"):
        return None
    decoded = "/" + dir_name[1:].replace("-", "/")
    return Path(decoded)


def _extract_cwd_from_jsonl(session_file: Path, max_lines: int = 20) -> Path | None:
    """Fallback: read 'cwd' from the first few lines of a JSONL file."""
    try:
        with session_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    obj = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                cwd = obj.get("cwd")
                if cwd and Path(cwd).is_dir():
                    return Path(cwd)
    except OSError:
        pass
    return None


class ClaudeAdapter(SessionAdapter):
    """Session discovery for Claude Code."""

    name = "claude"
    trigger_class = ClaudeTrigger

    def discover_sessions(self) -> list[Path]:
        if not CLAUDE_HOME.is_dir():
            return []

        sessions: list[Path] = []
        for project_dir in CLAUDE_HOME.iterdir():
            if not project_dir.is_dir():
                continue
            for f in project_dir.iterdir():
                if f.suffix == ".jsonl" and not f.name.startswith("agent-"):
                    sessions.append(f)

        # Sort by modification time (newest first)
        sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return sessions

    def discover_sessions_for_project(
        self, project_path: Path
    ) -> list[Path]:
        if not CLAUDE_HOME.is_dir():
            return []

        resolved = project_path.resolve()
        sessions: list[Path] = []

        for project_dir in CLAUDE_HOME.iterdir():
            if not project_dir.is_dir():
                continue

            decoded = _decode_project_path(project_dir.name)
            if decoded and decoded.resolve() == resolved:
                for f in project_dir.iterdir():
                    if (
                        f.suffix == ".jsonl"
                        and not f.name.startswith("agent-")
                    ):
                        sessions.append(f)

        # Fallback: scan all and check cwd
        if not sessions:
            for session in self.discover_sessions():
                cwd = _extract_cwd_from_jsonl(session)
                if cwd and cwd.resolve() == resolved:
                    sessions.append(session)

        sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return sessions

    def extract_project_path(self, session_file: Path) -> Path | None:
        # Try directory name decode
        parent = session_file.parent
        if parent.parent == CLAUDE_HOME:
            decoded = _decode_project_path(parent.name)
            if decoded and decoded.is_dir():
                return decoded

        # Fallback: read from JSONL content
        return _extract_cwd_from_jsonl(session_file)
