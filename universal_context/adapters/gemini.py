"""Gemini CLI session adapter.

Sessions live at ~/.gemini/tmp/{project_hash}/chats/session-*.json.
The {project_hash} is SHA256(absolute_project_path). We reverse it by
hashing candidate directories and comparing.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from ..triggers.gemini_trigger import GeminiTrigger
from .base import SessionAdapter

logger = logging.getLogger(__name__)

GEMINI_HOME = Path.home() / ".gemini" / "tmp"

# Module-level cache: sha256 hash -> resolved Path
_hash_cache: dict[str, Path] = {}

# Common parent directories to scan for project candidates
_SCAN_DIRS: list[Path] = [
    Path.home() / "Code",
    Path.home() / "code",
    Path.home() / "Projects",
    Path.home() / "projects",
    Path.home() / "src",
    Path.home() / "dev",
    Path.home() / "work",
    Path.home() / "repos",
    Path("/Volumes/MysticalTech/Code"),
]


def _hash_path(p: Path) -> str:
    """Compute the same SHA256 hash Gemini uses for project directories."""
    return hashlib.sha256(str(p).encode()).hexdigest()


def _candidate_paths(known_paths: list[str] | None = None) -> list[Path]:
    """Build a list of candidate project directories to try."""
    seen: set[str] = set()
    candidates: list[Path] = []

    def _add(p: Path) -> None:
        s = str(p)
        if s not in seen and p.is_dir():
            seen.add(s)
            candidates.append(p)

    # 1. Known scope paths from the DB
    for kp in (known_paths or []):
        if kp:
            _add(Path(kp))

    # 2. Home directory itself
    _add(Path.home())

    # 3. Immediate children of common parent directories
    for scan_dir in _SCAN_DIRS:
        if scan_dir.is_dir():
            try:
                for child in scan_dir.iterdir():
                    if child.is_dir() and not child.name.startswith("."):
                        _add(child)
            except PermissionError:
                pass

    return candidates


def _read_project_hash(session_file: Path) -> str | None:
    """Read projectHash from a Gemini session JSON file.

    Reads only the first 8KB to avoid loading the full session.
    """
    try:
        raw = session_file.read_bytes()[:8192]
        # Try to parse — if truncated, the JSON will fail, so try full load
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            with session_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        if isinstance(data, dict):
            return data.get("projectHash") or data.get("project_hash")
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        pass
    return None


def _extract_path_from_tool_calls(session_file: Path) -> Path | None:
    """Last resort: scan tool call outputs for absolute file paths and extract common root."""
    try:
        with session_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    messages = data.get("messages", [])
    abs_paths: list[str] = []

    for msg in messages:
        # Look for tool call results with file paths
        tool_calls = msg.get("toolCalls", []) or msg.get("tool_calls", [])
        for tc in tool_calls:
            result = tc.get("result", {}) or tc.get("output", {})
            if isinstance(result, dict):
                # Check common fields that contain file paths
                for key in ("path", "file", "filename", "cwd", "directory"):
                    val = result.get(key, "")
                    if isinstance(val, str) and val.startswith("/"):
                        abs_paths.append(val)
            elif isinstance(result, str) and "/" in result:
                # Scan string for absolute paths
                for word in result.split():
                    if word.startswith("/") and len(word) > 3:
                        abs_paths.append(word)

        # Also check content for file paths mentioned by the model
        content = msg.get("content", "")
        if isinstance(content, str):
            for word in content.split():
                cleaned = word.strip("`\"'(),;:")
                if cleaned.startswith("/") and len(cleaned) > 5 and "." in cleaned:
                    abs_paths.append(cleaned)

    if not abs_paths:
        return None

    # Find the common root of all absolute paths
    from os.path import commonpath
    try:
        common = Path(commonpath(abs_paths))
        # Walk up until we find a directory that actually exists
        while common != common.parent:
            if common.is_dir():
                # Don't return system paths
                if str(common) in ("/", "/Users", "/home", "/tmp", "/var"):
                    return None
                return common
            common = common.parent
    except ValueError:
        pass
    return None


class GeminiAdapter(SessionAdapter):
    """Session discovery for Gemini CLI."""

    name = "gemini"
    trigger_class = GeminiTrigger

    def discover_sessions(self) -> list[Path]:
        if not GEMINI_HOME.is_dir():
            return []

        sessions = list(GEMINI_HOME.glob("*/chats/session-*.json"))
        sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return sessions

    def discover_sessions_for_project(
        self, project_path: Path
    ) -> list[Path]:
        resolved = project_path.resolve()
        matching: list[Path] = []

        if not GEMINI_HOME.is_dir():
            return matching

        # Try hash-based matching first
        target_hash = _hash_path(resolved)
        hash_dir = GEMINI_HOME / target_hash
        if hash_dir.is_dir():
            chats_dir = hash_dir / "chats"
            if chats_dir.is_dir():
                matching.extend(chats_dir.glob("session-*.json"))

        # Fallback: name-based heuristic
        if not matching:
            for project_dir in GEMINI_HOME.iterdir():
                if not project_dir.is_dir():
                    continue
                chats_dir = project_dir / "chats"
                if not chats_dir.is_dir():
                    continue
                if resolved.name.lower() in project_dir.name.lower():
                    matching.extend(chats_dir.glob("session-*.json"))

        matching.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matching

    def extract_project_path(self, session_file: Path) -> Path | None:
        """Determine project path by reversing the SHA256 directory hash."""
        # The directory name under ~/.gemini/tmp/ is the project hash
        try:
            chats_dir = session_file.parent
            if chats_dir.name != "chats":
                return None
            project_hash_dir = chats_dir.parent
            dir_name = project_hash_dir.name
        except (AttributeError, IndexError):
            return None

        # Also try reading projectHash from the session JSON
        file_hash = _read_project_hash(session_file)
        project_hash = file_hash or dir_name

        # Check cache
        if project_hash in _hash_cache:
            return _hash_cache[project_hash]
        if dir_name in _hash_cache:
            return _hash_cache[dir_name]

        # Build candidates and try to reverse the hash
        known = self.config.get("known_paths", [])
        for candidate in _candidate_paths(known):
            candidate_hash = _hash_path(candidate)
            # Cache all hashes we compute (amortize future lookups)
            _hash_cache[candidate_hash] = candidate
            if candidate_hash == project_hash or candidate_hash == dir_name:
                return candidate

        # Last resort: extract paths from tool call outputs
        extracted = _extract_path_from_tool_calls(session_file)
        if extracted:
            # Verify it matches the hash
            if _hash_path(extracted) in (project_hash, dir_name):
                _hash_cache[project_hash] = extracted
                return extracted
            # Even if hash doesn't match, the extracted path is our best guess
            _hash_cache[project_hash] = extracted
            return extracted

        return None
