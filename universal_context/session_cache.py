"""Durable cache for turn summaries alongside raw session files.

Sidecars live next to raw session artifacts as:
  {session_file}.{extension}.summary.jsonl

Each line is a JSON object:
{
  "turn": 12,
  "raw_hash": "sha256:...",
  "summary": "...",
  "method": "cache|llm|extractive|extractive_fallback",
  "updated_at": "2026-02-13T...Z"
}
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .memory_repo import file_lock

_SUMMARY_CACHE_SUFFIX = ".summary.jsonl"


def _cache_path(session_file: Path) -> Path:
    """Return the sidecar path for a session file."""
    return session_file.with_name(f"{session_file.name}{_SUMMARY_CACHE_SUFFIX}")


def _content_hash(raw_content: str) -> str:
    return hashlib.sha256(raw_content.encode("utf-8")).hexdigest()


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_cached_summary(
    session_file: Path,
    sequence: int,
    raw_content: str,
) -> str | None:
    """Read the cached summary for turn N if raw content fingerprint matches."""
    if not session_file.exists():
        return None

    target = _cache_path(session_file)
    if not target.exists():
        return None

    target_hash = _content_hash(raw_content)
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for raw in reversed(lines):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if (
            payload.get("turn") == sequence
            and payload.get("raw_hash") == target_hash
            and isinstance(payload.get("summary"), str)
        ):
            summary = payload["summary"].strip()
            return summary or None

    return None


def write_summary_cache(
    session_file: Path,
    sequence: int,
    raw_content: str,
    summary: str,
    method: str,
) -> None:
    """Append one summary cache entry for the turn."""
    if not session_file.exists() or not summary.strip():
        return

    target = _cache_path(session_file)
    entry = {
        "turn": sequence,
        "raw_hash": _content_hash(raw_content),
        "summary": summary,
        "method": method,
        "updated_at": _now_utc(),
    }

    payload = json.dumps(entry, ensure_ascii=False)
    with file_lock(target):
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        target.write_text(f"{existing}{payload}\n", encoding="utf-8")
