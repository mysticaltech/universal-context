"""Base trigger interface for detecting turn boundaries in session files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models.types import TurnInfo


class TurnTrigger(ABC):
    """Detects turn boundaries and extracts turn content from session files.

    Each AI CLI stores transcripts differently — Claude Code uses JSONL
    with message pairs, Codex uses event-stream JSONL, Gemini uses JSON.
    Concrete triggers implement the parsing logic for each format.
    """

    name: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def count_complete_turns(self, session_file: Path) -> int:
        """Count the number of complete turns in a session file."""

    @abstractmethod
    def extract_turn_info(
        self, session_file: Path, turn_number: int
    ) -> TurnInfo | None:
        """Extract details for turn N (1-based)."""

    @abstractmethod
    def is_turn_complete(
        self, session_file: Path, turn_number: int
    ) -> bool:
        """Check whether turn N is complete (has full response)."""

    @abstractmethod
    def get_raw_transcript(
        self, session_file: Path, turn_number: int
    ) -> str | None:
        """Extract the raw transcript text for a turn."""

    def detect_format(self, session_file: Path) -> str | None:
        """Detect whether this trigger can handle the given file.

        Returns a format string (e.g. 'claude_jsonl') or None.
        Default implementation returns None — override for sniffing.
        """
        return None
