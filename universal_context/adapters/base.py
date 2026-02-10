"""Base adapter interface for discovering AI agent sessions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models.types import TurnInfo
from ..triggers.base import TurnTrigger


class SessionAdapter(ABC):
    """Discovers sessions for a specific AI CLI and delegates turn parsing.

    Adapter = "where are the sessions?" + "what project is this for?"
    Trigger = "how do I count/extract turns?"

    Each adapter composes a trigger via the trigger_class attribute.
    """

    name: str = ""
    trigger_class: type[TurnTrigger] | None = None

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._trigger: TurnTrigger | None = None

    @property
    def trigger(self) -> TurnTrigger:
        """Lazy-init the trigger instance."""
        if self._trigger is None:
            if self.trigger_class is None:
                raise RuntimeError(
                    f"Adapter {self.name} has no trigger_class set"
                )
            self._trigger = self.trigger_class(self.config)
        return self._trigger

    # --- Discovery ---

    @abstractmethod
    def discover_sessions(self) -> list[Path]:
        """Find all active sessions for this CLI."""

    @abstractmethod
    def discover_sessions_for_project(
        self, project_path: Path
    ) -> list[Path]:
        """Find sessions belonging to a specific project."""

    @abstractmethod
    def extract_project_path(self, session_file: Path) -> Path | None:
        """Determine which project a session file belongs to."""

    def is_session_valid(self, session_file: Path) -> bool:
        """Check if a file looks like a valid session for this adapter."""
        return self.trigger.detect_format(session_file) is not None

    # --- Delegated turn methods ---

    def count_turns(self, session_file: Path) -> int:
        return self.trigger.count_complete_turns(session_file)

    def is_turn_complete(self, session_file: Path, turn_number: int) -> bool:
        return self.trigger.is_turn_complete(session_file, turn_number)

    def extract_turn_info(
        self, session_file: Path, turn_number: int
    ) -> TurnInfo | None:
        return self.trigger.extract_turn_info(session_file, turn_number)

    def get_raw_transcript(
        self, session_file: Path, turn_number: int
    ) -> str | None:
        return self.trigger.get_raw_transcript(session_file, turn_number)
