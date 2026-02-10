"""Codex CLI turn detection.

Codex stores sessions as JSONL event streams in
~/.codex/sessions/{YYYY}/{MM}/{DD}/rollout-*.jsonl.

Events include:
  - session_meta: session metadata
  - event_msg with type=user_message: user input
  - event_msg with type=agent_message: assistant output
  - response_item with role=user/assistant: message content

A turn = user_message → agent_message sequence (state machine).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models.types import TurnInfo
from .base import TurnTrigger

logger = logging.getLogger(__name__)


@dataclass
class _CodexTurn:
    """Intermediate representation of a parsed Codex turn."""

    turn_number: int
    user_message: str
    user_line: int
    user_timestamp: str = ""
    assistant_message: str = ""
    assistant_line: int | None = None
    end_line: int | None = None


def _extract_user_message(data: dict[str, Any]) -> tuple[str, str]:
    """Extract user message text and timestamp from an event.

    Returns (message, timestamp) or ("", "") if not a user event.
    """
    payload = data.get("payload", data)

    # Pattern 1: event_msg with type=user_message
    if data.get("type") == "event_msg":
        if payload.get("type") == "user_message":
            content = payload.get("content", payload.get("text", ""))
            return content, data.get("timestamp", "")

    # Pattern 2: response_item with role=user
    if data.get("type") == "response_item":
        if payload.get("role") == "user":
            content = payload.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            return content, data.get("timestamp", "")

    return "", ""


def _extract_assistant_message(data: dict[str, Any]) -> tuple[str, str]:
    """Extract assistant message text and timestamp.

    Returns (message, timestamp) or ("", "") if not an assistant event.
    """
    payload = data.get("payload", data)

    # Pattern 1: event_msg with type=agent_message
    if data.get("type") == "event_msg":
        if payload.get("type") == "agent_message":
            content = payload.get("content", payload.get("text", ""))
            return content, data.get("timestamp", "")

    # Pattern 2: response_item with role=assistant
    if data.get("type") == "response_item":
        if payload.get("role") == "assistant":
            content = payload.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            return content, data.get("timestamp", "")

    return "", ""


def _is_turn_aborted(data: dict[str, Any]) -> bool:
    """Check if this event signals a turn abort."""
    return data.get("type") == "turn_aborted" or (
        data.get("type") == "event_msg"
        and data.get("payload", {}).get("type") == "turn_aborted"
    )


def _parse_turns(session_file: Path) -> list[_CodexTurn]:
    """Parse all complete turns from a Codex JSONL session file."""
    turns: list[_CodexTurn] = []
    current: _CodexTurn | None = None
    turn_number = 0

    try:
        with session_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # User message starts a new turn
                user_msg, user_ts = _extract_user_message(data)
                if user_msg:
                    # Finalize previous turn if it had an assistant response
                    if current and current.assistant_line is not None:
                        current.end_line = line_no - 1
                        turns.append(current)
                    elif current:
                        # Previous turn had no response — drop it
                        pass
                    turn_number += 1
                    current = _CodexTurn(
                        turn_number=turn_number,
                        user_message=user_msg,
                        user_line=line_no,
                        user_timestamp=user_ts,
                    )
                    continue

                # Assistant message completes current turn
                asst_msg, asst_ts = _extract_assistant_message(data)
                if asst_msg and current:
                    if current.assistant_message:
                        current.assistant_message += "\n" + asst_msg
                    else:
                        current.assistant_message = asst_msg
                    current.assistant_line = line_no
                    continue

                # Turn abort — discard incomplete turn
                if _is_turn_aborted(data) and current:
                    current = None
                    continue
    except OSError:
        logger.warning("Could not read session file: %s", session_file)

    # Finalize last turn
    if current and current.assistant_line is not None:
        current.end_line = current.assistant_line
        turns.append(current)

    return turns


class CodexTrigger(TurnTrigger):
    """Turn detection for Codex CLI event-stream sessions."""

    name = "codex"

    def count_complete_turns(self, session_file: Path) -> int:
        return len(_parse_turns(session_file))

    def extract_turn_info(
        self, session_file: Path, turn_number: int
    ) -> TurnInfo | None:
        turns = _parse_turns(session_file)
        if turn_number < 1 or turn_number > len(turns):
            return None

        turn = turns[turn_number - 1]
        started_at = None
        if turn.user_timestamp:
            try:
                from datetime import datetime

                started_at = datetime.fromisoformat(
                    turn.user_timestamp.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        raw = f"user: {turn.user_message}\nassistant: {turn.assistant_message}"
        return TurnInfo(
            sequence=turn.turn_number,
            user_message=turn.user_message,
            assistant_response=turn.assistant_message or None,
            raw_content=raw,
            started_at=started_at,
        )

    def is_turn_complete(
        self, session_file: Path, turn_number: int
    ) -> bool:
        turns = _parse_turns(session_file)
        return 1 <= turn_number <= len(turns)

    def get_raw_transcript(
        self, session_file: Path, turn_number: int
    ) -> str | None:
        turns = _parse_turns(session_file)
        if turn_number < 1 or turn_number > len(turns):
            return None
        turn = turns[turn_number - 1]
        return f"user: {turn.user_message}\nassistant: {turn.assistant_message}"

    def detect_format(self, session_file: Path) -> str | None:
        if session_file.suffix != ".jsonl":
            return None

        try:
            with session_file.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 20:
                        break
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    # session_meta with codex originator
                    if data.get("type") == "session_meta":
                        payload = data.get("payload", {})
                        if "codex" in str(
                            payload.get("originator", "")
                        ).lower():
                            return "codex_jsonl"

                    # response_item pattern
                    if data.get("type") == "response_item":
                        payload = data.get("payload", data)
                        if payload.get("role") in ("user", "assistant"):
                            return "codex_jsonl"

                    # event_msg pattern
                    if data.get("type") == "event_msg":
                        payload = data.get("payload", {})
                        if payload.get("type") in (
                            "user_message",
                            "agent_message",
                        ):
                            return "codex_jsonl"
        except OSError:
            pass
        return None
