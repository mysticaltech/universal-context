"""Claude Code turn detection.

Claude Code stores sessions as JSONL in ~/.claude/projects/{encoded-path}/.
Each line is a JSON object with a 'type' field ('human'/'assistant')
and a 'message' field containing the conversation content.

A turn = one user message + the assistant's full response.
The last turn requires an explicit end marker (next user message or
session end indicators) to avoid false positives from streaming.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models.types import TurnInfo
from .base import TurnTrigger

logger = logging.getLogger(__name__)


@dataclass
class _ClaudeMessage:
    """Parsed message from a Claude Code JSONL line."""

    line_no: int
    role: str  # "human" or "assistant"
    content: str
    uuid: str = ""
    parent_uuid: str = ""
    timestamp: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _parse_messages(session_file: Path) -> list[_ClaudeMessage]:
    """Parse all messages from a Claude Code JSONL file."""
    messages: list[_ClaudeMessage] = []
    try:
        with session_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                role = obj.get("type", "")
                if role not in ("human", "user", "assistant"):
                    continue
                # Normalize "user" → "human" for consistent grouping
                if role == "user":
                    role = "human"

                msg_obj = obj.get("message", {})
                content = ""
                if isinstance(msg_obj, dict):
                    # Content can be a string or list of content blocks
                    raw_content = msg_obj.get("content", "")
                    if isinstance(raw_content, str):
                        content = raw_content
                    elif isinstance(raw_content, list):
                        parts = []
                        for block in raw_content:
                            if isinstance(block, str):
                                parts.append(block)
                            elif isinstance(block, dict):
                                text = block.get("text", "")
                                if not text:
                                    inp = block.get("input", "")
                                    text = inp if isinstance(inp, str) else ""
                                parts.append(text)
                        content = "\n".join(p for p in parts if p)
                elif isinstance(msg_obj, str):
                    content = msg_obj

                messages.append(
                    _ClaudeMessage(
                        line_no=line_no,
                        role=role,
                        content=content,
                        uuid=obj.get("uuid", ""),
                        parent_uuid=obj.get("parentUUID", ""),
                        timestamp=obj.get("timestamp", ""),
                        raw=obj,
                    )
                )
    except OSError:
        logger.warning("Could not read session file: %s", session_file)
    return messages


def _group_into_turns(
    messages: list[_ClaudeMessage],
) -> list[list[_ClaudeMessage]]:
    """Group messages into turns.

    A turn starts with each 'human' message and includes all
    subsequent 'assistant' messages until the next 'human'.
    """
    turns: list[list[_ClaudeMessage]] = []
    current: list[_ClaudeMessage] = []

    for msg in messages:
        if msg.role == "human":
            if current:
                turns.append(current)
            current = [msg]
        else:
            current.append(msg)

    if current:
        turns.append(current)

    return turns


def _merge_retries(
    turns: list[list[_ClaudeMessage]], window_seconds: float = 120.0
) -> list[list[_ClaudeMessage]]:
    """Merge consecutive turns that are retries of the same message.

    Two consecutive turns are retries if:
    - Both start with a human message with identical content
    - They occur within the time window
    """
    if len(turns) <= 1:
        return turns

    merged: list[list[_ClaudeMessage]] = [turns[0]]
    for turn in turns[1:]:
        prev = merged[-1]
        if _is_retry(prev, turn, window_seconds):
            # Replace previous with this retry (keep the latest attempt)
            merged[-1] = turn
        else:
            merged.append(turn)

    return merged


def _is_retry(
    prev: list[_ClaudeMessage],
    curr: list[_ClaudeMessage],
    window: float,
) -> bool:
    """Check if curr is a retry of prev."""
    if not prev or not curr:
        return False
    if prev[0].role != "human" or curr[0].role != "human":
        return False

    # Content match
    prev_hash = hashlib.md5(
        prev[0].content.strip().encode()
    ).hexdigest()
    curr_hash = hashlib.md5(
        curr[0].content.strip().encode()
    ).hexdigest()
    if prev_hash != curr_hash:
        return False

    # Time window check
    try:
        from datetime import datetime

        t1 = datetime.fromisoformat(prev[0].timestamp.replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(curr[0].timestamp.replace("Z", "+00:00"))
        return abs((t2 - t1).total_seconds()) <= window
    except (ValueError, TypeError):
        # If timestamps are missing or invalid, fall back to content-only match
        return True


def _turn_to_transcript(turn: list[_ClaudeMessage]) -> str:
    """Convert a turn's messages to a readable transcript."""
    parts = []
    for msg in turn:
        label = "user" if msg.role == "human" else "assistant"
        parts.append(f"{label}: {msg.content}")
    return "\n".join(parts)


class ClaudeTrigger(TurnTrigger):
    """Turn detection for Claude Code JSONL sessions."""

    name = "claude"

    def count_complete_turns(self, session_file: Path) -> int:
        messages = _parse_messages(session_file)
        turns = _merge_retries(_group_into_turns(messages))

        complete = 0
        for i, turn in enumerate(turns):
            has_assistant = any(m.role == "assistant" for m in turn)
            if not has_assistant:
                continue

            if i < len(turns) - 1:
                # Non-last turns: complete if next user message exists
                complete += 1
            else:
                # Last turn: complete if it has assistant content
                # (conservative — could add end-marker detection later)
                assistant_content = "".join(
                    m.content for m in turn if m.role == "assistant"
                )
                if assistant_content.strip():
                    complete += 1

        return complete

    def extract_turn_info(
        self, session_file: Path, turn_number: int
    ) -> TurnInfo | None:
        messages = _parse_messages(session_file)
        turns = _merge_retries(_group_into_turns(messages))

        if turn_number < 1 or turn_number > len(turns):
            return None

        turn = turns[turn_number - 1]
        user_msg = next(
            (m.content for m in turn if m.role == "human"), None
        )
        assistant_parts = [
            m.content for m in turn if m.role == "assistant"
        ]
        assistant_response = "\n".join(assistant_parts) if assistant_parts else None

        timestamp = turn[0].timestamp if turn else None
        started_at = None
        if timestamp:
            try:
                from datetime import datetime

                started_at = datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        return TurnInfo(
            sequence=turn_number,
            user_message=user_msg,
            assistant_response=assistant_response,
            raw_content=_turn_to_transcript(turn),
            started_at=started_at,
        )

    def is_turn_complete(
        self, session_file: Path, turn_number: int
    ) -> bool:
        messages = _parse_messages(session_file)
        turns = _merge_retries(_group_into_turns(messages))

        if turn_number < 1 or turn_number > len(turns):
            return False

        turn = turns[turn_number - 1]
        has_assistant = any(m.role == "assistant" for m in turn)
        if not has_assistant:
            return False

        # Non-last turns are always complete
        if turn_number < len(turns):
            return True

        # Last turn: check for content
        assistant_content = "".join(
            m.content for m in turn if m.role == "assistant"
        )
        return bool(assistant_content.strip())

    def get_raw_transcript(
        self, session_file: Path, turn_number: int
    ) -> str | None:
        messages = _parse_messages(session_file)
        turns = _merge_retries(_group_into_turns(messages))

        if turn_number < 1 or turn_number > len(turns):
            return None

        return _turn_to_transcript(turns[turn_number - 1])

    def detect_format(self, session_file: Path) -> str | None:
        if session_file.suffix != ".jsonl":
            return None
        if ".claude" not in str(session_file):
            return None

        # Sniff first few lines for Claude Code format
        try:
            with session_file.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    try:
                        obj = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") in ("human", "user", "assistant") and isinstance(
                        obj.get("message"), (dict, str)
                    ):
                        return "claude_jsonl"
        except OSError:
            pass
        return None
