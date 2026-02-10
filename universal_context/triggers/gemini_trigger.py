"""Gemini CLI turn detection.

Gemini stores sessions as JSON files in
~/.gemini/tmp/{project_dir}/chats/session-*.json.

Each file is a JSON object with a 'messages' array.
Messages have 'type' (user/model) and 'content' fields.
Turn = one user message (simple counting).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..models.types import TurnInfo
from .base import TurnTrigger

logger = logging.getLogger(__name__)


def _load_messages(session_file: Path) -> list[dict]:
    """Load messages from a Gemini session JSON file."""
    try:
        with session_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("messages", [])
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read session file: %s", session_file)
    return []


_MODEL_TYPES = {"model", "assistant", "gemini"}


def _is_model_msg(msg: dict) -> bool:
    """Check if a message is from the model/assistant."""
    return msg.get("type") in _MODEL_TYPES or msg.get("role") in _MODEL_TYPES


def _extract_model_content(msg: dict) -> str:
    """Extract text content from a model message.

    Gemini model messages may have content in 'content', 'thoughts',
    or only in 'toolCalls'. We prefer content > thoughts > empty.
    """
    content = msg.get("content", "")
    if content:
        return content

    # Fall back to thoughts (reasoning trace)
    thoughts = msg.get("thoughts", [])
    if isinstance(thoughts, list) and thoughts:
        parts = []
        for t in thoughts:
            if isinstance(t, dict):
                desc = t.get("description", "")
                if desc:
                    parts.append(desc)
            elif isinstance(t, str):
                parts.append(t)
        if parts:
            return "\n".join(parts)

    return ""


def _extract_user_turns(
    messages: list[dict],
) -> list[tuple[int, dict, list[dict]]]:
    """Extract (index, user_msg, model_messages) tuples.

    Pairs each user message with ALL subsequent model messages
    until the next user message. Gemini often responds with
    multiple messages (tool calls, then text response).
    """
    turns: list[tuple[int, dict, list[dict]]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("type") == "user" or msg.get("role") == "user":
            # Collect all subsequent model messages
            model_msgs: list[dict] = []
            j = i + 1
            while j < len(messages) and _is_model_msg(messages[j]):
                model_msgs.append(messages[j])
                j += 1
            turns.append((len(turns) + 1, msg, model_msgs))
            i = j
        else:
            i += 1
    return turns


class GeminiTrigger(TurnTrigger):
    """Turn detection for Gemini CLI JSON sessions."""

    name = "gemini"

    def count_complete_turns(self, session_file: Path) -> int:
        messages = _load_messages(session_file)
        turns = _extract_user_turns(messages)
        return sum(1 for _, _, model_msgs in turns if model_msgs)

    def extract_turn_info(
        self, session_file: Path, turn_number: int
    ) -> TurnInfo | None:
        messages = _load_messages(session_file)
        turns = _extract_user_turns(messages)

        if turn_number < 1 or turn_number > len(turns):
            return None

        _, user_msg, model_msgs = turns[turn_number - 1]
        user_content = user_msg.get("content", "")

        # Merge all model messages into one response
        model_parts = [_extract_model_content(m) for m in model_msgs]
        model_content = "\n".join(p for p in model_parts if p) or None

        started_at = None
        ts = user_msg.get("timestamp")
        if ts:
            try:
                from datetime import datetime

                started_at = datetime.fromisoformat(
                    str(ts).replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        raw_parts = [f"user: {user_content}"]
        if model_content:
            raw_parts.append(f"assistant: {model_content}")

        return TurnInfo(
            sequence=turn_number,
            user_message=user_content,
            assistant_response=model_content,
            raw_content="\n".join(raw_parts),
            started_at=started_at,
        )

    def is_turn_complete(
        self, session_file: Path, turn_number: int
    ) -> bool:
        messages = _load_messages(session_file)
        turns = _extract_user_turns(messages)
        if turn_number < 1 or turn_number > len(turns):
            return False
        _, _, model_msgs = turns[turn_number - 1]
        return bool(model_msgs)

    def get_raw_transcript(
        self, session_file: Path, turn_number: int
    ) -> str | None:
        info = self.extract_turn_info(session_file, turn_number)
        return info.raw_content if info else None

    def detect_format(self, session_file: Path) -> str | None:
        if session_file.suffix != ".json":
            return None

        try:
            with session_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "messages" in data:
                msgs = data["messages"]
                if isinstance(msgs, list) and msgs:
                    first = msgs[0]
                    if first.get("type") in (
                        "user",
                        "model",
                        "gemini",
                    ) or first.get("role") in ("user", "model"):
                        return "gemini_json"
        except (OSError, json.JSONDecodeError):
            pass
        return None
