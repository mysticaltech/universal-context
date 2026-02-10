"""Tests for the adapter & trigger system."""

import json
from pathlib import Path

import pytest

from universal_context.adapters.claude import ClaudeAdapter, _decode_project_path
from universal_context.adapters.codex import CodexAdapter
from universal_context.adapters.gemini import GeminiAdapter
from universal_context.adapters.registry import AdapterRegistry, get_registry
from universal_context.triggers.claude_trigger import ClaudeTrigger
from universal_context.triggers.codex_trigger import CodexTrigger
from universal_context.triggers.gemini_trigger import GeminiTrigger

# ============================================================
# FIXTURES: Mock session files
# ============================================================


@pytest.fixture
def claude_session(tmp_path: Path) -> Path:
    """Create a mock Claude Code JSONL session file."""
    # Simulate ~/.claude/projects/-Users-testuser-myproject/abc123.jsonl
    project_dir = tmp_path / ".claude" / "projects" / "-Users-testuser-myproject"
    project_dir.mkdir(parents=True)
    session_file = project_dir / "abc123.jsonl"

    messages = [
        {
            "type": "human",
            "message": {"content": "Fix the login bug"},
            "uuid": "u1",
            "parentUUID": "",
            "timestamp": "2026-02-09T10:00:00Z",
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "I'll look into the login module."},
                    {"type": "text", "text": "The issue is in auth.py line 42."},
                ]
            },
            "uuid": "a1",
            "parentUUID": "u1",
            "timestamp": "2026-02-09T10:00:05Z",
        },
        {
            "type": "human",
            "message": {"content": "Great, now add tests"},
            "uuid": "u2",
            "parentUUID": "a1",
            "timestamp": "2026-02-09T10:01:00Z",
        },
        {
            "type": "assistant",
            "message": {"content": "Here are the tests for the auth module."},
            "uuid": "a2",
            "parentUUID": "u2",
            "timestamp": "2026-02-09T10:01:10Z",
        },
    ]

    with session_file.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")

    return session_file


@pytest.fixture
def claude_session_with_retry(tmp_path: Path) -> Path:
    """Claude session where the user retried the same message."""
    project_dir = tmp_path / ".claude" / "projects" / "-tmp-proj"
    project_dir.mkdir(parents=True)
    session_file = project_dir / "retry.jsonl"

    messages = [
        {
            "type": "human",
            "message": {"content": "explain this code"},
            "uuid": "u1",
            "timestamp": "2026-02-09T10:00:00Z",
        },
        {
            "type": "assistant",
            "message": {"content": "Sure, let me—"},
            "uuid": "a1",
            "timestamp": "2026-02-09T10:00:02Z",
        },
        # Retry: same message within 120s
        {
            "type": "human",
            "message": {"content": "explain this code"},
            "uuid": "u2",
            "timestamp": "2026-02-09T10:00:30Z",
        },
        {
            "type": "assistant",
            "message": {"content": "This code implements a binary search."},
            "uuid": "a2",
            "timestamp": "2026-02-09T10:00:35Z",
        },
    ]

    with session_file.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")

    return session_file


@pytest.fixture
def codex_session(tmp_path: Path) -> Path:
    """Create a mock Codex JSONL session file."""
    session_dir = tmp_path / ".codex" / "sessions" / "2026" / "02" / "09"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "rollout-abc123.jsonl"

    events = [
        {
            "type": "session_meta",
            "payload": {"originator": "codex-cli", "cwd": "/tmp/myproject"},
            "timestamp": "2026-02-09T10:00:00Z",
        },
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "content": "Create a hello world app"},
            "timestamp": "2026-02-09T10:00:01Z",
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "content": "I created main.py with a hello world program.",
            },
            "timestamp": "2026-02-09T10:00:10Z",
        },
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "content": "Add a test file"},
            "timestamp": "2026-02-09T10:01:00Z",
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "content": "Done. Created test_main.py.",
            },
            "timestamp": "2026-02-09T10:01:15Z",
        },
    ]

    with session_file.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    return session_file


@pytest.fixture
def codex_session_with_abort(tmp_path: Path) -> Path:
    """Codex session with an aborted turn."""
    session_dir = tmp_path / ".codex" / "sessions" / "2026" / "02" / "09"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "rollout-abort.jsonl"

    events = [
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "content": "Delete everything"},
            "timestamp": "2026-02-09T10:00:01Z",
        },
        {"type": "turn_aborted", "timestamp": "2026-02-09T10:00:02Z"},
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "content": "Just say hello"},
            "timestamp": "2026-02-09T10:00:05Z",
        },
        {
            "type": "event_msg",
            "payload": {"type": "agent_message", "content": "Hello!"},
            "timestamp": "2026-02-09T10:00:06Z",
        },
    ]

    with session_file.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    return session_file


@pytest.fixture
def gemini_session(tmp_path: Path) -> Path:
    """Create a mock Gemini session JSON file."""
    chats_dir = tmp_path / ".gemini" / "tmp" / "myproject" / "chats"
    chats_dir.mkdir(parents=True)
    session_file = chats_dir / "session-abc123.json"

    data = {
        "sessionId": "abc123",
        "messages": [
            {
                "type": "user",
                "content": "What does this function do?",
                "timestamp": "2026-02-09T10:00:00Z",
            },
            {
                "type": "model",
                "content": "This function calculates the fibonacci sequence.",
                "timestamp": "2026-02-09T10:00:05Z",
            },
            {
                "type": "user",
                "content": "Optimize it",
                "timestamp": "2026-02-09T10:01:00Z",
            },
            {
                "type": "model",
                "content": "Here's an O(1) space iterative version.",
                "timestamp": "2026-02-09T10:01:10Z",
            },
        ],
    }

    with session_file.open("w") as f:
        json.dump(data, f)

    return session_file


# ============================================================
# CLAUDE TRIGGER TESTS
# ============================================================


class TestClaudeTrigger:
    def test_count_turns(self, claude_session: Path):
        trigger = ClaudeTrigger()
        assert trigger.count_complete_turns(claude_session) == 2

    def test_extract_turn_info(self, claude_session: Path):
        trigger = ClaudeTrigger()
        info = trigger.extract_turn_info(claude_session, 1)
        assert info is not None
        assert info.sequence == 1
        assert info.user_message == "Fix the login bug"
        assert "login module" in info.assistant_response

    def test_extract_turn_info_second(self, claude_session: Path):
        trigger = ClaudeTrigger()
        info = trigger.extract_turn_info(claude_session, 2)
        assert info is not None
        assert info.user_message == "Great, now add tests"

    def test_turn_out_of_range(self, claude_session: Path):
        trigger = ClaudeTrigger()
        assert trigger.extract_turn_info(claude_session, 0) is None
        assert trigger.extract_turn_info(claude_session, 99) is None

    def test_is_turn_complete(self, claude_session: Path):
        trigger = ClaudeTrigger()
        assert trigger.is_turn_complete(claude_session, 1) is True
        assert trigger.is_turn_complete(claude_session, 2) is True

    def test_retry_merging(self, claude_session_with_retry: Path):
        trigger = ClaudeTrigger()
        # Should merge retries into 1 turn
        assert trigger.count_complete_turns(claude_session_with_retry) == 1
        info = trigger.extract_turn_info(claude_session_with_retry, 1)
        assert info is not None
        assert "binary search" in info.assistant_response

    def test_raw_transcript(self, claude_session: Path):
        trigger = ClaudeTrigger()
        raw = trigger.get_raw_transcript(claude_session, 1)
        assert raw is not None
        assert "user: Fix the login bug" in raw
        assert "assistant:" in raw

    def test_detect_format(self, claude_session: Path):
        trigger = ClaudeTrigger()
        fmt = trigger.detect_format(claude_session)
        assert fmt == "claude_jsonl"

    def test_detect_format_wrong_file(self, tmp_path: Path):
        wrong = tmp_path / "data.json"
        wrong.write_text("{}")
        trigger = ClaudeTrigger()
        assert trigger.detect_format(wrong) is None

    def test_content_blocks(self, claude_session: Path):
        """Claude uses content block arrays — verify we join them."""
        trigger = ClaudeTrigger()
        info = trigger.extract_turn_info(claude_session, 1)
        assert info is not None
        # Both text blocks should be joined
        assert "login module" in info.assistant_response
        assert "auth.py" in info.assistant_response


# ============================================================
# CODEX TRIGGER TESTS
# ============================================================


class TestCodexTrigger:
    def test_count_turns(self, codex_session: Path):
        trigger = CodexTrigger()
        assert trigger.count_complete_turns(codex_session) == 2

    def test_extract_turn_info(self, codex_session: Path):
        trigger = CodexTrigger()
        info = trigger.extract_turn_info(codex_session, 1)
        assert info is not None
        assert info.user_message == "Create a hello world app"
        assert "main.py" in info.assistant_response

    def test_second_turn(self, codex_session: Path):
        trigger = CodexTrigger()
        info = trigger.extract_turn_info(codex_session, 2)
        assert info is not None
        assert info.user_message == "Add a test file"

    def test_aborted_turn_skipped(self, codex_session_with_abort: Path):
        trigger = CodexTrigger()
        # Aborted turn should be skipped — only 1 complete turn
        assert trigger.count_complete_turns(codex_session_with_abort) == 1
        info = trigger.extract_turn_info(codex_session_with_abort, 1)
        assert info is not None
        assert info.user_message == "Just say hello"

    def test_detect_format(self, codex_session: Path):
        trigger = CodexTrigger()
        assert trigger.detect_format(codex_session) == "codex_jsonl"

    def test_raw_transcript(self, codex_session: Path):
        trigger = CodexTrigger()
        raw = trigger.get_raw_transcript(codex_session, 1)
        assert raw is not None
        assert "user:" in raw
        assert "assistant:" in raw


# ============================================================
# GEMINI TRIGGER TESTS
# ============================================================


class TestGeminiTrigger:
    def test_count_turns(self, gemini_session: Path):
        trigger = GeminiTrigger()
        assert trigger.count_complete_turns(gemini_session) == 2

    def test_extract_turn_info(self, gemini_session: Path):
        trigger = GeminiTrigger()
        info = trigger.extract_turn_info(gemini_session, 1)
        assert info is not None
        assert info.user_message == "What does this function do?"
        assert "fibonacci" in info.assistant_response

    def test_second_turn(self, gemini_session: Path):
        trigger = GeminiTrigger()
        info = trigger.extract_turn_info(gemini_session, 2)
        assert info is not None
        assert info.user_message == "Optimize it"

    def test_incomplete_turn(self, tmp_path: Path):
        """Gemini session with user message but no model response."""
        chats_dir = tmp_path / ".gemini" / "tmp" / "proj" / "chats"
        chats_dir.mkdir(parents=True)
        f = chats_dir / "session-incomplete.json"
        f.write_text(
            json.dumps(
                {
                    "sessionId": "x",
                    "messages": [
                        {"type": "user", "content": "hello"},
                    ],
                }
            )
        )
        trigger = GeminiTrigger()
        assert trigger.count_complete_turns(f) == 0
        assert trigger.is_turn_complete(f, 1) is False

    def test_detect_format(self, gemini_session: Path):
        trigger = GeminiTrigger()
        assert trigger.detect_format(gemini_session) == "gemini_json"

    def test_raw_transcript(self, gemini_session: Path):
        trigger = GeminiTrigger()
        raw = trigger.get_raw_transcript(gemini_session, 1)
        assert raw is not None
        assert "user:" in raw


# ============================================================
# ADAPTER TESTS
# ============================================================


class TestClaudeAdapter:
    def test_name_and_trigger(self):
        adapter = ClaudeAdapter()
        assert adapter.name == "claude"
        assert isinstance(adapter.trigger, ClaudeTrigger)

    def test_decode_project_path(self):
        assert _decode_project_path("-Users-alice-MyProject") == Path(
            "/Users/alice/MyProject"
        )
        assert _decode_project_path("not-encoded") is None

    def test_count_turns_delegation(self, claude_session: Path):
        adapter = ClaudeAdapter()
        assert adapter.count_turns(claude_session) == 2

    def test_is_session_valid(self, claude_session: Path):
        adapter = ClaudeAdapter()
        assert adapter.is_session_valid(claude_session) is True

    def test_is_session_valid_wrong_file(self, tmp_path: Path):
        wrong = tmp_path / "data.txt"
        wrong.write_text("not a session")
        adapter = ClaudeAdapter()
        assert adapter.is_session_valid(wrong) is False


class TestCodexAdapter:
    def test_name_and_trigger(self):
        adapter = CodexAdapter()
        assert adapter.name == "codex"
        assert isinstance(adapter.trigger, CodexTrigger)

    def test_count_turns_delegation(self, codex_session: Path):
        adapter = CodexAdapter()
        assert adapter.count_turns(codex_session) == 2


class TestGeminiAdapter:
    def test_name_and_trigger(self):
        adapter = GeminiAdapter()
        assert adapter.name == "gemini"
        assert isinstance(adapter.trigger, GeminiTrigger)

    def test_count_turns_delegation(self, gemini_session: Path):
        adapter = GeminiAdapter()
        assert adapter.count_turns(gemini_session) == 2


# ============================================================
# REGISTRY TESTS
# ============================================================


class TestAdapterRegistry:
    def test_register_and_get(self):
        reg = AdapterRegistry()
        reg.register(ClaudeAdapter)
        adapter = reg.get("claude")
        assert adapter is not None
        assert adapter.name == "claude"

    def test_get_unknown(self):
        reg = AdapterRegistry()
        assert reg.get("unknown") is None

    def test_names(self):
        reg = AdapterRegistry()
        reg.register(ClaudeAdapter)
        reg.register(CodexAdapter)
        assert set(reg.names()) == {"claude", "codex"}

    def test_global_registry_has_all(self):
        reg = get_registry()
        assert "claude" in reg.names()
        assert "codex" in reg.names()
        assert "gemini" in reg.names()

    def test_auto_detect_claude(self, claude_session: Path):
        reg = AdapterRegistry()
        reg.register(ClaudeAdapter)
        adapter = reg.auto_detect(claude_session)
        assert adapter is not None
        assert adapter.name == "claude"

    def test_auto_detect_codex(self, codex_session: Path):
        reg = AdapterRegistry()
        reg.register(CodexAdapter)
        adapter = reg.auto_detect(codex_session)
        assert adapter is not None
        assert adapter.name == "codex"

    def test_auto_detect_gemini(self, gemini_session: Path):
        reg = AdapterRegistry()
        reg.register(GeminiAdapter)
        adapter = reg.auto_detect(gemini_session)
        assert adapter is not None
        assert adapter.name == "gemini"


# ============================================================
# EDGE CASES
# ============================================================


class TestEdgeCases:
    def test_empty_session_file(self, tmp_path: Path):
        f = tmp_path / ".claude" / "projects" / "-tmp" / "empty.jsonl"
        f.parent.mkdir(parents=True)
        f.write_text("")
        trigger = ClaudeTrigger()
        assert trigger.count_complete_turns(f) == 0

    def test_malformed_json_lines(self, tmp_path: Path):
        f = tmp_path / ".claude" / "projects" / "-tmp" / "bad.jsonl"
        f.parent.mkdir(parents=True)
        f.write_text("not json\n{invalid\n")
        trigger = ClaudeTrigger()
        assert trigger.count_complete_turns(f) == 0

    def test_nonexistent_file(self, tmp_path: Path):
        f = tmp_path / "ghost.jsonl"
        trigger = ClaudeTrigger()
        assert trigger.count_complete_turns(f) == 0

    def test_human_only_turn(self, tmp_path: Path):
        """A turn with only a human message (no response) is incomplete."""
        f = tmp_path / ".claude" / "projects" / "-tmp" / "partial.jsonl"
        f.parent.mkdir(parents=True)
        with f.open("w") as fp:
            fp.write(
                json.dumps(
                    {
                        "type": "human",
                        "message": {"content": "hello"},
                        "uuid": "u1",
                        "timestamp": "2026-02-09T10:00:00Z",
                    }
                )
                + "\n"
            )
        trigger = ClaudeTrigger()
        assert trigger.count_complete_turns(f) == 0
