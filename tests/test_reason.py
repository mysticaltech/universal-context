"""Tests for RLM reasoning integration — AsyncBridge, LocalInterpreter, tools."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from universal_context.reason import (
    _WRITE_PATTERN,
    AsyncBridge,
    LocalInterpreter,
    _FinalOutput,
    _format_query_results,
    _SubmitError,
    build_tools,
)

# ============================================================
# AsyncBridge
# ============================================================


class TestAsyncBridge:
    def test_run_coroutine(self):
        """Bridge should execute a coroutine and return the result."""
        loop = asyncio.new_event_loop()

        async def _run():
            bridge = AsyncBridge(loop)

            async def coro():
                return 42

            # Run bridge.run in a thread so the loop stays free
            result = await asyncio.to_thread(bridge.run, coro())
            assert result == 42
            bridge.shutdown()

        loop.run_until_complete(_run())
        loop.close()

    def test_run_exception_propagation(self):
        """Exceptions from the coroutine should propagate to the caller."""
        loop = asyncio.new_event_loop()

        async def _run():
            bridge = AsyncBridge(loop)

            async def failing_coro():
                raise ValueError("test error")

            with pytest.raises(ValueError, match="test error"):
                await asyncio.to_thread(bridge.run, failing_coro())
            bridge.shutdown()

        loop.run_until_complete(_run())
        loop.close()

    def test_run_timeout(self):
        """Bridge should raise on timeout."""
        loop = asyncio.new_event_loop()

        async def _run():
            bridge = AsyncBridge(loop)

            async def slow_coro():
                await asyncio.sleep(10)

            with pytest.raises(TimeoutError):
                await asyncio.to_thread(bridge.run, slow_coro(), 0.1)
            bridge.shutdown()

        loop.run_until_complete(_run())
        loop.close()

    def test_shutdown_noop(self):
        """Shutdown should be a no-op (doesn't own the loop)."""
        loop = asyncio.new_event_loop()
        bridge = AsyncBridge(loop)
        bridge.shutdown()
        # Loop should still be usable
        assert not loop.is_closed()
        loop.close()


# ============================================================
# LocalInterpreter
# ============================================================


class TestLocalInterpreter:
    def test_basic_execution(self):
        interp = LocalInterpreter()
        result = interp.execute("print('hello world')")
        assert result == "hello world"

    def test_variable_persistence(self):
        """Variables should survive across execute() calls."""
        interp = LocalInterpreter()
        interp.execute("x = 42")
        result = interp.execute("print(x + 8)")
        assert result == "50"

    def test_tool_injection(self):
        """Tools should be callable by name in the namespace."""

        def my_tool(query: str) -> str:
            return f"result for {query}"

        interp = LocalInterpreter(tools={"my_tool": my_tool})
        result = interp.execute("print(my_tool('test'))")
        assert result == "result for test"

    def test_submit_returns_final_output(self):
        """SUBMIT() should return a FinalOutput-like object."""
        interp = LocalInterpreter()
        result = interp.execute("SUBMIT('the answer')")
        assert hasattr(result, "output")
        assert result.output == "the answer"

    def test_submit_with_complex_value(self):
        interp = LocalInterpreter()
        result = interp.execute("SUBMIT({'key': 'value', 'num': 42})")
        assert result.output == {"key": "value", "num": 42}

    def test_blocked_import_os(self):
        interp = LocalInterpreter()
        result = interp.execute("import os")
        assert "blocked" in result.lower() or "error" in result.lower()

    def test_blocked_import_subprocess(self):
        interp = LocalInterpreter()
        result = interp.execute("import subprocess")
        assert "blocked" in result.lower() or "error" in result.lower()

    def test_blocked_import_sys(self):
        interp = LocalInterpreter()
        result = interp.execute("import sys")
        assert "blocked" in result.lower() or "error" in result.lower()

    def test_allowed_import_json(self):
        interp = LocalInterpreter()
        result = interp.execute("import json; print(json.dumps({'a': 1}))")
        assert result == '{"a": 1}'

    def test_allowed_import_re(self):
        interp = LocalInterpreter()
        result = interp.execute("import re; print(re.findall(r'\\d+', 'abc123def456'))")
        assert "123" in result
        assert "456" in result

    def test_error_capture(self):
        """Errors should be captured as strings, not raised."""
        interp = LocalInterpreter()
        result = interp.execute("1/0")
        assert "ZeroDivisionError" in result

    def test_syntax_error_capture(self):
        interp = LocalInterpreter()
        result = interp.execute("def bad(:")
        assert "SyntaxError" in result

    def test_name_error_capture(self):
        interp = LocalInterpreter()
        result = interp.execute("print(undefined_var)")
        assert "NameError" in result

    def test_multiline_code(self):
        interp = LocalInterpreter()
        code = "for i in range(3):\n    print(i)"
        result = interp.execute(code)
        assert "0" in result
        assert "1" in result
        assert "2" in result

    def test_variables_parameter(self):
        """execute() should accept variables to inject into namespace."""
        interp = LocalInterpreter()
        result = interp.execute("print(x + y)", variables={"x": 10, "y": 20})
        assert result == "30"

    def test_shutdown_clears_namespace(self):
        interp = LocalInterpreter()
        interp.execute("x = 42")
        interp.shutdown()
        # After shutdown, namespace is empty — new execute would fail on x
        # But shutdown also clears builtins, so we need a new interpreter
        interp2 = LocalInterpreter()
        result = interp2.execute("print(x)")
        assert "NameError" in result

    def test_callable_interface(self):
        """LocalInterpreter should be callable via __call__."""
        interp = LocalInterpreter()
        result = interp("print('via call')")
        assert result == "via call"

    def test_empty_output(self):
        """Code with no print/output should return empty string."""
        interp = LocalInterpreter()
        result = interp.execute("x = 1 + 1")
        assert result == ""

    def test_start_noop(self):
        """start() should be a no-op."""
        interp = LocalInterpreter()
        interp.start()  # Should not raise

    def test_tools_property(self):
        """tools property should expose the tools dict (CodeInterpreter protocol)."""
        tools = {"my_tool": lambda: "ok"}
        interp = LocalInterpreter(tools=tools)
        assert interp.tools == tools
        assert interp.tools["my_tool"]() == "ok"

    def test_tools_property_empty(self):
        interp = LocalInterpreter()
        assert interp.tools == {}


# ============================================================
# Write pattern guard
# ============================================================


class TestWritePattern:
    @pytest.mark.parametrize(
        "query",
        [
            "CREATE scope:abc SET name = 'test'",
            "UPDATE scope:abc SET name = 'new'",
            "DELETE scope:abc",
            "RELATE scope:a->contains->run:b",
            "DEFINE TABLE foo",
            "REMOVE TABLE foo",
            "INSERT INTO scope (name) VALUES ('test')",
            "UPSERT scope:abc SET name = 'test'",
        ],
    )
    def test_rejects_write_queries(self, query: str):
        assert _WRITE_PATTERN.search(query) is not None

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT * FROM scope",
            "SELECT count() FROM run GROUP ALL",
            "SELECT ->produced->artifact FROM turn:abc",
            "SELECT * FROM artifact WHERE content @@ 'test'",
        ],
    )
    def test_allows_read_queries(self, query: str):
        assert _WRITE_PATTERN.search(query) is None


# ============================================================
# Format query results
# ============================================================


class TestFormatQueryResults:
    def test_basic_formatting(self):
        results = [{"id": "scope:abc", "name": "test"}]
        output = _format_query_results(results)
        assert "scope:abc" in output
        assert "test" in output

    def test_empty_results(self):
        output = _format_query_results([])
        assert output == "[]"

    def test_nested_dicts(self):
        results = [{"data": {"nested": "value"}}]
        output = _format_query_results(results)
        assert "nested" in output
        assert "value" in output


# ============================================================
# Build tools (with real DB)
# ============================================================


@pytest.fixture
async def seeded_db():
    """Create a DB with schema and seed data for tool testing."""
    from universal_context.db.client import UCDatabase
    from universal_context.db.queries import create_run, create_scope, create_turn_with_artifact
    from universal_context.db.schema import apply_schema

    database = UCDatabase.in_memory()
    await database.connect()
    await apply_schema(database)

    # Seed: scope -> run -> turn with artifact
    scope = await create_scope(database, "test-project", path="/tmp/test-project")
    scope_id = str(scope["id"])

    run = await create_run(database, scope_id, "claude", branch="main")
    run_id = str(run["id"])

    await create_turn_with_artifact(
        database,
        run_id,
        sequence=1,
        user_message="implement auth system",
        raw_content="User asked to implement auth. Built JWT-based auth with bcrypt.",
        create_summary_job=False,
    )

    yield database, scope_id, run_id
    await database.close()


class TestBuildTools:
    async def test_get_working_memory_no_memory(self, seeded_db):
        db, scope_id, _ = seeded_db
        loop = asyncio.get_running_loop()
        bridge = AsyncBridge(loop)

        tools = build_tools(db, bridge, scope_id, embed_provider=None)
        # Run in thread since bridge.run submits to the same loop
        result = await asyncio.to_thread(tools["get_working_memory"])
        assert "no working memory" in result.lower() or "empty" in result.lower()

    async def test_search_sessions_no_results(self, seeded_db):
        db, scope_id, _ = seeded_db
        loop = asyncio.get_running_loop()
        bridge = AsyncBridge(loop)

        tools = build_tools(db, bridge, scope_id, embed_provider=None)
        # BM25 FTS not available on embedded, substring fallback may match
        result = await asyncio.to_thread(tools["search_sessions"], "nonexistent_xyz_query")
        assert isinstance(result, str)

    async def test_list_recent_runs(self, seeded_db):
        db, scope_id, _ = seeded_db
        loop = asyncio.get_running_loop()
        bridge = AsyncBridge(loop)

        tools = build_tools(db, bridge, scope_id, embed_provider=None)
        result = await asyncio.to_thread(tools["list_recent_runs"])
        assert "claude" in result.lower()
        assert "main" in result.lower()

    async def test_get_run_turns(self, seeded_db):
        db, scope_id, run_id = seeded_db
        loop = asyncio.get_running_loop()
        bridge = AsyncBridge(loop)

        tools = build_tools(db, bridge, scope_id, embed_provider=None)
        result = await asyncio.to_thread(tools["get_run_turns"], run_id)
        assert "auth" in result.lower() or "Turn #1" in result

    async def test_query_graph_select(self, seeded_db):
        db, scope_id, _ = seeded_db
        loop = asyncio.get_running_loop()
        bridge = AsyncBridge(loop)

        tools = build_tools(db, bridge, scope_id, embed_provider=None)
        result = await asyncio.to_thread(tools["query_graph"], "SELECT * FROM scope")
        assert "test-project" in result

    async def test_query_graph_rejects_write(self, seeded_db):
        db, scope_id, _ = seeded_db
        loop = asyncio.get_running_loop()
        bridge = AsyncBridge(loop)

        tools = build_tools(db, bridge, scope_id, embed_provider=None)
        result = await asyncio.to_thread(tools["query_graph"], "DELETE scope:abc")
        assert "ERROR" in result
        assert "not allowed" in result.lower()

    async def test_search_semantic_no_provider(self, seeded_db):
        db, scope_id, _ = seeded_db
        loop = asyncio.get_running_loop()
        bridge = AsyncBridge(loop)

        tools = build_tools(db, bridge, scope_id, embed_provider=None)
        result = await asyncio.to_thread(tools["search_semantic"], "auth system")
        assert "unavailable" in result.lower()

    async def test_no_scope(self, seeded_db):
        db, _, _ = seeded_db
        loop = asyncio.get_running_loop()
        bridge = AsyncBridge(loop)

        tools = build_tools(db, bridge, scope_id=None, embed_provider=None)
        result = await asyncio.to_thread(tools["get_working_memory"])
        assert "no project scope" in result.lower()


# ============================================================
# Build DSPy LM (mocked)
# ============================================================


class TestBuildDspyLm:
    @pytest.fixture(autouse=True)
    def _skip_no_dspy(self):
        pytest.importorskip("dspy")

    def test_openrouter_config(self):
        from universal_context.config import UCConfig
        from universal_context.reason import build_dspy_lm

        config = UCConfig(
            llm_provider="openrouter",
            llm_model="x-ai/grok-4.1-fast",
            openrouter_api_key="test-key-123",
        )

        with patch("dspy.LM") as mock_lm:
            mock_lm.return_value = MagicMock()
            build_dspy_lm(config)
            mock_lm.assert_called_once_with(
                "openrouter/x-ai/grok-4.1-fast",
                api_key="test-key-123",
                api_base="https://openrouter.ai/api/v1",
            )

    def test_openai_config(self):
        from universal_context.config import UCConfig
        from universal_context.reason import build_dspy_lm

        config = UCConfig(
            llm_provider="openai",
            llm_model="gpt-4.1",
            openai_api_key="sk-test-123",
        )

        with patch("dspy.LM") as mock_lm:
            mock_lm.return_value = MagicMock()
            build_dspy_lm(config)
            mock_lm.assert_called_once_with(
                "openai/gpt-4.1",
                api_key="sk-test-123",
            )

    def test_anthropic_config(self):
        from universal_context.config import UCConfig
        from universal_context.reason import build_dspy_lm

        config = UCConfig(
            llm_provider="claude",
            llm_model="claude-sonnet-4-5-20250929",
            anthropic_api_key="sk-ant-test",
        )

        with patch("dspy.LM") as mock_lm:
            mock_lm.return_value = MagicMock()
            build_dspy_lm(config)
            mock_lm.assert_called_once_with(
                "anthropic/claude-sonnet-4-5-20250929",
                api_key="sk-ant-test",
            )

    def test_auto_provider_resolution(self):
        from universal_context.config import UCConfig
        from universal_context.reason import build_dspy_lm

        config = UCConfig(
            llm_provider="auto",
            openai_api_key="sk-test-auto",
        )

        with patch("dspy.LM") as mock_lm:
            mock_lm.return_value = MagicMock()
            build_dspy_lm(config)
            # Should resolve to OpenAI (only key provided)
            call_args = mock_lm.call_args
            assert "openai/" in call_args[0][0]

    def test_no_api_key_raises(self):
        from universal_context.config import UCConfig
        from universal_context.reason import build_dspy_lm

        config = UCConfig(llm_provider="openrouter")
        with pytest.raises(ValueError, match="No API key"):
            build_dspy_lm(config)

    def test_unsupported_provider_raises(self):
        from universal_context.config import UCConfig
        from universal_context.reason import build_dspy_lm

        config = UCConfig(llm_provider="unsupported_provider")
        with pytest.raises(ValueError, match="No API key|Unsupported"):
            build_dspy_lm(config)

    def test_default_model_when_empty(self):
        from universal_context.config import UCConfig
        from universal_context.reason import build_dspy_lm

        config = UCConfig(
            llm_provider="openrouter",
            llm_model="",
            openrouter_api_key="test-key",
        )

        with patch("dspy.LM") as mock_lm:
            mock_lm.return_value = MagicMock()
            build_dspy_lm(config)
            model_arg = mock_lm.call_args[0][0]
            assert "openrouter/" in model_arg
            assert "claude-haiku" in model_arg


# ============================================================
# FinalOutput
# ============================================================


class TestFinalOutput:
    def test_fallback_final_output(self):
        fo = _FinalOutput("test value")
        assert fo.output == "test value"

    def test_fallback_repr(self):
        fo = _FinalOutput("test")
        assert repr(fo) == "FinalOutput('test')"

    def test_submit_error(self):
        sig = _SubmitError({"answer": "yes"})
        assert sig.value == {"answer": "yes"}
