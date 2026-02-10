"""Tests for LLM client factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from universal_context.config import UCConfig
from universal_context.llm import _DEFAULT_MODELS, SYSTEM_PROMPT, create_llm_fn


@pytest.fixture
def _clean_env(monkeypatch):
    """Ensure no LLM API keys leak from the host environment."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.mark.usefixtures("_clean_env")
class TestCreateLlmFn:
    async def test_disabled_returns_none(self):
        config = UCConfig(use_llm=False)
        result = await create_llm_fn(config)
        assert result is None

    async def test_no_key_returns_none(self):
        config = UCConfig(llm_provider="openrouter")
        result = await create_llm_fn(config)
        assert result is None

    async def test_openrouter_with_key_returns_callable(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
        config = UCConfig(llm_provider="openrouter")
        with patch("openai.AsyncOpenAI"):
            fn = await create_llm_fn(config)
        assert fn is not None
        assert callable(fn)

    async def test_claude_with_key_returns_callable(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        config = UCConfig(llm_provider="claude")
        with patch("anthropic.AsyncAnthropic"):
            fn = await create_llm_fn(config)
        assert fn is not None
        assert callable(fn)

    async def test_openai_with_key_returns_callable(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config = UCConfig(llm_provider="openai")
        with patch("openai.AsyncOpenAI"):
            fn = await create_llm_fn(config)
        assert fn is not None
        assert callable(fn)

    async def test_unknown_provider_returns_none(self):
        config = UCConfig(llm_provider="unknown_provider")
        result = await create_llm_fn(config)
        assert result is None

    async def test_auto_picks_openrouter_first(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        config = UCConfig(llm_provider="auto")

        with patch("universal_context.llm._make_openrouter_fn") as mock_or:
            mock_or.return_value = AsyncMock()
            fn = await create_llm_fn(config)
            mock_or.assert_called_once()
            assert fn is not None

    async def test_auto_falls_to_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        config = UCConfig(llm_provider="auto")

        with patch("universal_context.llm._make_claude_fn") as mock_cl:
            mock_cl.return_value = AsyncMock()
            fn = await create_llm_fn(config)
            mock_cl.assert_called_once()
            assert fn is not None

    async def test_auto_falls_to_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = UCConfig(llm_provider="auto")

        with patch("universal_context.llm._make_openai_fn") as mock_oai:
            mock_oai.return_value = AsyncMock()
            fn = await create_llm_fn(config)
            mock_oai.assert_called_once()
            assert fn is not None

    async def test_auto_no_keys_returns_none(self):
        config = UCConfig(llm_provider="auto")
        result = await create_llm_fn(config)
        assert result is None

    async def test_custom_model_passed_through(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        config = UCConfig(llm_provider="openrouter", llm_model="meta-llama/llama-3-8b")

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            fn = await create_llm_fn(config)
            assert fn is not None

            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="Summary"))]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

            result = await fn("test content")
            assert result == "Summary"
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "meta-llama/llama-3-8b"


@pytest.mark.usefixtures("_clean_env")
class TestOpenRouterFn:
    async def test_calls_api_with_correct_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            config = UCConfig(llm_provider="openrouter")
            fn = await create_llm_fn(config)

            # Verify client created with OpenRouter base URL
            mock_cls.assert_called_once_with(
                api_key="sk-or-test-key",
                base_url="https://openrouter.ai/api/v1",
            )

            # Call the function and verify API interaction
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="A concise summary."))]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

            result = await fn("User asked to fix a bug in auth. Assistant found the issue.")
            assert result == "A concise summary."

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == _DEFAULT_MODELS["openrouter"]
            assert call_kwargs["messages"][0]["role"] == "system"
            assert call_kwargs["messages"][0]["content"] == SYSTEM_PROMPT
            assert call_kwargs["messages"][1]["role"] == "user"


@pytest.mark.usefixtures("_clean_env")
class TestClaudeFn:
    async def test_calls_anthropic_api(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            config = UCConfig(llm_provider="claude")
            fn = await create_llm_fn(config)

            mock_cls.assert_called_once_with(api_key="sk-ant-test-key")

            mock_resp = MagicMock()
            mock_resp.content = [MagicMock(text="Claude summary.")]
            mock_client.messages.create = AsyncMock(return_value=mock_resp)

            result = await fn("Some transcript content")
            assert result == "Claude summary."

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["model"] == _DEFAULT_MODELS["claude"]
            assert call_kwargs["system"] == SYSTEM_PROMPT
            assert call_kwargs["max_tokens"] == 300


@pytest.mark.usefixtures("_clean_env")
class TestLlmFnErrorHandling:
    async def test_api_error_propagates(self, monkeypatch):
        """LLM function raises on API error — TurnSummarizer handles the fallback."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            config = UCConfig(llm_provider="openrouter")
            fn = await create_llm_fn(config)

            mock_client.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("API connection error")
            )

            with pytest.raises(RuntimeError, match="API connection error"):
                await fn("test content")
