"""LLM client factory for turn summarization.

Supports three providers through a single factory function:
- OpenRouter (default): OpenAI SDK with custom base_url, one key for all models
- Claude: Anthropic SDK direct
- OpenAI: OpenAI SDK direct
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from .config import UCConfig

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize the following AI coding session transcript. "
    "Focus on: what was asked, what was done, key decisions made. "
    "Be concise (2-4 sentences)."
)

# Legacy alias
SYSTEM_PROMPT = SUMMARIZE_SYSTEM_PROMPT

ASK_SYSTEM_PROMPT = (
    "You are a project knowledge assistant. You have access to a project's "
    "working memory and relevant session summaries from its AI coding history.\n\n"
    "Answer the user's question using ONLY the provided context. "
    "Be specific — cite file names, function names, and decisions when possible. "
    "If the context doesn't contain enough information, say so clearly.\n\n"
    "Keep answers concise but thorough. Use markdown formatting."
)

# Default models per provider (cheap + fast + good at summarization)
_DEFAULT_MODELS: dict[str, str] = {
    "openrouter": "anthropic/claude-haiku-4-5-20251001",
    "claude": "claude-haiku-4-5-20251001",
    "openai": "gpt-4.1-mini",
}


async def create_llm_fn(
    config: UCConfig,
    system_prompt: str = SUMMARIZE_SYSTEM_PROMPT,
    max_tokens: int = 300,
) -> Callable[[str], Awaitable[str]] | None:
    """Create an LLM function based on config.

    Args:
        config: UC configuration.
        system_prompt: System prompt to use. Defaults to summarization prompt.
        max_tokens: Max tokens for the response.

    Returns None if LLM is disabled or no API key is available.
    """
    if not config.use_llm:
        logger.debug("LLM disabled by config")
        return None

    provider = config.llm_provider
    model = config.llm_model or None  # empty string → None (use default)

    if provider == "auto":
        return _resolve_auto(config, model, system_prompt, max_tokens)

    if provider == "openrouter":
        return _make_openrouter_fn(
            model, config.get_api_key("openrouter"), system_prompt, max_tokens,
        )

    if provider == "claude":
        return _make_claude_fn(model, config.get_api_key("claude"), system_prompt, max_tokens)

    if provider == "openai":
        return _make_openai_fn(model, config.get_api_key("openai"), system_prompt, max_tokens)

    logger.warning("Unknown LLM provider %r, LLM disabled", provider)
    return None


def _resolve_auto(
    config: UCConfig,
    model: str | None,
    system_prompt: str = SUMMARIZE_SYSTEM_PROMPT,
    max_tokens: int = 300,
) -> Callable[[str], Awaitable[str]] | None:
    """Try providers in priority order: OpenRouter → Anthropic → OpenAI."""
    key = config.get_api_key("openrouter")
    if key:
        return _make_openrouter_fn(model, key, system_prompt, max_tokens)

    key = config.get_api_key("claude")
    if key:
        return _make_claude_fn(model, key, system_prompt, max_tokens)

    key = config.get_api_key("openai")
    if key:
        return _make_openai_fn(model, key, system_prompt, max_tokens)

    logger.warning("No LLM API key found")
    return None


def _make_openrouter_fn(
    model: str | None,
    api_key: str | None,
    system_prompt: str = SUMMARIZE_SYSTEM_PROMPT,
    max_tokens: int = 300,
) -> Callable[[str], Awaitable[str]] | None:
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set, LLM disabled")
        return None

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    resolved_model = model or _DEFAULT_MODELS["openrouter"]

    async def _call(content: str) -> str:
        resp = await client.chat.completions.create(
            model=resolved_model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return resp.choices[0].message.content or ""

    return _call


def _make_claude_fn(
    model: str | None,
    api_key: str | None,
    system_prompt: str = SUMMARIZE_SYSTEM_PROMPT,
    max_tokens: int = 300,
) -> Callable[[str], Awaitable[str]] | None:
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, LLM disabled")
        return None

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key)
    resolved_model = model or _DEFAULT_MODELS["claude"]

    async def _call(content: str) -> str:
        resp = await client.messages.create(
            model=resolved_model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text

    return _call


def _make_openai_fn(
    model: str | None,
    api_key: str | None,
    system_prompt: str = SUMMARIZE_SYSTEM_PROMPT,
    max_tokens: int = 300,
) -> Callable[[str], Awaitable[str]] | None:
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, LLM disabled")
        return None

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    resolved_model = model or _DEFAULT_MODELS["openai"]

    async def _call(content: str) -> str:
        resp = await client.chat.completions.create(
            model=resolved_model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return resp.choices[0].message.content or ""

    return _call
