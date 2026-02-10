"""Working memory processor — distills session summaries into project context.

Takes recent turn summaries for a scope and uses an LLM to produce a
structured "working memory" document. When previous working memory exists,
the LLM evolves it rather than regenerating from scratch (Mastra-style
replace semantics).
"""

from __future__ import annotations

import logging
from typing import Any

from ...db.client import UCDatabase
from ...db.queries import (
    get_scope,
    get_scope_summaries_for_distillation,
    get_working_memory,
    store_embedding,
    upsert_working_memory,
)
from .base import BaseProcessor

logger = logging.getLogger(__name__)

MEMORY_SYSTEM_PROMPT = """\
You are maintaining a project's persistent working memory — a hyper-compressed, \
cumulative knowledge base that grows across every AI coding session. This document \
is automatically injected into every new session as system context (like CLAUDE.md \
or AGENTS.md), so it must be dense, precise, and immediately useful.

## Your Task

Merge the PREVIOUS working memory with NEW session summaries to produce an \
updated working memory document. This is cumulative — important knowledge from \
the previous memory MUST be preserved unless it's truly obsolete. New information \
should be integrated, not appended.

## Output Format

# Project Memory: {scope_name}

## Architecture & Key Decisions
- [Critical design decisions, patterns chosen, tech stack choices]
- [Include rationale when non-obvious: "X because Y"]

## Current State
- [What exists, what works, what's broken/in-progress]
- [Key file paths and module responsibilities]

## Recent Activity
- [Last 3-5 significant actions, most recent first]
- [Drop older items when they become irrelevant]

## Active Threads
- [Unfinished work, known bugs, planned next steps]
- [Remove threads that have been resolved]

## Gotchas & Learned
- [Things that broke and how they were fixed]
- [Non-obvious behaviors, workarounds, API quirks]
- [Patterns that should be followed or avoided]

## Compression Rules
- TOTAL output MUST be under 4000 words — ruthlessly compress
- Every bullet must earn its place: drop vague or redundant info
- Use terse technical shorthand: file names, function names, error codes
- Merge related points into single dense bullets
- Completed features → remove from Active Threads, keep in Architecture if significant
- Resolved bugs → remove from Active Threads, keep in Gotchas only if the lesson is reusable
- Old Recent Activity → drop unless it established a pattern worth preserving
- This will be read by an AI coding agent — optimize for machine comprehension

## Anti-Bloat (critical)
- The #1 failure mode is accumulating stale facts. Fight it aggressively
- "In progress" last time + done in new summaries → DELETE from Active Threads entirely
- "Added feature X" is NOT worth keeping once X is established — only if there's a gotcha
- Do NOT narrate history. This is a state snapshot, not a changelog
- When in doubt, cut. The agent can always query `uc context` for details
"""


def _format_summaries(summaries: list[dict[str, Any]]) -> str:
    """Format summaries into a readable block for the LLM prompt."""
    if not summaries:
        return "(no summaries available)"

    lines: list[str] = []
    for s in summaries:
        agent = s.get("agent_type", "unknown")
        seq = s.get("sequence", "?")
        user_msg = s.get("user_message") or "(no user message)"
        summary = s.get("summary") or "(no summary)"
        started = str(s.get("started_at", ""))[:19]
        run_id = s.get("run_id", "")

        lines.append(
            f"### [{agent}] Run {run_id} / Turn #{seq} ({started})\n"
            f"**User**: {user_msg}\n"
            f"**Summary**: {summary}\n"
        )
    return "\n".join(lines)


def build_distillation_prompt(
    scope_name: str,
    summaries: list[dict[str, Any]],
    previous_memory: str | None = None,
) -> str:
    """Build the full LLM prompt for working memory distillation."""
    system = MEMORY_SYSTEM_PROMPT.format(scope_name=scope_name)
    previous_block = (
        previous_memory
        or "None — this is the first distillation. "
        "Build the memory from scratch using only the summaries below."
    )
    summaries_block = _format_summaries(summaries)

    return (
        f"{system}\n\n"
        f"## Previous Working Memory\n{previous_block}\n\n"
        f"## Recent Session Summaries (newest first)\n{summaries_block}"
    )


class WorkingMemoryProcessor(BaseProcessor):
    """Distills recent session summaries into a project working memory."""

    def __init__(
        self,
        llm_fn: Any,
        embed_fn: Any | None = None,
        max_summaries: int = 30,
    ) -> None:
        self._llm_fn = llm_fn
        self._embed_fn = embed_fn
        self._max_summaries = max_summaries

    async def process(
        self, db: UCDatabase, job: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a memory_update job.

        job.target is the scope ID (e.g. "scope:abc123").
        """
        scope_id = job.get("target", "")
        if not scope_id:
            raise ValueError("Job has no target scope ID")

        # 1. Get scope info
        scope = await get_scope(db, scope_id)
        if not scope:
            raise ValueError(f"Scope not found: {scope_id}")

        scope_name = scope.get("name", "Unknown Project")

        # 2. Get recent summaries
        summaries = await get_scope_summaries_for_distillation(
            db, scope_id, limit=self._max_summaries,
        )
        if not summaries:
            return {"status": "skipped", "reason": "no_summaries"}

        # 3. Get previous working memory (if any)
        previous = await get_working_memory(db, scope_id)
        previous_content = previous.get("content") if previous else None

        # 4. Build prompt and call LLM
        prompt = build_distillation_prompt(
            scope_name, summaries, previous_content,
        )
        memory_content = await self._llm_fn(prompt)

        if not memory_content:
            raise ValueError("LLM returned empty working memory")

        # 5. Store new working memory artifact
        artifact_id = await upsert_working_memory(
            db, scope_id, memory_content, method="llm",
        )

        # 6. Optionally embed for semantic search
        embedded = False
        if self._embed_fn is not None:
            try:
                embedding = await self._embed_fn.embed_document(memory_content)
                await store_embedding(db, artifact_id, embedding)
                embedded = True
            except Exception as e:
                logger.warning("Embedding failed for %s: %s", artifact_id, e)

        return {
            "artifact_id": artifact_id,
            "scope_id": scope_id,
            "summaries_used": len(summaries),
            "method": "llm",
            "embedded": embedded,
            "had_previous": previous_content is not None,
        }
