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
from ...memory_repo import MEMORY_SECTIONS, append_section_entry
from .base import BaseProcessor

logger = logging.getLogger(__name__)
MAX_SUMMARY_EVIDENCE_ROWS = 16

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
- When in doubt, cut. The agent can always query `uc find`/`uc ask` for details
"""


def _classify_legacy_section(title: str) -> str:
    lowered = title.lower()
    if "architecture" in lowered:
        return "architecture"
    if "procedure" in lowered or "how to" in lowered:
        return "procedures"
    if "preference" in lowered:
        return "preferences"
    if "current" in lowered or "recent" in lowered or "active" in lowered:
        return "state"
    if "gotcha" in lowered or "learned" in lowered:
        return "state"
    if "question" in lowered:
        return "open_questions"
    if "decision" in lowered:
        return "architecture"
    return "state"


def _split_memory_by_heading(raw: str) -> list[tuple[str, str]]:
    import re

    title_re = re.compile(r"^##\s+(?P<title>.+)$")
    current_title = "state"
    current_body: list[str] = []
    blocks: list[tuple[str, str]] = []

    for line in raw.splitlines():
        match = title_re.match(line.strip())
        if not match:
            current_body.append(line)
            continue

        if any(token.strip() for token in current_body):
            blocks.append((current_title, "\n".join(current_body).strip()))
            current_body = []
        current_title = match.group("title")

    if any(token.strip() for token in current_body):
        blocks.append((current_title, "\n".join(current_body).strip()))

    if not blocks and raw.strip():
        blocks.append(("state", raw.strip()))
    return blocks


def _to_section_payload(raw: str) -> dict[str, str]:
    routed: dict[str, str] = {section: "" for section in MEMORY_SECTIONS}
    for title, block in _split_memory_by_heading(raw):
        section = _classify_legacy_section(title)
        if section in routed:
            section_buffer = routed[section].strip()
            routed[section] = f"{section_buffer}\n\n{block.strip()}".strip()
    return {section: text for section, text in routed.items() if text.strip()}


def _build_summary_evidence(
    summaries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Convert distillation inputs into compact evidence rows."""
    dedup: set[tuple[str, str]] = set()
    evidence: list[dict[str, str]] = []

    for item in summaries:
        run_id = item.get("run_id")
        turn_id = item.get("turn_id")

        if run_id:
            key = ("run_id", str(run_id))
            if key not in dedup:
                dedup.add(key)
                evidence.append({"run_id": str(run_id)})
                if len(evidence) >= MAX_SUMMARY_EVIDENCE_ROWS:
                    break

        if turn_id:
            key = ("turn_id", str(turn_id))
            if key not in dedup:
                dedup.add(key)
                evidence.append({"turn_id": str(turn_id)})
                if len(evidence) >= MAX_SUMMARY_EVIDENCE_ROWS:
                    break

        if len(evidence) >= MAX_SUMMARY_EVIDENCE_ROWS:
            break

    return evidence


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

        # 5. Persist durable memory sections first
        canonical_id = scope.get("canonical_id")
        scope_path = str(scope.get("path")) if scope.get("path") else None
        if not canonical_id and scope.get("path"):
            from pathlib import Path

            from ...git import resolve_canonical_id

            canonical_id = resolve_canonical_id(Path(scope_path))
        canonical_id = canonical_id or scope_id

        routed_sections = _to_section_payload(memory_content)
        if not routed_sections and memory_content.strip():
            routed_sections = {"state": memory_content.strip()}

        evidence = _build_summary_evidence(summaries)
        section_confidence = 0.9 if routed_sections else 0.6

        for section, text in routed_sections.items():
            append_section_entry(
                canonical_id=canonical_id,
                section=section,
                display_name=scope_name,
                content=text,
                memory_type="durable_fact",
                confidence=section_confidence,
                manual=False,
                source="distilled",
                produced_by_model="llm-distiller",
                evidence=evidence,
                scope_path=scope_path,
            )

        # 6. Store working memory artifact (derived index)
        artifact_id = await upsert_working_memory(
            db, scope_id, memory_content, method="llm",
        )

        # 7. Optionally embed for semantic search
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
