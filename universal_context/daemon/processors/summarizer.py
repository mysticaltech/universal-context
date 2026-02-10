"""Turn summarizer processor — creates summary artifacts from transcripts.

When LLM is unavailable, falls back to a simple extractive summary
(first N characters of the transcript).

When an embed_fn is provided, also generates and stores an embedding
vector on the summary artifact for semantic search.
"""

from __future__ import annotations

import logging
from typing import Any

from ...db.client import UCDatabase
from ...db.queries import (
    create_derived_artifact,
    get_run,
    get_turn,
    get_turn_artifacts,
    store_embedding,
)
from .base import BaseProcessor

logger = logging.getLogger(__name__)

FALLBACK_SUMMARY_LENGTH = 500


class TurnSummarizer(BaseProcessor):
    """Summarizes a turn's transcript into a derived artifact."""

    def __init__(
        self,
        llm_fn: Any | None = None,
        embed_fn: Any | None = None,
        max_chars: int = FALLBACK_SUMMARY_LENGTH,
    ) -> None:
        self._llm_fn = llm_fn
        self._embed_fn = embed_fn
        self._max_chars = max_chars

    async def process(
        self, db: UCDatabase, job: dict[str, Any]
    ) -> dict[str, Any]:
        target = job.get("target", "")
        if not target:
            raise ValueError("Job has no target")

        # Get the turn's transcript artifact
        turn = await get_turn(db, target)
        if not turn:
            raise ValueError(f"Turn not found: {target}")

        artifacts = await get_turn_artifacts(db, target)
        if not artifacts:
            raise ValueError(f"No artifacts for turn: {target}")

        # Find the transcript artifact
        transcript_id = self._find_transcript_id(artifacts)
        if not transcript_id:
            raise ValueError(f"No transcript artifact for turn: {target}")

        # Get the transcript content
        transcript_result = await db.query(f"SELECT content FROM {transcript_id}")
        if not transcript_result:
            raise ValueError(f"Could not read transcript: {transcript_id}")

        content = transcript_result[0].get("content", "")
        if not content:
            return {"summary": "", "method": "empty"}

        # Generate summary
        if self._llm_fn is not None:
            try:
                summary = await self._llm_fn(content)
                method = "llm"
            except Exception as e:
                logger.warning("LLM summarization failed, using fallback: %s", e)
                summary = self._extractive_summary(content)
                method = "extractive_fallback"
        else:
            summary = self._extractive_summary(content)
            method = "extractive"

        # Resolve scope from turn's run for denormalized scope field
        scope_id = None
        run_ref = turn.get("run")
        if run_ref:
            run = await get_run(db, str(run_ref))
            if run:
                scope_val = run.get("scope")
                if scope_val:
                    scope_id = str(scope_val)

        # Create derived artifact
        summary_id = await create_derived_artifact(
            db,
            kind="summary",
            content=summary,
            source_id=transcript_id,
            relationship="summarized_from",
            scope_id=scope_id,
        )

        # Generate and store embedding for semantic search
        embedded = False
        if self._embed_fn is not None:
            try:
                embedding = await self._embed_fn.embed_document(summary)
                await store_embedding(db, summary_id, embedding)
                embedded = True
            except Exception as e:
                logger.warning("Embedding failed for %s: %s", summary_id, e)

        return {
            "summary_id": summary_id,
            "method": method,
            "length": len(summary),
            "embedded": embedded,
        }

    def _extractive_summary(self, content: str) -> str:
        """Simple extractive summary: first N chars, truncated at word boundary."""
        if len(content) <= self._max_chars:
            return content
        truncated = content[: self._max_chars]
        # Cut at last space for clean truncation
        last_space = truncated.rfind(" ")
        if last_space > self._max_chars // 2:
            truncated = truncated[:last_space]
        return truncated + "..."

    @staticmethod
    def _find_transcript_id(artifacts: list[dict[str, Any]]) -> str | None:
        """Extract the transcript artifact ID from graph traversal results.

        get_turn_artifacts returns results like:
        [{'->produced': {'->artifact': [RecordID(...)]}}]
        """
        for item in artifacts:
            produced = item.get("->produced", {})
            if isinstance(produced, dict):
                artifact_ids = produced.get("->artifact", [])
                if artifact_ids:
                    return str(artifact_ids[0])
        return None
