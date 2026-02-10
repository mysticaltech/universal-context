"""Job worker — claims and processes queued jobs.

Uses a simple poll-and-claim loop. Each job is dispatched to a
registered processor based on job_type.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..db.client import UCDatabase
from ..db.queries import claim_next_job, complete_job, fail_job
from .processors.base import BaseProcessor

logger = logging.getLogger(__name__)


class Worker:
    """Claims and processes jobs from the job queue."""

    # Rebuild HNSW index after this many new embeddings are stored.
    # SurrealDB v3 beta has a build-once bug: data inserted after
    # DEFINE INDEX ... HNSW is invisible to KNN queries.
    HNSW_REBUILD_THRESHOLD = 25

    def __init__(
        self,
        db: UCDatabase,
        poll_interval: float = 1.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._db = db
        self._poll_interval = poll_interval
        self._config = config or {}
        self._processors: dict[str, BaseProcessor] = {}
        self._running = False
        self._embeddings_since_rebuild = 0

    def register_processor(self, job_type: str, processor: BaseProcessor) -> None:
        self._processors[job_type] = processor

    async def run(self) -> None:
        """Run the worker loop until cancelled."""
        self._running = True
        logger.info(
            "Worker started (poll_interval=%.1fs, processors=%s)",
            self._poll_interval,
            list(self._processors.keys()),
        )
        try:
            while self._running:
                processed = await self._claim_and_process()
                if not processed:
                    await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            logger.info("Worker stopping (cancelled)")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    async def _claim_and_process(self) -> bool:
        """Try to claim and process one job. Returns True if a job was processed."""
        job = await claim_next_job(self._db)
        if job is None:
            return False

        job_id = str(job["id"])
        job_type = job.get("job_type", "")
        target = job.get("target", "")

        processor = self._processors.get(job_type)
        if processor is None:
            logger.warning("No processor for job type: %s", job_type)
            await fail_job(self._db, job_id, f"No processor for type: {job_type}")
            return True

        logger.debug("Processing job %s (type=%s, target=%s)", job_id, job_type, target)
        try:
            result = await processor.process(self._db, job)
            await complete_job(self._db, job_id, result)
            logger.debug("Job %s completed", job_id)

            # Track new embeddings → auto-rebuild HNSW when threshold hit
            if isinstance(result, dict) and result.get("embedded"):
                self._embeddings_since_rebuild += 1
                if self._embeddings_since_rebuild >= self.HNSW_REBUILD_THRESHOLD:
                    await self._rebuild_hnsw()
        except Exception as e:
            logger.error("Job %s failed: %s", job_id, e, exc_info=True)
            await fail_job(self._db, job_id, str(e))

        return True

    async def _rebuild_hnsw(self) -> None:
        """Rebuild HNSW index to incorporate newly stored embeddings."""
        if not self._db.is_server:
            self._embeddings_since_rebuild = 0
            return

        from ..db.schema import rebuild_hnsw_index

        try:
            rebuilt = await rebuild_hnsw_index(self._db)
            if rebuilt:
                logger.info(
                    "HNSW index rebuilt after %d new embeddings",
                    self._embeddings_since_rebuild,
                )
            self._embeddings_since_rebuild = 0
        except Exception as e:
            logger.warning("HNSW rebuild failed: %s", e)
