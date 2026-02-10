"""UC Daemon — single-process watcher + worker.

Runs two async task groups in one process:
1. Watcher: polls adapters, discovers sessions, ingests turns
2. Worker: claims and processes jobs from the queue

Single-process design is required because SurrealDB's embedded RocksDB
storage uses fcntl locks, preventing multi-process access.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path

from ..config import UCConfig
from ..db.client import UCDatabase
from ..db.schema import apply_schema
from .processors.memory import WorkingMemoryProcessor
from .processors.summarizer import TurnSummarizer
from .watcher import Watcher
from .worker import Worker

logger = logging.getLogger(__name__)


class UCDaemon:
    """Main daemon process managing watcher + worker."""

    def __init__(self, config: UCConfig | None = None) -> None:
        self._config = config or UCConfig.load()
        self._db: UCDatabase | None = None
        self._watcher: Watcher | None = None
        self._worker: Worker | None = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the daemon: connect DB, apply schema, run watcher + worker."""
        # Initialize database — server URL takes priority over embedded path
        if self._config.db_url:
            self._db = UCDatabase.from_url(
                self._config.db_url, self._config.db_user, self._config.db_pass,
            )
            logger.info("Connecting to SurrealDB server: %s", self._config.db_url)
        else:
            db_path = self._config.resolved_db_path
            self._db = UCDatabase.from_path(Path(db_path))
            logger.info("Using embedded database: %s", db_path)
        await self._db.connect()

        # Initialize embed provider (needed for schema dimension + summarizer)
        from ..embed import create_embed_provider
        from ..llm import create_llm_fn

        embed_provider = await create_embed_provider(self._config)
        embedding_dim = embed_provider.dim if embed_provider else 768

        await apply_schema(self._db, embedding_dim=embedding_dim)
        logger.info("Database ready (embedding_dim=%d)", embedding_dim)

        # Create watcher (pass config as dict for adapter discovery)
        from dataclasses import asdict

        self._watcher = Watcher(
            db=self._db,
            poll_interval=self._config.watcher_poll_interval,
            config=asdict(self._config),
        )

        # Create worker with processors
        self._worker = Worker(
            db=self._db,
            poll_interval=self._config.worker_poll_interval,
        )

        llm_fn = await create_llm_fn(self._config)
        self._worker.register_processor(
            "turn_summary",
            TurnSummarizer(
                llm_fn=llm_fn,
                embed_fn=embed_provider,
                max_chars=self._config.summary_max_chars,
            ),
        )

        if llm_fn and self._config.memory_enabled:
            self._worker.register_processor(
                "memory_update",
                WorkingMemoryProcessor(
                    llm_fn=llm_fn,
                    embed_fn=embed_provider,
                    max_summaries=self._config.memory_max_summaries,
                ),
            )

        # Recover from crashes
        await self._watcher.recover_interrupted_runs()

        # Run both as concurrent tasks
        logger.info("UC Daemon starting...")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._watcher.run(), name="watcher")
            tg.create_task(self._worker.run(), name="worker")
            tg.create_task(self._wait_for_shutdown(), name="shutdown-monitor")

    async def _wait_for_shutdown(self) -> None:
        """Wait for shutdown signal, then cancel sibling tasks."""
        await self._shutdown_event.wait()
        logger.info("Shutdown signal received, stopping...")
        if self._watcher:
            self._watcher.stop()
        if self._worker:
            self._worker.stop()

    async def stop(self) -> None:
        """Signal graceful shutdown."""
        self._shutdown_event.set()
        if self._db:
            await self._db.close()
            logger.info("Database connection closed")

    def _handle_signal(self, sig: signal.Signals) -> None:
        logger.info("Received signal %s", sig.name)
        self._shutdown_event.set()


async def run_daemon(
    config: UCConfig | None = None,
    foreground: bool = True,
) -> None:
    """Entry point to run the daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    daemon = UCDaemon(config)

    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, daemon._handle_signal, sig)

    try:
        await daemon.start()
    except* BaseException as eg:
        # TaskGroup raises ExceptionGroup on cancellation
        for exc in eg.exceptions:
            if not isinstance(exc, asyncio.CancelledError):
                logger.error("Daemon error: %s", exc, exc_info=exc)
    finally:
        await daemon.stop()
