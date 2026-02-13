"""SurrealDB client wrapper for Universal Context.

Supports two connection modes:
  - Embedded (file://, mem://) — uses the Python SDK's built-in v2 engine.
    Good for development and single-process use. HNSW/KNN not available;
    vector search uses brute-force cosine.  BM25 uses SEARCH ANALYZER syntax.
  - Server (ws://, wss://) — connects to a SurrealDB server (v3+).
    Full feature set: HNSW KNN, BM25 via FULLTEXT ANALYZER, RRF hybrid search.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from surrealdb import AsyncSurreal

logger = logging.getLogger(__name__)


class UCDatabase:
    """Async SurrealDB client with embedded and server support."""

    NAMESPACE = "uc"
    DATABASE = "main"

    def __init__(
        self,
        url: str = "mem://",
        user: str = "",
        password: str = "",
    ) -> None:
        self._url = url
        self._user = user
        self._password = password
        self._db: AsyncSurreal | None = None

    @classmethod
    def from_path(cls, db_path: Path) -> UCDatabase:
        """Create an embedded client from a filesystem path."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(url=f"file://{db_path}")

    @classmethod
    def from_url(cls, url: str, user: str = "", password: str = "") -> UCDatabase:
        """Create a client from a SurrealDB server URL (ws:// or wss://)."""
        return cls(url=url, user=user, password=password)

    @classmethod
    def in_memory(cls) -> UCDatabase:
        """Create an in-memory client (for testing)."""
        return cls(url="mem://")

    @property
    def is_server(self) -> bool:
        """True when connected to a SurrealDB server (vs embedded engine)."""
        return self._url.startswith(("ws://", "wss://", "http://", "https://"))

    async def connect(self, timeout: float = 10.0) -> None:
        """Connect to the database and select namespace/database.

        Args:
            timeout: Max seconds to wait for connect and signin.
                     Pass 0 to disable (used by tests with mem://).
        """
        self._db = AsyncSurreal(self._url)
        if timeout > 0:
            await asyncio.wait_for(self._db.connect(), timeout=timeout)
        else:
            await self._db.connect()
        if self.is_server and self._user:
            if timeout > 0:
                await asyncio.wait_for(
                    self._db.signin({"user": self._user, "pass": self._password}),
                    timeout=timeout,
                )
            else:
                await self._db.signin({"user": self._user, "pass": self._password})
        await self._db.use(self.NAMESPACE, self.DATABASE)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> UCDatabase:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    @property
    def db(self) -> AsyncSurreal:
        """Get the underlying SurrealDB connection. Raises if not connected."""
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db

    async def query(
        self,
        surql: str,
        params: dict[str, Any] | None = None,
        *,
        _retries: int = 3,
    ) -> list[Any]:
        """Execute a SurrealQL query and return results.

        The SDK may return an error string instead of raising when the server
        reports a transaction/IO failure inside the result payload.  We detect
        that here so callers don't have to guard against non-list returns.

        Transient "Transaction conflict" and "IO error" responses are retried
        with linear backoff (100ms, 200ms, 300ms) up to ``_retries`` attempts.
        The IO errors are a known SurrealDB v3 beta issue where concurrent
        connections hit stale vlog pointers during surrealkv compaction
        (surrealdb/surrealdb#6872).
        """
        _retryable = ("Transaction conflict", "IO error")
        for attempt in range(_retries):
            result = await (self.db.query(surql, params) if params else self.db.query(surql))
            if isinstance(result, str):
                if any(msg in result for msg in _retryable) and attempt < _retries - 1:
                    delay = 0.1 * (attempt + 1)
                    logger.debug(
                        "Retryable error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        _retries,
                        delay,
                        result[:120],
                    )
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(f"SurrealDB query error: {result}")
            if not isinstance(result, list):
                return [result] if result is not None else []
            return result
        return []  # unreachable but satisfies type checker

    async def health(self) -> bool:
        """Check if the database is reachable."""
        try:
            await self.query("INFO FOR DB")
            return True
        except Exception:
            return False
