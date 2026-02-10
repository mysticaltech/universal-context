"""SurrealDB client wrapper for Universal Context.

Supports two connection modes:
  - Embedded (file://, mem://) — uses the Python SDK's built-in v2 engine.
    Good for development and single-process use. HNSW/KNN not available;
    vector search uses brute-force cosine.  BM25 uses SEARCH ANALYZER syntax.
  - Server (ws://, wss://) — connects to a SurrealDB server (v3+).
    Full feature set: HNSW KNN, BM25 via FULLTEXT ANALYZER, RRF hybrid search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from surrealdb import AsyncSurreal


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

    async def connect(self) -> None:
        """Connect to the database and select namespace/database."""
        self._db = AsyncSurreal(self._url)
        await self._db.connect()
        if self.is_server and self._user:
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

    async def query(self, surql: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Execute a SurrealQL query and return results."""
        if params:
            return await self.db.query(surql, params)
        return await self.db.query(surql)

    async def health(self) -> bool:
        """Check if the database is reachable."""
        try:
            await self.query("INFO FOR DB")
            return True
        except Exception:
            return False
