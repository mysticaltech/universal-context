"""Base processor interface for job processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...db.client import UCDatabase


class BaseProcessor(ABC):
    """Processes a single job type.

    Subclasses implement `process()` which receives the job record
    and returns a result dict on success.
    """

    @abstractmethod
    async def process(
        self, db: UCDatabase, job: dict[str, Any]
    ) -> dict[str, Any]:
        """Process a job. Returns result dict or raises on failure."""
