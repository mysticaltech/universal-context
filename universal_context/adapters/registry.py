"""Adapter registry for pluggable session discovery."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import SessionAdapter

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry of session adapters, keyed by name."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[SessionAdapter]] = {}

    def register(self, adapter_class: type[SessionAdapter]) -> None:
        self._adapters[adapter_class.name] = adapter_class

    def get(
        self, name: str, config: dict[str, Any] | None = None
    ) -> SessionAdapter | None:
        cls = self._adapters.get(name)
        return cls(config) if cls else None

    def names(self) -> list[str]:
        return list(self._adapters.keys())

    def discover_all_sessions(
        self, config: dict[str, Any] | None = None
    ) -> list[tuple[Path, SessionAdapter]]:
        """Discover sessions across all registered adapters."""
        results: list[tuple[Path, SessionAdapter]] = []
        for name, cls in self._adapters.items():
            try:
                adapter = cls(config)
                for session in adapter.discover_sessions():
                    results.append((session, adapter))
            except Exception:
                logger.warning("Error discovering %s sessions", name, exc_info=True)
        return results

    def auto_detect(
        self, session_file: Path, config: dict[str, Any] | None = None
    ) -> SessionAdapter | None:
        """Try each adapter until one recognizes the file."""
        for cls in self._adapters.values():
            try:
                adapter = cls(config)
                if adapter.is_session_valid(session_file):
                    return adapter
            except Exception:
                continue
        return None


# Singleton
_registry = AdapterRegistry()


def get_registry() -> AdapterRegistry:
    return _registry
