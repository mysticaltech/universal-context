"""Session adapters for AI CLI discovery."""

from .base import SessionAdapter
from .claude import ClaudeAdapter
from .codex import CodexAdapter
from .gemini import GeminiAdapter
from .registry import AdapterRegistry, get_registry

__all__ = [
    "SessionAdapter",
    "ClaudeAdapter",
    "CodexAdapter",
    "GeminiAdapter",
    "AdapterRegistry",
    "get_registry",
]

# Auto-register built-in adapters
_reg = get_registry()
_reg.register(ClaudeAdapter)
_reg.register(CodexAdapter)
_reg.register(GeminiAdapter)
