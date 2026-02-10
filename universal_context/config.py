"""Configuration management for Universal Context."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class UCConfig:
    """Universal Context configuration."""

    # Database
    db_path: str = "~/.uc/data/surreal.db"
    db_url: str = ""  # SurrealDB server URL (ws:// or wss://). Empty = use embedded file://
    db_user: str = ""  # Server auth username (ignored for embedded)
    db_pass: str = ""  # Server auth password (ignored for embedded)

    # API keys (config file takes priority, env vars as fallback)
    openrouter_api_key: str = ""  # or OPENROUTER_API_KEY env var
    openai_api_key: str = ""  # or OPENAI_API_KEY env var
    anthropic_api_key: str = ""  # or ANTHROPIC_API_KEY env var

    # LLM
    use_llm: bool = True
    llm_provider: str = "openrouter"  # "openrouter", "auto", "claude", "openai"
    llm_model: str = ""  # empty = use provider default

    # Embeddings (mandatory for semantic search)
    embed_provider: str = "local"  # "local" (default), "openai", "openrouter", "auto"
    embed_model: str = "text-embedding-3-small"  # Only used for openai/openrouter providers

    # Adapter auto-detection
    auto_detect_claude: bool = True
    auto_detect_codex: bool = True
    auto_detect_gemini: bool = True

    # Discovery
    codex_discovery_days: int = 365  # How many days back to scan for Codex sessions

    # Daemon settings
    watcher_poll_interval: float = 2.0  # seconds between adapter polls
    worker_poll_interval: float = 1.5  # seconds between job claim attempts
    max_job_attempts: int = 10

    # Summarization
    summary_max_chars: int = 500

    # Working memory
    memory_enabled: bool = True           # Enable working memory distillation
    memory_update_threshold: int = 5      # Turns between auto-updates
    memory_max_summaries: int = 30        # Max summaries to feed to LLM

    # Redaction
    redact_secrets: bool = True

    @classmethod
    def load(cls, config_path: Path | None = None) -> UCConfig:
        """Load configuration from YAML file with environment variable overrides."""
        if config_path is None:
            config_path = Path.home() / ".uc" / "config.yaml"

        config_dict: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}

        # Environment variable overrides (UC_ prefix)
        env_map = {
            "db_path": ("UC_DB_PATH", str),
            "db_url": ("UC_DB_URL", str),
            "db_user": ("UC_DB_USER", str),
            "db_pass": ("UC_DB_PASS", str),
            "openrouter_api_key": ("OPENROUTER_API_KEY", str),
            "openai_api_key": ("OPENAI_API_KEY", str),
            "anthropic_api_key": ("ANTHROPIC_API_KEY", str),
            "use_llm": ("UC_USE_LLM", _parse_bool),
            "llm_provider": ("UC_LLM_PROVIDER", str),
            "llm_model": ("UC_LLM_MODEL", str),
            "embed_provider": ("UC_EMBED_PROVIDER", str),
            "embed_model": ("UC_EMBED_MODEL", str),
            "auto_detect_claude": ("UC_AUTO_DETECT_CLAUDE", _parse_bool),
            "auto_detect_codex": ("UC_AUTO_DETECT_CODEX", _parse_bool),
            "auto_detect_gemini": ("UC_AUTO_DETECT_GEMINI", _parse_bool),
            "codex_discovery_days": ("UC_CODEX_DISCOVERY_DAYS", int),
            "watcher_poll_interval": ("UC_WATCHER_POLL_INTERVAL", float),
            "worker_poll_interval": ("UC_WORKER_POLL_INTERVAL", float),
            "max_job_attempts": ("UC_MAX_JOB_ATTEMPTS", int),
            "summary_max_chars": ("UC_SUMMARY_MAX_CHARS", int),
            "memory_enabled": ("UC_MEMORY_ENABLED", _parse_bool),
            "memory_update_threshold": ("UC_MEMORY_UPDATE_THRESHOLD", int),
            "memory_max_summaries": ("UC_MEMORY_MAX_SUMMARIES", int),
            "redact_secrets": ("UC_REDACT_SECRETS", _parse_bool),
        }

        for field_name, (env_var, converter) in env_map.items():
            value = os.getenv(env_var)
            if value is not None:
                config_dict[field_name] = converter(value)

        # Only pass known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in config_dict.items() if k in known})

    def save(self, config_path: Path | None = None) -> None:
        """Save configuration to YAML file."""
        if config_path is None:
            config_path = Path.home() / ".uc" / "config.yaml"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        from dataclasses import asdict

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for a provider: config value first, then env var fallback."""
        key_map = {
            "openrouter": (self.openrouter_api_key, "OPENROUTER_API_KEY"),
            "openai": (self.openai_api_key, "OPENAI_API_KEY"),
            "claude": (self.anthropic_api_key, "ANTHROPIC_API_KEY"),
            "anthropic": (self.anthropic_api_key, "ANTHROPIC_API_KEY"),
        }
        config_val, env_var = key_map.get(provider, ("", ""))
        return config_val or os.getenv(env_var, "") or None

    @property
    def resolved_db_path(self) -> Path:
        """Resolve the database path (expand ~ and make absolute)."""
        return Path(self.db_path).expanduser().resolve()


def _parse_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def get_default_config_content() -> str:
    """Get default config file content for `uc init`."""
    return """\
# Universal Context Configuration
db_path: "~/.uc/data/surreal.db"      # SurrealDB embedded storage (file:// backend)
db_url: ""                             # SurrealDB server URL (ws:// or wss://). Empty = embedded
db_user: ""                            # Server auth username
db_pass: ""                            # Server auth password

# API keys (env vars override these: OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)
openrouter_api_key: ""
openai_api_key: ""
anthropic_api_key: ""

# LLM summarization
use_llm: true                          # Use cloud LLM for summarization
llm_provider: "openrouter"             # "openrouter" (default), "auto", "claude", or "openai"
llm_model: ""                          # Empty = use provider default model

# Embeddings
embed_provider: "local"                # "local" (EmbeddingGemma), "openai", "openrouter", or "auto"
embed_model: "text-embedding-3-small"  # Only used for openai/openrouter providers

# Adapters
auto_detect_claude: true               # Auto-detect Claude Code sessions
auto_detect_codex: true                # Auto-detect Codex sessions
auto_detect_gemini: true               # Auto-detect Gemini CLI sessions

# Discovery
codex_discovery_days: 365              # How many days back to scan for Codex sessions

# Daemon
watcher_poll_interval: 2.0             # Seconds between adapter polls
worker_poll_interval: 1.5              # Seconds between job claim attempts
max_job_attempts: 10                   # Max retries for failed jobs
summary_max_chars: 500                 # Max length for generated summaries

# Working memory
memory_enabled: true                   # Enable working memory distillation
memory_update_threshold: 5             # Turns between auto-updates
memory_max_summaries: 30               # Max summaries to feed to LLM for distillation

redact_secrets: true                   # Redact API keys/tokens in captured content
"""
