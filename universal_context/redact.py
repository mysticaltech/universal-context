"""Secret redaction for captured content.

Scans text for common secret patterns and replaces them with
[REDACTED] markers before storing in the database.
"""

from __future__ import annotations

import re

# Patterns that match common secrets
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("API Key", re.compile(
        r"(?:sk-|pk-|api[_-]?key|apikey|api_secret|secret_key|access_key)"
        r"[\s:=]*['\"]?([A-Za-z0-9_\-]{20,})['\"]?",
        re.IGNORECASE,
    )),
    ("Bearer Token", re.compile(
        r"[Bb]earer\s+([A-Za-z0-9_\-.]{20,})"
    )),
    ("AWS Key", re.compile(
        r"(?:AKIA|ASIA)[A-Z0-9]{16}"
    )),
    ("Private Key", re.compile(
        r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"
        r".*?"
        r"-----END (?:RSA |EC |DSA )?PRIVATE KEY-----",
        re.DOTALL,
    )),
    ("Password Assignment", re.compile(
        r"(?:password|passwd|pwd)[\s:=]+['\"]([^'\"]{8,})['\"]",
        re.IGNORECASE,
    )),
    ("Connection String", re.compile(
        r"(?:postgres|mysql|mongodb|redis)://[^\s'\"]+:[^\s@'\"]+@",
        re.IGNORECASE,
    )),
    ("GitHub Token", re.compile(
        r"ghp_[A-Za-z0-9]{36}"
    )),
    ("Anthropic Key", re.compile(
        r"sk-ant-[A-Za-z0-9_\-]{20,}"
    )),
    ("OpenAI Key", re.compile(
        r"sk-[A-Za-z0-9]{20,}"
    )),
]


def redact_secrets(text: str) -> str:
    """Replace detected secrets in text with [REDACTED] markers."""
    result = text
    for label, pattern in _PATTERNS:
        result = pattern.sub(f"[REDACTED:{label}]", result)
    return result


def has_secrets(text: str) -> bool:
    """Check if text contains any detectable secrets."""
    for _, pattern in _PATTERNS:
        if pattern.search(text):
            return True
    return False
