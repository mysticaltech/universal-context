"""Durable memory repository helpers.

This module owns the canonical file-backed memory store at ~/.uc/memory/.
It provides deterministic scope-to-directory mapping, locked append/update
operations, and section-aware read/write helpers.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from . import get_uc_home

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover
    fcntl = None

MEMORY_DIR = "memory"
SCOPE_REGISTRY_FILE = "scope-map.json"
LOCK_SUFFIX = ".lock"

MEMORY_SECTIONS = (
    "architecture",
    "state",
    "procedures",
    "preferences",
    "open_questions",
)

MEMORY_TYPES = (
    "durable_fact",
    "procedure",
    "decision",
    "open_question",
)


class MemoryRepoError(RuntimeError):
    """Base memory repo error."""


@dataclass(frozen=True)
class ScopeRegistryEntry:
    """Persistent scope mapping entry."""

    canonical_id: str
    canonical_url: str
    dir_key: str
    display_name: str
    scope_name: str | None = None
    remote_url: str | None = None
    path: str | None = None
    scope_id: str | None = None
    created_at: str = ""
    updated_at: str = ""


def get_memory_root() -> Path:
    """Return the canonical memory repo root."""
    return get_uc_home() / MEMORY_DIR


def get_scope_map_path(root: Path | None = None) -> Path:
    """Return path to scope mapping file."""
    return (root or get_memory_root()) / SCOPE_REGISTRY_FILE


def get_memory_project_root(root: Path | None = None) -> Path:
    """Return the project-scoped memory bucket."""
    return (root or get_memory_root()) / "projects"


def get_memory_global_root(root: Path | None = None) -> Path:
    """Return the global memory bucket."""
    return (root or get_memory_root()) / "global"


def get_memory_skills_root(root: Path | None = None) -> Path:
    """Return the promoted skill bucket."""
    return (root or get_memory_root()) / "skills"


def get_memory_migrations_root(root: Path | None = None) -> Path:
    """Return migration checkpoint directory."""
    return (root or get_memory_root()) / ".migrations"


def _slugify_name(name: str) -> str:
    """Convert scope name to filesystem-safe slug."""
    slug = re.sub(r"[^0-9a-zA-Z._-]+", "-", name.strip().lower())
    return slug.strip(".-_") or "project"


def _strip_git_suffix(value: str) -> str:
    """Normalize git path-like identifiers by trimming trailing `.git` and slash."""
    if not value:
        return ""
    normalized = value.strip().rstrip("/")
    if normalized.lower().endswith(".git"):
        normalized = normalized[: -4]
    return normalized.strip("/")


def slugify_name(name: str) -> str:
    """Public slug helper for memory-backed artifacts."""
    return _slugify_name(name)


def normalize_canonical_id(raw: str) -> str:
    """Normalize canonical identifiers for stable scope mapping."""
    canonical = (raw or "").strip().strip("/")
    if not canonical:
        return ""

    lowered = canonical.lower()
    if lowered.startswith("path://") or lowered.startswith("git-local://"):
        return canonical

    parsed = urlparse(canonical)
    if parsed.scheme and parsed.netloc:
        host = parsed.netloc.split("@", 1)[-1].lower()
        path = _strip_git_suffix(parsed.path.lstrip("/"))
        return f"{host}/{path}" if path else host

    m = re.match(r"^[^@]+@([\w.-]+):(.+)$", canonical)
    if m:
        return f"{m.group(1).lower()}/{_strip_git_suffix(m.group(2))}"

    return _strip_git_suffix(canonical)


def _scope_entry_field(raw: dict[str, Any], key: str, default: str | None = None) -> str | None:
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        return default
    value = value.strip()
    return value or default


def _hash_scope_key(value: str) -> str:
    """Return a stable short hash for a scope identifier."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _now_utc() -> str:
    """Return RFC3339-ish UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _lock_path(path: Path) -> Path:
    """Return lockfile path for guarded writes."""
    return Path(f"{path}.lock")


@contextmanager
def file_lock(target: Path, timeout: float = 8.0):
    """Serialize writers across threads/processes.

    On platforms without fcntl, lock acquisition degrades to best-effort.
    """
    lock_file = _lock_path(target)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "a+", encoding="utf-8") as fp:
        if fcntl is None:
            yield
            return
        end_at = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= end_at:
                    raise TimeoutError(f"Timeout waiting for lock on {target}")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def bootstrap_memory_repo(root: Path | None = None, init_git: bool = True) -> Path:
    """Create canonical directories and optionally initialize git tracking."""
    base = root or get_memory_root()
    base.mkdir(parents=True, exist_ok=True)
    get_memory_project_root(base).mkdir(parents=True, exist_ok=True)
    get_memory_global_root(base).mkdir(parents=True, exist_ok=True)
    get_memory_skills_root(base).mkdir(parents=True, exist_ok=True)
    get_memory_migrations_root(base).mkdir(parents=True, exist_ok=True)
    (get_memory_global_root(base) / "human.md").touch(exist_ok=True)

    # Ensure .gitignore excludes lock files
    gitignore = base / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*.lock\n", encoding="utf-8")

    if init_git:
        _ensure_git_repo(base)

    return base


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "updated_at": _now_utc(), "scopes": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MemoryRepoError(f"Malformed scope map at {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    data = dict(data)
    data["updated_at"] = _now_utc()
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _scope_map_snapshot(root: Path | None = None) -> dict[str, ScopeRegistryEntry]:
    """Load current scope map into typed entries."""
    map_path = get_scope_map_path(root)
    payload = _read_json(map_path)
    raw_scopes = payload.get("scopes", {})
    result: dict[str, ScopeRegistryEntry] = {}
    if not isinstance(raw_scopes, dict):
        return result

    for canonical_id, raw in raw_scopes.items():
        if not isinstance(raw, dict):
            continue
        canonical_value = str(canonical_id)
        canonical_url = (
            _scope_entry_field(raw, "canonical_url", canonical_value)
            or canonical_value
        )
        result[canonical_value] = ScopeRegistryEntry(
            canonical_id=canonical_value,
            canonical_url=canonical_url,
            dir_key=str(raw.get("dir_key", "")),
            display_name=str(raw.get("display_name", canonical_value)),
            scope_name=_scope_entry_field(raw, "scope_name"),
            remote_url=raw.get("remote_url"),
            path=raw.get("path"),
            scope_id=_scope_entry_field(raw, "scope_id"),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
        )
    return result


def list_scope_registry(root: Path | None = None) -> list[ScopeRegistryEntry]:
    """Return all canonical scope-map entries as typed records."""
    return list(_scope_map_snapshot(root).values())


def _find_registry_entry(
    registry: dict[str, ScopeRegistryEntry],
    canonical_id: str,
) -> tuple[str, ScopeRegistryEntry] | tuple[None, None]:
    normalized = normalize_canonical_id(canonical_id)
    if not normalized:
        return None, None
    direct = registry.get(normalized)
    if direct is not None:
        return normalized, direct

    for key, entry in registry.items():
        if normalize_canonical_id(entry.canonical_id) == normalized:
            return key, entry
        if entry.canonical_url and normalize_canonical_id(entry.canonical_url) == normalized:
            return key, entry
        if entry.remote_url and normalize_canonical_id(entry.remote_url) == normalized:
            return key, entry
    return None, None


def _find_registry_entry_by_path(
    registry: dict[str, ScopeRegistryEntry],
    scope_path: str | None,
) -> tuple[str, ScopeRegistryEntry] | tuple[None, None]:
    if not scope_path:
        return None, None
    try:
        normalized_target = str(Path(scope_path).resolve())
    except OSError:
        normalized_target = str(scope_path)
    for key, entry in registry.items():
        if not entry.path:
            continue
        try:
            normalized_entry = str(Path(entry.path).resolve())
        except OSError:
            normalized_entry = str(entry.path)
        if normalized_entry == normalized_target:
            return key, entry
    return None, None


def _normalize_section(section: str) -> str:
    section = section.strip()
    if section not in MEMORY_SECTIONS:
        raise ValueError(f"Unknown memory section: {section}")
    return section


def _make_dir_key(canonical_id: str, display_name: str) -> str:
    canonical_id = normalize_canonical_id(canonical_id)
    return f"{_slugify_name(display_name)}--{_hash_scope_key(canonical_id)}"


def _serialize_entry(entry: dict[str, Any]) -> str:
    yaml_text = yaml.safe_dump(entry, sort_keys=False, allow_unicode=True)
    return f"---\n{yaml_text}"


def _parse_entry(raw: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    if "content" not in raw:
        return None
    return dict(raw)


def _scope_dir_candidate(
    canonical_id: str,
    display_name: str,
    *,
    root: Path | None = None,
) -> Path:
    repo_root = root or get_memory_root()
    project_root = get_memory_project_root(repo_root)
    normalized = normalize_canonical_id(canonical_id) or canonical_id
    return project_root / _make_dir_key(normalized, display_name)

def resolve_scope_directory(
    canonical_id: str,
    display_name: str,
    *,
    root: Path | None = None,
    remote_url: str | None = None,
    path: str | None = None,
) -> Path:
    """Map canonical scope identity to the canonical project directory."""
    repo_root = root or get_memory_root()
    repo_root.mkdir(parents=True, exist_ok=True)
    get_memory_project_root(repo_root).mkdir(parents=True, exist_ok=True)
    lock_path = get_scope_map_path(repo_root)

    normalized = normalize_canonical_id(canonical_id)
    with file_lock(lock_path):
        registry = _scope_map_snapshot(repo_root)
        registry_key, entry = _find_registry_entry(registry, canonical_id)
        now = _now_utc()
        if entry is None:
            canonical_resolved = normalized or canonical_id
            canonical_url = (
                _scope_entry_field(
                    {"canonical_url": remote_url},
                    "canonical_url",
                    canonical_resolved,
                )
                or canonical_resolved
            )
            entry = ScopeRegistryEntry(
                canonical_id=canonical_resolved,
                canonical_url=canonical_url,
                dir_key=_make_dir_key(canonical_resolved, display_name),
                display_name=display_name,
                scope_name=display_name,
                remote_url=remote_url,
                path=path,
                created_at=now,
                updated_at=now,
            )
        else:
            canonical_resolved = normalized or canonical_id or entry.canonical_id
            entry = ScopeRegistryEntry(
                canonical_id=canonical_resolved,
                canonical_url=(entry.canonical_url or canonical_resolved),
                dir_key=entry.dir_key or _make_dir_key(canonical_resolved, display_name),
                display_name=entry.display_name or display_name,
                scope_name=entry.scope_name or display_name,
                remote_url=remote_url or entry.remote_url,
                path=path or entry.path,
                scope_id=entry.scope_id,
                created_at=entry.created_at or now,
                updated_at=now,
            )
            if not entry.dir_key:
                entry = ScopeRegistryEntry(
                    canonical_id=canonical_resolved,
                    canonical_url=entry.canonical_url or canonical_resolved,
                    dir_key=_make_dir_key(canonical_resolved, display_name),
                    display_name=entry.display_name or display_name,
                    scope_name=entry.scope_name or display_name,
                    remote_url=entry.remote_url,
                    path=entry.path,
                    scope_id=entry.scope_id,
                    created_at=entry.created_at or now,
                    updated_at=now,
                )

        if registry_key and registry_key != normalized:
            registry.pop(registry_key, None)
        registry[canonical_resolved] = entry
        payload = {
            "version": 1,
            "updated_at": now,
            "scopes": {
                key: asdict(value)
                for key, value in registry.items()
            },
        }
        _write_json(lock_path, payload)

    scope_dir = get_memory_project_root(repo_root) / entry.dir_key
    scope_dir.mkdir(parents=True, exist_ok=True)
    return scope_dir


def lookup_scope_directory(
    canonical_id: str,
    display_name: str,
    *,
    root: Path | None = None,
    remote_url: str | None = None,
    path: str | None = None,
) -> Path | None:
    """Resolve scope directory without mutating scope-map or filesystem."""
    repo_root = root or get_memory_root()
    project_root = get_memory_project_root(repo_root)
    map_path = get_scope_map_path(repo_root)
    registry: dict[str, ScopeRegistryEntry] = {}

    if map_path.exists():
        with file_lock(map_path):
            registry = _scope_map_snapshot(repo_root)

    _, entry = _find_registry_entry(registry, canonical_id)
    if entry is None and remote_url:
        _, entry = _find_registry_entry(registry, remote_url)
    if entry is None and path:
        _, entry = _find_registry_entry_by_path(registry, path)

    if entry is not None and entry.dir_key:
        return project_root / entry.dir_key

    normalized = normalize_canonical_id(canonical_id) or canonical_id
    if not normalized:
        return None

    suffix = _hash_scope_key(normalized)
    by_hash = sorted(
        path_obj
        for path_obj in project_root.glob(f"*--{suffix}")
        if path_obj.is_dir()
    )
    if by_hash:
        return by_hash[0]

    candidate = _scope_dir_candidate(
        canonical_id=normalized,
        display_name=display_name,
        root=repo_root,
    )
    if candidate.exists():
        return candidate
    return None


def get_section_file(
    canonical_id: str,
    section: str,
    *,
    display_name: str,
    root: Path | None = None,
    remote_url: str | None = None,
    path: str | None = None,
    create: bool = True,
) -> Path:
    """Return canonical file path for one section.

    Also resolves and updates scope registry entry as needed.
    """
    section = _normalize_section(section)
    if create:
        scope_dir = resolve_scope_directory(
            canonical_id,
            display_name=display_name,
            root=root,
            remote_url=remote_url,
            path=path,
        )
        return scope_dir / f"{section}.md"

    scope_dir = lookup_scope_directory(
        canonical_id,
        display_name=display_name,
        root=root,
        remote_url=remote_url,
        path=path,
    )
    if scope_dir is None:
        return _scope_dir_candidate(canonical_id, display_name, root=root) / f"{section}.md"
    return scope_dir / f"{section}.md"


def append_section_entry(
    canonical_id: str,
    section: str,
    content: str,
    *,
    display_name: str,
    memory_type: str = "durable_fact",
    confidence: float = 0.8,
    manual: bool = False,
    source: str = "distilled",
    produced_by_model: str | None = None,
    evidence: list[dict[str, str]] | None = None,
    scope_remote_url: str | None = None,
    scope_path: str | None = None,
    commit: bool = False,
    root: Path | None = None,
) -> Path:
    """Append one canonical entry into a section file."""
    if memory_type not in MEMORY_TYPES:
        raise ValueError(f"Unsupported memory type: {memory_type}")
    section = _normalize_section(section)
    scope_dir = resolve_scope_directory(
        canonical_id,
        display_name=display_name,
        root=root,
        remote_url=scope_remote_url,
        path=scope_path,
    )
    path = scope_dir / f"{section}.md"
    entry = {
        "entry_id": str(uuid.uuid4()),
        "type": memory_type,
        "confidence": float(confidence),
        "manual": bool(manual),
        "source": source,
        "scope_canonical_id": canonical_id,
        "produced_by_model": produced_by_model,
        "evidence": evidence or [],
        "created_at": _now_utc(),
        "updated_at": _now_utc(),
        "content": content.strip(),
    }

    with file_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        serialized = _serialize_entry(entry)
        if existing and not existing.endswith("\n"):
            existing += "\n"
        path.write_text(f"{existing}{serialized}\n", encoding="utf-8")

    if commit:
        _commit_if_possible(path, f"uc memory: add {section} entry")

    return path

def read_section_entries(
    canonical_id: str,
    section: str,
    *,
    display_name: str,
    root: Path | None = None,
    scope_remote_url: str | None = None,
    scope_path: str | None = None,
) -> list[dict[str, Any]]:
    """Read all entries for one section."""
    section = _normalize_section(section)
    path = get_section_file(
        canonical_id=canonical_id,
        section=section,
        display_name=display_name,
        root=root,
        remote_url=scope_remote_url,
        path=scope_path,
        create=False,
    )
    if not path.exists():
        return []

    with file_lock(path):
        raw = path.read_text(encoding="utf-8")
    entries: list[dict[str, Any]] = []
    for doc in yaml.safe_load_all(raw):
        parsed = _parse_entry(doc)
        if parsed is not None:
            entries.append(parsed)
    return entries


def list_scope_sections(
    canonical_id: str,
    *,
    display_name: str,
    root: Path | None = None,
    scope_remote_url: str | None = None,
    scope_path: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Read all section entries for a scope."""
    return {
        section: read_section_entries(
            canonical_id=canonical_id,
            section=section,
            display_name=display_name,
            root=root,
            scope_remote_url=scope_remote_url,
            scope_path=scope_path,
        )
        for section in MEMORY_SECTIONS
    }


def render_section_text(entries: list[dict[str, Any]]) -> str:
    """Render entries for a section into compact markdown."""
    if not entries:
        return ""

    lines: list[str] = []
    for entry in entries:
        entry_type = entry.get("type", "durable_fact")
        confidence = entry.get("confidence")
        manual = "manual" if entry.get("manual") else "distilled"
        body = str(entry.get("content", "")).strip()
        lines.append(f"- [{entry_type}/{manual}][{confidence}] {body}")
    return "\n".join(lines)


def read_scope_file_text(
    canonical_id: str,
    section: str,
    *,
    display_name: str,
    root: Path | None = None,
    scope_remote_url: str | None = None,
    scope_path: str | None = None,
) -> str:
    """Render entire section file as text."""
    section = _normalize_section(section)
    entries = read_section_entries(
        canonical_id=canonical_id,
        section=section,
        display_name=display_name,
        root=root,
        scope_remote_url=scope_remote_url,
        scope_path=scope_path,
    )
    header = f"# {section.replace('_', ' ').title()}"
    if not entries:
        return f"{header}\n"
    return f"{header}\n\n{render_section_text(entries)}\n"


def _run_git(cwd: Path, args: list[str]) -> tuple[bool, str]:
    try:
        cp = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return cp.returncode == 0, (cp.stdout or cp.stderr)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return False, str(exc)


def _ensure_git_repo(root: Path) -> None:
    ok, _ = _run_git(root, ["rev-parse", "--git-dir"])
    if ok:
        return
    _run_git(root, ["init"])


def _commit_if_possible(path: Path, message: str) -> bool:
    current = path.resolve()
    if current.is_file():
        current = current.parent

    repo_root = current
    while repo_root != repo_root.parent:
        if (repo_root / ".git").exists():
            break
        repo_root = repo_root.parent

    if not (repo_root / ".git").exists():
        # Not a git repo; best-effort no-op.
        return False
    _run_git(repo_root, ["add", str(path)])
    ok, _ = _run_git(repo_root, ["commit", "-m", message])
    return ok
