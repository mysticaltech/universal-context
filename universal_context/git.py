"""Git-aware scope identity resolution.

Provides canonical identity for project scopes based on git remote URLs,
git-common-dir for local repos, or filesystem path as fallback.

Priority chain:
1. Normalized git remote URL (e.g. github.com/user/repo)
2. git-common-dir for local repos without remote (git-local://{path})
3. Filesystem path for non-git directories (path://{path})
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def resolve_canonical_id(path: Path) -> str:
    """Resolve a filesystem path to a canonical scope identity string.

    Always returns a string — never None. The result is suitable as a
    UNIQUE key for scope deduplication across worktrees and clones.
    """
    resolved = path.resolve()

    # 1. Try git remote origin URL
    remote_url = _run_git(["remote", "get-url", "origin"], cwd=resolved)
    if remote_url:
        normalized = _normalize_git_url(remote_url)
        if normalized:
            return normalized

    # 2. Try git-common-dir (shared .git dir across worktrees)
    common_dir = _run_git(["rev-parse", "--git-common-dir"], cwd=resolved)
    if common_dir:
        # Resolve relative paths (git returns relative for main worktree)
        common_path = (resolved / common_dir).resolve()
        return f"git-local://{common_path}"

    # 3. Fallback to filesystem path
    return f"path://{resolved}"


def _normalize_git_url(url: str) -> str | None:
    """Normalize a git remote URL to a canonical form.

    Returns host/user/repo (no protocol, no credentials, no .git suffix).
    Returns None for file:// URLs (fall through to git-common-dir).

    Examples:
        git@github.com:user/repo.git      -> github.com/user/repo
        https://user:tok@github.com/u/r   -> github.com/u/r
        ssh://git@github.com/user/repo.git -> github.com/user/repo
        file:///local/path                 -> None
    """
    url = url.strip()

    # file:// URLs → fall through to git-common-dir
    if url.startswith("file://"):
        return None

    # SSH shorthand: git@host:user/repo.git
    m = re.match(r"^[\w.-]+@([\w.-]+):(.+)$", url)
    if m:
        host = m.group(1)
        path = m.group(2)
        path = _strip_git_suffix(path)
        return f"{host}/{path}"

    # Protocol URLs: https://, ssh://, git://
    m = re.match(r"^(?:https?|ssh|git)://(?:[^@]+@)?([\w.-]+)(?::\d+)?/(.+)$", url)
    if m:
        host = m.group(1)
        path = m.group(2)
        path = _strip_git_suffix(path)
        return f"{host}/{path}"

    return None


def _strip_git_suffix(path: str) -> str:
    """Remove trailing .git and slashes."""
    path = path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path.rstrip("/")


def get_current_branch(path: Path) -> str | None:
    """Get the current git branch name, or None for detached HEAD / non-git."""
    result = _run_git(["branch", "--show-current"], cwd=path.resolve())
    return result if result else None


def get_head_sha(path: Path) -> str | None:
    """Get the current HEAD commit SHA (short form), or None for non-git."""
    return _run_git(["rev-parse", "--short", "HEAD"], cwd=path.resolve())


def get_merge_base(path: Path, branch: str, target: str = "main") -> str | None:
    """Get the merge-base commit between branch and target.

    Useful for determining where a feature branch diverged from main,
    which helps attribute work to the correct lineage after merges.
    Returns None if either ref doesn't exist or not a git repo.
    """
    return _run_git(["merge-base", branch, target], cwd=path.resolve())


def _run_git(args: list[str], cwd: Path) -> str | None:
    """Run a git command and return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            return output if output else None
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
