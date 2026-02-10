"""Universal Context — Operational memory engine for AI agents."""

import hashlib
from pathlib import Path

__version__ = "0.1.0"

_DEFAULT_UC_HOME = Path.home() / ".uc"


def get_uc_home() -> Path:
    """Get the global UC home directory (~/.uc/)."""
    return Path.home() / ".uc"


def get_uc_project_dir(project_root: Path) -> Path:
    """Get the per-project UC data directory.

    Resolution order:
    1. <project_root>/.uc-config — optional override file containing a custom path
    2. Default: ~/.uc/projects/<project_name>-<hash>/

    Args:
        project_root: Root directory of the project.

    Returns:
        Path to the project's UC data directory.
    """
    project_root = Path(project_root).absolute()

    override = project_root / ".uc-config"
    try:
        if override.exists():
            configured = override.read_text(encoding="utf-8").strip()
            if configured:
                return Path(configured).expanduser()
    except Exception:
        pass

    digest = hashlib.sha1(str(project_root).encode("utf-8")).hexdigest()[:10]
    return get_uc_home() / "projects" / f"{project_root.name}-{digest}"
