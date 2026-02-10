"""Tests for git-aware scope identity resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from universal_context.git import (
    _normalize_git_url,
    get_current_branch,
    get_head_sha,
    get_merge_base,
    resolve_canonical_id,
)

# ============================================================
# URL NORMALIZATION
# ============================================================


class TestNormalizeGitUrl:
    def test_ssh_shorthand(self):
        assert _normalize_git_url("git@github.com:user/repo.git") == "github.com/user/repo"

    def test_ssh_shorthand_no_suffix(self):
        assert _normalize_git_url("git@github.com:user/repo") == "github.com/user/repo"

    def test_https(self):
        assert _normalize_git_url("https://github.com/user/repo.git") == "github.com/user/repo"

    def test_https_no_suffix(self):
        assert _normalize_git_url("https://github.com/user/repo") == "github.com/user/repo"

    def test_https_with_credentials(self):
        assert (
            _normalize_git_url("https://user:token@github.com/user/repo.git")
            == "github.com/user/repo"
        )

    def test_ssh_protocol(self):
        assert (
            _normalize_git_url("ssh://git@github.com/user/repo.git")
            == "github.com/user/repo"
        )

    def test_git_protocol(self):
        assert _normalize_git_url("git://github.com/user/repo.git") == "github.com/user/repo"

    def test_file_url_returns_none(self):
        assert _normalize_git_url("file:///local/path/repo.git") is None

    def test_trailing_slash(self):
        assert _normalize_git_url("https://github.com/user/repo.git/") == "github.com/user/repo"

    def test_gitlab(self):
        assert (
            _normalize_git_url("git@gitlab.com:org/sub/repo.git")
            == "gitlab.com/org/sub/repo"
        )

    def test_custom_host(self):
        assert (
            _normalize_git_url("https://git.company.com/team/project.git")
            == "git.company.com/team/project"
        )

    def test_ssh_with_port(self):
        assert (
            _normalize_git_url("ssh://git@github.com:22/user/repo.git")
            == "github.com/user/repo"
        )

    def test_whitespace_stripped(self):
        assert (
            _normalize_git_url("  git@github.com:user/repo.git  \n")
            == "github.com/user/repo"
        )

    def test_empty_string(self):
        assert _normalize_git_url("") is None

    def test_nonsense(self):
        assert _normalize_git_url("not a url at all") is None


# ============================================================
# RESOLVE CANONICAL ID
# ============================================================


class TestResolveCanonicalId:
    def test_git_repo_with_remote(self, tmp_path):
        """A git repo with an origin remote → normalized URL."""
        def mock_run_git(args, cwd):
            if args == ["remote", "get-url", "origin"]:
                return "git@github.com:user/my-project.git"
            return None

        with patch("universal_context.git._run_git", side_effect=mock_run_git):
            result = resolve_canonical_id(tmp_path)

        assert result == "github.com/user/my-project"

    def test_git_repo_without_remote(self, tmp_path):
        """A git repo without remote → git-local:// with common dir."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        def mock_run_git(args, cwd):
            if args == ["remote", "get-url", "origin"]:
                return None
            if args == ["rev-parse", "--git-common-dir"]:
                return ".git"
            return None

        with patch("universal_context.git._run_git", side_effect=mock_run_git):
            result = resolve_canonical_id(tmp_path)

        assert result.startswith("git-local://")
        assert str(tmp_path.resolve()) in result

    def test_non_git_directory(self, tmp_path):
        """A non-git directory → path:// fallback."""
        def mock_run_git(args, cwd):
            return None

        with patch("universal_context.git._run_git", side_effect=mock_run_git):
            result = resolve_canonical_id(tmp_path)

        assert result == f"path://{tmp_path.resolve()}"

    def test_file_url_remote_falls_through(self, tmp_path):
        """A git repo with file:// remote falls through to git-common-dir."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        def mock_run_git(args, cwd):
            if args == ["remote", "get-url", "origin"]:
                return "file:///some/local/bare.git"
            if args == ["rev-parse", "--git-common-dir"]:
                return ".git"
            return None

        with patch("universal_context.git._run_git", side_effect=mock_run_git):
            result = resolve_canonical_id(tmp_path)

        assert result.startswith("git-local://")

    def test_cross_worktree_same_canonical_id(self, tmp_path):
        """Two worktrees sharing the same remote → same canonical_id."""
        wt1 = tmp_path / "worktree1"
        wt2 = tmp_path / "worktree2"
        wt1.mkdir()
        wt2.mkdir()

        def mock_run_git(args, cwd):
            if args == ["remote", "get-url", "origin"]:
                return "git@github.com:user/shared-repo.git"
            return None

        with patch("universal_context.git._run_git", side_effect=mock_run_git):
            id1 = resolve_canonical_id(wt1)
            id2 = resolve_canonical_id(wt2)

        assert id1 == id2 == "github.com/user/shared-repo"

    def test_never_returns_none(self, tmp_path):
        """resolve_canonical_id always returns a string, never None."""
        def mock_run_git(args, cwd):
            return None

        with patch("universal_context.git._run_git", side_effect=mock_run_git):
            result = resolve_canonical_id(tmp_path)

        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================
# GET CURRENT BRANCH
# ============================================================


class TestGetCurrentBranch:
    def test_normal_branch(self, tmp_path):
        """Returns branch name on a normal checkout."""
        with patch("universal_context.git._run_git", return_value="main"):
            assert get_current_branch(tmp_path) == "main"

    def test_feature_branch(self, tmp_path):
        with patch("universal_context.git._run_git", return_value="feature/git-scope"):
            assert get_current_branch(tmp_path) == "feature/git-scope"

    def test_detached_head(self, tmp_path):
        """Returns None for detached HEAD (git branch --show-current outputs empty)."""
        with patch("universal_context.git._run_git", return_value=None):
            assert get_current_branch(tmp_path) is None

    def test_non_git_dir(self, tmp_path):
        """Returns None for non-git directory."""
        with patch("universal_context.git._run_git", return_value=None):
            assert get_current_branch(tmp_path) is None


# ============================================================
# GET HEAD SHA
# ============================================================


class TestGetHeadSha:
    def test_returns_sha(self, tmp_path):
        with patch("universal_context.git._run_git", return_value="abc1234"):
            assert get_head_sha(tmp_path) == "abc1234"

    def test_non_git_returns_none(self, tmp_path):
        with patch("universal_context.git._run_git", return_value=None):
            assert get_head_sha(tmp_path) is None


# ============================================================
# GET MERGE BASE
# ============================================================


class TestGetMergeBase:
    def test_returns_merge_base_sha(self, tmp_path):
        with patch("universal_context.git._run_git", return_value="deadbeef123"):
            result = get_merge_base(tmp_path, "feature/x", "main")
            assert result == "deadbeef123"

    def test_no_common_ancestor_returns_none(self, tmp_path):
        with patch("universal_context.git._run_git", return_value=None):
            assert get_merge_base(tmp_path, "orphan", "main") is None


# ============================================================
# INTEGRATION (uses real git)
# ============================================================


class TestGitIntegration:
    """Tests that use the real git binary — only run if this repo is a git repo."""

    def test_resolve_canonical_id_on_real_repo(self):
        """Running from this repo should give a real canonical_id."""
        result = resolve_canonical_id(Path.cwd())
        assert isinstance(result, str)
        assert len(result) > 0
        # Should not be a bare path:// if we're in a git repo
        if (Path.cwd() / ".git").exists() or _is_git_dir(Path.cwd()):
            assert not result.startswith("path://")

    def test_get_current_branch_on_real_repo(self):
        """Should return a branch name or None (detached)."""
        result = get_current_branch(Path.cwd())
        # Could be None if detached, otherwise a string
        assert result is None or isinstance(result, str)

    def test_get_head_sha_on_real_repo(self):
        """Should return a short SHA string."""
        result = get_head_sha(Path.cwd())
        if _is_git_dir(Path.cwd()):
            assert result is not None
            assert len(result) >= 7  # short SHA


def _is_git_dir(path: Path) -> bool:
    """Check if path is inside a git repo."""
    try:
        import subprocess

        r = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=path, capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False
