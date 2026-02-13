"""Tests for durable memory repository helpers."""

from __future__ import annotations

from universal_context.memory_repo import (
    append_section_entry,
    bootstrap_memory_repo,
    get_scope_map_path,
    get_section_file,
    read_section_entries,
)


def test_read_paths_do_not_mutate_scope_map(tmp_path):
    memory_root = bootstrap_memory_repo(tmp_path / "memory", init_git=False)
    canonical_id = "https://github.com/acme/repo.git"

    append_section_entry(
        canonical_id=canonical_id,
        section="state",
        content="Durable fact",
        display_name="repo",
        memory_type="durable_fact",
        root=memory_root,
    )

    scope_map = get_scope_map_path(memory_root)
    before = scope_map.read_text(encoding="utf-8")

    entries = read_section_entries(
        canonical_id=canonical_id,
        section="state",
        display_name="repo",
        root=memory_root,
    )
    after = scope_map.read_text(encoding="utf-8")

    assert len(entries) == 1
    assert entries[0]["content"] == "Durable fact"
    assert before == after


def test_read_lookup_is_stable_across_display_name_changes(tmp_path):
    memory_root = bootstrap_memory_repo(tmp_path / "memory", init_git=False)
    canonical_id = "github.com/acme/repo"

    append_section_entry(
        canonical_id=canonical_id,
        section="state",
        content="Stable entry",
        display_name="repo-old-name",
        memory_type="durable_fact",
        root=memory_root,
    )

    entries = read_section_entries(
        canonical_id=canonical_id,
        section="state",
        display_name="repo-new-name",
        root=memory_root,
    )

    assert len(entries) == 1
    assert entries[0]["content"] == "Stable entry"


def test_get_section_file_read_only_does_not_create_missing_scope_dir(tmp_path):
    memory_root = bootstrap_memory_repo(tmp_path / "memory", init_git=False)

    section_path = get_section_file(
        canonical_id="github.com/acme/new-repo",
        section="state",
        display_name="new-repo",
        root=memory_root,
        create=False,
    )

    assert section_path.name == "state.md"
    assert not section_path.parent.exists()
