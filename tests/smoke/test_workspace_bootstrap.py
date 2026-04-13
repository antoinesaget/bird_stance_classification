from __future__ import annotations

from pathlib import Path

from birdsys_workspace.cli import SOURCE_RELATIVE_DIRS, workspace_source_paths


def test_workspace_bootstrap_paths_cover_all_source_roots() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    discovered = workspace_source_paths(repo_root)

    assert len(discovered) == len(SOURCE_RELATIVE_DIRS)
    for relative_dir, discovered_path in zip(SOURCE_RELATIVE_DIRS, discovered, strict=True):
        expected = repo_root / relative_dir
        assert discovered_path == str(expected)
        assert expected.is_dir()
