"""Purpose: Bootstrap the monorepo source roots and hand off to the umbrella CLI"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

SOURCE_RELATIVE_DIRS = (
    "ops/src",
    "shared/birdsys_core/src",
    "projects/labelstudio/src",
    "projects/datasets/src",
    "projects/ml_backend/src",
    "projects/ml_experiments/src",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def workspace_source_paths(root: Path | None = None) -> list[str]:
    base = root or repo_root()
    return [str(base / relative_dir) for relative_dir in SOURCE_RELATIVE_DIRS]


def extend_workspace_sys_path(root: Path | None = None) -> None:
    existing = set(sys.path)
    discovered = [path for path in workspace_source_paths(root) if Path(path).is_dir()]
    if not discovered:
        raise RuntimeError("BirdSys workspace source directories are not available.")

    for path in reversed(discovered):
        if path not in existing:
            sys.path.insert(0, path)
            existing.add(path)


def main(argv: Sequence[str] | None = None) -> int:
    extend_workspace_sys_path()
    from birdsys.cli.main import main as workspace_main

    return workspace_main(argv)
