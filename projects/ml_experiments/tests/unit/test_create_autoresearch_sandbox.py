"""Purpose: Verify sandbox creation copies the tracked template into a new run workspace"""
from __future__ import annotations

from pathlib import Path

from birdsys.ml_experiments.create_autoresearch_sandbox import main

def test_create_autoresearch_sandbox_creates_nested_repo(tmp_path: Path) -> None:
    sandbox_root = tmp_path / ".sandboxes"
    assert main(
        [
            "--sandbox-root",
            str(sandbox_root),
            "--name",
            "demo",
        ],
    ) == 0

    target = sandbox_root / "demo"
    assert (target / ".git").is_dir()
    assert (target / "README.md").is_file()
    assert (target / "train.py").is_file()
    assert (target / "baselines" / "reference_thresholds.json").is_file()
    assert (target / "best").is_dir()
    assert (target / "data").is_dir()
    assert (target / "runs").is_dir()
    assert not (target / "__pycache__").exists()
