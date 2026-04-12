from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "create_autoresearch_sandbox.py"


def test_create_autoresearch_sandbox_creates_nested_repo(tmp_path: Path) -> None:
    sandbox_root = tmp_path / ".sandboxes"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sandbox-root",
            str(sandbox_root),
            "--name",
            "demo",
        ],
        check=True,
    )

    target = sandbox_root / "demo"
    assert (target / ".git").is_dir()
    assert (target / "README.md").is_file()
    assert (target / "train.py").is_file()
    assert (target / "baselines" / "reference_thresholds.json").is_file()
    assert (target / "best").is_dir()
    assert (target / "data").is_dir()
    assert (target / "runs").is_dir()
    assert not (target / "__pycache__").exists()
