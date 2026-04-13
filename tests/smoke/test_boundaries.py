from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def test_no_legacy_root_code_files_remain() -> None:
    legacy_roots = [
        REPO_ROOT / "scripts",
        REPO_ROOT / "services",
        REPO_ROOT / "src",
        REPO_ROOT / "deploy",
        REPO_ROOT / "config",
        REPO_ROOT / "sql",
        REPO_ROOT / "sandbox_templates",
        REPO_ROOT / "labelstudio",
    ]
    for root in legacy_roots:
        assert not _iter_python_files(root), root


def test_subproject_import_boundaries() -> None:
    source_roots = {
        "labelstudio": REPO_ROOT / "projects" / "labelstudio" / "src",
        "datasets": REPO_ROOT / "projects" / "datasets" / "src",
        "ml_backend": REPO_ROOT / "projects" / "ml_backend" / "src",
    }
    forbidden = {
        "labelstudio": ("birdsys.ml_backend", "birdsys.ml_experiments"),
        "datasets": ("birdsys.ml_backend", "birdsys.ml_experiments"),
        "ml_backend": ("birdsys.ml_experiments",),
    }
    for name, root in source_roots.items():
        for path in _iter_python_files(root):
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden[name]:
                assert pattern not in text, f"{path} imports {pattern}"

