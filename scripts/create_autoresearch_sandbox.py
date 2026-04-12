#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_DIR = REPO_ROOT / "sandbox_templates" / "model_b_autoresearch_p7_v1"
DEFAULT_SANDBOX_ROOT = REPO_ROOT / ".sandboxes"
RUNTIME_DIRS = ("best", "data", "runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a nested-git autoresearch sandbox from the tracked Model B template"
    )
    parser.add_argument("--name", required=True, help="Sandbox directory name created under --sandbox-root")
    parser.add_argument(
        "--sandbox-root",
        default=str(DEFAULT_SANDBOX_ROOT),
        help="Parent directory for created sandboxes",
    )
    parser.add_argument(
        "--template-dir",
        default=str(DEFAULT_TEMPLATE_DIR),
        help="Tracked template directory to copy from",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing sandbox directory")
    parser.add_argument(
        "--init-git",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the sandbox as a nested git repository with an initial commit",
    )
    parser.add_argument(
        "--prepare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run prepare.py after the template is copied",
    )
    return parser.parse_args()


def require_valid_name(raw_name: str) -> str:
    name = raw_name.strip()
    if not name:
        raise ValueError("sandbox name must be non-empty")
    path = Path(name)
    if path.name != name or name in {".", ".."}:
        raise ValueError(f"invalid sandbox name: {raw_name!r}")
    return name


def ensure_template(template_dir: Path) -> None:
    required = [
        template_dir / "README.md",
        template_dir / "program.md",
        template_dir / "common.py",
        template_dir / "prepare.py",
        template_dir / "train.py",
        template_dir / ".gitignore",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"template missing required files: {missing}")


def copy_template(*, template_dir: Path, target_dir: Path, force: bool) -> None:
    if target_dir.exists():
        if not force:
            raise FileExistsError(target_dir)
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        template_dir,
        target_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache", ".ruff_cache"),
    )
    for dirname in RUNTIME_DIRS:
        (target_dir / dirname).mkdir(parents=True, exist_ok=True)


def init_git_repo(target_dir: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=target_dir, check=True)
    subprocess.run(["git", "add", "."], cwd=target_dir, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=BirdSys Sandbox",
            "-c",
            "user.email=birdsys-sandbox@local",
            "commit",
            "-qm",
            "Initialize sandbox from tracked template",
        ],
        cwd=target_dir,
        check=True,
    )


def repo_python() -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def run_prepare(target_dir: Path) -> None:
    subprocess.run([repo_python(), "prepare.py"], cwd=target_dir, check=True)


def main() -> int:
    args = parse_args()
    name = require_valid_name(args.name)
    template_dir = Path(args.template_dir).expanduser().resolve()
    sandbox_root = Path(args.sandbox_root).expanduser().resolve()
    target_dir = sandbox_root / name

    ensure_template(template_dir)
    copy_template(template_dir=template_dir, target_dir=target_dir, force=args.force)

    if args.init_git:
        init_git_repo(target_dir)

    if args.prepare:
        run_prepare(target_dir)

    print(f"created_sandbox={target_dir}")
    print(f"template_dir={template_dir}")
    print("next_step=cd {0} && {1} prepare.py".format(target_dir, repo_python()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
