#!/usr/bin/env python3
"""Locate an image directory for bird annotation workflows.

Priority:
1) --images-dir CLI argument
2) environment override (default: BIRD_IMAGES_DIR)
3) common repo-relative directories
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_OVERRIDE_ENV = "BIRD_IMAGES_DIR"
DEFAULT_CANDIDATES = [
    "images",
    "data/images",
    "dataset/images",
    "scraped_images",
    "scraped_images/comsan_100",
]


@dataclass
class DetectionResult:
    directory: pathlib.Path
    image_count: int


def iter_image_files(directory: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def evaluate_candidate(directory: pathlib.Path) -> DetectionResult | None:
    if not directory.exists() or not directory.is_dir():
        return None

    count = sum(1 for _ in iter_image_files(directory))
    if count == 0:
        return None

    return DetectionResult(directory=directory.resolve(), image_count=count)


def detect_images_dir(
    repo_root: pathlib.Path,
    images_dir_arg: str | None = None,
    override_env: str = DEFAULT_OVERRIDE_ENV,
) -> DetectionResult:
    repo_root = repo_root.resolve()

    candidates: list[pathlib.Path] = []

    if images_dir_arg:
        candidates.append(pathlib.Path(images_dir_arg).expanduser())

    env_override = os.getenv(override_env)
    if env_override:
        candidates.append(pathlib.Path(env_override).expanduser())

    for rel_path in DEFAULT_CANDIDATES:
        candidates.append(repo_root / rel_path)

    checked = []
    for candidate in candidates:
        candidate = candidate if candidate.is_absolute() else (repo_root / candidate)
        checked.append(str(candidate))
        result = evaluate_candidate(candidate)
        if result:
            return result

    joined = "\n  - ".join(checked)
    raise FileNotFoundError(
        "Could not find any image directory with supported files. "
        "Checked:\n  - "
        f"{joined}"
    )


def maybe_update_symlink(repo_root: pathlib.Path, target: pathlib.Path, link_name: str) -> None:
    symlink_path = repo_root / link_name
    target = target.resolve()

    if symlink_path.exists() and not symlink_path.is_symlink():
        return

    desired = os.path.relpath(target, start=repo_root)

    if symlink_path.is_symlink():
        current = os.readlink(symlink_path)
        if current == desired:
            return
        symlink_path.unlink()
    elif symlink_path.exists():
        return

    symlink_path.symlink_to(desired)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect the repo image directory.")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (default: inferred from this script location)",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Explicit image directory override (highest priority)",
    )
    parser.add_argument(
        "--override-env",
        default=DEFAULT_OVERRIDE_ENV,
        help=f"Environment variable override name (default: {DEFAULT_OVERRIDE_ENV})",
    )
    parser.add_argument(
        "--format",
        choices=["human", "json", "path", "env"],
        default="human",
        help="Output format",
    )
    parser.add_argument(
        "--create-images-symlink",
        action="store_true",
        help="Create/update ./images symlink to point at detected directory",
    )
    parser.add_argument(
        "--symlink-name",
        default="images",
        help="Symlink name in repo root when --create-images-symlink is used",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = (
        pathlib.Path(args.repo_root).expanduser()
        if args.repo_root
        else pathlib.Path(__file__).resolve().parents[2]
    )

    try:
        result = detect_images_dir(
            repo_root=repo_root,
            images_dir_arg=args.images_dir,
            override_env=args.override_env,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.create_images_symlink:
        maybe_update_symlink(repo_root, result.directory, args.symlink_name)

    if args.format == "json":
        print(
            json.dumps(
                {"image_dir": str(result.directory), "image_count": result.image_count},
                indent=2,
            )
        )
    elif args.format == "path":
        print(str(result.directory))
    elif args.format == "env":
        print(f"IMAGE_DIR={result.directory}")
        print(f"IMAGE_COUNT={result.image_count}")
    else:
        print(f"Selected image directory: {result.directory}")
        print(f"Image files found: {result.image_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
