#!/usr/bin/env python3
"""Purpose: Promote model artifacts into the served release layout used by the ML backend"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from birdsys.core import PromotionMetadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a model artifact into the served model slot.")
    parser.add_argument("--source", required=True, help="Path to the candidate model artifact file or directory")
    parser.add_argument("--served-dir", required=True, help="Directory containing current/ and releases/")
    parser.add_argument("--artifact-name", default="weights.pt", help="Filename to write for file artifacts")
    parser.add_argument("--label", default="manual", help="Promotion label written to metadata")
    parser.add_argument("--notes", default="", help="Optional promotion notes")
    return parser.parse_args(argv)


def sha256sum_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256sum_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        rel = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(sha256sum_file(child).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clear_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_artifact(source: Path, target_dir: Path, artifact_name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        for child in source.iterdir():
            target_path = target_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target_path)
            else:
                shutil.copy2(child, target_path)
        return target_dir
    target_path = target_dir / artifact_name
    shutil.copy2(source, target_path)
    return target_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    served_dir = Path(args.served_dir).expanduser().resolve()
    release_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release_dir = served_dir / "releases" / release_id
    current_dir = served_dir / "current"
    release_dir.mkdir(parents=True, exist_ok=False)
    current_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(current_dir)

    artifact_kind = "directory" if source.is_dir() else "file"
    release_artifact = copy_artifact(source, release_dir, args.artifact_name)
    current_artifact = copy_artifact(source, current_dir, args.artifact_name)
    source_sha256 = sha256sum_tree(source) if source.is_dir() else sha256sum_file(source)

    metadata = PromotionMetadata(
        release_id=release_id,
        promoted_at=datetime.now(timezone.utc).isoformat(),
        label=args.label,
        source_path=str(source),
        source_kind=artifact_kind,
        source_sha256=source_sha256,
        active_artifact=str(current_artifact),
        notes=args.notes,
    ).to_dict()
    metadata.update(
        {
            "source_filename": source.name,
            "artifact_kind": artifact_kind,
            "release_artifact": str(release_artifact),
            "current_artifact": str(current_artifact),
            "release_weights": str(release_artifact) if artifact_kind == "file" else None,
            "current_weights": str(current_artifact) if artifact_kind == "file" else None,
        }
    )
    write_metadata(release_dir / "promotion.json", metadata)
    write_metadata(current_dir / "promotion.json", metadata)

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
