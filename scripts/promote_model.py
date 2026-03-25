#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a model artifact into the served model slot.")
    parser.add_argument("--source", required=True, help="Path to the candidate model artifact")
    parser.add_argument("--served-dir", required=True, help="Directory containing current/ and releases/")
    parser.add_argument("--label", default="manual", help="Promotion label written to metadata")
    parser.add_argument("--notes", default="", help="Optional promotion notes")
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    served_dir = Path(args.served_dir).expanduser().resolve()
    release_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release_dir = served_dir / "releases" / release_id
    current_dir = served_dir / "current"
    release_dir.mkdir(parents=True, exist_ok=False)
    current_dir.mkdir(parents=True, exist_ok=True)

    release_weights = release_dir / "weights.pt"
    current_weights = current_dir / "weights.pt"
    shutil.copy2(source, release_weights)
    shutil.copy2(source, current_weights)

    metadata = {
        "release_id": release_id,
        "label": args.label,
        "notes": args.notes or None,
        "source_path": str(source),
        "source_filename": source.name,
        "source_sha256": sha256sum(source),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "release_weights": str(release_weights),
        "current_weights": str(current_weights),
    }
    write_metadata(release_dir / "promotion.json", metadata)
    write_metadata(current_dir / "promotion.json", metadata)

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
