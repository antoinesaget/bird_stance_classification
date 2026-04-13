#!/usr/bin/env python3
"""Purpose: Sync species-scoped source and Label Studio data from TrueNAS to a local host."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from birdsys.core import default_data_home, default_species_slug, ensure_layout


SYNC_TIERS = (
    "originals",
    "labelstudio/images_compressed",
    "labelstudio/imports",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync one species dataset from TrueNAS or a local source root")
    parser.add_argument("--species-slug", default=default_species_slug())
    parser.add_argument("--source-host", default="truenas", help="Remote host alias, or 'local' for a local source path")
    parser.add_argument("--source-data-home", default="/mnt/tank/media/birds")
    parser.add_argument("--dest-data-home", default=str(default_data_home()))
    parser.add_argument("--delete", action="store_true", help="Mirror mode: delete destination files missing from source")
    return parser.parse_args(argv)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    counts = manifest["counts"]
    lines = [
        f"# Species Sync Report: {manifest['species_slug']}",
        "",
        f"- Source host: `{manifest['source_host']}`",
        f"- Source root: `{manifest['source_species_root']}`",
        f"- Destination root: `{manifest['dest_species_root']}`",
        f"- Delete mode: `{manifest['delete']}`",
        "",
        "## Counts",
        "",
    ]
    for tier, payload in counts.items():
        lines.append(f"- `{tier}`: `{payload['files']}` files, `{payload['bytes']}` bytes")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Compression profiles: `{manifest['compression_profiles']}`",
            f"- Import manifests present: `{manifest['import_manifests_present']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def walk_counts(root: Path) -> dict[str, int]:
    if not root.exists():
        return {"files": 0, "bytes": 0}
    files = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        files += 1
        total_bytes += int(path.stat().st_size)
    return {"files": files, "bytes": total_bytes}


def sync_tree_local(*, source: Path, dest: Path, delete: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    source_paths: set[Path] = set()
    if source.exists():
        for path in source.rglob("*"):
            rel = path.relative_to(source)
            source_paths.add(rel)
            target = dest / rel
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    if not delete:
        return
    for path in sorted(dest.rglob("*"), reverse=True):
        rel = path.relative_to(dest)
        if rel in source_paths:
            continue
        if path.is_file():
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def sync_tree_rsync(*, source_host: str, source: Path, dest: Path, delete: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a"]
    if delete:
        cmd.append("--delete")
    cmd.extend([f"{source_host}:{source.as_posix().rstrip('/')}/", f"{dest.as_posix().rstrip('/')}/"])
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    layout = ensure_layout(Path(args.dest_data_home), args.species_slug)
    source_species_root = Path(args.source_data_home).expanduser() / args.species_slug
    dest_species_root = layout.root

    for tier in SYNC_TIERS:
        source = source_species_root / tier
        dest = dest_species_root / tier
        if args.source_host == "local":
            sync_tree_local(source=source, dest=dest, delete=bool(args.delete))
        else:
            sync_tree_rsync(source_host=args.source_host, source=source, dest=dest, delete=bool(args.delete))

    counts = {
        tier: walk_counts(dest_species_root / tier)
        for tier in SYNC_TIERS
    }
    compressed_root = layout.labelstudio_images_compressed
    compression_profiles = sorted(path.name for path in compressed_root.iterdir() if path.is_dir()) if compressed_root.exists() else []
    import_manifests_present = bool(list(layout.labelstudio_imports.glob("*.manifest.json")))
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "species_slug": layout.species_slug,
        "source_host": args.source_host,
        "source_species_root": str(source_species_root),
        "dest_species_root": str(dest_species_root),
        "delete": bool(args.delete),
        "synced_roots": [str(dest_species_root / tier) for tier in SYNC_TIERS],
        "counts": counts,
        "compression_profiles": compression_profiles,
        "import_manifests_present": import_manifests_present,
    }

    manifest_path = layout.metadata / "sync_manifest.json"
    report_path = layout.metadata / "sync_report.md"
    write_json(manifest_path, manifest)
    write_report(report_path, manifest)

    print(f"species_slug={layout.species_slug}")
    print(f"dest_species_root={dest_species_root}")
    print(f"sync_manifest={manifest_path}")
    print(f"sync_report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
