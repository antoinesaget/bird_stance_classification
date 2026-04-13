#!/usr/bin/env python3
"""Purpose: Build species-aware local-files batches and manifests for Label Studio."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Sequence
from pathlib import Path

from birdsys.core import (
    LabelStudioBatchSummary,
    default_data_home,
    default_species_slug,
    ensure_layout,
    normalize_relative_path,
    resolve_species_relative_path,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Label Studio local-files import batch with a JPEG mirror."
    )
    parser.add_argument("--data-home", default=str(default_data_home()))
    parser.add_argument("--species-slug", default=default_species_slug())
    parser.add_argument(
        "--source-relative-root",
        default="originals",
        help="Source image root under the species root.",
    )
    parser.add_argument(
        "--mirror-relative-root",
        default="",
        help="Optional mirror root under the species root. Defaults to labelstudio/images_compressed/<compression-profile>.",
    )
    parser.add_argument(
        "--import-relative-root",
        default="labelstudio/imports",
        help="Import artifact root under the species root.",
    )
    parser.add_argument("--compression-profile", default="q60")
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=60)
    parser.add_argument(
        "--ls-local-files-prefix",
        default="/data/local-files/?d=",
        help="Label Studio local-files URL prefix.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively enumerate source images under source-relative-root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def iter_images(source_root: Path, *, recursive: bool) -> list[Path]:
    iterator = source_root.rglob("*") if recursive else source_root.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_images(
    images: list[Path],
    *,
    count: int,
    sample_mode: str,
    seed: int,
) -> list[Path]:
    if count <= 0 or count >= len(images):
        if sample_mode == "random":
            shuffled = list(images)
            random.Random(seed).shuffle(shuffled)
            return shuffled
        return list(images)

    if sample_mode == "first":
        return images[:count]

    return random.Random(seed).sample(images, count)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.jpeg_quality < 1 or args.jpeg_quality > 95:
        raise ValueError("--jpeg-quality must be in [1, 95]")

    try:
        from PIL import Image, ImageOps
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required. Install dependencies with `make bootstrap` or `pip install -e .[dev]`.") from exc

    layout = ensure_layout(Path(args.data_home), args.species_slug)
    source_relative_root = normalize_relative_path(args.source_relative_root, field_name="source-relative-root")
    import_relative_root = normalize_relative_path(args.import_relative_root, field_name="import-relative-root")
    compression_profile = args.compression_profile.strip()
    if not compression_profile:
        raise ValueError("--compression-profile cannot be empty")
    mirror_relative_root = (
        normalize_relative_path(args.mirror_relative_root, field_name="mirror-relative-root")
        if args.mirror_relative_root.strip()
        else f"labelstudio/images_compressed/{compression_profile}"
    )

    source_root = resolve_species_relative_path(layout.root, source_relative_root, field_name="source-relative-root")
    if not source_root.exists():
        raise FileNotFoundError(source_root)

    mirror_root = resolve_species_relative_path(layout.root, mirror_relative_root, field_name="mirror-relative-root")
    import_root = resolve_species_relative_path(layout.root, import_relative_root, field_name="import-relative-root")
    mirror_root.mkdir(parents=True, exist_ok=True)
    import_root.mkdir(parents=True, exist_ok=True)

    all_images = iter_images(source_root, recursive=args.recursive)
    if not all_images:
        raise RuntimeError(f"No images found in {source_root}")

    selected = select_images(
        all_images,
        count=args.sample_size,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

    rows: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    seen_urls: set[str] = set()

    for sample_index, source_path in enumerate(selected):
        source_rel_from_root = source_path.relative_to(layout.root)
        source_rel_from_source = source_path.relative_to(source_root)
        target_rel_from_mirror = source_rel_from_source.with_suffix(".jpg")
        target_path = mirror_root / target_rel_from_mirror
        ensure_parent(target_path)

        if target_path.exists() and not args.overwrite:
            pass
        else:
            with Image.open(source_path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                rgb.save(
                    target_path,
                    format="JPEG",
                    quality=args.jpeg_quality,
                    optimize=True,
                    progressive=True,
                    subsampling=0,
                )

        served_rel_from_root = target_path.relative_to(layout.root)
        localfiles_relpath = Path("birds") / layout.species_slug / served_rel_from_root
        served_url = f"{args.ls_local_files_prefix}{localfiles_relpath.as_posix()}"
        if served_url in seen_urls:
            raise ValueError(f"duplicate served url generated: {served_url}")
        seen_urls.add(served_url)

        row = {
            "sample_index": sample_index,
            "sample_seed": args.seed,
            "batch_name": args.batch_name,
            "species_slug": layout.species_slug,
            "compression_profile": compression_profile,
            "image_id": source_path.stem,
            "source_filename": source_path.name,
            "original_relative_path": source_rel_from_root.as_posix(),
            "served_relative_path": served_rel_from_root.as_posix(),
            "served_localfiles_relative_path": localfiles_relpath.as_posix(),
            "served_url": served_url,
            "original_bytes": int(source_path.stat().st_size),
            "served_bytes": int(target_path.stat().st_size),
            "jpeg_quality": args.jpeg_quality,
        }
        rows.append(row)
        tasks.append(
            {
                "data": {
                    "image": served_url,
                    "image_id": source_path.stem,
                    "source_filename": source_path.name,
                },
                "meta": {
                    "sample_index": sample_index,
                    "sample_seed": args.seed,
                    "batch_name": args.batch_name,
                    "species_slug": layout.species_slug,
                    "compression_profile": compression_profile,
                    "original_relative_path": row["original_relative_path"],
                    "served_relative_path": row["served_relative_path"],
                },
            }
        )

    manifest_json = import_root / f"{args.batch_name}.manifest.json"
    manifest_csv = import_root / f"{args.batch_name}.manifest.csv"
    tasks_json = import_root / f"{args.batch_name}.tasks.json"
    summary_json = import_root / f"{args.batch_name}.summary.json"

    manifest_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    tasks_json.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")

    total_original = sum(int(row["original_bytes"]) for row in rows)
    total_served = sum(int(row["served_bytes"]) for row in rows)
    summary = LabelStudioBatchSummary(
        batch_name=args.batch_name,
        data_home=str(layout.data_home),
        species_slug=layout.species_slug,
        species_root=str(layout.root),
        source_root=str(source_root),
        mirror_root=str(mirror_root),
        import_root=str(import_root),
        compression_profile=compression_profile,
        images_available=len(all_images),
        images_selected=len(selected),
        jpeg_quality=args.jpeg_quality,
        total_original_bytes=total_original,
        total_served_bytes=total_served,
        size_ratio=(total_served / total_original) if total_original else 0.0,
    ).to_dict()
    summary.update(
        {
            "sample_mode": args.sample_mode,
            "sample_seed": args.seed,
            "tasks_json": str(tasks_json),
            "manifest_csv": str(manifest_csv),
            "manifest_json": str(manifest_json),
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"species_root={layout.root}")
    print(f"source_root={source_root}")
    print(f"mirror_root={mirror_root}")
    print(f"import_root={import_root}")
    print(f"images_available={len(all_images)}")
    print(f"images_selected={len(selected)}")
    print(f"tasks_json={tasks_json}")
    print(f"manifest_csv={manifest_csv}")
    print(f"manifest_json={manifest_json}")
    print(f"summary_json={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
