#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
from dataclasses import asdict, dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from birdsys.paths import ensure_layout

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class MirrorRow:
    site_id: str
    image_id: str
    source_path: str
    target_path: str
    width: int
    height: int
    bytes_before: int
    bytes_after: int
    size_ratio: float
    quality: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build JPEG compressed mirror for annotation UI while keeping geometry unchanged"
    )
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--quality", type=int, default=60)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument(
        "--source-relative-root",
        default="raw_images",
        help="Source root under data_root (default: raw_images)",
    )
    parser.add_argument(
        "--output-relative-root",
        default="labelstudio/images_compressed",
        help="Output root under data_root (default: labelstudio/images_compressed)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_images(site_dir: pathlib.Path) -> list[pathlib.Path]:
    return [
        path
        for path in sorted(site_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main() -> int:
    args = parse_args()
    if args.quality < 1 or args.quality > 95:
        raise ValueError("--quality must be in [1, 95]")

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required. Install dependencies with `uv sync --python 3.11`.") from exc
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required. Install dependencies with `uv sync --python 3.11`.") from exc

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    ensure_layout(data_root)

    src_rel = args.source_relative_root.strip().strip("/")
    out_rel = args.output_relative_root.strip().strip("/")
    if not src_rel:
        raise ValueError("--source-relative-root cannot be empty")
    if not out_rel:
        raise ValueError("--output-relative-root cannot be empty")

    source_site_dir = data_root / src_rel / args.site_id
    if not source_site_dir.exists():
        raise FileNotFoundError(source_site_dir)

    output_site_dir = data_root / out_rel / f"q{args.quality}" / args.site_id
    output_site_dir.mkdir(parents=True, exist_ok=True)

    candidates = iter_images(source_site_dir)
    if not candidates:
        raise RuntimeError(f"No images found in {source_site_dir}")

    if args.max_images > 0:
        candidates = candidates[: args.max_images]

    seen_ids: set[str] = set()
    rows: list[MirrorRow] = []

    for src in candidates:
        image_id = src.stem
        if image_id in seen_ids:
            raise ValueError(f"duplicate image_id in source set: {image_id}")
        seen_ids.add(image_id)

        dst = output_site_dir / f"{image_id}.jpg"
        if dst.exists() and not args.overwrite:
            before = int(src.stat().st_size)
            after = int(dst.stat().st_size)
            with Image.open(dst) as out_img:
                width, height = out_img.size
            rows.append(
                MirrorRow(
                    site_id=args.site_id,
                    image_id=image_id,
                    source_path=str(src),
                    target_path=str(dst),
                    width=int(width),
                    height=int(height),
                    bytes_before=before,
                    bytes_after=after,
                    size_ratio=(after / before) if before > 0 else 0.0,
                    quality=int(args.quality),
                )
            )
            continue

        with Image.open(src) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            rgb.save(dst, format="JPEG", quality=args.quality, optimize=True)

        before = int(src.stat().st_size)
        after = int(dst.stat().st_size)
        rows.append(
            MirrorRow(
                site_id=args.site_id,
                image_id=image_id,
                source_path=str(src),
                target_path=str(dst),
                width=int(width),
                height=int(height),
                bytes_before=before,
                bytes_after=after,
                size_ratio=(after / before) if before > 0 else 0.0,
                quality=int(args.quality),
            )
        )

    manifest = pd.DataFrame([asdict(row) for row in rows]).sort_values(["site_id", "image_id"]).reset_index(drop=True)
    manifest_path = output_site_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    total_before = int(manifest["bytes_before"].sum()) if not manifest.empty else 0
    total_after = int(manifest["bytes_after"].sum()) if not manifest.empty else 0
    ratio = (total_after / total_before) if total_before > 0 else 0.0

    print(f"source_site_dir={source_site_dir}")
    print(f"output_site_dir={output_site_dir}")
    print(f"images_processed={len(rows)}")
    print(f"bytes_before={total_before}")
    print(f"bytes_after={total_after}")
    print(f"size_ratio={ratio:.6f}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
