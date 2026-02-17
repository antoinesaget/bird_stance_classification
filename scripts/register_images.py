#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
from dataclasses import asdict, dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from birdsys.paths import ensure_layout

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    filepath: str
    file_name: str
    site_id: str
    width: int
    height: int


def iter_images(root: pathlib.Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register raw images into metadata/images.parquet")
    parser.add_argument("--source-dir", default="scraped_images/scolop2_10k", help="Source image directory")
    parser.add_argument("--site-id", default="scolop2", help="Site identifier")
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
        help="How to expose source images under data_root/raw_images/site_id",
    )
    parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="Allow rewriting metadata/images.parquet",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for smoke runs")
    return parser.parse_args()


def ensure_raw_site(
    *,
    source_dir: pathlib.Path,
    raw_site_dir: pathlib.Path,
    mode: str,
) -> pathlib.Path:
    raw_site_dir.parent.mkdir(parents=True, exist_ok=True)

    if mode == "none":
        return source_dir

    if mode == "symlink":
        if raw_site_dir.exists() or raw_site_dir.is_symlink():
            if raw_site_dir.is_symlink() and raw_site_dir.resolve() == source_dir.resolve():
                return raw_site_dir
            raise RuntimeError(
                f"{raw_site_dir} already exists and cannot be replaced automatically."
            )
        raw_site_dir.symlink_to(source_dir.resolve())
        return raw_site_dir

    if raw_site_dir.exists() and any(raw_site_dir.iterdir()):
        return raw_site_dir

    raw_site_dir.mkdir(parents=True, exist_ok=True)
    for src in iter_images(source_dir):
        dst = raw_site_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    return raw_site_dir


def read_size(path: pathlib.Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required. Install dependencies with `uv sync --python 3.11`.") from exc

    with Image.open(path) as img:
        return img.width, img.height


def main() -> int:
    args = parse_args()
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required. Install dependencies with `uv sync --python 3.11`.") from exc

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    source_dir = pathlib.Path(args.source_dir).expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"source dir does not exist: {source_dir}")

    layout = ensure_layout(data_root)
    raw_site_dir = layout.raw_images / args.site_id
    scan_dir = ensure_raw_site(source_dir=source_dir, raw_site_dir=raw_site_dir, mode=args.link_mode)

    rows: list[ImageRecord] = []
    seen_ids: set[str] = set()
    for idx, image_path in enumerate(iter_images(scan_dir)):
        if args.max_images > 0 and idx >= args.max_images:
            break

        image_id = image_path.stem
        if image_id in seen_ids:
            raise ValueError(f"duplicate image_id detected: {image_id}")
        seen_ids.add(image_id)

        width, height = read_size(image_path)
        rows.append(
            ImageRecord(
                image_id=image_id,
                filepath=str(image_path.absolute()),
                file_name=image_path.name,
                site_id=args.site_id,
                width=width,
                height=height,
            )
        )

    if not rows:
        raise RuntimeError(f"No images found in {scan_dir}")

    df = pd.DataFrame([asdict(r) for r in rows]).sort_values(["image_id"]).reset_index(drop=True)

    out_path = layout.metadata / "images.parquet"
    if out_path.exists() and not args.overwrite_metadata:
        raise FileExistsError(f"{out_path} already exists. Use --overwrite-metadata to replace it.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"registered_images={len(df)}")
    print(f"metadata_path={out_path}")
    print(f"scan_dir={scan_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
