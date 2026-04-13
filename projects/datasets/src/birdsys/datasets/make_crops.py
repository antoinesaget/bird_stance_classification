#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
from collections.abc import Sequence

from birdsys.core import ensure_layout


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bird crops from normalized bird boxes")
    parser.add_argument("--annotation-version", required=True, help="ann_vXXX")
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument("--margin", type=float, default=1.2, help="Crop expansion factor around bbox center")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-crops", type=int, default=0)
    return parser.parse_args(argv)


def safe_bird_id(bird_id: str) -> str:
    return bird_id.replace(":", "_").replace("/", "_")


def clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def expanded_crop(
    *,
    img_w: int,
    img_h: int,
    x: float,
    y: float,
    w: float,
    h: float,
    margin: float,
) -> tuple[int, int, int, int]:
    x1 = x * img_w
    y1 = y * img_h
    x2 = (x + w) * img_w
    y2 = (y + h) * img_h

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, (x2 - x1) * margin)
    bh = max(1.0, (y2 - y1) * margin)

    nx1 = int(round(cx - bw / 2.0))
    ny1 = int(round(cy - bh / 2.0))
    nx2 = int(round(cx + bw / 2.0))
    ny2 = int(round(cy + bh / 2.0))

    nx1 = clip(nx1, 0, img_w - 1)
    ny1 = clip(ny1, 0, img_h - 1)
    nx2 = clip(nx2, nx1 + 1, img_w)
    ny2 = clip(ny2, ny1 + 1, img_h)
    return nx1, ny1, nx2, ny2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required. Install dependencies with `uv sync --python 3.11`.") from exc
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required. Install dependencies with `uv sync --python 3.11`.") from exc

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    layout = ensure_layout(data_root)

    birds_path = layout.labelstudio_normalized / args.annotation_version / "birds.parquet"
    metadata_path = layout.metadata / "images.parquet"
    images_path = layout.labelstudio_normalized / args.annotation_version / "images_labels.parquet"

    if not birds_path.exists():
        raise FileNotFoundError(birds_path)
    if not metadata_path.exists() and not images_path.exists():
        raise FileNotFoundError(metadata_path)

    birds_df = pd.read_parquet(birds_path)
    if metadata_path.exists():
        meta_df = pd.read_parquet(metadata_path)[["image_id", "filepath"]]
    else:
        meta_df = pd.read_parquet(images_path)[["image_id", "filepath"]]
    merged = birds_df.merge(meta_df, on="image_id", how="left", validate="many_to_one")

    missing = merged["filepath"].isna().sum()
    if missing:
        raise RuntimeError(f"{missing} birds rows missing filepaths in metadata")

    out_dir = layout.derived_crops / args.annotation_version
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    total = len(merged)
    if args.max_crops > 0:
        total = min(total, args.max_crops)

    for i, row in merged.head(total).iterrows():
        image_path = pathlib.Path(str(row["filepath"])).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        bird_id = str(row["bird_id"])
        out_path = out_dir / f"{safe_bird_id(bird_id)}.jpg"
        if out_path.exists() and not args.overwrite:
            manifest_rows.append({
                "annotation_version": args.annotation_version,
                "image_id": row["image_id"],
                "bird_id": bird_id,
                "crop_path": str(out_path),
                "status": "exists",
            })
            continue

        with Image.open(image_path).convert("RGB") as img:
            img_w, img_h = img.size
            x1, y1, x2, y2 = expanded_crop(
                img_w=img_w,
                img_h=img_h,
                x=float(row["bbox_x"]),
                y=float(row["bbox_y"]),
                w=float(row["bbox_w"]),
                h=float(row["bbox_h"]),
                margin=args.margin,
            )
            crop = img.crop((x1, y1, x2, y2))
            crop.save(out_path, format="JPEG", quality=95)

        manifest_rows.append(
            {
                "annotation_version": args.annotation_version,
                "image_id": row["image_id"],
                "bird_id": bird_id,
                "crop_path": str(out_path),
                "crop_width": x2 - x1,
                "crop_height": y2 - y1,
                "status": "written",
            }
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["image_id", "bird_id"]).reset_index(drop=True)
    manifest_path = out_dir / "_crops.parquet"
    manifest_df.to_parquet(manifest_path, index=False)

    print(f"birds_rows={len(merged)}")
    print(f"processed={len(manifest_rows)}")
    print(f"manifest={manifest_path}")
    print(f"output_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
