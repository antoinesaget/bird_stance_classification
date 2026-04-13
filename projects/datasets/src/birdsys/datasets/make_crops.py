#!/usr/bin/env python3
"""Purpose: Generate tunable crop artifacts from normalized bird annotations."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from collections.abc import Sequence
from typing import Any

from birdsys.core import (
    default_data_home,
    default_species_slug,
    ensure_layout,
    normalize_relative_path,
    resolve_species_relative_path,
)
from birdsys.datasets.dataset_common import add_bar_headroom, annotate_bars, embed_plot_block, format_count, import_pyplot, save_plot


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate named crop-spec artifacts from normalized bird boxes")
    parser.add_argument("--annotation-version", required=True, help="ann_vNNN")
    parser.add_argument("--crop-spec-id", required=True, help="Stable crop artifact name, e.g. crop_margin120_q95")
    parser.add_argument("--data-home", default=str(default_data_home()))
    parser.add_argument("--species-slug", default=default_species_slug())
    parser.add_argument("--image-source", required=True, choices=["original", "compressed"])
    parser.add_argument("--margin", type=float, default=1.2, help="Crop expansion factor around bbox center")
    parser.add_argument("--clip-mode", default="image_bounds", choices=["image_bounds"])
    parser.add_argument("--jpeg-quality", type=int, default=95)
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


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def render_crop_plots(*, out_dir: pathlib.Path, manifest_rows) -> dict[str, dict[str, str]]:
    plt = import_pyplot()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, str]] = {}

    status_counts = manifest_rows["status"].value_counts().sort_index().to_dict() if not manifest_rows.empty else {}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(status_counts) or ["none"]
    values = [int(status_counts[label]) for label in labels] or [0]
    bars = ax.bar(range(len(labels)), values, color="#2F6B8A")
    add_bar_headroom(ax, values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Crop Status Counts")
    annotate_bars(ax, bars, [format_count(value) for value in values], fontsize=9)
    out["crop_status_counts"] = save_plot(fig, plots_dir / "crop_status_counts")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    if manifest_rows.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No crops", ha="center", va="center")
            ax.set_axis_off()
    else:
        widths = manifest_rows["crop_width"].fillna(0).astype(int).tolist()
        heights = manifest_rows["crop_height"].fillna(0).astype(int).tolist()
        axes[0].hist(widths, bins=min(12, max(3, len(set(widths)) or 3)), color="#2A9D8F")
        axes[0].set_title("Crop Widths")
        axes[1].hist(heights, bins=min(12, max(3, len(set(heights)) or 3)), color="#E76F51")
        axes[1].set_title("Crop Heights")
    out["crop_size_distribution"] = save_plot(fig, plots_dir / "crop_size_distribution")
    plt.close(fig)
    return out


def write_crop_report_md(*, path: pathlib.Path, manifest: dict[str, Any], plot_files: dict[str, dict[str, str]]) -> None:
    counts = manifest["counts"]
    lines = [
        f"# Crop Report: {manifest['crop_spec_id']}",
        "",
        f"- Species slug: `{manifest['species_slug']}`",
        f"- Annotation version: `{manifest['annotation_version']}`",
        f"- Image source: `{manifest['image_source']}`",
        f"- Crop output dir: `{manifest['output_dir']}`",
        f"- Margin: `{manifest['crop_spec']['margin']}`",
        f"- Clip mode: `{manifest['crop_spec']['clip_mode']}`",
        f"- JPEG quality: `{manifest['crop_spec']['jpeg_quality']}`",
        f"- Birds rows seen: `{counts['birds_rows']}`",
        f"- Crops processed: `{counts['processed_rows']}`",
        f"- Crops written: `{counts['written_rows']}`",
        f"- Crops reused: `{counts['existing_rows']}`",
        "",
    ]
    lines.extend(embed_plot_block("Crop Status Counts", "crop_status_counts", plot_files))
    lines.extend(embed_plot_block("Crop Size Distribution", "crop_size_distribution", plot_files))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relpath_column_for_source(image_source: str) -> str:
    if image_source == "original":
        return "original_relpath"
    if image_source == "compressed":
        return "compressed_relpath"
    raise ValueError(f"Unsupported image source {image_source!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required for crop generation.") from exc
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for crop generation.") from exc

    if args.margin <= 0:
        raise ValueError("--margin must be > 0")
    if not 1 <= int(args.jpeg_quality) <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    layout = ensure_layout(pathlib.Path(args.data_home), args.species_slug)
    relpath_column = relpath_column_for_source(args.image_source)

    birds_path = layout.labelstudio_normalized / args.annotation_version / "birds.parquet"
    images_path = layout.labelstudio_normalized / args.annotation_version / "images_labels.parquet"

    if not birds_path.exists():
        raise FileNotFoundError(birds_path)
    if not images_path.exists():
        raise FileNotFoundError(images_path)

    out_dir = layout.derived_crops / args.annotation_version / args.crop_spec_id
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists; use --overwrite to regenerate the crop artifact")
    out_dir.mkdir(parents=True, exist_ok=True)

    birds_df = pd.read_parquet(birds_path)
    images_df = pd.read_parquet(images_path)[["image_id", "species_slug", "original_relpath", "compressed_relpath"]]
    merged = birds_df.merge(images_df, on="image_id", how="left", validate="many_to_one")

    missing = int(merged[relpath_column].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} birds rows missing {relpath_column} in normalized images table")

    total = len(merged)
    if args.max_crops > 0:
        total = min(total, int(args.max_crops))

    manifest_rows: list[dict[str, object]] = []
    for _, row in merged.head(total).iterrows():
        source_relpath = normalize_relative_path(str(row[relpath_column]), field_name=relpath_column)
        image_path = resolve_species_relative_path(layout.root, source_relpath, field_name=relpath_column)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        bird_id = str(row["bird_id"])
        out_path = out_dir / f"{safe_bird_id(bird_id)}.jpg"
        if out_path.exists() and not args.overwrite:
            with Image.open(out_path) as existing:
                crop_width, crop_height = existing.size
            manifest_rows.append(
                {
                    "annotation_version": args.annotation_version,
                    "species_slug": layout.species_slug,
                    "crop_spec_id": args.crop_spec_id,
                    "image_source": args.image_source,
                    "image_id": str(row["image_id"]),
                    "bird_id": bird_id,
                    "source_relpath": source_relpath,
                    "crop_path": str(out_path),
                    "crop_width": int(crop_width),
                    "crop_height": int(crop_height),
                    "status": "exists",
                }
            )
            continue

        with Image.open(image_path).convert("RGB") as image:
            img_w, img_h = image.size
            x1, y1, x2, y2 = expanded_crop(
                img_w=img_w,
                img_h=img_h,
                x=float(row["bbox_x"]),
                y=float(row["bbox_y"]),
                w=float(row["bbox_w"]),
                h=float(row["bbox_h"]),
                margin=float(args.margin),
            )
            crop = image.crop((x1, y1, x2, y2))
            crop.save(out_path, format="JPEG", quality=int(args.jpeg_quality))
            crop_width, crop_height = crop.size

        manifest_rows.append(
            {
                "annotation_version": args.annotation_version,
                "species_slug": layout.species_slug,
                "crop_spec_id": args.crop_spec_id,
                "image_source": args.image_source,
                "image_id": str(row["image_id"]),
                "bird_id": bird_id,
                "source_relpath": source_relpath,
                "crop_path": str(out_path),
                "crop_width": int(crop_width),
                "crop_height": int(crop_height),
                "status": "written",
            }
        )

    manifest_rows_df = pd.DataFrame(
        manifest_rows,
        columns=[
            "annotation_version",
            "species_slug",
            "crop_spec_id",
            "image_source",
            "image_id",
            "bird_id",
            "source_relpath",
            "crop_path",
            "crop_width",
            "crop_height",
            "status",
        ],
    ).sort_values(["image_id", "bird_id"]).reset_index(drop=True)
    manifest_path = out_dir / "_crops.parquet"
    manifest_rows_df.to_parquet(manifest_path, index=False)

    counts = {
        "birds_rows": int(len(merged)),
        "processed_rows": int(len(manifest_rows_df)),
        "written_rows": int((manifest_rows_df["status"] == "written").sum()) if not manifest_rows_df.empty else 0,
        "existing_rows": int((manifest_rows_df["status"] == "exists").sum()) if not manifest_rows_df.empty else 0,
    }
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "species_slug": layout.species_slug,
        "annotation_version": args.annotation_version,
        "crop_spec_id": args.crop_spec_id,
        "image_source": args.image_source,
        "output_dir": str(out_dir),
        "inputs": {
            "birds_parquet": str(birds_path),
            "images_labels_parquet": str(images_path),
        },
        "crop_spec": {
            "margin": float(args.margin),
            "clip_mode": args.clip_mode,
            "jpeg_quality": int(args.jpeg_quality),
        },
        "counts": counts,
    }

    plot_files = render_crop_plots(out_dir=out_dir, manifest_rows=manifest_rows_df)
    crop_manifest_json = out_dir / "crop_manifest.json"
    crop_report_json = out_dir / "crop_report.json"
    crop_report_md = out_dir / "crop_report.md"
    write_json(crop_manifest_json, manifest)
    write_json(crop_report_json, manifest)
    write_crop_report_md(path=crop_report_md, manifest=manifest, plot_files=plot_files)

    print(f"species_slug={layout.species_slug}")
    print(f"annotation_version={args.annotation_version}")
    print(f"crop_spec_id={args.crop_spec_id}")
    print(f"image_source={args.image_source}")
    print(f"output_dir={out_dir}")
    print(f"crop_manifest={crop_manifest_json}")
    print(f"crop_report_md={crop_report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
