#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from birdsys.paths import ensure_layout

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic Label Studio import JSON for a sample image subset"
    )
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument("--site-id", default="scolop2")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument(
        "--ls-local-files-prefix",
        default="/data/local-files/?d=",
        help="Prefix for Label Studio local-files URL",
    )
    return parser.parse_args()


def iter_site_images(site_dir: pathlib.Path):
    for path in sorted(site_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def select_images(
    images: list[pathlib.Path],
    *,
    count: int,
    sample_mode: str,
    seed: int,
) -> list[pathlib.Path]:
    if count <= 0 or count >= len(images):
        if sample_mode == "random":
            rng = random.Random(seed)
            shuffled = list(images)
            rng.shuffle(shuffled)
            return shuffled
        return images

    if sample_mode == "first":
        return images[:count]

    rng = random.Random(seed)
    return rng.sample(images, count)


def main() -> int:
    args = parse_args()

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    layout = ensure_layout(data_root)

    site_dir = layout.raw_images / args.site_id
    if not site_dir.exists():
        raise FileNotFoundError(site_dir)

    all_images = list(iter_site_images(site_dir))
    if not all_images:
        raise RuntimeError(f"No images found in {site_dir}")

    selected = select_images(
        all_images,
        count=args.count,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else data_root / "labelstudio" / "imports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix or f"{args.site_id}_sample{len(selected)}"
    out_json = output_dir / f"{prefix}.tasks.json"
    out_csv = output_dir / f"{prefix}.files.csv"

    data_root_name = data_root.name
    tasks: list[dict[str, object]] = []
    csv_rows: list[dict[str, str]] = []

    for image in selected:
        rel_for_ls = f"{data_root_name}/raw_images/{args.site_id}/{image.name}"
        ls_url = f"{args.ls_local_files_prefix}{rel_for_ls}"
        task = {
            "image_id": image.stem,
            "site_id": args.site_id,
            "image": ls_url,
        }
        tasks.append(task)
        csv_rows.append(
            {
                "image_id": image.stem,
                "site_id": args.site_id,
                "absolute_path": str(image),
                "label_studio_image": ls_url,
            }
        )

    tasks = sorted(tasks, key=lambda row: str(row["image_id"]))
    csv_rows = sorted(csv_rows, key=lambda row: row["image_id"])

    out_json.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "site_id", "absolute_path", "label_studio_image"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"site_dir={site_dir}")
    print(f"images_available={len(all_images)}")
    print(f"images_selected={len(selected)}")
    print(f"tasks_json={out_json}")
    print(f"files_csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
