#!/usr/bin/env python3
"""Convert COCO detection JSON exports into YOLO txt labels.

This helper is intentionally lightweight for POC export workflows.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert COCO JSON to YOLO labels.")
    parser.add_argument("--coco-json", required=True, help="Path to COCO annotations JSON")
    parser.add_argument("--out-dir", required=True, help="Output directory for YOLO labels")
    parser.add_argument(
        "--only-category",
        default="bird",
        help="Optional category name filter (default: bird). Use '*' for all.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    coco_path = pathlib.Path(args.coco_json)
    out_dir = pathlib.Path(args.out_dir)
    out_labels = out_dir / "labels"
    out_labels.mkdir(parents=True, exist_ok=True)

    data = json.loads(coco_path.read_text())

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}

    selected = [
        c for c in data.get("categories", []) if args.only_category == "*" or c["name"] == args.only_category
    ]
    if not selected:
        raise SystemExit(
            f"No categories matched filter '{args.only_category}'. Available: {sorted(categories.values())}"
        )

    yolo_ids = {cat["id"]: i for i, cat in enumerate(selected)}

    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image = defaultdict(list)

    for ann in data.get("annotations", []):
        cat_id = ann.get("category_id")
        if cat_id not in yolo_ids:
            continue
        anns_by_image[ann.get("image_id")].append(ann)

    classes_path = out_dir / "classes.txt"
    classes_path.write_text("\n".join(cat["name"] for cat in selected) + "\n")

    converted_images = 0
    converted_boxes = 0

    for image_id, image in images.items():
        width = float(image["width"])
        height = float(image["height"])
        yolo_lines = []

        for ann in anns_by_image.get(image_id, []):
            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2.0) / width
            y_center = (y + h / 2.0) / height
            norm_w = w / width
            norm_h = h / height

            yolo_lines.append(
                f"{yolo_ids[ann['category_id']]} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

        image_name = pathlib.Path(image["file_name"]).stem
        (out_labels / f"{image_name}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        converted_images += 1
        converted_boxes += len(yolo_lines)

    print(f"Wrote labels for {converted_images} images")
    print(f"Wrote {converted_boxes} boxes")
    print(f"Classes file: {classes_path}")
    print(f"Labels dir: {out_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
