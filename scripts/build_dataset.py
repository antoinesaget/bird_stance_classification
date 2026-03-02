#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from birdsys.paths import ensure_layout, next_version_dir
from birdsys.reporting import diff_numeric_dict, find_previous_version_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build versioned train/val/test datasets from normalized parquet")
    parser.add_argument("--annotation-version", required=True, help="ann_vXXX")
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument("--dataset-version", default="", help="Optional explicit output folder name, e.g. ds_v001")
    parser.add_argument("--train-pct", type=int, default=80)
    parser.add_argument("--val-pct", type=int, default=10)
    parser.add_argument("--test-pct", type=int, default=10)
    return parser.parse_args()


def choose_split(bucket: int, train_pct: int, val_pct: int) -> str:
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "val"
    return "test"


def _pct(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round((part / whole) * 100.0, 3)


def _label_count_deltas(
    current: dict[str, dict[str, int]],
    previous: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for field in sorted(set(current) | set(previous)):
        cur_counts = current.get(field, {})
        prev_counts = previous.get(field, {})
        labels = sorted(set(cur_counts) | set(prev_counts))
        out[field] = {
            label: int(cur_counts.get(label, 0)) - int(prev_counts.get(label, 0))
            for label in labels
        }
    return out


def write_dataset_report(
    *,
    out_dir: pathlib.Path,
    manifest: dict[str, Any],
    null_counts: dict[str, int],
    previous_manifest: dict[str, Any] | None,
) -> tuple[pathlib.Path, pathlib.Path]:
    counts = manifest["counts"]
    rows_total = int(counts["rows_total"])
    split_percentages = {
        "train_pct_actual": _pct(int(counts["rows_train"]), rows_total),
        "val_pct_actual": _pct(int(counts["rows_val"]), rows_total),
        "test_pct_actual": _pct(int(counts["rows_test"]), rows_total),
    }

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_version": manifest["dataset_version"],
        "source_annotation_version": manifest["source_annotation_version"],
        "data_root": manifest["data_root"],
        "counts": counts,
        "split_percentages": split_percentages,
        "null_counts": null_counts,
        "label_counts": manifest["label_counts"],
        "comparison_to_previous": None,
    }

    if previous_manifest is not None:
        prev_counts = previous_manifest.get("counts", {})
        prev_label_counts = previous_manifest.get("label_counts", {})
        comparison = {
            "previous_dataset_version": previous_manifest.get("dataset_version"),
            "delta_counts": diff_numeric_dict(counts, prev_counts),
            "delta_label_counts": _label_count_deltas(manifest["label_counts"], prev_label_counts),
        }
        report["comparison_to_previous"] = comparison

    report_json = out_dir / "dataset_report.json"
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Dataset Report: {manifest['dataset_version']}",
        "",
        f"- Source annotation version: `{manifest['source_annotation_version']}`",
        f"- Rows: total `{counts['rows_total']}`, train `{counts['rows_train']}`, val `{counts['rows_val']}`, test `{counts['rows_test']}`",
        f"- Images: total `{counts['images_total']}`, train `{counts['images_train']}`, val `{counts['images_val']}`, test `{counts['images_test']}`",
        f"- Split percentages (actual): train `{split_percentages['train_pct_actual']:.3f}%`, val `{split_percentages['val_pct_actual']:.3f}%`, test `{split_percentages['test_pct_actual']:.3f}%`",
        f"- Null counts: `{null_counts}`",
    ]
    comparison = report["comparison_to_previous"]
    if comparison:
        lines.append(
            f"- Delta vs `{comparison['previous_dataset_version']}` counts: `{comparison['delta_counts']}`"
        )

    report_md = out_dir / "dataset_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_json, report_md


def main() -> int:
    args = parse_args()
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("duckdb is required. Install dependencies with `uv sync --python 3.11`.") from exc
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required. Install dependencies with `uv sync --python 3.11`.") from exc

    if args.train_pct + args.val_pct + args.test_pct != 100:
        raise ValueError("train/val/test percentages must sum to 100")

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    layout = ensure_layout(data_root)

    birds_path = layout.labelstudio_normalized / args.annotation_version / "birds.parquet"
    images_path = layout.labelstudio_normalized / args.annotation_version / "images_labels.parquet"
    meta_path = layout.metadata / "images.parquet"
    crop_root = layout.derived_crops / args.annotation_version

    for p in (birds_path, images_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(p)

    if args.dataset_version:
        out_dir = layout.derived_datasets / args.dataset_version
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        out_dir = next_version_dir(layout.derived_datasets, "ds")

    conn = duckdb.connect(database=":memory:")
    query = f"""
        SELECT
            b.annotation_version,
            b.image_id,
            b.bird_id,
            b.bbox_x,
            b.bbox_y,
            b.bbox_w,
            b.bbox_h,
            b.isbird,
            b.readability,
            b.specie,
            b.behavior,
            b.substrate,
            b.legs,
            i.image_usable,
            m.filepath,
            m.site_id,
            abs(hash(coalesce(m.site_id, '') || ':' || b.image_id)) % 100 AS split_bucket,
            '{crop_root.as_posix()}/' || replace(b.bird_id, ':', '_') || '.jpg' AS crop_path
        FROM read_parquet('{birds_path.as_posix()}') b
        LEFT JOIN read_parquet('{images_path.as_posix()}') i
          ON b.annotation_version = i.annotation_version AND b.image_id = i.image_id
        LEFT JOIN read_parquet('{meta_path.as_posix()}') m
          ON b.image_id = m.image_id
        ORDER BY b.image_id, b.bird_id
    """
    df = conn.execute(query).fetch_df()

    df["split"] = df["split_bucket"].map(lambda b: choose_split(int(b), args.train_pct, args.val_pct))
    df = df.drop(columns=["split_bucket"])

    split_frames = {
        "train": df[df["split"] == "train"].copy(),
        "val": df[df["split"] == "val"].copy(),
        "test": df[df["split"] == "test"].copy(),
    }

    for split, frame in split_frames.items():
        out_path = out_dir / f"{split}.parquet"
        frame.to_parquet(out_path, index=False)

    split_image_ids: dict[str, set[str]] = {
        split: set(frame["image_id"].dropna().astype(str).tolist()) for split, frame in split_frames.items()
    }
    if split_image_ids["train"] & split_image_ids["val"]:
        raise RuntimeError("train/val image leakage detected")
    if split_image_ids["train"] & split_image_ids["test"]:
        raise RuntimeError("train/test image leakage detected")
    if split_image_ids["val"] & split_image_ids["test"]:
        raise RuntimeError("val/test image leakage detected")

    label_counts = {}
    for column in ["isbird", "readability", "specie", "behavior", "substrate", "legs", "image_usable"]:
        vc = df[column].fillna("<null>").value_counts().sort_index().to_dict()
        label_counts[column] = {str(k): int(v) for k, v in vc.items()}

    null_counts = {
        column: int(df[column].isna().sum())
        for column in ["isbird", "readability", "specie", "behavior", "substrate", "legs", "image_usable", "filepath", "crop_path"]
    }

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_version": out_dir.name,
        "source_annotation_version": args.annotation_version,
        "data_root": str(data_root),
        "inputs": {
            "birds_parquet": str(birds_path),
            "images_labels_parquet": str(images_path),
            "metadata_parquet": str(meta_path),
        },
        "filters": {
            "include_all_birds": True,
            "include_unreadable": True,
        },
        "split_policy": {
            "strategy": "stable_hash",
            "formula": "abs(hash(coalesce(site_id,'') || ':' || image_id)) % 100",
            "train_pct": args.train_pct,
            "val_pct": args.val_pct,
            "test_pct": args.test_pct,
        },
        "counts": {
            "rows_total": int(len(df)),
            "rows_train": int(len(split_frames["train"])),
            "rows_val": int(len(split_frames["val"])),
            "rows_test": int(len(split_frames["test"])),
            "images_total": int(df["image_id"].nunique()),
            "images_train": int(split_frames["train"]["image_id"].nunique()),
            "images_val": int(split_frames["val"]["image_id"].nunique()),
            "images_test": int(split_frames["test"]["image_id"].nunique()),
        },
        "label_counts": label_counts,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    previous_manifest: dict[str, Any] | None = None
    prev_dir = find_previous_version_dir(layout.derived_datasets, "ds", out_dir.name)
    if prev_dir is not None:
        prev_manifest_path = prev_dir / "manifest.json"
        if prev_manifest_path.exists():
            previous_manifest = json.loads(prev_manifest_path.read_text(encoding="utf-8"))
    report_json, report_md = write_dataset_report(
        out_dir=out_dir,
        manifest=manifest,
        null_counts=null_counts,
        previous_manifest=previous_manifest,
    )

    print(f"dataset_dir={out_dir}")
    print(f"rows_total={len(df)}")
    print(f"rows_train={len(split_frames['train'])}")
    print(f"rows_val={len(split_frames['val'])}")
    print(f"rows_test={len(split_frames['test'])}")
    print(f"manifest={manifest_path}")
    print(f"report_json={report_json}")
    print(f"report_md={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
