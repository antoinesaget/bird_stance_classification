#!/usr/bin/env python3
"""Purpose: Build versioned pooled dataset artifacts from split and crop artifacts."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from collections.abc import Sequence
from typing import Any

from birdsys.core import default_data_home, default_species_slug, diff_numeric_dict, ensure_layout, find_previous_version_dir, next_version_dir
from birdsys.datasets.build_split import render_split_plots
from birdsys.datasets.dataset_common import (
    LABEL_FIELDS,
    absent_class_warnings,
    diff_nested_counts,
    embed_plot_block,
    import_pyplot,
    save_plot,
    visible_label_counts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled dataset artifacts from normalized, split, and crop inputs")
    parser.add_argument("--annotation-version", required=True, help="ann_vNNN")
    parser.add_argument("--split-version", required=True, help="split_vNNN")
    parser.add_argument("--crop-spec-id", required=True, help="Named crop artifact id")
    parser.add_argument("--data-home", default=str(default_data_home()))
    parser.add_argument("--species-slug", default=default_species_slug())
    parser.add_argument("--dataset-version", default="", help="Optional explicit output folder name, e.g. ds_v001")
    return parser.parse_args(argv)


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _group_id(site_id: Any, image_id: str) -> str:
    prefix = "" if site_id is None else str(site_id)
    return f"{prefix}:{image_id}"


def subset_counts(frame, image_ids: set[str]) -> dict[str, dict[str, int]]:
    if not image_ids:
        return {field: {} for field in LABEL_FIELDS}
    return visible_label_counts(frame[frame["image_id"].isin(sorted(image_ids))].copy())


def fold_support_counts(frame, assignments_df) -> dict[int, dict[str, dict[str, int]]]:
    supports: dict[int, dict[str, dict[str, int]]] = {}
    for fold_id in sorted(assignments_df["fold_id"].dropna().astype(int).unique().tolist()):
        image_ids = set(assignments_df[assignments_df["fold_id"] == fold_id]["image_id"].astype(str).tolist())
        supports[int(fold_id)] = subset_counts(frame, image_ids)
    return supports


def render_crop_spec_plot(
    *,
    out_dir: pathlib.Path,
    current_manifest: dict[str, Any],
    previous_manifest: dict[str, Any] | None,
) -> dict[str, str] | None:
    if previous_manifest is None:
        return None
    current_spec = current_manifest.get("crop_spec") or {}
    previous_spec = previous_manifest.get("crop_spec") or {}
    numeric_keys = [
        key
        for key in ("margin", "jpeg_quality")
        if isinstance(current_spec.get(key), (int, float)) and isinstance(previous_spec.get(key), (int, float))
    ]
    if not numeric_keys:
        return None

    plt = import_pyplot()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.32
    positions = list(range(len(numeric_keys)))
    current_values = [float(current_spec[key]) for key in numeric_keys]
    previous_values = [float(previous_spec[key]) for key in numeric_keys]
    ax.bar([position - (width / 2.0) for position in positions], previous_values, width=width, label="previous", color="#9C6644")
    ax.bar([position + (width / 2.0) for position in positions], current_values, width=width, label="current", color="#457B9D")
    ax.set_xticks(positions)
    ax.set_xticklabels(numeric_keys, rotation=15, ha="right")
    ax.set_title("Crop Spec Comparison")
    ax.legend(loc="upper right")
    plot = save_plot(fig, plots_dir / "crop_spec_comparison")
    plt.close(fig)
    return plot


def write_dataset_report_md(
    *,
    path: pathlib.Path,
    manifest: dict[str, Any],
    warnings: list[str],
    plot_files: dict[str, dict[str, str]],
    crop_plot: dict[str, str] | None,
) -> None:
    counts = manifest["counts"]
    membership = manifest["test_membership"]
    lines = [
        f"# Dataset Report: {manifest['dataset_version']}",
        "",
        f"- Species slug: `{manifest['species_slug']}`",
        f"- Source annotation version: `{manifest['source_annotation_version']}`",
        f"- Source split version: `{manifest['source_split_version']}`",
        f"- Crop spec id: `{manifest['crop_spec_id']}`",
        f"- Image source: `{manifest['image_source']}`",
        f"- Rows: total `{counts['rows_total']}`, test `{counts['rows_test']}`, train_pool `{counts['rows_train_pool']}`",
        f"- Images: total `{counts['images_total']}`, test `{counts['images_test']}`, train_pool `{counts['images_train_pool']}`",
        f"- Test membership: preserved `{membership['preserved']}`, added `{membership['added']}`, removed `{membership['removed']}`",
        f"- Leakage checks: `{manifest['leakage_checks']}`",
        "",
    ]
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings[:50]:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(embed_plot_block("Current Split Counts", "current_split_counts", plot_files))
    lines.extend(embed_plot_block("Current Class Supports", "current_class_supports", plot_files))
    lines.extend(embed_plot_block("Per-Fold Class Supports", "fold_class_supports", plot_files))
    lines.extend(embed_plot_block("Fold Sizes", "fold_sizes", plot_files))
    lines.extend(embed_plot_block("Test Support Coverage", "test_support_coverage", plot_files))
    lines.extend(embed_plot_block("Test Membership Churn", "test_membership_churn", plot_files))
    if "comparison_count_deltas" in plot_files:
        lines.extend(embed_plot_block("Comparison Count Deltas", "comparison_count_deltas", plot_files))
    if "comparison_class_support_deltas" in plot_files:
        lines.extend(embed_plot_block("Comparison Class Support Deltas", "comparison_class_support_deltas", plot_files))
    if "comparison_fold_support_deltas" in plot_files:
        lines.extend(embed_plot_block("Comparison Fold Support Deltas", "comparison_fold_support_deltas", plot_files))
    if crop_plot is not None:
        lines.extend(
            [
                "### Crop Spec Comparison",
                "",
                f"![Crop Spec Comparison](plots/{crop_plot['png']})",
                "",
                f"`SVG:` `plots/{crop_plot['svg']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required for dataset building.") from exc

    layout = ensure_layout(pathlib.Path(args.data_home), args.species_slug)

    birds_path = layout.labelstudio_normalized / args.annotation_version / "birds.parquet"
    images_path = layout.labelstudio_normalized / args.annotation_version / "images_labels.parquet"
    split_dir = layout.derived_splits / args.split_version
    crop_dir = layout.derived_crops / args.annotation_version / args.crop_spec_id
    test_groups_path = split_dir / "test_groups.parquet"
    pool_groups_path = split_dir / "train_pool_groups.parquet"
    folds_path = split_dir / "fold_assignments.parquet"
    split_manifest_path = split_dir / "split_manifest.json"
    crop_rows_path = crop_dir / "_crops.parquet"
    crop_manifest_path = crop_dir / "crop_manifest.json"
    for path in (birds_path, images_path, test_groups_path, pool_groups_path, folds_path, split_manifest_path, crop_rows_path, crop_manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.dataset_version:
        out_dir = layout.derived_datasets / args.dataset_version
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        out_dir = next_version_dir(layout.derived_datasets, "ds")

    birds_df = pd.read_parquet(birds_path)
    images_df = pd.read_parquet(images_path)
    images_df = images_df.drop_duplicates(subset=["image_id"]).copy()
    test_groups_df = pd.read_parquet(test_groups_path)
    pool_groups_df = pd.read_parquet(pool_groups_path)
    fold_assignments_df = pd.read_parquet(folds_path)
    crop_rows_df = pd.read_parquet(crop_rows_path)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    crop_manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))

    if str(split_manifest.get("source_annotation_version")) != args.annotation_version:
        raise ValueError("split artifact annotation version mismatch")
    if str(crop_manifest.get("annotation_version")) != args.annotation_version:
        raise ValueError("crop artifact annotation version mismatch")
    if str(crop_manifest.get("crop_spec_id")) != args.crop_spec_id:
        raise ValueError("crop artifact crop spec id mismatch")

    test_ids = set(test_groups_df["image_id"].astype(str).tolist())
    pool_ids = set(pool_groups_df["image_id"].astype(str).tolist())
    if test_ids & pool_ids:
        raise RuntimeError("test/train_pool image leakage detected in split artifact")

    birds_df["image_id"] = birds_df["image_id"].astype(str)
    birds_df["bird_id"] = birds_df["bird_id"].astype(str)
    images_df["image_id"] = images_df["image_id"].astype(str)
    crop_rows_df["bird_id"] = crop_rows_df["bird_id"].astype(str)
    crop_rows_df["image_id"] = crop_rows_df["image_id"].astype(str)
    fold_assignments_df["image_id"] = fold_assignments_df["image_id"].astype(str)
    fold_assignments_df["fold_id"] = fold_assignments_df["fold_id"].astype(int)

    all_df = birds_df.merge(images_df, on=["annotation_version", "image_id"], how="left", validate="many_to_one", suffixes=("", "_image"))
    all_df = all_df.merge(crop_rows_df[["bird_id", "crop_path", "crop_width", "crop_height"]], on="bird_id", how="left", validate="one_to_one")
    missing_crops = int(all_df["crop_path"].isna().sum())
    if missing_crops:
        raise RuntimeError(f"{missing_crops} dataset rows missing crop_path in crop artifact")

    def split_role_for_image(image_id: str) -> str:
        if image_id in test_ids:
            return "test"
        if image_id in pool_ids:
            return "train_pool"
        raise RuntimeError(f"image_id {image_id!r} missing from split artifact")

    fold_map = {
        str(row["image_id"]): int(row["fold_id"])
        for _, row in fold_assignments_df.iterrows()
    }
    all_df["split_role"] = all_df["image_id"].map(split_role_for_image)
    all_df["fold_id"] = all_df["image_id"].map(lambda image_id: fold_map.get(str(image_id)))
    all_df["row_id"] = all_df["bird_id"].astype(str)
    all_df["group_id"] = all_df.apply(
        lambda row: _group_id(None if pd.isna(row.get("site_id")) else row.get("site_id"), str(row["image_id"])),
        axis=1,
    )
    all_df["crop_spec_id"] = args.crop_spec_id

    test_df = all_df[all_df["split_role"] == "test"].copy().reset_index(drop=True)
    train_pool_df = all_df[all_df["split_role"] == "train_pool"].copy().reset_index(drop=True)
    if train_pool_df["fold_id"].isna().any():
        raise RuntimeError("train_pool rows missing fold assignments")
    test_df["fold_id"] = None
    all_data_df = all_df.copy().reset_index(drop=True)

    test_out = out_dir / "test.parquet"
    train_pool_out = out_dir / "train_pool.parquet"
    all_data_out = out_dir / "all_data.parquet"
    folds_out = out_dir / "fold_assignments.parquet"
    test_df.to_parquet(test_out, index=False)
    train_pool_df.to_parquet(train_pool_out, index=False)
    all_data_df.to_parquet(all_data_out, index=False)
    fold_assignments_df.to_parquet(folds_out, index=False)

    split_supports = {
        "all_data": visible_label_counts(all_data_df),
        "test": visible_label_counts(test_df),
        "train_pool": visible_label_counts(train_pool_df),
    }
    fold_supports = fold_support_counts(train_pool_df, fold_assignments_df)
    warnings = absent_class_warnings(
        baseline=split_supports["all_data"],
        test_counts=split_supports["test"],
        fold_counts=fold_supports,
    )
    counts = {
        "rows_total": int(len(all_data_df)),
        "rows_test": int(len(test_df)),
        "rows_train_pool": int(len(train_pool_df)),
        "images_total": int(all_data_df["image_id"].nunique()),
        "images_test": int(test_df["image_id"].nunique()),
        "images_train_pool": int(train_pool_df["image_id"].nunique()),
    }
    leakage_checks = {
        "train_pool_test_image_overlap": bool(set(train_pool_df["image_id"]) & set(test_df["image_id"])),
        "fold_ids_present_on_test": bool(test_df["fold_id"].notna().any()),
        "missing_fold_ids_on_train_pool": bool(train_pool_df["fold_id"].isna().any()),
    }
    if any(leakage_checks.values()):
        raise RuntimeError(f"Dataset leakage check failed: {leakage_checks}")

    previous_dir = find_previous_version_dir(layout.derived_datasets, "ds", out_dir.name)
    previous_manifest: dict[str, Any] | None = None
    if previous_dir is not None:
        previous_manifest_path = previous_dir / "manifest.json"
        if previous_manifest_path.exists():
            previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_version": out_dir.name,
        "source_annotation_version": args.annotation_version,
        "species_slug": layout.species_slug,
        "species_root": str(layout.root),
        "source_split_version": args.split_version,
        "crop_spec_id": args.crop_spec_id,
        "image_source": crop_manifest.get("image_source"),
        "data_home": str(layout.data_home),
        "inputs": {
            "birds_parquet": str(birds_path),
            "images_labels_parquet": str(images_path),
            "split_manifest": str(split_manifest_path),
            "crop_manifest": str(crop_manifest_path),
        },
        "crop_spec": crop_manifest.get("crop_spec"),
        "counts": counts,
        "split_supports": {
            "all_data": split_supports["all_data"],
            "test": split_supports["test"],
            "train_pool": split_supports["train_pool"],
            "folds": {str(fold_id): fold_supports[fold_id] for fold_id in sorted(fold_supports)},
        },
        "test_membership": split_manifest.get("test_membership", {}),
        "leakage_checks": leakage_checks,
        "warnings": warnings,
        "comparison_to_previous": None,
    }
    if previous_manifest is not None:
        manifest["comparison_to_previous"] = {
            "previous_dataset_version": previous_manifest.get("dataset_version"),
            "delta_counts": diff_numeric_dict(counts, previous_manifest.get("counts", {})),
            "delta_supports": {
                role: diff_nested_counts(split_supports[role], (previous_manifest.get("split_supports") or {}).get(role, {}))
                for role in ("all_data", "test", "train_pool")
            },
            "previous_crop_spec_id": previous_manifest.get("crop_spec_id"),
        }

    plot_files = render_split_plots(
        out_dir=out_dir,
        counts=counts,
        split_supports=split_supports,
        fold_supports=fold_supports,
        assignments_df=fold_assignments_df,
        membership_summary={
            "preserved": int(split_manifest.get("test_membership", {}).get("preserved", 0)),
            "added": int(split_manifest.get("test_membership", {}).get("added", 0)),
            "removed": int(split_manifest.get("test_membership", {}).get("removed", 0)),
        },
        previous_counts=None if previous_manifest is None else previous_manifest.get("counts"),
        previous_supports=None if previous_manifest is None else previous_manifest.get("split_supports"),
    )
    crop_plot = render_crop_spec_plot(out_dir=out_dir, current_manifest=manifest, previous_manifest=previous_manifest)

    manifest_path = out_dir / "manifest.json"
    report_json = out_dir / "dataset_report.json"
    report_md = out_dir / "dataset_report.md"
    write_json(manifest_path, manifest)
    write_json(report_json, manifest)
    write_dataset_report_md(path=report_md, manifest=manifest, warnings=warnings, plot_files=plot_files, crop_plot=crop_plot)

    print(f"dataset_version={out_dir.name}")
    print(f"test_out={test_out}")
    print(f"train_pool_out={train_pool_out}")
    print(f"all_data_out={all_data_out}")
    print(f"fold_assignments_out={folds_out}")
    print(f"manifest={manifest_path}")
    print(f"dataset_report_md={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
