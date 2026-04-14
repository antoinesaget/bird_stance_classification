#!/usr/bin/env python3
"""Purpose: Build stable leakage-safe split artifacts from normalized annotations."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from collections import Counter
from collections.abc import Sequence
from typing import Any

from birdsys.core import default_data_home, default_species_slug, diff_numeric_dict, ensure_layout, next_version_dir
from birdsys.datasets.dataset_common import (
    LABEL_FIELDS,
    absent_class_warnings,
    add_bar_headroom,
    annotate_bars,
    diff_nested_counts,
    embed_plot_block,
    format_count,
    import_pyplot,
    percentage_distribution,
    json_safe,
    save_plot,
    visible_label_counts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stable grouped test/train-pool/fold split artifacts")
    parser.add_argument("--annotation-version", required=True, help="ann_vNNN")
    parser.add_argument("--data-home", default=str(default_data_home()))
    parser.add_argument("--species-slug", default=default_species_slug())
    parser.add_argument("--split-version", default="", help="Optional explicit output folder name, e.g. split_v001")
    parser.add_argument("--test-pct", type=float, default=20.0, help="Approximate test target as a percent of bird rows")
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args(argv)


def _group_id(site_id: Any, image_id: str) -> str:
    prefix = "" if site_id is None else str(site_id)
    return f"{prefix}:{image_id}"


def _presence_tokens(frame) -> set[str]:
    tokens: set[str] = set()
    for field in LABEL_FIELDS:
        if field not in frame.columns:
            continue
        values = frame[field].dropna().astype(str).tolist()
        for value in values:
            tokens.add(f"{field}={value}")
    return tokens


def load_frames(*, birds_path: pathlib.Path, images_path: pathlib.Path):
    import pandas as pd

    birds_df = pd.read_parquet(birds_path)
    images_df = pd.read_parquet(images_path)
    birds_df["image_id"] = birds_df["image_id"].astype(str)
    birds_df["bird_id"] = birds_df["bird_id"].astype(str)
    images_df["image_id"] = images_df["image_id"].astype(str)
    return birds_df, images_df


def version_index(name: str, prefix: str) -> int | None:
    token = f"{prefix}_v"
    if not name.startswith(token):
        return None
    suffix = name[len(token) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def list_previous_version_dirs(parent: pathlib.Path, *, prefix: str, current_name: str) -> list[pathlib.Path]:
    current_idx = version_index(current_name, prefix)
    if current_idx is None or not parent.exists():
        return []

    candidates: list[tuple[int, pathlib.Path]] = []
    for path in parent.iterdir():
        if not path.is_dir():
            continue
        idx = version_index(path.name, prefix)
        if idx is None or idx >= current_idx:
            continue
        candidates.append((idx, path))
    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates]


def read_split_image_ids(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()

    import pandas as pd

    return set(pd.read_parquet(path)["image_id"].astype(str).tolist())


def build_group_table(*, birds_df, images_df, annotation_version: str):
    import pandas as pd

    image_lookup = images_df.drop_duplicates(subset=["image_id"]).set_index("image_id")
    rows: list[dict[str, Any]] = []
    for image_id, frame in birds_df.groupby("image_id", sort=True):
        image_meta = image_lookup.loc[image_id] if image_id in image_lookup.index else None
        species_slug = None if image_meta is None else image_meta.get("species_slug")
        site_id = None if image_meta is None else image_meta.get("site_id")
        source_filename = None if image_meta is None else image_meta.get("source_filename")
        original_relpath = None if image_meta is None else image_meta.get("original_relpath")
        compressed_relpath = None if image_meta is None else image_meta.get("compressed_relpath")
        image_usable = None if image_meta is None else image_meta.get("image_usable")
        tokens = sorted(_presence_tokens(frame))
        rows.append(
            {
                "annotation_version": annotation_version,
                "species_slug": None if pd.isna(species_slug) else str(species_slug),
                "image_id": str(image_id),
                "site_id": None if pd.isna(site_id) else str(site_id),
                "source_filename": None if pd.isna(source_filename) else str(source_filename),
                "original_relpath": None if pd.isna(original_relpath) else str(original_relpath),
                "compressed_relpath": None if pd.isna(compressed_relpath) else str(compressed_relpath),
                "image_usable": None if pd.isna(image_usable) else bool(image_usable),
                "group_id": _group_id(None if pd.isna(site_id) else str(site_id), str(image_id)),
                "bird_rows": int(len(frame)),
                "tokens": tokens,
                "tokens_json": json.dumps(tokens),
            }
        )
    return pd.DataFrame(rows).sort_values(["image_id"]).reset_index(drop=True)


def token_totals(groups_df) -> Counter[str]:
    totals: Counter[str] = Counter()
    for tokens in groups_df["tokens"]:
        totals.update(tokens)
    return totals


def rarity_score(tokens: list[str], totals: Counter[str]) -> float:
    score = 0.0
    for token in tokens:
        count = max(1, int(totals.get(token, 0)))
        score += 1.0 / float(count)
    return score


def incremental_label_cost(
    *,
    current: Counter[str],
    desired: dict[str, float],
    tokens: list[str],
) -> float:
    delta = 0.0
    for token in tokens:
        target = float(desired.get(token, 0.0))
        before = abs(float(current.get(token, 0)) - target)
        after = abs(float(current.get(token, 0) + 1) - target)
        delta += after - before
    return delta


def select_test_groups(
    groups_df,
    *,
    historical_test_ids: set[str],
    historical_train_pool_ids: set[str],
    target_rows: int,
):
    records = groups_df.to_dict("records")
    by_image_id = {str(row["image_id"]): row for row in records}
    current_ids = set(by_image_id)
    preserved = sorted(current_ids & set(historical_test_ids))
    removed = sorted(set(historical_test_ids) - current_ids)
    blocked_historical_train = current_ids & (set(historical_train_pool_ids) - set(historical_test_ids))
    eligible_new_ids = current_ids - set(historical_test_ids) - set(historical_train_pool_ids)

    total_rows = int(groups_df["bird_rows"].sum())
    target_fraction = (float(target_rows) / float(total_rows)) if total_rows > 0 else 0.0
    totals = token_totals(groups_df)
    desired = {token: float(count) * target_fraction for token, count in totals.items()}

    test_ids = set(preserved)
    current_rows = sum(int(by_image_id[image_id]["bird_rows"]) for image_id in preserved)
    current_counts: Counter[str] = Counter()
    for image_id in preserved:
        current_counts.update(by_image_id[image_id]["tokens"])

    remaining = [row for row in records if str(row["image_id"]) in eligible_new_ids]
    while current_rows < target_rows and remaining:
        best_row: dict[str, Any] | None = None
        best_score: tuple[float, float, float] | None = None
        for row in sorted(remaining, key=lambda item: str(item["group_id"])):
            row_tokens = list(row["tokens"])
            new_rows = current_rows + int(row["bird_rows"])
            row_gap = abs(float(target_rows - new_rows)) / max(1.0, float(target_rows))
            overshoot = max(0.0, float(new_rows - target_rows)) / max(1.0, float(target_rows))
            label_cost = incremental_label_cost(current=current_counts, desired=desired, tokens=row_tokens)
            score = (
                label_cost + (0.35 * row_gap) + (0.80 * overshoot),
                -rarity_score(row_tokens, totals),
                float(row["bird_rows"]),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_row = row
        if best_row is None:
            break
        image_id = str(best_row["image_id"])
        test_ids.add(image_id)
        current_rows += int(best_row["bird_rows"])
        current_counts.update(best_row["tokens"])
        remaining = [row for row in remaining if str(row["image_id"]) != image_id]

    membership = {
        image_id: ("preserved" if image_id in preserved else "added")
        for image_id in sorted(test_ids)
    }
    return test_ids, membership, removed, {
        "eligible_new_ids": sorted(eligible_new_ids),
        "blocked_historical_train_ids": sorted(blocked_historical_train),
        "target_rows": int(target_rows),
        "actual_rows": int(current_rows),
        "shortfall_rows": max(0, int(target_rows) - int(current_rows)),
    }


def assign_pool_folds(pool_df, *, folds: int):
    if pool_df.empty:
        return {}

    totals = token_totals(pool_df)
    desired_rows = float(pool_df["bird_rows"].sum()) / float(max(1, folds))
    desired = {token: float(count) / float(max(1, folds)) for token, count in totals.items()}
    fold_rows = {fold_id: 0 for fold_id in range(folds)}
    fold_counts = {fold_id: Counter() for fold_id in range(folds)}
    assignments: dict[str, int] = {}

    order = sorted(
        pool_df.to_dict("records"),
        key=lambda row: (
            -rarity_score(list(row["tokens"]), totals),
            -int(row["bird_rows"]),
            str(row["group_id"]),
        ),
    )

    for row in order:
        best_fold = 0
        best_score: tuple[float, float, int] | None = None
        for fold_id in range(folds):
            row_gap = abs(float(fold_rows[fold_id] + int(row["bird_rows"])) - desired_rows) / max(1.0, desired_rows)
            label_cost = incremental_label_cost(current=fold_counts[fold_id], desired=desired, tokens=list(row["tokens"]))
            score = (label_cost + (0.25 * row_gap), row_gap, fold_id)
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold_id
        image_id = str(row["image_id"])
        assignments[image_id] = int(best_fold)
        fold_rows[best_fold] += int(row["bird_rows"])
        fold_counts[best_fold].update(row["tokens"])
    return assignments


def subset_counts(birds_df, image_ids: set[str]) -> dict[str, dict[str, int]]:
    if not image_ids:
        return {field: {} for field in LABEL_FIELDS}
    return visible_label_counts(birds_df[birds_df["image_id"].isin(sorted(image_ids))].copy())


def fold_support_counts(birds_df, assignments: dict[str, int], *, folds: int) -> dict[int, dict[str, dict[str, int]]]:
    out: dict[int, dict[str, dict[str, int]]] = {}
    for fold_id in range(folds):
        image_ids = {image_id for image_id, assigned in assignments.items() if int(assigned) == fold_id}
        out[fold_id] = subset_counts(birds_df, image_ids)
    return out


def split_counts(*, birds_df, test_ids: set[str], pool_ids: set[str]) -> dict[str, int]:
    test_mask = birds_df["image_id"].isin(sorted(test_ids))
    pool_mask = birds_df["image_id"].isin(sorted(pool_ids))
    return {
        "rows_total": int(len(birds_df)),
        "rows_test": int(test_mask.sum()),
        "rows_train_pool": int(pool_mask.sum()),
        "images_total": int(birds_df["image_id"].nunique()),
        "images_test": int(len(test_ids)),
        "images_train_pool": int(len(pool_ids)),
    }


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def render_split_plots(
    *,
    out_dir: pathlib.Path,
    counts: dict[str, int],
    split_supports: dict[str, dict[str, dict[str, int]]],
    fold_supports: dict[int, dict[str, dict[str, int]]],
    assignments_df,
    membership_summary: dict[str, int],
    previous_counts: dict[str, int] | None,
    previous_supports: dict[str, dict[str, dict[str, int]]] | None,
) -> dict[str, dict[str, str]]:
    plt = import_pyplot()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, str]] = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    row_labels = ["test", "train_pool", "all_data"]
    row_values = [counts["rows_test"], counts["rows_train_pool"], counts["rows_total"]]
    bars = axes[0].bar(range(len(row_labels)), row_values, color="#2F6B8A")
    add_bar_headroom(axes[0], row_values)
    axes[0].set_xticks(range(len(row_labels)))
    axes[0].set_xticklabels(row_labels, rotation=15, ha="right")
    axes[0].set_title("Split Row Counts")
    annotate_bars(axes[0], bars, [format_count(value) for value in row_values], fontsize=9)

    image_labels = ["test", "train_pool", "all_images"]
    image_values = [counts["images_test"], counts["images_train_pool"], counts["images_total"]]
    bars = axes[1].bar(range(len(image_labels)), image_values, color="#5F8A45")
    add_bar_headroom(axes[1], image_values)
    axes[1].set_xticks(range(len(image_labels)))
    axes[1].set_xticklabels(image_labels, rotation=15, ha="right")
    axes[1].set_title("Split Image Counts")
    annotate_bars(axes[1], bars, [format_count(value) for value in image_values], fontsize=9)
    out["current_split_counts"] = save_plot(fig, plots_dir / "current_split_counts")
    plt.close(fig)

    roles = ["all_data", "test", "train_pool"]
    colors = {"all_data": "#457B9D", "test": "#E76F51", "train_pool": "#2A9D8F"}
    fig, axes = plt.subplots(3, 2, figsize=(14, 12.5))
    for ax, field in zip(axes.flat, LABEL_FIELDS):
        labels = sorted(
            set(split_supports["all_data"].get(field, {}))
            | set(split_supports["test"].get(field, {}))
            | set(split_supports["train_pool"].get(field, {}))
        )
        if not labels:
            ax.text(0.5, 0.5, "No visible labels", ha="center", va="center")
            ax.set_axis_off()
            continue
        width = 0.24
        positions = list(range(len(labels)))
        for idx, role in enumerate(roles):
            distribution = percentage_distribution(split_supports[role].get(field, {}))
            values = [float(distribution.get(label, 0.0)) for label in labels]
            x_values = [position + ((idx - 1) * width) for position in positions]
            bars = ax.bar(x_values, values, width=width, label=role, color=colors[role])
            annotate_bars(ax, bars, [f"{value:.1f}%" for value in values], fontsize=7, padding=2.0)
        add_bar_headroom(ax, [100.0], extra_ratio=0.05, min_top=100.0)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(field)
        ax.set_ylabel("Percent")
    axes[0][0].legend(loc="upper right")
    fig.suptitle("Current Class Supports", y=0.995)
    out["current_class_supports"] = save_plot(fig, plots_dir / "current_class_supports", tight_rect=(0.0, 0.0, 1.0, 0.97))
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12.5))
    fold_ids = sorted(fold_supports)
    palette = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
    for ax, field in zip(axes.flat, LABEL_FIELDS):
        labels = sorted({label for counts_map in fold_supports.values() for label in counts_map.get(field, {})})
        if not labels:
            ax.text(0.5, 0.5, "No visible labels", ha="center", va="center")
            ax.set_axis_off()
            continue
        width = 0.14
        positions = list(range(len(labels)))
        center_offset = (len(fold_ids) - 1) / 2.0
        for idx, fold_id in enumerate(fold_ids):
            values = [int(fold_supports[fold_id].get(field, {}).get(label, 0)) for label in labels]
            x_values = [position + ((idx - center_offset) * width) for position in positions]
            ax.bar(x_values, values, width=width, label=f"fold {fold_id}", color=palette[idx % len(palette)])
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(field)
    axes[0][0].legend(loc="upper right")
    fig.suptitle("Per-Fold Class Supports", y=0.995)
    out["fold_class_supports"] = save_plot(fig, plots_dir / "fold_class_supports", tight_rect=(0.0, 0.0, 1.0, 0.97))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fold_row_counts = [int(assignments_df[assignments_df["fold_id"] == fold_id]["bird_rows"].sum()) for fold_id in fold_ids]
    bars = axes[0].bar(range(len(fold_ids)), fold_row_counts, color="#6D597A")
    add_bar_headroom(axes[0], fold_row_counts)
    axes[0].set_xticks(range(len(fold_ids)))
    axes[0].set_xticklabels([f"fold {fold_id}" for fold_id in fold_ids], rotation=15, ha="right")
    axes[0].set_title("Fold Rows")
    annotate_bars(axes[0], bars, [format_count(value) for value in fold_row_counts], fontsize=8)
    fold_image_counts = [int((assignments_df["fold_id"] == fold_id).sum()) for fold_id in fold_ids]
    bars = axes[1].bar(range(len(fold_ids)), fold_image_counts, color="#B56576")
    add_bar_headroom(axes[1], fold_image_counts)
    axes[1].set_xticks(range(len(fold_ids)))
    axes[1].set_xticklabels([f"fold {fold_id}" for fold_id in fold_ids], rotation=15, ha="right")
    axes[1].set_title("Fold Images")
    annotate_bars(axes[1], bars, [format_count(value) for value in fold_image_counts], fontsize=8)
    out["fold_sizes"] = save_plot(fig, plots_dir / "fold_sizes")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12.5))
    for ax, field in zip(axes.flat, LABEL_FIELDS):
        labels = sorted(split_supports["all_data"].get(field, {}))
        if not labels:
            ax.text(0.5, 0.5, "No visible labels", ha="center", va="center")
            ax.set_axis_off()
            continue
        width = 0.34
        positions = list(range(len(labels)))
        all_values = [int(split_supports["all_data"].get(field, {}).get(label, 0)) for label in labels]
        test_values = [int(split_supports["test"].get(field, {}).get(label, 0)) for label in labels]
        all_bars = ax.bar(
            [position - (width / 2.0) for position in positions],
            all_values,
            width=width,
            color="#C9D6DF",
            label="all_data",
        )
        test_bars = ax.bar(
            [position + (width / 2.0) for position in positions],
            test_values,
            width=width,
            color="#8AB17D",
            label="test",
        )
        add_bar_headroom(ax, all_values + test_values, extra_ratio=0.22, min_top=5.0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(field)
        annotate_bars(ax, all_bars, [format_count(value) for value in all_values], fontsize=7)
        annotate_bars(ax, test_bars, [format_count(value) for value in test_values], fontsize=7)
        ax.set_ylabel("Count")
    axes[0][0].legend(loc="upper right")
    fig.suptitle("Test Support Coverage vs All Data (Counts)", y=0.995)
    out["test_support_coverage"] = save_plot(fig, plots_dir / "test_support_coverage", tight_rect=(0.0, 0.0, 1.0, 0.97))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    churn_labels = ["preserved", "added", "removed"]
    churn_values = [int(membership_summary.get(label, 0)) for label in churn_labels]
    bars = ax.bar(range(len(churn_labels)), churn_values, color="#BC6C25")
    add_bar_headroom(ax, churn_values)
    ax.set_xticks(range(len(churn_labels)))
    ax.set_xticklabels(churn_labels, rotation=15, ha="right")
    ax.set_title("Test Membership Churn")
    annotate_bars(ax, bars, [format_count(value) for value in churn_values], fontsize=9)
    out["test_membership_churn"] = save_plot(fig, plots_dir / "test_membership_churn")
    plt.close(fig)

    if previous_counts is not None and previous_supports is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        row_delta = {
            "test": int(counts["rows_test"]) - int(previous_counts.get("rows_test", 0)),
            "train_pool": int(counts["rows_train_pool"]) - int(previous_counts.get("rows_train_pool", 0)),
            "all_data": int(counts["rows_total"]) - int(previous_counts.get("rows_total", 0)),
        }
        row_labels = list(row_delta)
        row_values = [row_delta[label] for label in row_labels]
        axes[0].bar(range(len(row_labels)), row_values, color=["#2A9D8F" if value >= 0 else "#E76F51" for value in row_values])
        axes[0].axhline(0, color="#333333", linewidth=1)
        axes[0].set_xticks(range(len(row_labels)))
        axes[0].set_xticklabels(row_labels, rotation=15, ha="right")
        axes[0].set_title("Row Count Deltas")

        image_delta = {
            "test": int(counts["images_test"]) - int(previous_counts.get("images_test", 0)),
            "train_pool": int(counts["images_train_pool"]) - int(previous_counts.get("images_train_pool", 0)),
            "all_images": int(counts["images_total"]) - int(previous_counts.get("images_total", 0)),
        }
        image_labels = list(image_delta)
        image_values = [image_delta[label] for label in image_labels]
        axes[1].bar(range(len(image_labels)), image_values, color=["#2A9D8F" if value >= 0 else "#E76F51" for value in image_values])
        axes[1].axhline(0, color="#333333", linewidth=1)
        axes[1].set_xticks(range(len(image_labels)))
        axes[1].set_xticklabels(image_labels, rotation=15, ha="right")
        axes[1].set_title("Image Count Deltas")
        out["comparison_count_deltas"] = save_plot(fig, plots_dir / "comparison_count_deltas")
        plt.close(fig)

        support_delta = {
            role: diff_nested_counts(split_supports[role], previous_supports.get(role, {}))
            for role in roles
        }
        fig, axes = plt.subplots(3, 2, figsize=(14, 12.5))
        for ax, field in zip(axes.flat, LABEL_FIELDS):
            labels = sorted({label for role in roles for label in support_delta[role].get(field, {})})
            if not labels:
                ax.text(0.5, 0.5, "No visible labels", ha="center", va="center")
                ax.set_axis_off()
                continue
            width = 0.24
            positions = list(range(len(labels)))
            for idx, role in enumerate(roles):
                values = [int(support_delta[role].get(field, {}).get(label, 0)) for label in labels]
                x_values = [position + ((idx - 1) * width) for position in positions]
                colors = ["#2A9D8F" if value >= 0 else "#E76F51" for value in values]
                ax.bar(x_values, values, width=width, label=role, color=colors)
            ax.axhline(0, color="#333333", linewidth=1)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_title(field)
        axes[0][0].legend(loc="upper right")
        fig.suptitle("Class Support Deltas", y=0.995)
        out["comparison_class_support_deltas"] = save_plot(
            fig,
            plots_dir / "comparison_class_support_deltas",
            tight_rect=(0.0, 0.0, 1.0, 0.97),
        )
        plt.close(fig)

        previous_fold_supports = previous_supports.get("folds", {})
        fig, axes = plt.subplots(3, 2, figsize=(14, 12.5))
        for ax, field in zip(axes.flat, LABEL_FIELDS):
            labels = sorted({label for counts_map in fold_supports.values() for label in counts_map.get(field, {})})
            if not labels:
                ax.text(0.5, 0.5, "No visible labels", ha="center", va="center")
                ax.set_axis_off()
                continue
            width = 0.14
            positions = list(range(len(labels)))
            center_offset = (len(fold_ids) - 1) / 2.0
            for idx, fold_id in enumerate(fold_ids):
                previous_fold = previous_fold_supports.get(str(fold_id), {})
                values = [
                    int(fold_supports[fold_id].get(field, {}).get(label, 0)) - int(previous_fold.get(field, {}).get(label, 0))
                    for label in labels
                ]
                x_values = [position + ((idx - center_offset) * width) for position in positions]
                colors = ["#2A9D8F" if value >= 0 else "#E76F51" for value in values]
                ax.bar(x_values, values, width=width, label=f"fold {fold_id}", color=colors)
            ax.axhline(0, color="#333333", linewidth=1)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_title(field)
        axes[0][0].legend(loc="upper right")
        fig.suptitle("Per-Fold Support Deltas", y=0.995)
        out["comparison_fold_support_deltas"] = save_plot(
            fig,
            plots_dir / "comparison_fold_support_deltas",
            tight_rect=(0.0, 0.0, 1.0, 0.97),
        )
        plt.close(fig)

    return out


def write_split_report_md(
    *,
    path: pathlib.Path,
    manifest: dict[str, Any],
    warnings: list[str],
    plot_files: dict[str, dict[str, str]],
) -> None:
    counts = manifest["counts"]
    membership = manifest["test_membership"]
    comparison = manifest.get("comparison_to_previous")
    lines = [
        f"# Split Report: {manifest['split_version']}",
        "",
        f"- Species slug: `{manifest['species_slug']}`",
        f"- Source annotation version: `{manifest['source_annotation_version']}`",
        f"- Grouping key: `{manifest['split_policy']['grouping_key']}`",
        f"- Test target percent: `{manifest['split_policy']['test_pct_target']:.3f}%`",
        f"- Test growth policy: `{manifest['split_policy']['test_membership_policy']}`",
        f"- Rows: total `{counts['rows_total']}`, test `{counts['rows_test']}`, train_pool `{counts['rows_train_pool']}`",
        f"- Images: total `{counts['images_total']}`, test `{counts['images_test']}`, train_pool `{counts['images_train_pool']}`",
        f"- Test membership: preserved `{membership['preserved']}`, added `{membership['added']}`, removed `{membership['removed']}`, shortfall rows `{membership['shortfall_rows']}`",
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
    if comparison is not None:
        lines.extend(embed_plot_block("Comparison Count Deltas", "comparison_count_deltas", plot_files))
        lines.extend(embed_plot_block("Comparison Class Support Deltas", "comparison_class_support_deltas", plot_files))
        lines.extend(embed_plot_block("Comparison Fold Support Deltas", "comparison_fold_support_deltas", plot_files))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    layout = ensure_layout(pathlib.Path(args.data_home), args.species_slug)

    birds_path = layout.labelstudio_normalized / args.annotation_version / "birds.parquet"
    images_path = layout.labelstudio_normalized / args.annotation_version / "images_labels.parquet"
    if not birds_path.exists():
        raise FileNotFoundError(birds_path)
    if not images_path.exists():
        raise FileNotFoundError(images_path)

    if args.split_version:
        out_dir = layout.derived_splits / args.split_version
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        out_dir = next_version_dir(layout.derived_splits, "split")

    birds_df, images_df = load_frames(birds_path=birds_path, images_path=images_path)
    groups_df = build_group_table(birds_df=birds_df, images_df=images_df, annotation_version=args.annotation_version)

    previous_split_dirs = list_previous_version_dirs(layout.derived_splits, prefix="split", current_name=out_dir.name)
    previous_dir = previous_split_dirs[-1] if previous_split_dirs else None
    historical_test_ids: set[str] = set()
    historical_train_pool_ids: set[str] = set()
    previous_counts: dict[str, int] | None = None
    previous_supports: dict[str, dict[str, dict[str, int]]] | None = None
    for prior_dir in previous_split_dirs:
        historical_test_ids.update(read_split_image_ids(prior_dir / "test_groups.parquet"))
        historical_train_pool_ids.update(read_split_image_ids(prior_dir / "train_pool_groups.parquet"))
    if previous_dir is not None:
        prev_manifest_path = previous_dir / "split_manifest.json"
        if prev_manifest_path.exists():
            previous_manifest = json.loads(prev_manifest_path.read_text(encoding="utf-8"))
            previous_counts = previous_manifest.get("counts")
            previous_supports = previous_manifest.get("split_supports")

    target_rows = int(round((float(args.test_pct) / 100.0) * float(len(birds_df))))
    test_ids, membership_status, removed_previous, selection_stats = select_test_groups(
        groups_df,
        historical_test_ids=historical_test_ids,
        historical_train_pool_ids=historical_train_pool_ids,
        target_rows=target_rows,
    )
    pool_ids = set(groups_df["image_id"].astype(str).tolist()) - set(test_ids)
    pool_df = groups_df[groups_df["image_id"].isin(sorted(pool_ids))].copy().reset_index(drop=True)
    fold_map = assign_pool_folds(pool_df, folds=int(args.folds))

    test_groups_df = groups_df[groups_df["image_id"].isin(sorted(test_ids))].copy().reset_index(drop=True)
    test_groups_df["split_role"] = "test"
    test_groups_df["membership_status"] = test_groups_df["image_id"].map(lambda value: membership_status[str(value)])

    train_pool_groups_df = pool_df.copy()
    train_pool_groups_df["split_role"] = "train_pool"
    train_pool_groups_df["membership_status"] = "train_pool"

    import pandas as pd

    fold_assignments_df = train_pool_groups_df[
        ["annotation_version", "image_id", "site_id", "group_id", "bird_rows"]
    ].copy()
    fold_assignments_df["split_version"] = out_dir.name
    fold_assignments_df["fold_id"] = fold_assignments_df["image_id"].map(lambda value: int(fold_map[str(value)]))
    fold_assignments_df = fold_assignments_df.sort_values(["fold_id", "image_id"]).reset_index(drop=True)

    split_supports = {
        "all_data": visible_label_counts(birds_df),
        "test": subset_counts(birds_df, set(test_ids)),
        "train_pool": subset_counts(birds_df, set(pool_ids)),
    }
    fold_supports = fold_support_counts(birds_df, fold_map, folds=int(args.folds))
    warnings = absent_class_warnings(
        baseline=split_supports["all_data"],
        test_counts=split_supports["test"],
        fold_counts=fold_supports,
    )
    if int(selection_stats["shortfall_rows"]) > 0:
        warnings.append(
            "Test split is below target rows because only never-before-seen images are eligible for new test admissions."
        )
    counts = split_counts(birds_df=birds_df, test_ids=set(test_ids), pool_ids=set(pool_ids))

    membership_summary = {
        "preserved": sum(1 for status in membership_status.values() if status == "preserved"),
        "added": sum(1 for status in membership_status.values() if status == "added"),
        "removed": len(removed_previous),
    }
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split_version": out_dir.name,
        "species_slug": layout.species_slug,
        "source_annotation_version": args.annotation_version,
        "inputs": {
            "birds_parquet": str(birds_path),
            "images_labels_parquet": str(images_path),
        },
        "split_policy": {
            "grouping_key": "image_id",
            "test_pct_target": float(args.test_pct),
            "folds": int(args.folds),
            "test_membership_policy": "preserve_historical_test_and_top_up_from_new_images_only",
            "historical_membership_policy": "historical_train_pool_images_cannot_move_into_test",
            "stratification": "grouped_multilabel_visible_class_presence",
            "rare_class_policy": "best_effort_report_only",
        },
        "counts": counts,
        "split_supports": {
            "all_data": split_supports["all_data"],
            "test": split_supports["test"],
            "train_pool": split_supports["train_pool"],
            "folds": {str(fold_id): fold_supports[fold_id] for fold_id in sorted(fold_supports)},
        },
        "test_membership": {
            "preserved": membership_summary["preserved"],
            "added": membership_summary["added"],
            "removed": membership_summary["removed"],
            "removed_image_ids": removed_previous,
            "eligible_new_image_ids": selection_stats["eligible_new_ids"],
            "blocked_historical_train_image_ids": selection_stats["blocked_historical_train_ids"],
            "target_rows": selection_stats["target_rows"],
            "actual_rows": selection_stats["actual_rows"],
            "shortfall_rows": selection_stats["shortfall_rows"],
        },
        "warnings": warnings,
        "previous_split_version": None if previous_dir is None else previous_dir.name,
        "comparison_to_previous": None,
    }
    if previous_counts is not None and previous_supports is not None:
        manifest["comparison_to_previous"] = {
            "previous_split_version": previous_dir.name if previous_dir is not None else None,
            "delta_counts": diff_numeric_dict(counts, previous_counts),
            "delta_supports": {
                role: diff_nested_counts(split_supports[role], previous_supports.get(role, {}))
                for role in ("all_data", "test", "train_pool")
            },
        }

    plot_files = render_split_plots(
        out_dir=out_dir,
        counts=counts,
        split_supports=split_supports,
        fold_supports=fold_supports,
        assignments_df=fold_assignments_df,
        membership_summary=membership_summary,
        previous_counts=previous_counts,
        previous_supports=previous_supports,
    )

    test_path = out_dir / "test_groups.parquet"
    train_pool_path = out_dir / "train_pool_groups.parquet"
    fold_path = out_dir / "fold_assignments.parquet"
    test_groups_df.drop(columns=["tokens"]).to_parquet(test_path, index=False)
    train_pool_groups_df.drop(columns=["tokens"]).to_parquet(train_pool_path, index=False)
    fold_assignments_df.to_parquet(fold_path, index=False)

    split_manifest_path = out_dir / "split_manifest.json"
    split_report_json = out_dir / "split_report.json"
    split_report_md = out_dir / "split_report.md"
    write_json(split_manifest_path, manifest)
    write_json(split_report_json, manifest)
    write_split_report_md(path=split_report_md, manifest=manifest, warnings=warnings, plot_files=plot_files)

    print(f"split_version={out_dir.name}")
    print(f"test_groups={test_path}")
    print(f"train_pool_groups={train_pool_path}")
    print(f"fold_assignments={fold_path}")
    print(f"split_manifest={split_manifest_path}")
    print(f"split_report_md={split_report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
