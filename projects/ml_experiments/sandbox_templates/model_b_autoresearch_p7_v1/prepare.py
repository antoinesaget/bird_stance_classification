#!/usr/bin/env python3
"""Purpose: Create the run directory and baseline inputs needed before a sandbox search starts"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

from common import (
    HEADS,
    RESULTS_HEADER,
    ROOT,
    SOURCE_CV_REPORT_JSON,
    SOURCE_DATASET_DIR,
    SOURCE_FINAL_REPORT_JSON,
    SOURCE_OLD_MODEL_EVAL_JSON,
    collect_visible_label_counts,
    sanitize_positive_frame,
    stable_fold_id,
    stable_hash_fraction,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the self-contained Model B autoresearch sandbox")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--force-copy", action="store_true")
    parser.add_argument("--inner-val-fraction", type=float, default=0.15)
    parser.add_argument("--inner-val-min-fraction", type=float, default=0.10)
    return parser.parse_args()


def _target_crop_name(row_id: str, source_crop: Path) -> str:
    digest = hashlib.sha1(str(source_crop).encode("utf-8")).hexdigest()[:10]
    return f"{row_id}_{digest}{source_crop.suffix.lower()}"


def build_inner_validation_split(
    fold_frame: pd.DataFrame,
    *,
    group_column: str = "group_id",
    preferred_val_fraction: float = 0.15,
    minimum_val_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    ranked_groups = sorted(fold_frame[group_column].astype(str).unique().tolist(), key=stable_hash_fraction)
    total_visible_stance = int(sum(collect_visible_label_counts(fold_frame)["stance"].values()))
    min_stance_rows = 1 if total_visible_stance < 10 else min(3, total_visible_stance)
    candidate_fractions = [preferred_val_fraction, 0.14, 0.13, 0.12, 0.11, minimum_val_fraction]

    chosen_train = fold_frame
    chosen_val = fold_frame.iloc[0:0].copy()
    chosen_fraction = minimum_val_fraction
    for fraction in candidate_fractions:
        val_group_count = max(1, round(len(ranked_groups) * fraction))
        val_groups = set(ranked_groups[:val_group_count])
        val_df = fold_frame[fold_frame[group_column].astype(str).isin(val_groups)].reset_index(drop=True)
        train_df = fold_frame[~fold_frame[group_column].astype(str).isin(val_groups)].reset_index(drop=True)
        if train_df.empty or val_df.empty:
            continue
        chosen_train = train_df
        chosen_val = val_df
        chosen_fraction = fraction
        visible_stance = int(sum(collect_visible_label_counts(val_df)["stance"].values()))
        if visible_stance >= min_stance_rows:
            break
    return chosen_train, chosen_val, float(chosen_fraction)


def main() -> int:
    args = parse_args()
    data_dir = ROOT / "data"
    crops_dir = data_dir / "crops"
    folds_dir = data_dir / "folds"
    baselines_dir = ROOT / "baselines"
    best_dir = ROOT / "best"
    runs_dir = ROOT / "runs"
    for path in [data_dir, crops_dir, folds_dir, baselines_dir, best_dir, runs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    source_train = SOURCE_DATASET_DIR / "train.parquet"
    frame = pd.read_parquet(source_train)
    frame = sanitize_positive_frame(frame)
    frame = frame.reset_index(drop=True)
    frame["row_id"] = [f"row_{idx:04d}" for idx in range(len(frame))]

    copied_paths: list[str] = []
    for _, row in frame.iterrows():
        source_crop = Path(str(row["crop_path"]))
        target_crop = crops_dir / _target_crop_name(str(row["row_id"]), source_crop)
        if args.force_copy or not target_crop.exists():
            shutil.copy2(source_crop, target_crop)
        copied_paths.append(str(target_crop))
    frame["source_crop_path"] = frame["crop_path"].astype(str)
    frame["crop_path"] = copied_paths
    frame["fold_id"] = frame["group_id"].astype(str).map(lambda value: stable_fold_id(value, args.folds))

    full_pool_path = data_dir / "full_pool.parquet"
    frame.to_parquet(full_pool_path, index=False)

    visible_label_counts = collect_visible_label_counts(frame)
    folds_manifest: dict[str, object] = {
        "source_dataset": str(SOURCE_DATASET_DIR),
        "source_train": str(source_train),
        "folds": args.folds,
        "rows": int(len(frame)),
        "images": int(frame["image_id"].nunique()),
        "visible_label_counts": visible_label_counts,
        "folds_detail": [],
    }
    for fold_idx in range(args.folds):
        train_df = frame[frame["fold_id"] != fold_idx].drop(columns=["fold_id"]).reset_index(drop=True)
        test_df = frame[frame["fold_id"] == fold_idx].drop(columns=["fold_id"]).reset_index(drop=True)
        inner_train_df, inner_val_df, inner_val_fraction = build_inner_validation_split(
            train_df,
            preferred_val_fraction=float(args.inner_val_fraction),
            minimum_val_fraction=float(args.inner_val_min_fraction),
        )
        train_path = folds_dir / f"fold_{fold_idx}_train.parquet"
        test_path = folds_dir / f"fold_{fold_idx}_test.parquet"
        inner_train_path = folds_dir / f"fold_{fold_idx}_inner_train.parquet"
        inner_val_path = folds_dir / f"fold_{fold_idx}_inner_val.parquet"
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        inner_train_df.to_parquet(inner_train_path, index=False)
        inner_val_df.to_parquet(inner_val_path, index=False)
        folds_manifest["folds_detail"].append(
            {
                "fold_id": fold_idx,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "inner_train_rows": int(len(inner_train_df)),
                "inner_val_rows": int(len(inner_val_df)),
                "train_images": int(train_df["image_id"].nunique()),
                "test_images": int(test_df["image_id"].nunique()),
                "train_groups": sorted(train_df["group_id"].astype(str).unique().tolist()),
                "test_groups": sorted(test_df["group_id"].astype(str).unique().tolist()),
                "inner_train_groups": sorted(inner_train_df["group_id"].astype(str).unique().tolist()),
                "inner_val_groups": sorted(inner_val_df["group_id"].astype(str).unique().tolist()),
                "train_parquet": str(train_path),
                "test_parquet": str(test_path),
                "inner_train_parquet": str(inner_train_path),
                "inner_val_parquet": str(inner_val_path),
                "inner_val_fraction": float(inner_val_fraction),
                "inner_val_visible_label_counts": collect_visible_label_counts(inner_val_df),
            }
        )
    write_json(folds_dir / "folds_manifest.json", folds_manifest)

    shutil.copy2(SOURCE_OLD_MODEL_EVAL_JSON, baselines_dir / "old_served_model_full_pool_eval.json")
    shutil.copy2(SOURCE_CV_REPORT_JSON, baselines_dir / "project7_convnext_baseline_cv.json")
    thresholds = {
        "objective_weights": {"stance_f1": 0.50, "behavior_f1": 0.30, "substrate_f1": 0.20},
        "guardrails": {"readability_f1_min": 0.30, "specie_f1_min": 0.30},
        "delta_to_keep": 0.020,
        "delta_to_keep_policy": "Empirical 3-seed baseline std on current baseline recipe, rounded up to nearest 0.001.",
        "baseline_seed_search_score_std": 0.019616862848070266,
        "baseline_anchor": {
            "run_id": "20260326_150758_sampler-cap-2p5-v1",
            "search_score": 0.5183739491703461,
            "score_floor": 0.5083739491703461,
        },
        "dynamics_acceptance": {
            "criterion": "selected_checkpoint_stability",
            "transition_spike_ratio_max": 2.0,
            "best_step_min_fraction": 0.20,
            "final_val_loss_worse_pct_max": 0.10,
            "final_val_search_score_retention_min": 0.95,
            "minimum_passing_folds": 4,
        },
        "heads": HEADS,
        "baseline_report": str(SOURCE_FINAL_REPORT_JSON),
    }
    write_json(baselines_dir / "reference_thresholds.json", thresholds)

    results_path = ROOT / "results.tsv"
    if not results_path.exists():
        results_path.write_text("\t".join(RESULTS_HEADER) + "\n", encoding="utf-8")

    best_default = {
        "run_id": None,
        "search_score": None,
        "status": "empty",
    }
    for best_name in ["research_best.json", "servable_best.json"]:
        target = best_dir / best_name
        if not target.exists():
            write_json(target, best_default)

    summary = {
        "rows": int(len(frame)),
        "images": int(frame["image_id"].nunique()),
        "folds": args.folds,
        "full_pool": str(full_pool_path),
        "results_tsv": str(results_path),
        "visible_label_counts": visible_label_counts,
    }
    write_json(ROOT / "prepare_report.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
