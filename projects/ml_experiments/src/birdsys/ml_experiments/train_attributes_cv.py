#!/usr/bin/env python3
"""Purpose: Run grouped cross-validation for Model B and compare against the served artifact."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
from collections.abc import Sequence
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from birdsys.core import ensure_layout, next_version_dir

from .common import CONFUSION_HEADS, HEADS, ID_TO_LABELS, collect_visible_label_counts, resolve_default_served_model_b_artifact_path
from .model_b_evaluation import (
    compare_evaluation_results,
    evaluate_checkpoint_on_frame,
    evaluation_result_to_dict,
)
from .reports import summarize_metric_distribution, write_cv_report_artifacts, write_evaluation_report
from .train_attributes import (
    TrainCfg,
    dataframe_from_split,
    load_cfg,
    pick_device,
    save_checkpoint,
    summarize_labels,
    train_model,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped 5-fold CV for Model B attributes on the train pool")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset dir with train_pool/test/all_data parquet files")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "train_attributes.yaml"))
    parser.add_argument("--data-home", default=os.getenv("BIRD_DATA_HOME", "/data/birds"))
    parser.add_argument("--species-slug", default=os.getenv("BIRD_SPECIES_SLUG", "black_winged_stilt"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--schema-version", default="annotation_schema_v2")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-weighted-sampling", action="store_true")
    parser.add_argument("--progress-every-batches", type=int, default=20)
    parser.add_argument(
        "--old-model-checkpoint",
        help="Path to the baseline checkpoint file or served artifact directory",
        default="",
    )
    return parser.parse_args(argv)


def load_train_pool_with_folds(*, dataset_dir: pathlib.Path, smoke: bool) -> pd.DataFrame:
    train_pool_path = dataset_dir / "train_pool.parquet"
    fold_assignments_path = dataset_dir / "fold_assignments.parquet"
    pool_df = dataframe_from_split(train_pool_path, smoke)
    if pool_df.empty:
        raise RuntimeError("train_pool.parquet is empty after filtering")
    if not fold_assignments_path.exists():
        raise FileNotFoundError(fold_assignments_path)

    assignments = pd.read_parquet(fold_assignments_path)[["image_id", "fold_id"]].drop_duplicates(subset=["image_id"]).copy()
    assignments["image_id"] = assignments["image_id"].astype(str)
    assignments["fold_id"] = assignments["fold_id"].astype(int)

    pool_df = pool_df.copy()
    pool_df["image_id"] = pool_df["image_id"].astype(str)
    pool_df = pool_df.merge(assignments, on="image_id", how="left", validate="many_to_one")
    if pool_df["fold_id"].isna().any():
        raise RuntimeError("train_pool rows missing fold_id from fold_assignments.parquet")
    return pool_df.reset_index(drop=True)


def _fold_row(
    *,
    fold_number: int,
    fold_train: pd.DataFrame,
    fold_eval: pd.DataFrame,
    candidate_result,
    baseline_result,
) -> dict[str, Any]:
    comparison = compare_evaluation_results(candidate_result, baseline_result)
    row: dict[str, Any] = {
        "fold": int(fold_number),
        "train_rows": int(len(fold_train)),
        "eval_rows": int(len(fold_eval)),
        "candidate_primary_score": float(candidate_result.aggregate_metrics["primary_score"]),
        "baseline_primary_score": float(baseline_result.aggregate_metrics["primary_score"]),
        "delta_primary_score": float(comparison.delta_aggregate_metrics.get("primary_score", 0.0)),
        "candidate_mean_balanced_accuracy": float(candidate_result.aggregate_metrics["mean_balanced_accuracy"]),
        "baseline_mean_balanced_accuracy": float(baseline_result.aggregate_metrics["mean_balanced_accuracy"]),
        "delta_mean_balanced_accuracy": float(comparison.delta_aggregate_metrics.get("mean_balanced_accuracy", 0.0)),
        "candidate_visible_support_total": float(candidate_result.aggregate_metrics["total_visible_support"]),
        "baseline_visible_support_total": float(baseline_result.aggregate_metrics["total_visible_support"]),
    }
    row.update({f"candidate_{key}": float(value) for key, value in candidate_result.summary_metrics.items()})
    row.update({f"baseline_{key}": float(value) for key, value in baseline_result.summary_metrics.items()})
    row.update({f"delta_{key}": float(value) for key, value in comparison.delta_summary_metrics.items()})
    row["candidate_report_summary"] = f"fold_{fold_number:02d}/candidate_eval/summary.json"
    row["baseline_report_summary"] = f"fold_{fold_number:02d}/baseline_eval/summary.json"
    return row


def _numeric_metric_keys(rows: list[dict[str, Any]], prefix: str) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if not key.startswith(prefix):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                keys.add(key.removeprefix(prefix))
    return sorted(keys)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_cfg(pathlib.Path(args.config).resolve())
    dataset_dir = pathlib.Path(args.dataset_dir).expanduser().resolve()
    pool_df = load_train_pool_with_folds(dataset_dir=dataset_dir, smoke=args.smoke)

    effective = cfg
    if args.smoke:
        effective = TrainCfg(
            backbone=cfg.backbone,
            image_size=min(cfg.image_size, 224),
            batch_size=min(cfg.batch_size, 8),
            num_workers=0,
            head_epochs=1,
            finetune_epochs=1,
            head_lr=cfg.head_lr,
            finetune_lr=cfg.finetune_lr,
            weight_decay=cfg.weight_decay,
            device=cfg.device,
        )

    data_home = pathlib.Path(args.data_home).expanduser().resolve()
    layout = ensure_layout(data_home, args.species_slug)
    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        out_dir = next_version_dir(layout.models_attributes / "cv_reports", "attributes_cv")

    device = pick_device(effective.device)
    fold_ids = sorted(pool_df["fold_id"].dropna().astype(int).unique().tolist())
    if not fold_ids:
        raise RuntimeError("No fold assignments found in train_pool.parquet")

    baseline_checkpoint = (
        pathlib.Path(args.old_model_checkpoint).expanduser().resolve()
        if args.old_model_checkpoint
        else resolve_default_served_model_b_artifact_path(data_home=data_home, species_slug=args.species_slug)
    )
    if not baseline_checkpoint.exists():
        raise FileNotFoundError(baseline_checkpoint)

    fold_rows: list[dict[str, Any]] = []
    aggregate_candidate_confusions = {
        head: np.zeros((len(ID_TO_LABELS[head]), len(ID_TO_LABELS[head])), dtype=np.int64)
        for head in CONFUSION_HEADS
    }
    aggregate_baseline_confusions = {
        head: np.zeros((len(ID_TO_LABELS[head]), len(ID_TO_LABELS[head])), dtype=np.int64)
        for head in CONFUSION_HEADS
    }

    for fold_idx in fold_ids:
        fold_number = int(fold_idx) + 1
        fold_train = pool_df[pool_df["fold_id"] != fold_idx].drop(columns=["fold_id"]).reset_index(drop=True)
        fold_eval = pool_df[pool_df["fold_id"] == fold_idx].drop(columns=["fold_id"]).reset_index(drop=True)
        if fold_train.empty or fold_eval.empty:
            raise RuntimeError(f"Fold {fold_idx} is empty")

        print(
            "cv_fold_start "
            f"fold={fold_number}/{len(fold_ids)} "
            f"train_rows={len(fold_train)} "
            f"eval_rows={len(fold_eval)}",
            flush=True,
        )

        result = train_model(
            train_df=fold_train,
            eval_df=fold_eval,
            cfg=effective,
            smoke=bool(args.smoke),
            pretrained=not args.no_pretrained,
            device=device,
            progress_every_batches=args.progress_every_batches,
            weighted_sampling=not args.no_weighted_sampling,
        )

        fold_dir = out_dir / f"fold_{fold_number:02d}"
        fold_dir.mkdir(parents=True, exist_ok=False)
        with TemporaryDirectory(prefix=f"model_b_cv_fold_{fold_number}_") as tmp_dir_name:
            tmp_dir = pathlib.Path(tmp_dir_name)
            temp_checkpoint = save_checkpoint(
                out_dir=tmp_dir,
                model=result.model,
                cfg=effective,
                pretrained=not args.no_pretrained,
                schema_version=args.schema_version,
                supported_labels=result.supported_labels,
                train_visible_label_counts=result.train_visible_label_counts,
            )
            candidate_result = evaluate_checkpoint_on_frame(
                checkpoint_path=temp_checkpoint,
                frame=fold_eval,
                device=str(device),
                baseline_checkpoint_path=baseline_checkpoint,
            )
        baseline_result = evaluate_checkpoint_on_frame(
            checkpoint_path=baseline_checkpoint,
            frame=fold_eval,
            device=str(device),
        )

        write_evaluation_report(fold_dir / "candidate_eval", candidate_result)
        write_evaluation_report(fold_dir / "baseline_eval", baseline_result)

        fold_rows.append(
            _fold_row(
                fold_number=fold_number,
                fold_train=fold_train,
                fold_eval=fold_eval,
                candidate_result=candidate_result,
                baseline_result=baseline_result,
            )
        )
        for head, matrix in candidate_result.confusion_matrices.items():
            aggregate_candidate_confusions[head] += matrix
        for head, matrix in baseline_result.confusion_matrices.items():
            aggregate_baseline_confusions[head] += matrix

    candidate_metric_keys = _numeric_metric_keys(fold_rows, "candidate_")
    baseline_metric_keys = _numeric_metric_keys(fold_rows, "baseline_")
    delta_metric_keys = _numeric_metric_keys(fold_rows, "delta_")

    summary_payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_version": dataset_dir.name,
        "schema_version": args.schema_version,
        "folds": len(fold_ids),
        "device": str(device),
        "pretrained": not args.no_pretrained,
        "weighted_sampling": not args.no_weighted_sampling,
        "baseline_checkpoint": str(baseline_checkpoint),
        "pool_rows": int(len(pool_df)),
        "pool_label_counts": summarize_labels(pool_df.drop(columns=["fold_id"])),
        "pool_visible_label_counts": collect_visible_label_counts(pool_df.drop(columns=["fold_id"])),
        "candidate_metrics_summary": {
            key: summarize_metric_distribution(fold_rows, f"candidate_{key}")
            for key in candidate_metric_keys
        },
        "baseline_metrics_summary": {
            key: summarize_metric_distribution(fold_rows, f"baseline_{key}")
            for key in baseline_metric_keys
        },
        "delta_metrics_summary": {
            key: summarize_metric_distribution(fold_rows, f"delta_{key}")
            for key in delta_metric_keys
        },
        "folds_report": fold_rows,
    }

    outputs = write_cv_report_artifacts(
        out_dir,
        summary_payload=summary_payload,
        fold_rows=fold_rows,
    )

    cv_report_json = out_dir / "cv_report.json"
    cv_report_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    for head, matrix in aggregate_candidate_confusions.items():
        pd.DataFrame(matrix, index=ID_TO_LABELS[head], columns=ID_TO_LABELS[head]).to_csv(
            out_dir / f"candidate_{head}_confusion_matrix.csv"
        )
    for head, matrix in aggregate_baseline_confusions.items():
        pd.DataFrame(matrix, index=ID_TO_LABELS[head], columns=ID_TO_LABELS[head]).to_csv(
            out_dir / f"baseline_{head}_confusion_matrix.csv"
        )

    print(f"output_dir={out_dir}")
    print(f"summary_json={outputs['summary_json']}")
    print(f"summary_csv={outputs['summary_csv']}")
    print(f"fold_metrics_csv={outputs['fold_metrics_csv']}")
    print(f"cv_report_json={cv_report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
