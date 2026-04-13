#!/usr/bin/env python3
"""Purpose: Run grouped cross-validation for Model B and compare against the served artifact"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
from collections.abc import Sequence
from statistics import mean, pstdev
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from birdsys.core import ensure_layout, next_version_dir
from birdsys.ml_experiments.model_b_evaluation import (
    CONFUSION_HEADS,
    evaluation_result_to_dict,
    evaluate_checkpoint_on_frame,
)
from birdsys.ml_experiments.train_attributes import (
    HEADS,
    ID_TO_LABELS,
    TrainCfg,
    collect_visible_label_counts,
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
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--schema-version", default="annotation_schema_v2")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-weighted-sampling", action="store_true")
    parser.add_argument("--progress-every-batches", type=int, default=20)
    parser.add_argument(
        "--old-model-checkpoint",
        help="Path to the old-model checkpoint file or served artifact directory",
        default=os.getenv(
            "MODEL_B_CHECKPOINT",
            "/home/antoine/bird_stance_classification/data/birds_project/models/attributes/served/model_b/current",
        ),
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

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    layout = ensure_layout(data_root)
    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        out_dir = next_version_dir(layout.models_attributes / "cv_reports", "attributes_cv")

    device = pick_device(effective.device)
    pool_df = pool_df.copy()
    fold_ids = sorted(pool_df["fold_id"].dropna().astype(int).unique().tolist())
    if not fold_ids:
        raise RuntimeError("No fold assignments found in train_pool.parquet")

    fold_rows: list[dict[str, Any]] = []
    aggregate_confusions = {
        head: np.zeros((len(ID_TO_LABELS[head]), len(ID_TO_LABELS[head])), dtype=np.int64)
        for head in CONFUSION_HEADS
    }
    old_aggregate_confusions = {
        head: np.zeros((len(ID_TO_LABELS[head]), len(ID_TO_LABELS[head])), dtype=np.int64)
        for head in CONFUSION_HEADS
    }
    old_model_checkpoint = pathlib.Path(args.old_model_checkpoint).expanduser().resolve()
    if not old_model_checkpoint.exists():
        raise FileNotFoundError(old_model_checkpoint)

    for fold_idx in fold_ids:
        fold_train = pool_df[pool_df["fold_id"] != fold_idx].drop(columns=["fold_id"]).reset_index(drop=True)
        fold_eval = pool_df[pool_df["fold_id"] == fold_idx].drop(columns=["fold_id"]).reset_index(drop=True)
        if fold_train.empty or fold_eval.empty:
            raise RuntimeError(f"Fold {fold_idx} is empty")

        print(
            "cv_fold_start "
            f"fold={fold_idx + 1}/{len(fold_ids)} "
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
        with TemporaryDirectory(prefix=f"model_b_cv_fold_{fold_idx + 1}_") as tmp_dir_name:
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
            new_eval = evaluate_checkpoint_on_frame(
                checkpoint_path=temp_checkpoint,
                frame=fold_eval,
                device=str(device),
            )
        old_eval = evaluate_checkpoint_on_frame(
            checkpoint_path=old_model_checkpoint,
            frame=fold_eval,
            device=str(device),
        )
        delta_metrics = {
            key: float(new_eval.metrics.get(key, 0.0)) - float(old_eval.metrics.get(key, 0.0))
            for key in sorted(set(new_eval.metrics) | set(old_eval.metrics))
        }

        row: dict[str, Any] = {
            "fold": fold_idx + 1,
            "train_rows": len(fold_train),
            "eval_rows": len(fold_eval),
            "supported_labels": result.supported_labels,
            "eval_visible_label_counts": result.eval_visible_label_counts,
            "fold_eval_label_counts": summarize_labels(fold_eval),
            "new_model_fold_metrics": new_eval.metrics,
            "old_model_fold_metrics": old_eval.metrics,
            "delta_metrics": delta_metrics,
            "new_model_diagnostics": evaluation_result_to_dict(new_eval)["diagnostics"],
            "old_model_diagnostics": evaluation_result_to_dict(old_eval)["diagnostics"],
        }
        fold_rows.append(row)
        for head, matrix in new_eval.confusion_matrices.items():
            aggregate_confusions[head] += matrix
        for head, matrix in old_eval.confusion_matrices.items():
            old_aggregate_confusions[head] += matrix

    metrics_summary: dict[str, dict[str, dict[str, float]]] = {"new_model": {}, "old_model": {}, "delta": {}}
    for scope, key_name in (
        ("new_model", "new_model_fold_metrics"),
        ("old_model", "old_model_fold_metrics"),
        ("delta", "delta_metrics"),
    ):
        for head in HEADS:
            for suffix in ("accuracy", "f1"):
                metric_key = f"{head}_{suffix}"
                values = [float((row.get(key_name) or {}).get(metric_key, 0.0)) for row in fold_rows]
                metrics_summary[scope][metric_key] = {
                    "mean": float(mean(values)) if values else 0.0,
                    "std": float(pstdev(values)) if len(values) > 1 else 0.0,
                    "min": float(min(values)) if values else 0.0,
                    "max": float(max(values)) if values else 0.0,
                }

    full_visible_counts = collect_visible_label_counts(pool_df.drop(columns=["fold_id"]))
    unsupported_labels = {
        head: [label for label, count in counts.items() if int(count) <= 0]
        for head, counts in full_visible_counts.items()
    }
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_version": dataset_dir.name,
        "schema_version": args.schema_version,
        "folds": len(fold_ids),
        "device": str(device),
        "pretrained": not args.no_pretrained,
        "weighted_sampling": not args.no_weighted_sampling,
        "old_model_checkpoint": str(old_model_checkpoint),
        "pool_rows": int(len(pool_df)),
        "pool_label_counts": summarize_labels(pool_df.drop(columns=["fold_id"])),
        "pool_visible_label_counts": full_visible_counts,
        "unsupported_labels": unsupported_labels,
        "folds_report": fold_rows,
        "metrics_summary": metrics_summary,
    }

    report_json = out_dir / "cv_report.json"
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fold_metrics_csv = out_dir / "cv_fold_metrics.csv"
    pd.DataFrame(fold_rows).to_csv(fold_metrics_csv, index=False)
    for head, matrix in aggregate_confusions.items():
        pd.DataFrame(matrix, index=ID_TO_LABELS[head], columns=ID_TO_LABELS[head]).to_csv(
            out_dir / f"cv_new_model_{head}_confusion_matrix.csv"
        )
    for head, matrix in old_aggregate_confusions.items():
        pd.DataFrame(matrix, index=ID_TO_LABELS[head], columns=ID_TO_LABELS[head]).to_csv(
            out_dir / f"cv_old_model_{head}_confusion_matrix.csv"
        )

    print(f"output_dir={out_dir}")
    print(f"report_json={report_json}")
    print(f"fold_metrics_csv={fold_metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
