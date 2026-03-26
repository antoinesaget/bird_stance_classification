#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
from statistics import mean, pstdev
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from birdsys.paths import ensure_layout, next_version_dir
from train_attributes import (
    CONFUSION_HEADS,
    HEADS,
    ID_TO_LABELS,
    TrainCfg,
    collect_visible_label_counts,
    dataframe_from_split,
    load_cfg,
    pick_device,
    summarize_labels,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped 5-fold CV for Model B attributes on the train pool")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset dir with train/test parquet")
    parser.add_argument("--config", default="config/train_attributes.yaml")
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--schema-version", default="annotation_schema_v2")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-weighted-sampling", action="store_true")
    parser.add_argument("--progress-every-batches", type=int, default=20)
    return parser.parse_args()


def stable_fold_id(group_id: str, folds: int) -> int:
    digest = hashlib.sha1(group_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def main() -> int:
    args = parse_args()
    cfg = load_cfg(pathlib.Path(args.config).resolve())
    dataset_dir = pathlib.Path(args.dataset_dir).expanduser().resolve()
    train_path = dataset_dir / "train.parquet"
    pool_df = dataframe_from_split(train_path, args.smoke)
    if pool_df.empty:
        raise RuntimeError("train.parquet is empty after filtering")

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
    pool_df["fold_id"] = pool_df["group_id"].astype(str).map(lambda value: stable_fold_id(value, args.folds))

    fold_rows: list[dict[str, Any]] = []
    aggregate_confusions = {
        head: np.zeros((len(ID_TO_LABELS[head]), len(ID_TO_LABELS[head])), dtype=np.int64)
        for head in CONFUSION_HEADS
    }

    for fold_idx in range(args.folds):
        fold_train = pool_df[pool_df["fold_id"] != fold_idx].drop(columns=["fold_id"]).reset_index(drop=True)
        fold_eval = pool_df[pool_df["fold_id"] == fold_idx].drop(columns=["fold_id"]).reset_index(drop=True)
        if fold_train.empty or fold_eval.empty:
            raise RuntimeError(f"Fold {fold_idx} is empty")

        print(
            "cv_fold_start "
            f"fold={fold_idx + 1}/{args.folds} "
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

        row: dict[str, Any] = {
            "fold": fold_idx + 1,
            "train_rows": len(fold_train),
            "eval_rows": len(fold_eval),
            "supported_labels": result.supported_labels,
            "eval_visible_label_counts": result.eval_visible_label_counts,
            **result.final_metrics,
        }
        fold_rows.append(row)
        for head, matrix in result.confusion_matrices.items():
            aggregate_confusions[head] += matrix

    metrics_summary: dict[str, dict[str, float]] = {}
    for head in HEADS:
        for suffix in ("accuracy", "f1"):
            key = f"{head}_{suffix}"
            values = [float(row.get(key, 0.0)) for row in fold_rows]
            metrics_summary[key] = {
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
        "folds": args.folds,
        "device": str(device),
        "pretrained": not args.no_pretrained,
        "weighted_sampling": not args.no_weighted_sampling,
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
            out_dir / f"cv_{head}_confusion_matrix.csv"
        )

    print(f"output_dir={out_dir}")
    print(f"report_json={report_json}")
    print(f"fold_metrics_csv={fold_metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
