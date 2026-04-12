#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from birdsys.paths import ensure_layout
from birdsys.training.model_b_evaluation import (
    CONFUSION_HEADS,
    ID_TO_LABELS,
    evaluation_result_to_dict,
    evaluate_checkpoint_on_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Model B artifact on a dataset split")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or artifact directory")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--schema-version", default="annotation_schema_v2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    layout = ensure_layout(data_root)
    dataset_dir = pathlib.Path(args.dataset_dir).expanduser().resolve()
    checkpoint_path = pathlib.Path(args.checkpoint).expanduser().resolve()
    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        artifact_label = checkpoint_path.stem if checkpoint_path.is_file() else checkpoint_path.name
        out_dir = layout.models_attributes / "evaluations" / f"{dataset_dir.name}_{artifact_label}_{args.split}"
        out_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate_checkpoint_on_dataset(
        checkpoint_path=checkpoint_path,
        dataset_dir=dataset_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    metrics_out = out_dir / "metrics.json"
    metrics_out.write_text(json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8")

    report_payload = evaluation_result_to_dict(result)
    report_payload.update(
        {
            "dataset_dir": str(dataset_dir),
            "split": args.split,
            "requested_schema_version": args.schema_version,
        }
    )
    report_json = out_dir / "report.json"
    report_json.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Model B Evaluation: {checkpoint_path.name}",
        "",
        f"- Dataset: `{dataset_dir.name}` split `{args.split}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Artifact mode: `{result.artifact_mode}`",
        f"- Device: `{result.device}`",
        f"- Schema version: `{result.schema_version}`",
        f"- Metrics: `{result.metrics}`",
        f"- Diagnostics: `{report_payload['diagnostics']}`",
    ]
    report_md = out_dir / "report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for head in CONFUSION_HEADS:
        csv_path = out_dir / f"{head}_confusion_matrix.csv"
        pd.DataFrame(
            result.confusion_matrices[head],
            index=ID_TO_LABELS[head],
            columns=ID_TO_LABELS[head],
        ).to_csv(csv_path)

    print(f"output_dir={out_dir}")
    print(f"metrics={metrics_out}")
    print(f"report_json={report_json}")
    print(f"report_md={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
