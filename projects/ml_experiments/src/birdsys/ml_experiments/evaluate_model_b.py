#!/usr/bin/env python3
"""Purpose: Evaluate a Model B artifact on a dataset split and write a standard report package."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
from collections.abc import Sequence

from birdsys.core import ensure_layout, next_version_dir
from birdsys.ml_experiments.model_b_evaluation import (
    evaluate_checkpoint_on_dataset,
    evaluation_result_to_dict,
)
from birdsys.ml_experiments.common import resolve_default_served_model_b_artifact_path
from birdsys.ml_experiments.reports import write_evaluation_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Model B artifact on a dataset split.")
    parser.add_argument("--dataset-dir", required=True, help="Path to ds_vXXX with train_pool/test/all_data parquet files")
    parser.add_argument("--split", default="test", choices=["train_pool", "test", "all_data"])
    parser.add_argument("--artifact-path", default="")
    parser.add_argument("--baseline-artifact-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--data-home", default=os.getenv("BIRD_DATA_HOME", "/data/birds"))
    parser.add_argument("--species-slug", default=os.getenv("BIRD_SPECIES_SLUG", "black_winged_stilt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def default_served_artifact_path(*, data_home: pathlib.Path, species_slug: str) -> pathlib.Path:
    return resolve_default_served_model_b_artifact_path(data_home=data_home, species_slug=species_slug)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = pathlib.Path(args.dataset_dir).expanduser().resolve()
    data_home = pathlib.Path(args.data_home).expanduser().resolve()
    artifact_path = (
        pathlib.Path(args.artifact_path).expanduser().resolve()
        if args.artifact_path
        else default_served_artifact_path(data_home=data_home, species_slug=args.species_slug)
    )
    baseline_path = pathlib.Path(args.baseline_artifact_path).expanduser().resolve() if args.baseline_artifact_path else None
    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)
    if baseline_path is not None and not baseline_path.exists():
        raise FileNotFoundError(baseline_path)

    layout = ensure_layout(data_home, args.species_slug)
    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        out_dir = next_version_dir(layout.models_attributes / "evaluations", "attributes_eval")

    result = evaluate_checkpoint_on_dataset(
        checkpoint_path=artifact_path,
        dataset_dir=dataset_dir,
        split=args.split,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=args.device,
        baseline_checkpoint_path=baseline_path,
    )
    outputs = write_evaluation_report(out_dir, result)

    raw_json = out_dir / "evaluation.json"
    raw_json.write_text(json.dumps(evaluation_result_to_dict(result), indent=2) + "\n", encoding="utf-8")

    print(f"output_dir={out_dir}")
    print(f"summary_json={outputs['summary_json']}")
    print(f"summary_csv={outputs['summary_csv']}")
    print(f"report_md={outputs['report_md']}")
    print(f"per_class_metrics_csv={outputs['per_class_metrics_csv']}")
    print(f"predictions_parquet={outputs['predictions_parquet']}")
    print(f"evaluation_json={raw_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
