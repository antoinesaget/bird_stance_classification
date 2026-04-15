#!/usr/bin/env python3
"""Purpose: Reproduce the served Model B setup with the new train/CV pipeline and summarize the audit."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from birdsys.core import ensure_layout, next_version_dir

from .common import resolve_default_served_model_b_artifact_path


DEFAULT_DATASET_NAMES = ("ds_v002", "ds_v001")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
APPROXIMATION_NOTES = (
    "This audit uses the new minimal trainer, not the archived autoresearch loop.",
    "It does not reproduce historical inner-validation model selection, capped weighted sampling, or time-budgeted training.",
    "It does not reproduce separate backbone/head finetune learning rates, light train augmentations, or early stopping from the archived run.",
    "Candidate-versus-served comparisons are still meaningful for evaluating the current logging, CV, and report surfaces.",
)


@dataclass(frozen=True)
class DatasetAuditRun:
    dataset_dir: pathlib.Path
    dataset_name: str
    train_dir: pathlib.Path
    cv_dir: pathlib.Path
    served_eval_dir: pathlib.Path
    candidate_eval_dir: pathlib.Path
    checkpoint_path: pathlib.Path


def _train_main(argv: list[str]) -> int:
    from . import train_attributes as train_cli

    return int(train_cli.main(argv))


def _cv_main(argv: list[str]) -> int:
    from . import train_attributes_cv as cv_cli

    return int(cv_cli.main(argv))


def _evaluate_main(argv: list[str]) -> int:
    from . import evaluate_model_b as evaluate_cli

    return int(evaluate_cli.main(argv))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a served-model audit with the new train, CV, and evaluation pipeline."
    )
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=[],
        help="Explicit dataset dirs to audit. When omitted, defaults to ds_v002 then ds_v001 under data-home/species.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "train_attributes_served_audit.yaml"),
    )
    parser.add_argument("--data-home", default=os.getenv("BIRD_DATA_HOME", "/data/birds"))
    parser.add_argument("--species-slug", default=os.getenv("BIRD_SPECIES_SLUG", "black_winged_stilt"))
    parser.add_argument("--served-artifact-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every-batches", type=int, default=20)
    parser.add_argument("--smoke-first", action="store_true", help="Run a smoke train and smoke CV pass on the first dataset.")
    parser.add_argument("--smoke-dataset-name", default="ds_v002")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-weighted-sampling", action="store_true")
    return parser.parse_args(argv)


def _resolve_dataset_dirs(*, data_home: pathlib.Path, species_slug: str, requested: Sequence[str]) -> list[pathlib.Path]:
    if requested:
        return [pathlib.Path(item).expanduser().resolve() for item in requested]
    datasets_root = data_home / species_slug / "derived" / "datasets"
    return [datasets_root / name for name in DEFAULT_DATASET_NAMES]


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_path(path: pathlib.Path, *, start: pathlib.Path) -> str:
    return path.relative_to(start).as_posix()


def _metric_from_summary(summary: dict[str, Any], key: str) -> float:
    value = summary.get("aggregate_metrics", {}).get(key)
    return float(value) if value is not None else 0.0


def _summary_metric(summary: dict[str, Any], key: str) -> float:
    value = summary.get("summary_metrics", {}).get(key)
    return float(value) if value is not None else 0.0


def _distribution_mean(summary: dict[str, Any], key: str) -> float:
    value = summary.get("candidate_metrics_summary", {}).get(key, {}).get("mean")
    return float(value) if value is not None else 0.0


def _distribution_delta_mean(summary: dict[str, Any], key: str) -> float:
    value = summary.get("delta_metrics_summary", {}).get(key, {}).get("mean")
    return float(value) if value is not None else 0.0


def _read_served_provenance(served_artifact_path: pathlib.Path) -> dict[str, Any]:
    provenance: dict[str, Any] = {"served_artifact_path": str(served_artifact_path)}
    promotion_path = served_artifact_path / "promotion.json" if served_artifact_path.is_dir() else served_artifact_path.with_name("promotion.json")
    if promotion_path.exists():
        promotion = _load_json(promotion_path)
        provenance["promotion"] = promotion
        source_path = promotion.get("source_path")
        if source_path:
            source_checkpoint = pathlib.Path(str(source_path))
            provenance["archived_source_path"] = str(source_checkpoint)
            if source_checkpoint.name == "checkpoint.pt":
                provenance["archived_run_dir"] = str(source_checkpoint.parents[2])
    return provenance


def _run_smoke_pass(
    *,
    dataset_dir: pathlib.Path,
    config_path: pathlib.Path,
    served_artifact_path: pathlib.Path,
    out_dir: pathlib.Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    smoke_root = out_dir / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=False)

    smoke_train_dir = smoke_root / "train"
    smoke_cv_dir = smoke_root / "cv"

    train_rc = _train_main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--config",
            str(config_path),
            "--eval-split",
            "test",
            "--output-dir",
            str(smoke_train_dir),
            "--data-home",
            str(args.data_home),
            "--species-slug",
            args.species_slug,
            "--progress-every-batches",
            str(args.progress_every_batches),
            "--smoke",
            *(["--no-pretrained"] if args.no_pretrained else []),
            *(["--no-weighted-sampling"] if args.no_weighted_sampling else []),
        ]
    )
    if train_rc != 0:
        raise RuntimeError(f"Smoke train_attributes failed for {dataset_dir.name} with exit code {train_rc}")

    cv_rc = _cv_main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--config",
            str(config_path),
            "--old-model-checkpoint",
            str(served_artifact_path),
            "--output-dir",
            str(smoke_cv_dir),
            "--data-home",
            str(args.data_home),
            "--species-slug",
            args.species_slug,
            "--progress-every-batches",
            str(args.progress_every_batches),
            "--smoke",
            *(["--no-pretrained"] if args.no_pretrained else []),
            *(["--no-weighted-sampling"] if args.no_weighted_sampling else []),
        ]
    )
    if cv_rc != 0:
        raise RuntimeError(f"Smoke train_attributes_cv failed for {dataset_dir.name} with exit code {cv_rc}")

    return {
        "train_dir": str(smoke_train_dir),
        "cv_dir": str(smoke_cv_dir),
    }


def _run_dataset_audit(
    *,
    dataset_dir: pathlib.Path,
    config_path: pathlib.Path,
    served_artifact_path: pathlib.Path,
    out_dir: pathlib.Path,
    args: argparse.Namespace,
) -> DatasetAuditRun:
    train_dir = out_dir / "train_run"
    cv_dir = out_dir / "cv_run"
    served_eval_dir = out_dir / "served_test_eval"
    candidate_eval_dir = out_dir / "candidate_test_eval"

    train_rc = _train_main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--config",
            str(config_path),
            "--eval-split",
            "test",
            "--output-dir",
            str(train_dir),
            "--data-home",
            str(args.data_home),
            "--species-slug",
            args.species_slug,
            "--progress-every-batches",
            str(args.progress_every_batches),
            *(["--no-pretrained"] if args.no_pretrained else []),
            *(["--no-weighted-sampling"] if args.no_weighted_sampling else []),
        ]
    )
    if train_rc != 0:
        raise RuntimeError(f"train_attributes failed for {dataset_dir.name} with exit code {train_rc}")

    checkpoint_path = train_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    cv_rc = _cv_main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--config",
            str(config_path),
            "--old-model-checkpoint",
            str(served_artifact_path),
            "--output-dir",
            str(cv_dir),
            "--data-home",
            str(args.data_home),
            "--species-slug",
            args.species_slug,
            "--progress-every-batches",
            str(args.progress_every_batches),
            *(["--no-pretrained"] if args.no_pretrained else []),
            *(["--no-weighted-sampling"] if args.no_weighted_sampling else []),
        ]
    )
    if cv_rc != 0:
        raise RuntimeError(f"train_attributes_cv failed for {dataset_dir.name} with exit code {cv_rc}")

    served_eval_rc = _evaluate_main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--artifact-path",
            str(served_artifact_path),
            "--output-dir",
            str(served_eval_dir),
            "--data-home",
            str(args.data_home),
            "--species-slug",
            args.species_slug,
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
    )
    if served_eval_rc != 0:
        raise RuntimeError(f"evaluate_model_b failed for served artifact on {dataset_dir.name} with exit code {served_eval_rc}")

    candidate_eval_rc = _evaluate_main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--artifact-path",
            str(checkpoint_path),
            "--baseline-artifact-path",
            str(served_artifact_path),
            "--output-dir",
            str(candidate_eval_dir),
            "--data-home",
            str(args.data_home),
            "--species-slug",
            args.species_slug,
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
    )
    if candidate_eval_rc != 0:
        raise RuntimeError(
            f"evaluate_model_b failed for candidate artifact on {dataset_dir.name} with exit code {candidate_eval_rc}"
        )

    return DatasetAuditRun(
        dataset_dir=dataset_dir,
        dataset_name=dataset_dir.name,
        train_dir=train_dir,
        cv_dir=cv_dir,
        served_eval_dir=served_eval_dir,
        candidate_eval_dir=candidate_eval_dir,
        checkpoint_path=checkpoint_path,
    )


def _dataset_audit_payload(run: DatasetAuditRun, *, root_dir: pathlib.Path) -> dict[str, Any]:
    train_summary = _load_json(run.train_dir / "summary.json")
    train_report = _load_json(run.train_dir / "report.json")
    train_eval_summary = _load_json(run.train_dir / "train_eval" / "summary.json")
    cv_summary = _load_json(run.cv_dir / "summary.json")
    served_summary = _load_json(run.served_eval_dir / "summary.json")
    candidate_summary = _load_json(run.candidate_eval_dir / "summary.json")

    final_epoch = train_report.get("epoch_history", [])[-1] if train_report.get("epoch_history") else {}
    dataset_payload = {
        "dataset_name": run.dataset_name,
        "paths": {
            "dataset_dir": str(run.dataset_dir),
            "train_dir": str(run.train_dir),
            "cv_dir": str(run.cv_dir),
            "served_eval_dir": str(run.served_eval_dir),
            "candidate_eval_dir": str(run.candidate_eval_dir),
            "checkpoint_path": str(run.checkpoint_path),
        },
        "train": {
            "test_primary_score": _metric_from_summary(train_summary, "primary_score"),
            "test_mean_balanced_accuracy": _metric_from_summary(train_summary, "mean_balanced_accuracy"),
            "train_primary_score": _metric_from_summary(train_eval_summary, "primary_score"),
            "train_mean_balanced_accuracy": _metric_from_summary(train_eval_summary, "mean_balanced_accuracy"),
            "epoch_count": int(len(train_report.get("epoch_history", []))),
            "final_epoch": final_epoch,
        },
        "cv": {
            "folds": int(cv_summary.get("folds", 0)),
            "candidate_primary_score_mean": _distribution_mean(cv_summary, "primary_score"),
            "candidate_primary_score_std": float(cv_summary.get("candidate_metrics_summary", {}).get("primary_score", {}).get("std", 0.0)),
            "baseline_primary_score_mean": float(cv_summary.get("baseline_metrics_summary", {}).get("primary_score", {}).get("mean", 0.0)),
            "delta_primary_score_mean": _distribution_delta_mean(cv_summary, "primary_score"),
            "candidate_mean_balanced_accuracy_mean": _distribution_mean(cv_summary, "mean_balanced_accuracy"),
            "delta_mean_balanced_accuracy_mean": _distribution_delta_mean(cv_summary, "mean_balanced_accuracy"),
        },
        "test_eval": {
            "served_primary_score": _metric_from_summary(served_summary, "primary_score"),
            "served_mean_balanced_accuracy": _metric_from_summary(served_summary, "mean_balanced_accuracy"),
            "candidate_primary_score": _metric_from_summary(candidate_summary, "primary_score"),
            "candidate_mean_balanced_accuracy": _metric_from_summary(candidate_summary, "mean_balanced_accuracy"),
            "candidate_minus_served_primary_score": _metric_from_summary(candidate_summary, "primary_score")
            - _metric_from_summary(served_summary, "primary_score"),
            "candidate_minus_served_mean_balanced_accuracy": _metric_from_summary(candidate_summary, "mean_balanced_accuracy")
            - _metric_from_summary(served_summary, "mean_balanced_accuracy"),
            "served_rows_scored": int(served_summary.get("diagnostics", {}).get("rows_scored", 0)),
            "candidate_rows_scored": int(candidate_summary.get("diagnostics", {}).get("rows_scored", 0)),
        },
        "per_head_macro_f1": {
            "served": {
                head: _summary_metric(served_summary, f"{head}_macro_f1")
                for head in ("readability", "specie", "behavior", "substrate", "stance")
            },
            "candidate": {
                head: _summary_metric(candidate_summary, f"{head}_macro_f1")
                for head in ("readability", "specie", "behavior", "substrate", "stance")
            },
        },
        "reports": {
            "train_report_md": _relative_path(run.train_dir / "report.md", start=root_dir),
            "train_eval_report_md": _relative_path(run.train_dir / "train_eval" / "report.md", start=root_dir),
            "cv_summary_json": _relative_path(run.cv_dir / "summary.json", start=root_dir),
            "served_eval_report_md": _relative_path(run.served_eval_dir / "report.md", start=root_dir),
            "candidate_eval_report_md": _relative_path(run.candidate_eval_dir / "report.md", start=root_dir),
        },
    }
    return dataset_payload


def _write_audit_report(
    out_dir: pathlib.Path,
    *,
    config_path: pathlib.Path,
    served_artifact_path: pathlib.Path,
    provenance: dict[str, Any],
    smoke_result: dict[str, str] | None,
    dataset_payloads: list[dict[str, Any]],
) -> dict[str, pathlib.Path]:
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(config_path),
        "served_artifact_path": str(served_artifact_path),
        "provenance": provenance,
        "approximation_notes": list(APPROXIMATION_NOTES),
        "smoke_result": smoke_result,
        "datasets": dataset_payloads,
    }
    summary_json = out_dir / "audit_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    markdown_lines = [
        f"# Served Model B Audit: {out_dir.name}",
        "",
        "## Overview",
        "",
        f"- Served artifact: `{served_artifact_path}`",
        f"- Audit config: `{config_path}`",
        f"- Archived source: `{provenance.get('archived_source_path', 'unknown')}`",
        f"- Archived run dir: `{provenance.get('archived_run_dir', 'unknown')}`",
        "",
        "## Audit Caveats",
        "",
    ]
    markdown_lines.extend([f"- {note}" for note in APPROXIMATION_NOTES])
    markdown_lines.extend(["", "## Dataset Summary", ""])

    for dataset in dataset_payloads:
        test_eval = dataset["test_eval"]
        train = dataset["train"]
        cv = dataset["cv"]
        markdown_lines.extend(
            [
                f"### {dataset['dataset_name']}",
                "",
                f"- Candidate test primary score: `{test_eval['candidate_primary_score']:.4f}`",
                f"- Served test primary score: `{test_eval['served_primary_score']:.4f}`",
                f"- Candidate minus served primary score: `{test_eval['candidate_minus_served_primary_score']:.4f}`",
                f"- Candidate test mean balanced accuracy: `{test_eval['candidate_mean_balanced_accuracy']:.4f}`",
                f"- Served test mean balanced accuracy: `{test_eval['served_mean_balanced_accuracy']:.4f}`",
                f"- Final train primary score: `{train['train_primary_score']:.4f}`",
                f"- Final test primary score from train run: `{train['test_primary_score']:.4f}`",
                f"- CV candidate primary score mean: `{cv['candidate_primary_score_mean']:.4f}`",
                f"- CV baseline primary score mean: `{cv['baseline_primary_score_mean']:.4f}`",
                f"- CV delta primary score mean: `{cv['delta_primary_score_mean']:.4f}`",
                "",
                "| Head | Served Macro F1 | Candidate Macro F1 | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for head in ("readability", "specie", "behavior", "substrate", "stance"):
            served_value = float(dataset["per_head_macro_f1"]["served"].get(head, 0.0))
            candidate_value = float(dataset["per_head_macro_f1"]["candidate"].get(head, 0.0))
            markdown_lines.append(
                f"| `{head}` | `{served_value:.4f}` | `{candidate_value:.4f}` | `{candidate_value - served_value:.4f}` |"
            )
        markdown_lines.extend(
            [
                "",
                f"- Train report: `{dataset['reports']['train_report_md']}`",
                f"- Train-eval report: `{dataset['reports']['train_eval_report_md']}`",
                f"- Served test report: `{dataset['reports']['served_eval_report_md']}`",
                f"- Candidate test report: `{dataset['reports']['candidate_eval_report_md']}`",
                f"- CV summary JSON: `{dataset['reports']['cv_summary_json']}`",
                "",
            ]
        )

    if smoke_result is not None:
        markdown_lines.extend(
            [
                "## Smoke Pass",
                "",
                f"- Smoke train dir: `{pathlib.Path(smoke_result['train_dir']).name}`",
                f"- Smoke CV dir: `{pathlib.Path(smoke_result['cv_dir']).name}`",
                "",
            ]
        )

    summary_md = out_dir / "audit_summary.md"
    summary_md.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return {"summary_json": summary_json, "summary_md": summary_md}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = pathlib.Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    data_home = pathlib.Path(args.data_home).expanduser().resolve()
    dataset_dirs = _resolve_dataset_dirs(data_home=data_home, species_slug=args.species_slug, requested=args.dataset_dirs)
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            raise FileNotFoundError(dataset_dir)

    served_artifact_path = (
        pathlib.Path(args.served_artifact_path).expanduser().resolve()
        if args.served_artifact_path
        else resolve_default_served_model_b_artifact_path(data_home=data_home, species_slug=args.species_slug)
    )
    if not served_artifact_path.exists():
        raise FileNotFoundError(served_artifact_path)

    layout = ensure_layout(data_home, args.species_slug)
    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        out_dir = next_version_dir(layout.models_attributes / "audit_reports", "served_model_b_audit")

    provenance = _read_served_provenance(served_artifact_path)
    smoke_result: dict[str, str] | None = None
    if args.smoke_first:
        smoke_dataset = next((path for path in dataset_dirs if path.name == args.smoke_dataset_name), dataset_dirs[0])
        print(f"audit_smoke_start dataset={smoke_dataset.name}", flush=True)
        smoke_result = _run_smoke_pass(
            dataset_dir=smoke_dataset,
            config_path=config_path,
            served_artifact_path=served_artifact_path,
            out_dir=out_dir,
            args=args,
        )

    runs: list[DatasetAuditRun] = []
    for dataset_dir in dataset_dirs:
        dataset_out = out_dir / dataset_dir.name
        dataset_out.mkdir(parents=True, exist_ok=False)
        print(f"audit_dataset_start dataset={dataset_dir.name}", flush=True)
        runs.append(
            _run_dataset_audit(
                dataset_dir=dataset_dir,
                config_path=config_path,
                served_artifact_path=served_artifact_path,
                out_dir=dataset_out,
                args=args,
            )
        )

    dataset_payloads = [_dataset_audit_payload(run, root_dir=out_dir) for run in runs]
    outputs = _write_audit_report(
        out_dir,
        config_path=config_path,
        served_artifact_path=served_artifact_path,
        provenance=provenance,
        smoke_result=smoke_result,
        dataset_payloads=dataset_payloads,
    )

    print(f"output_dir={out_dir}")
    print(f"audit_summary_json={outputs['summary_json']}")
    print(f"audit_summary_md={outputs['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
