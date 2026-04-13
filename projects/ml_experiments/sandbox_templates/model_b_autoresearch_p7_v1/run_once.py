#!/usr/bin/env python3
"""Purpose: Execute one fixed-budget autoresearch attempt inside the sandbox template"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from common import FoldContext, ROOT, slugify, write_json
from eval import (
    aggregate_fold_metrics,
    append_result_row,
    compute_search_score,
    decide_keep_discard,
    evaluate_fold_artifact,
    evaluate_guardrails,
    load_reference_thresholds,
    summary_to_row,
    update_best_pointers,
)

TOTAL_BUDGET_SECONDS = 300.0
TOTAL_HARD_TIMEOUT_SECONDS = 600.0
EXPORT_RESERVE_SECONDS = 45.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fixed-budget Model B autoresearch experiment")
    parser.add_argument("--description", default="manual")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=20260326)
    return parser.parse_args()


def load_train_module():
    spec = importlib.util.spec_from_file_location("sandbox_train", ROOT / "train.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_train"] = module
    spec.loader.exec_module(module)
    return module


def git_info() -> dict[str, str | None]:
    info = {"branch": None, "commit": None}
    try:
        info["branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT).decode().strip()
        info["commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return info
    return info


def _empty_run_summary(run_id: str, description: str, experiment: Any, git_state: dict[str, str | None], best_score_before: float | None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "description": description,
        "status": "crash",
        "keep_discard": "crash",
        "search_score": 0.0,
        "metric_summary": {},
        "metric_summary_mean": {},
        "metric_summary_std": {},
        "guardrail_status": "fail",
        "guardrail_failures": ["crash"],
        "peak_vram_gb": 0.0,
        "wall_seconds": 0.0,
        "model_family": getattr(experiment, "model_family", "unknown"),
        "supports_export_now": getattr(experiment, "supports_export_now", False),
        "export_status": "crash",
        "candidate_dir": "",
        "git": git_state,
        "best_search_score_before": best_score_before,
        "fold_evaluations": [],
    }


def main() -> int:
    args = parse_args()
    thresholds = load_reference_thresholds()
    train_mod = load_train_module()
    experiment = train_mod.build_experiment()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{slugify(args.description)}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    folds_manifest = json.loads((ROOT / "data" / "folds" / "folds_manifest.json").read_text(encoding="utf-8"))
    full_pool_path = ROOT / "data" / "full_pool.parquet"
    results_path = ROOT / "results.tsv"
    best_score_before = None
    try:
        from eval import best_kept_score

        best_score_before = best_kept_score(results_path)
    except Exception:
        best_score_before = None

    started = time.monotonic()
    hard_deadline = started + TOTAL_HARD_TIMEOUT_SECONDS
    fold_evaluations: list[dict[str, Any]] = []
    export_status = "not_supported"
    status = "success"
    git_state = git_info()
    summary = _empty_run_summary(run_id, args.description, experiment, git_state, best_score_before)

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    (run_dir / "train.py").write_text((ROOT / "train.py").read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "program.md").write_text((ROOT / "program.md").read_text(encoding="utf-8"), encoding="utf-8")

    try:
        log(f"run_id={run_id}")
        log(f"description={args.description}")
        log(f"git_branch={git_state['branch']}")
        log(f"git_commit={git_state['commit']}")
        cv_budget = TOTAL_BUDGET_SECONDS - (EXPORT_RESERVE_SECONDS if experiment.supports_export_now else 0.0)
        fold_metric_inputs: list[dict[str, float]] = []
        for idx, fold_detail in enumerate(folds_manifest["folds_detail"]):
            if time.monotonic() > hard_deadline:
                raise TimeoutError("run exceeded hard timeout")
            elapsed = time.monotonic() - started
            remaining_cv = max(15.0, cv_budget - elapsed)
            remaining_folds = max(1, len(folds_manifest["folds_detail"]) - idx)
            fold_budget = remaining_cv / remaining_folds
            fold_id = int(fold_detail["fold_id"])
            train_parquet = str(fold_detail.get("inner_train_parquet") or (ROOT / "data" / "folds" / f"fold_{fold_id}_train.parquet"))
            fold_run_dir = run_dir / "folds" / f"fold_{fold_id}"
            fold_ctx = FoldContext(
                train_parquet=train_parquet,
                test_parquet=str(ROOT / "data" / "folds" / f"fold_{fold_id}_test.parquet"),
                run_dir=str(fold_run_dir),
                device=args.device,
                budget_seconds=float(fold_budget),
                seed=args.seed_base + fold_id,
                fold_id=str(fold_id),
            )
            log(f"fold_start fold={fold_id} budget_seconds={fold_budget:.2f}")
            train_output = train_mod.run_fold(fold_ctx)
            fold_payload = {
                "fold_id": fold_id,
                "budget_seconds": fold_budget,
                "train_parquet": fold_ctx.train_parquet,
                "test_parquet": fold_ctx.test_parquet,
                "train_output": train_output.__dict__,
            }
            write_json(fold_run_dir / "fold_output.json", fold_payload)
            eval_report = evaluate_fold_artifact(
                fold_dir=fold_run_dir,
                test_parquet=Path(fold_ctx.test_parquet),
            )
            write_json(fold_run_dir / "fold_eval.json", eval_report)
            fold_metric_inputs.append(eval_report["metrics"])
            fold_evaluations.append(
                {
                    "fold_id": fold_id,
                    "budget_seconds": fold_budget,
                    "train_output": train_output.__dict__,
                    "eval_report": eval_report,
                }
            )
            log(f"fold_done fold={fold_id} fixed_metrics={eval_report['metrics']}")

        metric_summary = aggregate_fold_metrics(fold_metric_inputs)
        search_score = compute_search_score(metric_summary, thresholds)
        guardrail_status, guardrail_failures = evaluate_guardrails(metric_summary, thresholds)

        candidate_dir = run_dir / "candidate"
        if experiment.supports_export_now:
            remaining = max(0.0, TOTAL_BUDGET_SECONDS - (time.monotonic() - started))
            if remaining >= 20.0:
                log(f"full_export_start budget_seconds={remaining:.2f}")
                export_ctx = FoldContext(
                    train_parquet=str(full_pool_path),
                    test_parquet=str(full_pool_path),
                    run_dir=str(run_dir),
                    device=args.device,
                    budget_seconds=float(remaining),
                    seed=args.seed_base + 99,
                    fold_id="full_export",
                )
                export_output = train_mod.run_fold(export_ctx)
                write_json(
                    candidate_dir / "full_export_output.json",
                    {
                        "fold_id": "full_export",
                        "train_parquet": export_ctx.train_parquet,
                        "test_parquet": export_ctx.test_parquet,
                        "train_output": export_output.__dict__,
                    },
                )
                if (candidate_dir / "current_backend" / "checkpoint.pt").exists():
                    export_status = "ready"
                else:
                    export_status = "failed"
            else:
                export_status = "skipped_budget"

        keep_discard = decide_keep_discard(
            status="success",
            search_score=search_score,
            guardrail_status=guardrail_status,
            results_path=results_path,
            delta_to_keep=float(thresholds["delta_to_keep"]),
        )
        wall_seconds = time.monotonic() - started
        peak_vram_mb = max(float(item["train_output"].get("peak_vram_mb", 0.0)) for item in fold_evaluations) if fold_evaluations else 0.0
        summary = {
            "run_id": run_id,
            "description": args.description,
            "status": status,
            "keep_discard": keep_discard,
            "search_score": search_score,
            "metric_summary": metric_summary,
            "metric_summary_mean": {key: value["mean"] for key, value in metric_summary.items()},
            "metric_summary_std": {key: value["std"] for key, value in metric_summary.items()},
            "guardrail_status": guardrail_status,
            "guardrail_failures": guardrail_failures,
            "peak_vram_gb": peak_vram_mb / 1024.0,
            "wall_seconds": wall_seconds,
            "model_family": experiment.model_family,
            "supports_export_now": experiment.supports_export_now,
            "export_status": export_status,
            "candidate_dir": str(candidate_dir),
            "git": git_state,
            "seed_base": args.seed_base,
            "best_search_score_before": best_score_before,
            "fold_evaluations": fold_evaluations,
        }
    except Exception as exc:  # noqa: BLE001
        status = "crash"
        error_text = "".join(traceback.format_exception(exc))
        log(error_text)
        summary = _empty_run_summary(run_id, args.description, experiment, git_state, best_score_before)
        summary["status"] = status
        summary["wall_seconds"] = time.monotonic() - started
        summary["candidate_dir"] = str(run_dir / "candidate")
        summary["error"] = error_text
        summary["fold_evaluations"] = fold_evaluations
        summary["seed_base"] = args.seed_base

    write_json(run_dir / "fold_metrics.json", {"fold_evaluations": fold_evaluations})
    write_json(run_dir / "summary.json", summary)
    append_result_row(results_path, summary_to_row(summary))
    update_best_pointers(ROOT, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
