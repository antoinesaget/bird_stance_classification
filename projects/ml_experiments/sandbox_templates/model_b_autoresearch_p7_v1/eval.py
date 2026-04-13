#!/usr/bin/env python3
"""Purpose: Evaluate sandbox fold outputs and summarize keep-or-discard decisions"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import pandas as pd

from common import (
    CONFUSION_HEADS,
    HEADS,
    ID_TO_LABELS,
    LABEL_MAPS,
    RESULTS_HEADER,
    ROOT,
    coerce_supported_label,
    collect_visible_label_counts,
    compute_head_masks,
    confusion_matrix,
    fallback_label,
    macro_f1,
    normalize_behavior,
    normalize_choice,
    normalize_stance,
    normalize_substrate,
    summarize_dataset_labels,
    write_json,
)


def load_reference_thresholds() -> dict[str, Any]:
    return json.loads((ROOT / "baselines" / "reference_thresholds.json").read_text(encoding="utf-8"))


def aggregate_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    metric_keys = sorted({key for item in fold_metrics for key in item.keys()})
    summary: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = [float(item.get(key, 0.0)) for item in fold_metrics]
        summary[key] = {
            "mean": float(mean(values)) if values else 0.0,
            "std": float(pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)) if values else 0.0,
            "max": float(max(values)) if values else 0.0,
        }
    return summary


def compute_search_score(metric_summary: dict[str, dict[str, float]], thresholds: dict[str, Any]) -> float:
    weights = thresholds["objective_weights"]
    return float(
        weights["stance_f1"] * metric_summary.get("stance_f1", {}).get("mean", 0.0)
        + weights["behavior_f1"] * metric_summary.get("behavior_f1", {}).get("mean", 0.0)
        + weights["substrate_f1"] * metric_summary.get("substrate_f1", {}).get("mean", 0.0)
    )


def evaluate_guardrails(metric_summary: dict[str, dict[str, float]], thresholds: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    if metric_summary.get("readability_f1", {}).get("mean", 0.0) < thresholds["guardrails"]["readability_f1_min"]:
        failures.append("readability_f1")
    if metric_summary.get("specie_f1", {}).get("mean", 0.0) < thresholds["guardrails"]["specie_f1_min"]:
        failures.append("specie_f1")
    return ("pass" if not failures else "fail"), failures


def read_results_tsv(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        return pd.DataFrame(columns=RESULTS_HEADER)
    frame = pd.read_csv(results_path, sep="\t")
    if frame.empty:
        return pd.DataFrame(columns=RESULTS_HEADER)
    return frame


def best_kept_score(results_path: Path) -> float | None:
    frame = read_results_tsv(results_path)
    if frame.empty:
        return None
    kept = frame[frame["keep_discard"] == "keep"]
    if kept.empty:
        return None
    return float(kept["search_score"].astype(float).max())


def decide_keep_discard(
    *,
    status: str,
    search_score: float,
    guardrail_status: str,
    results_path: Path,
    delta_to_keep: float,
) -> str:
    if status != "success":
        return "crash"
    if guardrail_status != "pass":
        return "discard"
    current_best = best_kept_score(results_path)
    if current_best is None:
        return "keep"
    if search_score > current_best + delta_to_keep:
        return "keep"
    return "discard"


def append_result_row(results_path: Path, row: dict[str, Any]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_path.exists():
        results_path.write_text("\t".join(RESULTS_HEADER) + "\n", encoding="utf-8")
    ordered = [str(row.get(column, "")) for column in RESULTS_HEADER]
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(ordered) + "\n")


def update_best_pointers(root: Path, summary: dict[str, Any]) -> None:
    best_dir = root / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    research_path = best_dir / "research_best.json"
    servable_path = best_dir / "servable_best.json"
    current_research = json.loads(research_path.read_text(encoding="utf-8")) if research_path.exists() else {"search_score": None}
    if summary["keep_discard"] == "keep":
        current_score = current_research.get("search_score")
        if current_score is None or float(summary["search_score"]) > float(current_score):
            write_json(research_path, summary)
        if summary.get("export_status") == "ready":
            current_servable = json.loads(servable_path.read_text(encoding="utf-8")) if servable_path.exists() else {"search_score": None}
            servable_score = current_servable.get("search_score")
            if servable_score is None or float(summary["search_score"]) > float(servable_score):
                write_json(servable_path, summary)


def _normalize_supported_labels(raw: Any) -> dict[str, set[str]]:
    supported = {head: set(ID_TO_LABELS[head]) for head in HEADS}
    if not isinstance(raw, dict):
        return supported
    for head in HEADS:
        values = raw.get(head)
        if isinstance(values, list) and values:
            supported[head] = {str(item) for item in values}
    return supported


def _load_predictions(prediction_path: Path) -> pd.DataFrame:
    if prediction_path.suffix == ".parquet":
        return pd.read_parquet(prediction_path)
    if prediction_path.suffix == ".jsonl":
        records = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(records)
    raise ValueError(f"Unsupported predictions format: {prediction_path}")


def _canonical_true_labels(row: pd.Series) -> dict[str, str | None]:
    return {
        "readability": normalize_choice(str(row["readability"])) if pd.notna(row["readability"]) else None,
        "specie": normalize_choice(str(row["specie"])) if pd.notna(row["specie"]) else None,
        "behavior": normalize_behavior(str(row["behavior"])) if pd.notna(row["behavior"]) else None,
        "substrate": normalize_substrate(str(row["substrate"])) if pd.notna(row["substrate"]) else None,
        "stance": normalize_stance(str(row["stance"])) if pd.notna(row["stance"]) else None,
    }


def _canonical_prediction_labels(
    prediction_row: dict[str, Any],
    *,
    supported_labels: dict[str, set[str]],
    unsupported_label_coercions: dict[str, dict[str, int]],
    unreadable_or_incorrect_suppressions: dict[str, int],
    stance_suppressions_non_relevant: list[int],
) -> dict[str, str]:
    labels = {
        "readability": normalize_choice(str(prediction_row.get("readability", ""))) or fallback_label("readability"),
        "specie": normalize_choice(str(prediction_row.get("specie", ""))) or fallback_label("specie"),
        "behavior": coerce_supported_label(
            "behavior",
            None if pd.isna(prediction_row.get("behavior")) else str(prediction_row.get("behavior")),
            supported_labels,
            unsupported_label_coercions,
        ),
        "substrate": coerce_supported_label(
            "substrate",
            None if pd.isna(prediction_row.get("substrate")) else str(prediction_row.get("substrate")),
            supported_labels,
            unsupported_label_coercions,
        ),
        "stance": coerce_supported_label(
            "stance",
            None if pd.isna(prediction_row.get("stance")) else str(prediction_row.get("stance")),
            supported_labels,
            unsupported_label_coercions,
        ),
    }
    labels["readability"] = coerce_supported_label("readability", labels["readability"], supported_labels, unsupported_label_coercions)
    labels["specie"] = coerce_supported_label("specie", labels["specie"], supported_labels, unsupported_label_coercions)

    if labels["readability"] == "unreadable" or labels["specie"] == "incorrect":
        labels["behavior"] = fallback_label("behavior")
        labels["substrate"] = fallback_label("substrate")
        labels["stance"] = fallback_label("stance")
        unreadable_or_incorrect_suppressions["behavior"] += 1
        unreadable_or_incorrect_suppressions["substrate"] += 1
        unreadable_or_incorrect_suppressions["stance"] += 1
    elif not (labels["behavior"] in {"resting", "backresting"} and labels["substrate"] in {"bare_ground", "water", "unsure"}):
        labels["stance"] = fallback_label("stance")
        stance_suppressions_non_relevant[0] += 1
    return labels


def evaluate_predictions(
    *,
    test_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    supported_labels: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    if "row_id" not in test_frame.columns:
        raise ValueError("Test frame missing row_id")
    if "row_id" not in prediction_frame.columns:
        raise ValueError("Prediction frame missing row_id")
    required_prediction_columns = ["readability", "specie", "behavior", "substrate", "stance"]
    missing_prediction_columns = [column for column in required_prediction_columns if column not in prediction_frame.columns]
    if missing_prediction_columns:
        raise ValueError(f"Prediction frame missing columns: {missing_prediction_columns}")

    test_df = test_frame.copy().reset_index(drop=True)
    pred_df = prediction_frame.copy().reset_index(drop=True)
    pred_df["row_id"] = pred_df["row_id"].astype(str)
    if pred_df["row_id"].duplicated().any():
        duplicates = sorted(pred_df.loc[pred_df["row_id"].duplicated(), "row_id"].unique().tolist())
        raise ValueError(f"Duplicate row_id values in predictions: {duplicates[:10]}")
    extra_prediction_rows = sorted(set(pred_df["row_id"].tolist()) - set(test_df["row_id"].astype(str).tolist()))
    if extra_prediction_rows:
        raise ValueError(f"Unexpected prediction row_ids: {extra_prediction_rows[:10]}")

    joined = test_df.merge(pred_df, on="row_id", how="left", suffixes=("", "_pred"), validate="one_to_one")
    missing_prediction_rows = joined.loc[joined["readability_pred"].isna(), "row_id"].astype(str).tolist()
    if missing_prediction_rows:
        raise ValueError(f"Missing predictions for row_ids: {missing_prediction_rows[:10]}")

    supported = supported_labels or {head: set(ID_TO_LABELS[head]) for head in HEADS}
    unsupported_label_coercions = {head: {} for head in HEADS}
    unreadable_or_incorrect_suppressions = {"behavior": 0, "substrate": 0, "stance": 0}
    stance_suppressions_non_relevant = [0]
    prediction_label_counts = {head: {label: 0 for label in ID_TO_LABELS[head]} for head in HEADS}
    storage = {head: {"true": [], "pred": []} for head in HEADS}
    confusion_matrices = {
        head: np.zeros((len(ID_TO_LABELS[head]), len(ID_TO_LABELS[head])), dtype=np.int64)
        for head in CONFUSION_HEADS
    }

    for _, row in joined.iterrows():
        true_labels = _canonical_true_labels(row)
        pred_labels = _canonical_prediction_labels(
            {
                "readability": row["readability_pred"],
                "specie": row["specie_pred"],
                "behavior": row["behavior_pred"],
                "substrate": row["substrate_pred"],
                "stance": row["stance_pred"],
            },
            supported_labels=supported,
            unsupported_label_coercions=unsupported_label_coercions,
            unreadable_or_incorrect_suppressions=unreadable_or_incorrect_suppressions,
            stance_suppressions_non_relevant=stance_suppressions_non_relevant,
        )
        for head in HEADS:
            prediction_label_counts[head][pred_labels[head]] += 1

        masks = compute_head_masks(
            isbird=str(row["isbird"]),
            readability=str(row["readability"]),
            specie=str(row["specie"]),
            behavior=str(row["behavior"]) if pd.notna(row["behavior"]) else None,
            substrate=str(row["substrate"]) if pd.notna(row["substrate"]) else None,
        )
        mask_map = {
            "readability": masks.readability,
            "specie": masks.specie,
            "behavior": masks.behavior,
            "substrate": masks.substrate,
            "stance": masks.stance,
        }
        for head in HEADS:
            if not mask_map[head]:
                continue
            true_label = true_labels[head]
            if true_label not in LABEL_MAPS[head]:
                continue
            storage[head]["true"].append(LABEL_MAPS[head][true_label])
            storage[head]["pred"].append(LABEL_MAPS[head][pred_labels[head]])

    metrics: dict[str, float] = {}
    for head in HEADS:
        y_true = np.array(storage[head]["true"], dtype=np.int64)
        y_pred = np.array(storage[head]["pred"], dtype=np.int64)
        if len(y_true) == 0:
            metrics[f"{head}_accuracy"] = 0.0
            metrics[f"{head}_f1"] = 0.0
            continue
        metrics[f"{head}_accuracy"] = float((y_true == y_pred).mean())
        metrics[f"{head}_f1"] = macro_f1(y_true, y_pred, len(ID_TO_LABELS[head]), ignore_absent_classes=True)
        if head in confusion_matrices:
            confusion_matrices[head] = confusion_matrix(y_true, y_pred, len(ID_TO_LABELS[head]))

    return {
        "metrics": metrics,
        "visible_label_counts": collect_visible_label_counts(test_df),
        "dataset_label_counts": summarize_dataset_labels(test_df),
        "diagnostics": {
            "rows_scored": int(len(test_df)),
            "unsupported_label_coercions": unsupported_label_coercions,
            "unreadable_or_incorrect_suppressions": unreadable_or_incorrect_suppressions,
            "stance_suppressions_non_relevant": int(stance_suppressions_non_relevant[0]),
            "prediction_label_counts": prediction_label_counts,
        },
        "confusion_matrices": {head: confusion_matrices[head].tolist() for head in CONFUSION_HEADS},
    }


def evaluate_fold_artifact(*, fold_dir: Path, test_parquet: Path) -> dict[str, Any]:
    candidate_dir = fold_dir / "candidate"
    meta_path = candidate_dir / "artifact_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prediction_path = candidate_dir / meta["prediction_file"]
    prediction_frame = _load_predictions(prediction_path)
    test_frame = pd.read_parquet(test_parquet)
    supported = _normalize_supported_labels(meta.get("supported_labels"))
    report = evaluate_predictions(test_frame=test_frame, prediction_frame=prediction_frame, supported_labels=supported)
    report["artifact_meta"] = meta
    report["prediction_file"] = str(prediction_path)
    report["test_parquet"] = str(test_parquet)
    return report


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    fold_entries: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, float]] = []
    for fold_dir in sorted((run_dir / "folds").glob("fold_*")):
        fold_output_path = fold_dir / "fold_output.json"
        if not fold_output_path.exists():
            continue
        fold_output = json.loads(fold_output_path.read_text(encoding="utf-8"))
        test_parquet = Path(fold_output["test_parquet"])
        eval_report = evaluate_fold_artifact(fold_dir=fold_dir, test_parquet=test_parquet)
        fold_entry = {
            "fold_id": fold_output["fold_id"],
            "test_parquet": str(test_parquet),
            "budget_seconds": fold_output["budget_seconds"],
            "train_output": fold_output["train_output"],
            "eval_report": eval_report,
        }
        fold_entries.append(fold_entry)
        fold_metrics.append(eval_report["metrics"])
    metric_summary = aggregate_fold_metrics(fold_metrics)
    thresholds = load_reference_thresholds()
    search_score = compute_search_score(metric_summary, thresholds)
    guardrail_status, guardrail_failures = evaluate_guardrails(metric_summary, thresholds)
    return {
        "fold_evaluations": fold_entries,
        "metric_summary": metric_summary,
        "metric_summary_mean": {key: value["mean"] for key, value in metric_summary.items()},
        "metric_summary_std": {key: value["std"] for key, value in metric_summary.items()},
        "search_score": search_score,
        "guardrail_status": guardrail_status,
        "guardrail_failures": guardrail_failures,
    }


def summary_to_row(summary: dict[str, Any]) -> dict[str, Any]:
    mean_metrics = summary["metric_summary_mean"]
    return {
        "run_id": summary["run_id"],
        "search_score": f"{summary['search_score']:.6f}",
        "stance_f1": f"{mean_metrics.get('stance_f1', 0.0):.6f}",
        "behavior_f1": f"{mean_metrics.get('behavior_f1', 0.0):.6f}",
        "substrate_f1": f"{mean_metrics.get('substrate_f1', 0.0):.6f}",
        "readability_f1": f"{mean_metrics.get('readability_f1', 0.0):.6f}",
        "specie_f1": f"{mean_metrics.get('specie_f1', 0.0):.6f}",
        "stance_acc": f"{mean_metrics.get('stance_accuracy', 0.0):.6f}",
        "peak_vram_gb": f"{summary.get('peak_vram_gb', 0.0):.2f}",
        "wall_seconds": f"{summary.get('wall_seconds', 0.0):.2f}",
        "status": summary["status"],
        "keep_discard": summary["keep_discard"],
        "guardrail_status": summary["guardrail_status"],
        "export_status": summary.get("export_status", "not_checked"),
        "model_family": summary.get("model_family", "unknown"),
        "description": summary.get("description", ""),
    }


if __name__ == "__main__":
    payload = evaluate_run(ROOT / "runs")
    print(json.dumps(payload, indent=2))
