"""Purpose: Write file-based evaluation and training reports for Model B workflows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from birdsys.core import diff_numeric_dict

from .common import flatten_dict


def _load_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("matplotlib is required to render ml_experiments plots.") from exc
    return plt


def _result_to_payload(result: Any) -> dict[str, Any]:
    payload = {
        "checkpoint_path": result.checkpoint_path,
        "artifact_mode": result.artifact_mode,
        "device": result.device,
        "image_size": result.image_size,
        "schema_version": result.schema_version,
        "supported_labels": result.supported_labels,
        "train_label_counts": result.train_label_counts,
        "summary_metrics": result.summary_metrics,
        "aggregate_metrics": result.aggregate_metrics,
        "per_head_metrics": result.per_head_metrics,
        "per_class_metrics": result.per_class_metrics,
        "visible_label_counts": result.visible_label_counts,
        "dataset_label_counts": result.dataset_label_counts,
        "diagnostics": result.diagnostics.to_dict(),
        "confusion_matrices": {
            head: matrix.tolist()
            for head, matrix in result.confusion_matrices.items()
        },
        "comparison_to_baseline": None,
    }
    comparison = getattr(result, "comparison_to_baseline", None)
    if comparison is not None:
        payload["comparison_to_baseline"] = comparison.to_dict()
    return payload


def _summary_row(result: Any) -> dict[str, Any]:
    row = {
        "checkpoint_path": result.checkpoint_path,
        "artifact_mode": result.artifact_mode,
        "device": result.device,
        "image_size": result.image_size,
        "schema_version": result.schema_version,
    }
    row.update(flatten_dict("", result.aggregate_metrics))
    row.update(flatten_dict("", result.summary_metrics))
    comparison = getattr(result, "comparison_to_baseline", None)
    if comparison is not None:
        row.update(flatten_dict("delta_", comparison.delta_summary_metrics))
        row.update(flatten_dict("delta_", comparison.delta_aggregate_metrics))
        row["baseline_checkpoint_path"] = comparison.baseline_checkpoint_path
        row["baseline_artifact_mode"] = comparison.baseline_artifact_mode
    return row


def _plot_confusion_matrix(out_path: Path, *, title: str, labels: list[str], matrix) -> None:
    plt = _load_plt()
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), max(5, len(labels) * 0.8)))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    max_value = max(int(matrix.max()), 1)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            color = "white" if value > (max_value / 2.0) else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_metric_delta(out_path: Path, *, title: str, deltas: dict[str, float]) -> None:
    if not deltas:
        return
    plt = _load_plt()
    keys = sorted(deltas)
    values = [float(deltas[key]) for key in keys]
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.6), 5))
    colors = ["#2A9D8F" if value >= 0 else "#E76F51" for value in values]
    bars = ax.bar(range(len(keys)), values, color=colors)
    ax.axhline(0.0, color="#444444", linewidth=1)
    ax.set_title(title)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    for bar, value in zip(bars, values):
        y_pos = value + (0.01 if value >= 0 else -0.01)
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + (bar.get_width() / 2.0), y_pos, f"{value:.3f}", ha="center", va=va, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_per_head_metrics(out_path: Path, *, result: Any) -> None:
    head_rows = []
    for head, metrics in (result.per_head_metrics or {}).items():
        head_rows.append(
            {
                "head": head,
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
            }
        )
    if not head_rows:
        return
    df = pd.DataFrame(head_rows)
    plt = _load_plt()
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.6), 5))
    x = np.arange(len(df))
    width = 0.24
    ax.bar(x - width, df["macro_f1"], width=width, label="macro_f1", color="#2A9D8F")
    ax.bar(x, df["accuracy"], width=width, label="accuracy", color="#264653")
    ax.bar(x + width, df["balanced_accuracy"], width=width, label="balanced_accuracy", color="#E9C46A")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["head"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Per-Head Summary Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_visible_supports(out_path: Path, *, result: Any) -> None:
    rows = []
    for head, metrics in (result.per_head_metrics or {}).items():
        rows.append({"head": head, "visible_support": float(metrics.get("visible_support", 0.0))})
    if not rows:
        return
    df = pd.DataFrame(rows)
    plt = _load_plt()
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.4), 4.5))
    bars = ax.bar(df["head"], df["visible_support"], color="#6D597A")
    ax.set_ylabel("Rows")
    ax.set_title("Visible Support by Head")
    for bar, value in zip(bars, df["visible_support"].tolist()):
        ax.text(bar.get_x() + (bar.get_width() / 2.0), value, f"{int(value)}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_per_class_f1(out_path: Path, *, result: Any) -> None:
    if not result.per_class_metrics:
        return
    df = pd.DataFrame(result.per_class_metrics)
    if df.empty or "head" not in df.columns or "f1" not in df.columns:
        return
    heads = df["head"].dropna().unique().tolist()
    if not heads:
        return
    plt = _load_plt()
    fig, axes = plt.subplots(len(heads), 1, figsize=(10, max(4.5, len(heads) * 3.6)))
    if len(heads) == 1:
        axes = [axes]
    for ax, head in zip(axes, heads):
        head_df = df[df["head"] == head].copy().sort_values(["support", "label"], ascending=[False, True])
        ax.bar(head_df["label"], head_df["f1"], color="#2A9D8F", alpha=0.85, label="f1")
        ax.plot(head_df["label"], head_df["recall"], color="#E76F51", marker="o", linewidth=1.5, label="recall")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{head.title()} Per-Class Metrics")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=35)
        for idx, (_, row) in enumerate(head_df.iterrows()):
            ax.text(idx, float(row["f1"]) + 0.02, f"n={int(row['support'])}", ha="center", va="bottom", fontsize=7)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _relative_md_path(path: Path, *, from_dir: Path) -> str:
    return path.relative_to(from_dir).as_posix()


def _write_evaluation_markdown_report(out_dir: Path, *, result: Any, outputs: dict[str, Path]) -> Path:
    report_md = out_dir / "report.md"
    lines = [
        f"# Model B Evaluation Report: {out_dir.name}",
        "",
        "## Overview",
        "",
        f"- Checkpoint: `{result.checkpoint_path}`",
        f"- Artifact mode: `{result.artifact_mode}`",
        f"- Device: `{result.device}`",
        f"- Image size: `{result.image_size}`",
        f"- Schema version: `{result.schema_version}`",
        f"- Primary score: `{_format_metric(result.aggregate_metrics.get('primary_score'))}`",
        f"- Mean macro-F1: `{_format_metric(result.aggregate_metrics.get('mean_macro_f1'))}`",
        f"- Support-weighted macro-F1: `{_format_metric(result.aggregate_metrics.get('support_weighted_macro_f1'))}`",
        f"- Mean balanced accuracy: `{_format_metric(result.aggregate_metrics.get('mean_balanced_accuracy'))}`",
        f"- Rows scored: `{result.diagnostics.to_dict().get('rows_scored')}`",
        "",
        "## Per-Head Metrics",
        "",
        "| Head | Visible Support | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for head, metrics in sorted((result.per_head_metrics or {}).items()):
        lines.append(
            f"| `{head}` | `{int(metrics.get('visible_support', 0.0))}` | `{_format_metric(metrics.get('accuracy'))}` | "
            f"`{_format_metric(metrics.get('balanced_accuracy'))}` | `{_format_metric(metrics.get('macro_f1'))}` | "
            f"`{_format_metric(metrics.get('weighted_f1'))}` |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
        ]
    )
    if "per_head_metrics_png" in outputs:
        lines.append(f"![Per-head metrics]({_relative_md_path(outputs['per_head_metrics_png'], from_dir=out_dir)})")
        lines.append("")
    if "visible_supports_png" in outputs:
        lines.append(f"![Visible supports]({_relative_md_path(outputs['visible_supports_png'], from_dir=out_dir)})")
        lines.append("")
    if "per_class_f1_png" in outputs:
        lines.append(f"![Per-class metrics]({_relative_md_path(outputs['per_class_f1_png'], from_dir=out_dir)})")
        lines.append("")
    for key in sorted(outputs):
        if key.endswith("_confusion_matrix_png"):
            lines.append(f"### {key.replace('_confusion_matrix_png', '').replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"![{key}]({_relative_md_path(outputs[key], from_dir=out_dir)})")
            lines.append("")
    if "metric_delta_png" in outputs:
        lines.extend(
            [
                "## Baseline Delta",
                "",
                f"![Metric deltas vs baseline]({_relative_md_path(outputs['metric_delta_png'], from_dir=out_dir)})",
                "",
            ]
        )
    lines.extend(
        [
            "## Files",
            "",
            f"- Markdown report: `{report_md.name}`",
            f"- Summary JSON: `{outputs['summary_json'].name}`",
            f"- Summary CSV: `{outputs['summary_csv'].name}`",
            f"- Per-class metrics CSV: `{outputs['per_class_metrics_csv'].name}`",
            f"- Predictions Parquet: `{outputs['predictions_parquet'].name}`",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_md


def write_evaluation_report(out_dir: Path, result: Any) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    payload = _result_to_payload(result)
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary_csv = out_dir / "summary.csv"
    pd.DataFrame([_summary_row(result)]).to_csv(summary_csv, index=False)

    per_class_metrics_csv = out_dir / "per_class_metrics.csv"
    pd.DataFrame(result.per_class_metrics).to_csv(per_class_metrics_csv, index=False)

    predictions_parquet = out_dir / "predictions.parquet"
    prediction_frame = result.prediction_frame if isinstance(result.prediction_frame, pd.DataFrame) else pd.DataFrame()
    prediction_frame.to_parquet(predictions_parquet, index=False)

    outputs = {
        "summary_json": summary_json,
        "summary_csv": summary_csv,
        "per_class_metrics_csv": per_class_metrics_csv,
        "predictions_parquet": predictions_parquet,
    }

    per_head_metrics_png = plots_dir / "per_head_metrics.png"
    _plot_per_head_metrics(per_head_metrics_png, result=result)
    if per_head_metrics_png.exists():
        outputs["per_head_metrics_png"] = per_head_metrics_png

    visible_supports_png = plots_dir / "visible_supports.png"
    _plot_visible_supports(visible_supports_png, result=result)
    if visible_supports_png.exists():
        outputs["visible_supports_png"] = visible_supports_png

    per_class_f1_png = plots_dir / "per_class_metrics.png"
    _plot_per_class_f1(per_class_f1_png, result=result)
    if per_class_f1_png.exists():
        outputs["per_class_f1_png"] = per_class_f1_png

    for head, matrix in result.confusion_matrices.items():
        confusion_csv = out_dir / f"{head}_confusion_matrix.csv"
        pd.DataFrame(matrix, index=result.id_to_label[head], columns=result.id_to_label[head]).to_csv(confusion_csv)
        outputs[f"{head}_confusion_matrix_csv"] = confusion_csv
        confusion_png = plots_dir / f"{head}_confusion_matrix.png"
        _plot_confusion_matrix(
            confusion_png,
            title=f"{head.title()} Confusion Matrix",
            labels=result.id_to_label[head],
            matrix=matrix,
        )
        outputs[f"{head}_confusion_matrix_png"] = confusion_png

    comparison = getattr(result, "comparison_to_baseline", None)
    if comparison is not None:
        delta_png = plots_dir / "metric_deltas_vs_baseline.png"
        combined = dict(comparison.delta_aggregate_metrics)
        combined.update(comparison.delta_summary_metrics)
        _plot_metric_delta(delta_png, title="Metric Deltas vs Baseline", deltas=combined)
        outputs["metric_delta_png"] = delta_png

    outputs["report_md"] = _write_evaluation_markdown_report(out_dir, result=result, outputs=outputs)
    return outputs


def write_training_debug_artifacts(
    out_dir: Path,
    *,
    epoch_history: list[dict[str, Any]],
    batch_history: list[dict[str, Any]],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epoch_history_json = out_dir / "epoch_history.json"
    epoch_history_json.write_text(json.dumps(epoch_history, indent=2) + "\n", encoding="utf-8")
    epoch_history_csv = out_dir / "epoch_history.csv"
    pd.DataFrame(epoch_history).to_csv(epoch_history_csv, index=False)

    batch_history_jsonl = out_dir / "batch_history.jsonl"
    with batch_history_jsonl.open("w", encoding="utf-8") as handle:
        for row in batch_history:
            handle.write(json.dumps(row) + "\n")

    batch_history_csv = out_dir / "batch_history.csv"
    pd.DataFrame(batch_history).to_csv(batch_history_csv, index=False)

    outputs = {
        "epoch_history_json": epoch_history_json,
        "epoch_history_csv": epoch_history_csv,
        "batch_history_jsonl": batch_history_jsonl,
        "batch_history_csv": batch_history_csv,
    }

    epoch_df = pd.DataFrame(epoch_history)
    if not epoch_df.empty:
        plt = _load_plt()

        if {"epoch_index", "train_loss", "eval_loss"} <= set(epoch_df.columns):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(epoch_df["epoch_index"], epoch_df["train_loss"], label="train_loss", color="#264653")
            ax.plot(epoch_df["epoch_index"], epoch_df["eval_loss"], label="eval_loss", color="#E76F51")
            ax.set_title("Train vs Eval Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            fig.tight_layout()
            loss_png = plots_dir / "loss_curves.png"
            fig.savefig(loss_png, dpi=180)
            plt.close(fig)
            outputs["loss_curves_png"] = loss_png

        if {"epoch_index", "train_primary_score", "eval_primary_score"} <= set(epoch_df.columns):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(epoch_df["epoch_index"], epoch_df["train_primary_score"], label="train_primary_score", color="#2A9D8F")
            ax.plot(epoch_df["epoch_index"], epoch_df["eval_primary_score"], label="eval_primary_score", color="#F4A261")
            ax.set_title("Train vs Eval Primary Score")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Primary Score")
            ax.legend()
            fig.tight_layout()
            primary_png = plots_dir / "primary_metric_curves.png"
            fig.savefig(primary_png, dpi=180)
            plt.close(fig)
            outputs["primary_metric_curves_png"] = primary_png

    batch_df = pd.DataFrame(batch_history)
    if not batch_df.empty:
        plt = _load_plt()

        if {"global_step", "learning_rate"} <= set(batch_df.columns):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(batch_df["global_step"], batch_df["learning_rate"], color="#264653")
            ax.set_title("Learning Rate")
            ax.set_xlabel("Global Step")
            ax.set_ylabel("LR")
            fig.tight_layout()
            lr_png = plots_dir / "learning_rate_curve.png"
            fig.savefig(lr_png, dpi=180)
            plt.close(fig)
            outputs["learning_rate_curve_png"] = lr_png

        if {"global_step", "global_grad_norm"} <= set(batch_df.columns):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(batch_df["global_step"], batch_df["global_grad_norm"], color="#E76F51", label="global")
            if "backbone_grad_norm" in batch_df.columns:
                ax.plot(batch_df["global_step"], batch_df["backbone_grad_norm"], color="#2A9D8F", label="backbone")
            ax.set_title("Gradient Norms")
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Grad Norm")
            ax.legend()
            fig.tight_layout()
            grad_png = plots_dir / "grad_norm_curves.png"
            fig.savefig(grad_png, dpi=180)
            plt.close(fig)
            outputs["grad_norm_curves_png"] = grad_png

    return outputs


def write_cv_report_artifacts(
    out_dir: Path,
    *,
    summary_payload: dict[str, Any],
    fold_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    summary_csv = out_dir / "summary.csv"
    summary_row = {
        "dataset_version": summary_payload.get("dataset_version"),
        "folds": summary_payload.get("folds"),
        "device": summary_payload.get("device"),
        "pretrained": summary_payload.get("pretrained"),
        "weighted_sampling": summary_payload.get("weighted_sampling"),
    }
    summary_row.update(flatten_dict("candidate_", summary_payload.get("candidate_metrics_summary", {})))
    summary_row.update(flatten_dict("baseline_", summary_payload.get("baseline_metrics_summary", {})))
    summary_row.update(flatten_dict("delta_", summary_payload.get("delta_metrics_summary", {})))
    pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

    fold_metrics_csv = out_dir / "fold_metrics.csv"
    pd.DataFrame(fold_rows).to_csv(fold_metrics_csv, index=False)

    outputs = {
        "summary_json": summary_json,
        "summary_csv": summary_csv,
        "fold_metrics_csv": fold_metrics_csv,
    }

    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        plt = _load_plt()
        candidate_key = "candidate_primary_score"
        baseline_key = "baseline_primary_score"
        delta_key = "delta_primary_score"
        if {candidate_key, baseline_key, "fold"} <= set(fold_df.columns):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(fold_df["fold"], fold_df[candidate_key], marker="o", label="candidate", color="#2A9D8F")
            ax.plot(fold_df["fold"], fold_df[baseline_key], marker="o", label="baseline", color="#264653")
            ax.set_title("CV Primary Score by Fold")
            ax.set_xlabel("Fold")
            ax.set_ylabel("Primary Score")
            ax.legend()
            fig.tight_layout()
            score_png = plots_dir / "cv_primary_scores.png"
            fig.savefig(score_png, dpi=180)
            plt.close(fig)
            outputs["cv_primary_scores_png"] = score_png

        if {delta_key, "fold"} <= set(fold_df.columns):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            colors = ["#2A9D8F" if value >= 0 else "#E76F51" for value in fold_df[delta_key].tolist()]
            ax.bar(fold_df["fold"], fold_df[delta_key], color=colors)
            ax.axhline(0.0, color="#444444", linewidth=1)
            ax.set_title("CV Primary Score Delta vs Baseline")
            ax.set_xlabel("Fold")
            ax.set_ylabel("Delta")
            fig.tight_layout()
            delta_png = plots_dir / "cv_primary_score_deltas.png"
            fig.savefig(delta_png, dpi=180)
            plt.close(fig)
            outputs["cv_primary_score_deltas_png"] = delta_png

    return outputs


def summarize_metric_distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row.get(key, 0.0)) for row in rows]
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    series = pd.Series(values, dtype="float64")
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)) if len(values) > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
    }


def build_metric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return diff_numeric_dict(current=current, previous=baseline)
