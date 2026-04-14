"""Purpose: Provide shared metric helpers used by Model B training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import CONFUSION_HEADS, HEADS


@dataclass(frozen=True)
class MultiHeadMetrics:
    summary_metrics: dict[str, float]
    aggregate_metrics: dict[str, float]
    per_head_metrics: dict[str, dict[str, float]]
    per_class_metrics: list[dict[str, Any]]
    confusion_matrices: dict[str, np.ndarray]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def class_supports(y_true: np.ndarray, num_classes: int) -> np.ndarray:
    supports = np.zeros((num_classes,), dtype=np.int64)
    for value in y_true:
        if 0 <= value < num_classes:
            supports[int(value)] += 1
    return supports


def predicted_counts(y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.zeros((num_classes,), dtype=np.int64)
    for value in y_pred:
        if 0 <= value < num_classes:
            counts[int(value)] += 1
    return counts


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    *,
    ignore_absent_classes: bool = False,
) -> float:
    metrics = compute_head_metrics(
        head="__macro_f1__",
        y_true=y_true,
        y_pred=y_pred,
        labels=[str(idx) for idx in range(num_classes)],
    )
    if ignore_absent_classes:
        supported = [row["f1"] for row in metrics.per_class_metrics if int(row["support"]) > 0]
        return float(np.mean(supported)) if supported else 0.0
    return float(metrics.summary_metrics["macro_f1"])


@dataclass(frozen=True)
class HeadMetrics:
    summary_metrics: dict[str, float]
    per_class_metrics: list[dict[str, Any]]
    confusion_matrix: np.ndarray


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1e-12:
        return 0.0
    return float(numerator) / float(denominator)


def compute_head_metrics(
    *,
    head: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> HeadMetrics:
    num_classes = len(labels)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    supports = class_supports(y_true, num_classes)
    predictions = predicted_counts(y_pred, num_classes)

    total_support = int(supports.sum())
    per_class_metrics: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for class_idx, label in enumerate(labels):
        tp = float(cm[class_idx, class_idx])
        fp = float(cm[:, class_idx].sum() - tp)
        fn = float(cm[class_idx, :].sum() - tp)
        support = int(supports[class_idx])
        predicted = int(predictions[class_idx])
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        if support > 0:
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
            weighted_precision += precision * support
            weighted_recall += recall * support
            weighted_f1 += f1 * support
        per_class_metrics.append(
            {
                "head": head,
                "label": label,
                "label_id": int(class_idx),
                "support": support,
                "predicted_count": predicted,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    accuracy = _safe_div(float((y_true == y_pred).sum()), float(len(y_true))) if len(y_true) else 0.0
    summary_metrics = {
        "visible_support": float(total_support),
        "predicted_rows": float(len(y_pred)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(np.mean(recall_values)) if recall_values else 0.0,
        "macro_precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "macro_recall": float(np.mean(recall_values)) if recall_values else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "weighted_precision": _safe_div(weighted_precision, total_support),
        "weighted_recall": _safe_div(weighted_recall, total_support),
        "weighted_f1": _safe_div(weighted_f1, total_support),
    }
    return HeadMetrics(
        summary_metrics=summary_metrics,
        per_class_metrics=per_class_metrics,
        confusion_matrix=cm,
    )


def compute_multihead_metrics(
    *,
    storage: dict[str, dict[str, list[int]]],
    id_to_labels: dict[str, list[str]],
    heads: list[str] | None = None,
    confusion_heads: list[str] | None = None,
) -> MultiHeadMetrics:
    active_heads = heads or list(HEADS)
    active_confusion_heads = set(confusion_heads or list(CONFUSION_HEADS))

    summary_metrics: dict[str, float] = {}
    aggregate_metrics: dict[str, float] = {}
    per_head_metrics: dict[str, dict[str, float]] = {}
    per_class_metrics: list[dict[str, Any]] = []
    confusion_matrices: dict[str, np.ndarray] = {}

    macro_f1_values: list[float] = []
    balanced_accuracy_values: list[float] = []
    supports: list[float] = []
    weighted_macro_f1_sum = 0.0

    for head in active_heads:
        labels = id_to_labels[head]
        y_true = np.asarray(storage[head]["true"], dtype=np.int64)
        y_pred = np.asarray(storage[head]["pred"], dtype=np.int64)
        head_metrics = compute_head_metrics(head=head, y_true=y_true, y_pred=y_pred, labels=labels)
        per_head_metrics[head] = head_metrics.summary_metrics
        per_class_metrics.extend(head_metrics.per_class_metrics)
        if head in active_confusion_heads:
            confusion_matrices[head] = head_metrics.confusion_matrix

        support = float(head_metrics.summary_metrics["visible_support"])
        for metric_name, value in head_metrics.summary_metrics.items():
            summary_metrics[f"{head}_{metric_name}"] = float(value)
        if support > 0:
            macro_f1_values.append(float(head_metrics.summary_metrics["macro_f1"]))
            balanced_accuracy_values.append(float(head_metrics.summary_metrics["balanced_accuracy"]))
            supports.append(support)
            weighted_macro_f1_sum += float(head_metrics.summary_metrics["macro_f1"]) * support

    total_visible_support = float(sum(supports))
    mean_macro_f1 = float(np.mean(macro_f1_values)) if macro_f1_values else 0.0
    support_weighted_macro_f1 = _safe_div(weighted_macro_f1_sum, total_visible_support)
    mean_balanced_accuracy = float(np.mean(balanced_accuracy_values)) if balanced_accuracy_values else 0.0
    aggregate_metrics.update(
        {
            "primary_score": mean_macro_f1,
            "mean_macro_f1": mean_macro_f1,
            "support_weighted_macro_f1": support_weighted_macro_f1,
            "mean_balanced_accuracy": mean_balanced_accuracy,
            "heads_with_support": float(len(macro_f1_values)),
            "total_visible_support": total_visible_support,
        }
    )

    return MultiHeadMetrics(
        summary_metrics=summary_metrics,
        aggregate_metrics=aggregate_metrics,
        per_head_metrics=per_head_metrics,
        per_class_metrics=per_class_metrics,
        confusion_matrices=confusion_matrices,
    )
