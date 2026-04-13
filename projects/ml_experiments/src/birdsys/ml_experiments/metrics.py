from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def class_supports(y_true: np.ndarray, num_classes: int) -> np.ndarray:
    supports = np.zeros((num_classes,), dtype=np.int64)
    for value in y_true:
        if 0 <= value < num_classes:
            supports[int(value)] += 1
    return supports


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    *,
    ignore_absent_classes: bool = False,
) -> float:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    supports = class_supports(y_true, num_classes)
    f1s = []
    for i in range(num_classes):
        if ignore_absent_classes and supports[i] <= 0:
            continue
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        if tp == 0 and (fp == 0 or fn == 0):
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(float(f1))
    return float(np.mean(f1s)) if f1s else 0.0
