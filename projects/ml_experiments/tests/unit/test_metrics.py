from __future__ import annotations

import numpy as np

from birdsys.ml_experiments.common import HEADS, ID_TO_LABELS
from birdsys.ml_experiments.metrics import compute_multihead_metrics


def _empty_storage() -> dict[str, dict[str, list[int]]]:
    return {head: {"true": [], "pred": []} for head in HEADS}


def test_compute_multihead_metrics_tracks_per_class_and_aggregate_values() -> None:
    storage = _empty_storage()
    storage["behavior"] = {"true": [0, 0, 2, 2], "pred": [0, 2, 2, 2]}
    storage["substrate"] = {"true": [0, 2, 2], "pred": [0, 2, 1]}
    storage["stance"] = {"true": [], "pred": []}

    result = compute_multihead_metrics(storage=storage, id_to_labels=ID_TO_LABELS)

    assert result.summary_metrics["behavior_visible_support"] == 4.0
    assert result.summary_metrics["behavior_accuracy"] == 0.75
    assert result.summary_metrics["behavior_macro_f1"] < 1.0
    assert result.summary_metrics["stance_visible_support"] == 0.0
    assert result.aggregate_metrics["heads_with_support"] == 2.0
    assert result.aggregate_metrics["primary_score"] == result.aggregate_metrics["mean_macro_f1"]
    assert "behavior" in result.confusion_matrices

    behavior_rows = [row for row in result.per_class_metrics if row["head"] == "behavior"]
    assert any(row["label"] == ID_TO_LABELS["behavior"][0] and row["support"] == 2 for row in behavior_rows)
    assert all("precision" in row and "recall" in row and "f1" in row for row in behavior_rows)


def test_compute_multihead_metrics_excludes_absent_heads_from_primary_score() -> None:
    storage = _empty_storage()
    storage["behavior"] = {"true": [0, 0], "pred": [0, 1]}

    result = compute_multihead_metrics(storage=storage, id_to_labels=ID_TO_LABELS)

    expected_behavior_macro_f1 = result.summary_metrics["behavior_macro_f1"]
    assert np.isclose(result.aggregate_metrics["primary_score"], expected_behavior_macro_f1)
    assert result.aggregate_metrics["support_weighted_macro_f1"] == expected_behavior_macro_f1
