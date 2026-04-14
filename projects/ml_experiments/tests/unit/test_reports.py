from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from birdsys.ml_experiments.reports import write_evaluation_report


def test_write_evaluation_report_emits_standard_artifacts(tmp_path: Path) -> None:
    result = SimpleNamespace(
        checkpoint_path=str(tmp_path / "checkpoint.pt"),
        artifact_mode="legacy_single_checkpoint",
        device="cpu",
        image_size=224,
        schema_version="annotation_schema_v2",
        id_to_label={"behavior": ["flying", "foraging"], "substrate": ["bare_ground", "water"], "stance": ["unipedal", "bipedal"], "readability": ["readable"], "specie": ["correct"]},
        supported_labels={"behavior": ["flying", "foraging"], "substrate": ["bare_ground", "water"], "stance": ["unipedal", "bipedal"], "readability": ["readable"], "specie": ["correct"]},
        train_label_counts={head: {} for head in ["behavior", "substrate", "stance", "readability", "specie"]},
        summary_metrics={"behavior_accuracy": 1.0},
        aggregate_metrics={"primary_score": 1.0, "mean_macro_f1": 1.0, "support_weighted_macro_f1": 1.0, "mean_balanced_accuracy": 1.0, "heads_with_support": 1.0, "total_visible_support": 2.0},
        per_head_metrics={"behavior": {"accuracy": 1.0}},
        per_class_metrics=[{"head": "behavior", "label": "flying", "support": 1, "predicted_count": 1, "tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}],
        confusion_matrices={"behavior": np.array([[1, 0], [0, 1]], dtype=np.int64)},
        visible_label_counts={head: {} for head in ["behavior", "substrate", "stance", "readability", "specie"]},
        dataset_label_counts={head: {} for head in ["behavior", "substrate", "stance", "readability", "specie"]},
        diagnostics=SimpleNamespace(to_dict=lambda: {"rows_scored": 2}),
        prediction_frame=pd.DataFrame([{"row_id": "row-1", "behavior_pred": "flying"}]),
        comparison_to_baseline=SimpleNamespace(
            baseline_checkpoint_path=str(tmp_path / "baseline.pt"),
            baseline_artifact_mode="legacy_single_checkpoint",
            delta_summary_metrics={"behavior_accuracy": 0.5},
            delta_aggregate_metrics={"primary_score": 0.5},
            to_dict=lambda: {"delta_summary_metrics": {"behavior_accuracy": 0.5}},
        ),
    )

    outputs = write_evaluation_report(tmp_path / "report", result)

    assert outputs["summary_json"].exists()
    assert outputs["summary_csv"].exists()
    assert outputs["per_class_metrics_csv"].exists()
    assert outputs["predictions_parquet"].exists()
    assert (tmp_path / "report" / "plots" / "behavior_confusion_matrix.png").exists()
    assert (tmp_path / "report" / "plots" / "metric_deltas_vs_baseline.png").exists()
