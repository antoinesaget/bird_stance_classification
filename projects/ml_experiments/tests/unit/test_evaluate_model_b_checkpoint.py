from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from birdsys.ml_experiments.evaluate_model_b_checkpoint import main
from birdsys.ml_experiments.model_b_evaluation import EvaluationDiagnostics, EvaluationResult


def _dummy_result(checkpoint_path: Path) -> EvaluationResult:
    metrics = {
        "readability_accuracy": 0.8,
        "readability_f1": 0.8,
        "specie_accuracy": 0.8,
        "specie_f1": 0.8,
        "behavior_accuracy": 0.7,
        "behavior_f1": 0.7,
        "substrate_accuracy": 0.6,
        "substrate_f1": 0.6,
        "stance_accuracy": 0.5,
        "stance_f1": 0.5,
    }
    return EvaluationResult(
        checkpoint_path=str(checkpoint_path),
        artifact_mode="specialist_bundle",
        device="cpu",
        image_size=224,
        schema_version="annotation_schema_v2",
        supported_labels={head: [] for head in ["readability", "specie", "behavior", "substrate", "stance"]},
        train_label_counts={},
        metrics=metrics,
        confusion_matrices={
            "behavior": np.zeros((12, 12), dtype=np.int64),
            "substrate": np.zeros((5, 5), dtype=np.int64),
            "stance": np.zeros((4, 4), dtype=np.int64),
        },
        visible_label_counts={},
        dataset_label_counts={},
        diagnostics=EvaluationDiagnostics(
            rows_scored=0,
            unsupported_label_coercions={},
            unreadable_or_incorrect_suppressions={"behavior": 0, "substrate": 0, "stance": 0},
            stance_suppressions_non_relevant=0,
            prediction_label_counts={},
        ),
        prediction_frame=pd.DataFrame(),
    )


def test_evaluate_model_b_checkpoint_accepts_artifact_directory(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    artifact_dir = tmp_path / "served_current"
    artifact_dir.mkdir()
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        "birdsys.ml_experiments.evaluate_model_b_checkpoint.evaluate_checkpoint_on_dataset",
        lambda checkpoint_path, dataset_dir, split, batch_size, num_workers, device: _dummy_result(checkpoint_path),
    )
    assert main(
        [
            "--checkpoint",
            str(artifact_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--data-root",
            str(tmp_path / "data_root"),
        ]
    ) == 0
    payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["artifact_mode"] == "specialist_bundle"
    assert payload["checkpoint_path"] == str(artifact_dir.resolve())
