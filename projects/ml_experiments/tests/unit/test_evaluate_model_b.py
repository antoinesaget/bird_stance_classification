from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from birdsys.ml_experiments import evaluate_model_b as cli_mod


class DummyResult:
    checkpoint_path = "/tmp/checkpoint.pt"
    artifact_mode = "legacy_single_checkpoint"
    device = "cpu"
    image_size = 224
    schema_version = "annotation_schema_v2"
    id_to_label = {head: [] for head in ["readability", "specie", "behavior", "substrate", "stance"]}
    supported_labels = {head: [] for head in ["readability", "specie", "behavior", "substrate", "stance"]}
    train_label_counts = {head: {} for head in ["readability", "specie", "behavior", "substrate", "stance"]}
    summary_metrics = {}
    aggregate_metrics = {"primary_score": 0.0, "mean_macro_f1": 0.0, "support_weighted_macro_f1": 0.0, "mean_balanced_accuracy": 0.0, "heads_with_support": 0.0, "total_visible_support": 0.0}
    per_head_metrics = {}
    per_class_metrics = []
    confusion_matrices = {}
    visible_label_counts = {}
    dataset_label_counts = {}
    prediction_frame = None
    comparison_to_baseline = None

    class _Diagnostics:
        def to_dict(self):
            return {}

    diagnostics = _Diagnostics()


def test_evaluate_model_b_uses_default_served_artifact_path(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    layout_root = tmp_path / "birds_home"
    artifact_path = layout_root / "black_winged_stilt" / "models" / "attributes" / "served" / "model_b" / "current"
    artifact_path.mkdir(parents=True)

    called = {}

    def _evaluate_checkpoint_on_dataset(**kwargs):
        called.update(kwargs)
        return DummyResult()

    monkeypatch.setattr(cli_mod, "evaluate_checkpoint_on_dataset", _evaluate_checkpoint_on_dataset)
    monkeypatch.setattr(
        cli_mod,
        "write_evaluation_report",
        lambda out_dir, result: {
            "summary_json": out_dir / "summary.json",
            "summary_csv": out_dir / "summary.csv",
            "per_class_metrics_csv": out_dir / "per_class_metrics.csv",
            "predictions_parquet": out_dir / "predictions.parquet",
        },
    )

    assert cli_mod.main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--data-home",
            str(layout_root),
            "--species-slug",
            "black_winged_stilt",
            "--output-dir",
            str(tmp_path / "eval_out"),
        ]
    ) == 0

    assert called["checkpoint_path"] == artifact_path.resolve()
    assert called["split"] == "test"
