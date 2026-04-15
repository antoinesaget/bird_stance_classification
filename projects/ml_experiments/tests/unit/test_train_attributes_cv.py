from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import birdsys.ml_experiments.train_attributes_cv as cv_mod
from birdsys.ml_experiments.model_b_evaluation import EvaluationDiagnostics, EvaluationResult


def _empty_visible_counts() -> dict[str, dict[str, int]]:
    return {head: {label: 0 for label in cv_mod.ID_TO_LABELS[head]} for head in cv_mod.HEADS}


def _dummy_eval_result(checkpoint_path: Path, score: float) -> EvaluationResult:
    summary_metrics = {}
    for head in cv_mod.HEADS:
        summary_metrics[f"{head}_accuracy"] = score
        summary_metrics[f"{head}_visible_support"] = 1.0
        summary_metrics[f"{head}_macro_f1"] = score
        summary_metrics[f"{head}_balanced_accuracy"] = score
    return EvaluationResult(
        checkpoint_path=str(checkpoint_path),
        artifact_mode="specialist_bundle" if checkpoint_path.is_dir() else "legacy_single_checkpoint",
        device="cpu",
        image_size=224,
        schema_version="annotation_schema_v2",
        id_to_label={head: cv_mod.ID_TO_LABELS[head][:] for head in cv_mod.HEADS},
        supported_labels={head: cv_mod.ID_TO_LABELS[head][:] for head in cv_mod.HEADS},
        train_label_counts={head: {} for head in cv_mod.HEADS},
        summary_metrics=summary_metrics,
        aggregate_metrics={
            "primary_score": score,
            "mean_macro_f1": score,
            "support_weighted_macro_f1": score,
            "mean_balanced_accuracy": score,
            "heads_with_support": float(len(cv_mod.HEADS)),
            "total_visible_support": float(len(cv_mod.HEADS)),
        },
        per_head_metrics={head: {"macro_f1": score} for head in cv_mod.HEADS},
        per_class_metrics=[],
        confusion_matrices={
            head: np.zeros((len(cv_mod.ID_TO_LABELS[head]), len(cv_mod.ID_TO_LABELS[head])), dtype=np.int64)
            for head in cv_mod.CONFUSION_HEADS
        },
        visible_label_counts=_empty_visible_counts(),
        dataset_label_counts={},
        diagnostics=EvaluationDiagnostics(
            rows_scored=0,
            unsupported_label_coercions={head: {} for head in cv_mod.HEADS},
            unreadable_or_incorrect_suppressions={"behavior": 0, "substrate": 0, "stance": 0},
            stance_suppressions_non_relevant=0,
            prediction_label_counts=_empty_visible_counts(),
        ),
        prediction_frame=pd.DataFrame(),
    )


def test_train_attributes_cv_writes_fold_and_summary_reports(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    baseline_dir = tmp_path / "served_model_b"
    baseline_dir.mkdir()
    output_dir = tmp_path / "cv_out"
    frame = pd.DataFrame(
        [
            {"image_id": "img-0", "group_id": "g0", "crop_path": __file__, "isbird": "yes", "readability": "readable", "specie": "correct", "behavior": "resting", "substrate": "water", "stance": "bipedal"},
            {"image_id": "img-1", "group_id": "g1", "crop_path": __file__, "isbird": "yes", "readability": "readable", "specie": "correct", "behavior": "resting", "substrate": "water", "stance": "bipedal"},
            {"image_id": "img-2", "group_id": "g2", "crop_path": __file__, "isbird": "yes", "readability": "readable", "specie": "correct", "behavior": "resting", "substrate": "water", "stance": "bipedal"},
            {"image_id": "img-3", "group_id": "g3", "crop_path": __file__, "isbird": "yes", "readability": "readable", "specie": "correct", "behavior": "resting", "substrate": "water", "stance": "bipedal"},
        ]
    )
    folds = pd.DataFrame(
        [
            {"image_id": "img-0", "fold_id": 0},
            {"image_id": "img-1", "fold_id": 1},
            {"image_id": "img-2", "fold_id": 0},
            {"image_id": "img-3", "fold_id": 1},
        ]
    )
    frame.to_parquet(dataset_dir / "train_pool.parquet", index=False)
    folds.to_parquet(dataset_dir / "fold_assignments.parquet", index=False)

    monkeypatch.setattr(cv_mod, "load_cfg", lambda path: cv_mod.TrainCfg("dummy", 64, 2, 0, 1, 1, 1e-3, 1e-4, 0.0, "cpu"))
    monkeypatch.setattr(cv_mod, "pick_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(
        cv_mod,
        "train_model",
        lambda **kwargs: SimpleNamespace(
            model=object(),
            supported_labels={head: cv_mod.ID_TO_LABELS[head][:] for head in cv_mod.HEADS},
            eval_visible_label_counts=_empty_visible_counts(),
            train_visible_label_counts=_empty_visible_counts(),
        ),
    )

    def _save_checkpoint(out_dir: Path, **kwargs) -> Path:
        checkpoint = out_dir / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        return checkpoint

    monkeypatch.setattr(cv_mod, "save_checkpoint", _save_checkpoint)
    monkeypatch.setattr(
        cv_mod,
        "evaluate_checkpoint_on_frame",
        lambda checkpoint_path, frame, device, baseline_checkpoint_path=None, **kwargs: _dummy_eval_result(
            checkpoint_path,
            0.8 if Path(checkpoint_path).suffix == ".pt" else 0.4,
        ),
    )

    assert cv_mod.main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--old-model-checkpoint",
            str(baseline_dir),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "fold_metrics.csv").exists()
    assert (output_dir / "fold_01" / "candidate_eval" / "summary.json").exists()


def test_load_train_pool_with_folds_overwrites_existing_fold_id_column(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    frame = pd.DataFrame(
        [
            {
                "image_id": "img-0",
                "group_id": "g0",
                "crop_path": __file__,
                "isbird": "yes",
                "readability": "readable",
                "specie": "correct",
                "behavior": "resting",
                "substrate": "water",
                "stance": "bipedal",
                "fold_id": 99,
            },
            {
                "image_id": "img-1",
                "group_id": "g1",
                "crop_path": __file__,
                "isbird": "yes",
                "readability": "readable",
                "specie": "correct",
                "behavior": "resting",
                "substrate": "water",
                "stance": "bipedal",
                "fold_id": 99,
            },
        ]
    )
    assignments = pd.DataFrame(
        [
            {"image_id": "img-0", "fold_id": 0},
            {"image_id": "img-1", "fold_id": 1},
        ]
    )
    frame.to_parquet(dataset_dir / "train_pool.parquet", index=False)
    assignments.to_parquet(dataset_dir / "fold_assignments.parquet", index=False)

    merged = cv_mod.load_train_pool_with_folds(dataset_dir=dataset_dir, smoke=False)

    assert list(merged["fold_id"]) == [0, 1]
