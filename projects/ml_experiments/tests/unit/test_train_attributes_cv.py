"""Purpose: Verify grouped cross-validation reporting and baseline comparison behavior"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import birdsys.ml_experiments.train_attributes_cv as cv_mod
from birdsys.ml_experiments.model_b_evaluation import EvaluationDiagnostics, EvaluationResult


def _empty_visible_counts() -> dict[str, dict[str, int]]:
    return {head: {label: 0 for label in cv_mod.ID_TO_LABELS[head]} for head in cv_mod.HEADS}


def _dummy_eval_result(checkpoint_path: Path, score: float) -> EvaluationResult:
    metrics = {}
    for head in cv_mod.HEADS:
        metrics[f"{head}_accuracy"] = score
        metrics[f"{head}_f1"] = score
    return EvaluationResult(
        checkpoint_path=str(checkpoint_path),
        artifact_mode="specialist_bundle" if checkpoint_path.is_dir() else "legacy_single_checkpoint",
        device="cpu",
        image_size=224,
        schema_version="annotation_schema_v2",
        supported_labels={head: cv_mod.ID_TO_LABELS[head][:] for head in cv_mod.HEADS},
        train_label_counts={head: {} for head in cv_mod.HEADS},
        metrics=metrics,
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


def test_train_attributes_cv_accepts_old_model_artifact_directory(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    artifact_dir = tmp_path / "served_model_b"
    artifact_dir.mkdir()
    output_dir = tmp_path / "cv_out"
    frame = pd.DataFrame(
        [
            {"group_id": "g0", "isbird": "yes", "readability": "readable", "specie": "correct", "behavior": "resting", "substrate": "water", "stance": "bipedal"},
            {"group_id": "g1", "isbird": "yes", "readability": "readable", "specie": "correct", "behavior": "resting", "substrate": "water", "stance": "bipedal"},
            {"group_id": "g2", "isbird": "yes", "readability": "readable", "specie": "correct", "behavior": "resting", "substrate": "water", "stance": "bipedal"},
            {"group_id": "g3", "isbird": "yes", "readability": "readable", "specie": "correct", "behavior": "resting", "substrate": "water", "stance": "bipedal"},
        ]
    )

    monkeypatch.setattr(cv_mod, "load_cfg", lambda path: cv_mod.TrainCfg("dummy", 224, 4, 0, 1, 1, 1e-3, 1e-4, 0.0, "cpu"))
    monkeypatch.setattr(cv_mod, "dataframe_from_split", lambda path, smoke: frame.copy())
    monkeypatch.setattr(cv_mod, "pick_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(cv_mod, "stable_fold_id", lambda group_id, folds: int(str(group_id)[-1]) % folds)
    monkeypatch.setattr(
        cv_mod,
        "train_model",
        lambda **_kwargs: SimpleNamespace(
            model=object(),
            supported_labels={head: cv_mod.ID_TO_LABELS[head][:] for head in cv_mod.HEADS},
            eval_visible_label_counts=_empty_visible_counts(),
            train_visible_label_counts=_empty_visible_counts(),
        ),
    )
    monkeypatch.setattr(cv_mod, "collect_visible_label_counts", lambda frame: _empty_visible_counts())
    monkeypatch.setattr(cv_mod, "summarize_labels", lambda frame: {"isbird": {"yes": int(len(frame))}})

    def _save_checkpoint(out_dir: Path, **_kwargs) -> Path:
        checkpoint = out_dir / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        return checkpoint

    monkeypatch.setattr(cv_mod, "save_checkpoint", _save_checkpoint)
    monkeypatch.setattr(
        cv_mod,
        "evaluate_checkpoint_on_frame",
        lambda checkpoint_path, frame, device, **_kwargs: _dummy_eval_result(
            checkpoint_path, 0.4 if Path(checkpoint_path) == artifact_dir else 0.8
        ),
    )
    assert cv_mod.main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--old-model-checkpoint",
            str(artifact_dir),
            "--output-dir",
            str(output_dir),
            "--folds",
            "2",
            "--data-root",
            str(tmp_path / "data_root"),
        ]
    ) == 0
    payload = json.loads((output_dir / "cv_report.json").read_text(encoding="utf-8"))
    assert payload["old_model_checkpoint"] == str(artifact_dir.resolve())
